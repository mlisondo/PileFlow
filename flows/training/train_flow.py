"""
flows/training/train_flow.py
============================

Training and generation utilities for mixed-resolution image-only PileFlow.

This module consumes generator image outputs through:

    flows.data.dataset.PileFlowDataset

PileFlow conditions only on three flattened image channels:

    81    contaminated neutral image, 9x9
    1296  charged-pileup image, 36x36
    1296  charged-LV image, 36x36

Total context dimension:

    81 + 1296 + 1296 = 2673

The flow generates only:

    81  neutral-LV 9x9 image

Outputs written under cfg.outdir:

    checkpoints/pileflow_best.pt
    checkpoints/pileflow_best_history.npz
    plots/pileflow_loss.png
    data/generated_jets.npz

Generated PileFlow images are inverse-standardized and clamped to nonnegative
pT, but no positive pixel threshold is applied here. Any detector-cell
threshold used for physics evaluation must be applied uniformly to every
method in comparison/observable_comparison.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..data.dataset import PileFlowDataset
from ..models.pileflow import (
    CHARGED_DIM,
    CHARGED_SIDE,
    CNNContextEncoder,
    CRTVelocityField,
    ContextEncoder,
    NEUTRAL_DIM,
    NEUTRAL_SIDE,
    N_CONTEXT,
    N_IMAGES,
    N_SCALARS,
    N_TARGET,
    TargetCFM,
    TargetPreprocessor,
)
from ..models.pileflow_attention import TransformerVelocityField
from ..models.pileflow_unet import UNetVelocityField
from ..models.pileflow_set_transformer import PatchSetVelocityField

FLOW_CONTRACT = "image-only-neutral9-charged36-v1"


class _EMA:
    """
    Exponential moving average of model parameters.

    Maintains a shadow copy of each trainable parameter:

        shadow_t = decay * shadow_{t-1} + (1 - decay) * param_t

    Usage
    -----
    ema = _EMA(model, decay=0.999)

    # after each optimizer.step():
    ema.update(model)

    # for validation / checkpointing with EMA weights:
    original = ema.apply(model)
    ...
    _EMA.restore(model, original)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                f"EMA decay must satisfy 0 <= decay < 1, got {decay}"
            )
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        """Update shadow parameters after an optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Swap model weights for EMA shadow weights.

        Returns the original weights so they can be restored afterwards.
        """
        original: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return original

    @staticmethod
    def restore(
        model: nn.Module,
        original: dict[str, torch.Tensor],
    ) -> None:
        """Restore the original (non-EMA) weights after apply()."""
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    """Load a PyTorch checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _validate_checkpoint_contract(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Confirm that a checkpoint uses the mixed-resolution 2673-to-81 contract.

    Previous PileFlow checkpoints using either the legacy 253-context,
    97-target contract or the image-only 243-context, 81-target contract
    cannot be loaded by this model.
    """
    saved_cfg = state.get("cfg")

    if not isinstance(saved_cfg, dict):
        raise ValueError(
            "PileFlow checkpoint does not contain valid configuration metadata."
        )

    # n_context is NOT checked against a hardcoded value because the CNN
    # context encoder produces a different (smaller) context dimension.
    # We check only the fixed physics dimensions.
    fixed_expected = {
        "n_target": N_TARGET,
        "n_scalars": N_SCALARS,
        "neutral_dim": NEUTRAL_DIM,
        "charged_dim": CHARGED_DIM,
    }

    fixed_actual = {
        "n_target": saved_cfg.get("n_target"),
        "n_scalars": saved_cfg.get("n_scalars"),
        "neutral_dim": saved_cfg.get("neutral_dim"),
        "charged_dim": saved_cfg.get("charged_dim"),
    }

    if fixed_actual != fixed_expected:
        raise ValueError(
            "Checkpoint is incompatible with mixed-resolution PileFlow.\n"
            f"Expected: {fixed_expected}\n"
            f"Found:    {fixed_actual}\n"
            "Older 253-to-97 and 243-to-81 checkpoints must be retrained."
        )

    if "n_context" not in saved_cfg:
        raise ValueError(
            "Checkpoint missing 'n_context' in saved cfg."
        )

    saved_contract = state.get("contract")

    if saved_contract is not None and saved_contract != FLOW_CONTRACT:
        raise ValueError(
            "Checkpoint contract mismatch: "
            f"expected {FLOW_CONTRACT!r}, found {saved_contract!r}."
        )

    return saved_cfg


def _batch_to_device(
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Move a PileFlowDataset batch to the requested device.

    Dataset order:
        neutral_lv, neutral_all, charged_pu, charged_lv
    """
    neutral_lv, neutral_all, charged_pu, charged_lv = [
        item.to(device) for item in batch
    ]

    return neutral_lv, neutral_all, charged_pu, charged_lv


def _make_model(
    cfg,
    dropout: float,
    device: torch.device,
    context_dim: int = N_CONTEXT,
) -> "CRTVelocityField | TransformerVelocityField | UNetVelocityField | PatchSetVelocityField":
    """Construct the PileFlow velocity-field model."""
    use_transformer = getattr(cfg, "use_transformer", False)
    use_unet = getattr(cfg, "use_unet", False)
    use_set_transformer = getattr(cfg, "use_set_transformer", False)

    if use_set_transformer:
        return PatchSetVelocityField(
            n_features=N_TARGET,
            context_dim=N_CONTEXT,
            d_model=getattr(cfg, "st_d_model", 256),
            n_heads=getattr(cfg, "st_n_heads", 4),
            n_layers=getattr(cfg, "st_n_layers", 6),
            n_inducing=getattr(cfg, "st_n_inducing", 0),
            time_emb_dim=cfg.flow_time_emb,
            dropout=dropout,
        ).to(device)

    if use_unet:
        return UNetVelocityField(
            n_features=N_TARGET,
            context_dim=N_CONTEXT,
            base_channels=getattr(cfg, "unet_base_channels", 64),
            time_emb_dim=cfg.flow_time_emb,
            dropout=dropout,
        ).to(device)

    if use_transformer:
        return TransformerVelocityField(
            n_features=N_TARGET,
            context_dim=N_CONTEXT,           # always flat context for transformer
            cnn_channels=getattr(cfg, "transformer_cnn_channels", 32),
            d_model=getattr(cfg, "transformer_d_model", 256),
            n_heads=getattr(cfg, "transformer_n_heads", 4),
            n_self_attn=getattr(cfg, "transformer_n_self_attn", 2),
            n_cross_attn=getattr(cfg, "transformer_n_cross_attn", 4),
            time_emb_dim=cfg.flow_time_emb,
            dropout=dropout,
        ).to(device)

    return CRTVelocityField(
        n_features=N_TARGET,
        context_dim=context_dim,
        hidden_dim=cfg.flow_hidden,
        n_blocks=cfg.flow_blocks,
        time_emb_dim=cfg.flow_time_emb,
        dropout=dropout,
    ).to(device)


def _checkpoint_payload(
    model: "CRTVelocityField | TransformerVelocityField | UNetVelocityField",
    ctx_enc: "ContextEncoder | CNNContextEncoder",
    tgt_prep: TargetPreprocessor,
    cfg,
    dropout: float,
    ema: "_EMA | None" = None,
    context_dim: int = N_CONTEXT,
) -> dict[str, Any]:
    """Build the checkpoint payload saved during training."""
    use_transformer = getattr(cfg, "use_transformer", False)
    use_unet = getattr(cfg, "use_unet", False)
    use_set_transformer = getattr(cfg, "use_set_transformer", False)

    # All drastic architectures use the flat context encoder
    if use_transformer or use_unet or use_set_transformer:
        enc_type = "flat"
    else:
        enc_type = "cnn" if getattr(cfg, "use_cnn_context", False) else "flat"

    if use_set_transformer:
        vf_type = "set_transformer"
    elif use_unet:
        vf_type = "unet"
    elif use_transformer:
        vf_type = "transformer"
    else:
        vf_type = "crt"

    return {
        "contract": FLOW_CONTRACT,
        "model": model.state_dict(),
        "ctx_enc": ctx_enc.state_dict(),
        "tgt_prep": tgt_prep.state_dict(),
        "cfg": {
            "n_context": context_dim,
            "n_target": N_TARGET,
            "n_images": N_IMAGES,
            "n_scalars": N_SCALARS,
            "n_img": NEUTRAL_DIM,
            "neutral_side": NEUTRAL_SIDE,
            "neutral_dim": NEUTRAL_DIM,
            "charged_side": CHARGED_SIDE,
            "charged_dim": CHARGED_DIM,
            "channel_order": [
                "ch_neutral_all_raw",
                "ch_charged_pu",
                "ch_charged_lv",
            ],
            "channel_shapes": {
                "ch_neutral_all_raw": [NEUTRAL_SIDE, NEUTRAL_SIDE],
                "ch_charged_pu": [CHARGED_SIDE, CHARGED_SIDE],
                "ch_charged_lv": [CHARGED_SIDE, CHARGED_SIDE],
            },
            "target_key": "ch_neutral_lv",
            "target_shape": [NEUTRAL_SIDE, NEUTRAL_SIDE],
            "flow_hidden": cfg.flow_hidden,
            "flow_blocks": cfg.flow_blocks,
            "flow_time_emb": cfg.flow_time_emb,
            "flow_sigma_min": cfg.flow_sigma_min,
            "flow_dropout": dropout,
            "use_log_pt": getattr(cfg, "use_log_pt", False),
            "use_ema": getattr(cfg, "use_ema", False),
            "ema_decay": getattr(cfg, "ema_decay", 0.999),
            "pt_threshold": getattr(cfg, "pt_threshold", 0.0),
            "ctx_enc_type": enc_type,
            "cnn_neutral_feat": getattr(cfg, "cnn_neutral_feat", 128),
            "cnn_charged_feat": getattr(cfg, "cnn_charged_feat", 256),
            "decode_postprocessing": "clamp-nonnegative-no-threshold",
            # Velocity field type and architecture metadata
            "velocity_field_type": vf_type,
            # Set-transformer metadata
            "st_d_model": getattr(cfg, "st_d_model", 256),
            "st_n_heads": getattr(cfg, "st_n_heads", 4),
            "st_n_layers": getattr(cfg, "st_n_layers", 6),
            "st_n_inducing": getattr(cfg, "st_n_inducing", 0),
            # U-Net metadata
            "unet_base_channels": getattr(cfg, "unet_base_channels", 64),
            # Transformer metadata
            "transformer_d_model": getattr(cfg, "transformer_d_model", 256),
            "transformer_n_heads": getattr(cfg, "transformer_n_heads", 4),
            "transformer_n_self_attn": getattr(cfg, "transformer_n_self_attn", 2),
            "transformer_n_cross_attn": getattr(cfg, "transformer_n_cross_attn", 4),
            "transformer_cnn_channels": getattr(cfg, "transformer_cnn_channels", 32),
        },
        "ema_shadow": {
            name: tensor.clone()
            for name, tensor in ema.shadow.items()
        } if ema is not None else None,
    }


def train_pileflow(
    npz_path: str,
    flow_ckpt: str,
    cfg,
) -> "CRTVelocityField | TransformerVelocityField":
    """
    Train mixed-resolution image-only PileFlow.

    Parameters
    ----------
    npz_path:
        Path to the generator image/constituent .npz file.

    flow_ckpt:
        Path where the best PileFlow checkpoint will be saved.

    cfg:
        Config object containing training hyperparameters and runtime options.

    Returns
    -------
    CRTVelocityField
        Trained velocity-field model with the best validation weights loaded.
    """
    if cfg.flow_epochs <= 0:
        raise ValueError(
            "cfg.flow_epochs must be positive when training PileFlow."
        )

    if cfg.flow_batch <= 0:
        raise ValueError("cfg.flow_batch must be positive.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    ckpt_dir = Path(flow_ckpt).expanduser().resolve().parent
    plot_dir = Path(cfg.outdir) / "plots"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset = PileFlowDataset(
        npz_path=npz_path,
        max_n=getattr(cfg, "max_jets", None),
    )

    n_total = len(dataset)

    if n_total < 2:
        raise ValueError(
            "Need at least 2 jets to train and validate PileFlow."
        )

    n_train = int(0.9 * n_total)
    n_train = max(1, min(n_train, n_total - 1))
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print(f"  [flow] Train / Val: {n_train:,} / {n_val:,}")

    # Fit normalization statistics using training rows only.
    train_indices = train_ds.indices

    neutral_all_train = dataset.neutral_all_9x9[train_indices]
    charged_pu_train = dataset.charged_pu_36x36[train_indices]
    charged_lv_train = dataset.charged_lv_36x36[train_indices]
    neutral_lv_train = dataset.neutral_lv[train_indices]

    use_log_pt = getattr(cfg, "use_log_pt", False)
    use_minibatch_ot = getattr(cfg, "use_minibatch_ot", False)
    use_ema = getattr(cfg, "use_ema", False)
    ema_decay = getattr(cfg, "ema_decay", 0.999)
    use_transformer = getattr(cfg, "use_transformer", False)
    use_unet = getattr(cfg, "use_unet", False)
    use_set_transformer = getattr(cfg, "use_set_transformer", False)
    use_cnn_context = getattr(cfg, "use_cnn_context", False)
    cnn_neutral_feat = getattr(cfg, "cnn_neutral_feat", 128)
    cnn_charged_feat = getattr(cfg, "cnn_charged_feat", 256)

    # All drastic architectures split context internally.
    # Use the flat ContextEncoder so the training loop always passes a 2673-d
    # context vector; the model splits it back to images internally.
    if use_transformer or use_unet or use_set_transformer or not use_cnn_context:
        ctx_enc = ContextEncoder(
            neutral_dim=NEUTRAL_DIM,
            charged_dim=CHARGED_DIM,
        )
        actual_context_dim = N_CONTEXT
    else:
        ctx_enc = CNNContextEncoder(
            neutral_feat_dim=cnn_neutral_feat,
            charged_feat_dim=cnn_charged_feat,
        )
        actual_context_dim = ctx_enc.context_dim

    tgt_prep = TargetPreprocessor(
        n_img=N_TARGET,
        use_log_pt=use_log_pt,
    )


    ctx_enc.fit(
        neutral_all_train,
        charged_pu_train,
        charged_lv_train,
    )

    tgt_prep.fit(neutral_lv_train)

    ctx_enc = ctx_enc.to(device)
    tgt_prep = tgt_prep.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.flow_batch,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.flow_batch,
        shuffle=False,
        num_workers=0,
    )

    dropout = getattr(cfg, "flow_dropout", 0.1)

    model = _make_model(
        cfg=cfg,
        dropout=dropout,
        device=device,
        context_dim=actual_context_dim,
    )

    cfm = TargetCFM(
        sigma_min=cfg.flow_sigma_min,
        use_minibatch_ot=use_minibatch_ot,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.flow_lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.flow_epochs,
    )

    ema = _EMA(model, decay=ema_decay) if use_ema else None

    print(f"  [flow] Parameters  : {model.count_parameters():,}")
    if use_set_transformer:
        d = getattr(cfg, "st_d_model", 256)
        nh = getattr(cfg, "st_n_heads", 4)
        nl = getattr(cfg, "st_n_layers", 6)
        ni = getattr(cfg, "st_n_inducing", 0)
        attn_type = f"ISAB(m={ni})" if ni > 0 else "SAB (global O(N²))"
        print(
            f"  [flow] Architecture: PatchSetVelocityField "
            f"(d={d}, heads={nh}, layers={nl}, attn={attn_type})"
        )
        print(
            f"  [flow] Tokens      : 81 patch tokens — each: z_t(1) + neutral(1) "
            f"+ pu_patch(16) + lv_patch(16) + η,φ(2) → Linear → {d}-d"
        )
        print(
            f"  [flow] Pos enc     : explicit (η,φ) in raw features + "
            f"sinusoidal 2D PE added after projection"
        )
    elif use_unet:
        C = getattr(cfg, "unet_base_channels", 64)
        print(
            f"  [flow] Architecture: UNetVelocityField "
            f"(base_channels={C}, encoder C→2C→4C, bottleneck 8C, skip@9×9)"
        )
        print(
            f"  [flow] Context     : flat {NEUTRAL_DIM}+{CHARGED_DIM}+{CHARGED_DIM}={actual_context_dim}-d "
            f"→ charged 36×36→18×18→9×9, neutral+z_t merged at 9×9"
        )
    elif use_transformer:
        d_model = getattr(cfg, "transformer_d_model", 256)
        n_heads = getattr(cfg, "transformer_n_heads", 4)
        n_sa = getattr(cfg, "transformer_n_self_attn", 2)
        n_ca = getattr(cfg, "transformer_n_cross_attn", 4)
        n_cnn = getattr(cfg, "transformer_cnn_channels", 32)
        print(
            f"  [flow] Architecture: TransformerVelocityField "
            f"(d={d_model}, heads={n_heads}, self-attn={n_sa}, cross-attn={n_ca}, cnn-ch={n_cnn})"
        )
        print(
            f"  [flow] Context     : flat {NEUTRAL_DIM}+{CHARGED_DIM}+{CHARGED_DIM}={actual_context_dim}-d "
            f"→ 243 tokens (81×3 channels, CNN inside model)"
        )
    elif use_cnn_context:
        print(
            f"  [flow] Architecture: CRTVelocityField "
            f"({cfg.flow_blocks} blocks × {cfg.flow_hidden} hidden)"
        )
        print(
            f"  [flow] Context enc : CNN "
            f"(neutral {cnn_neutral_feat}-d + "
            f"charged-PU {cnn_charged_feat}-d + "
            f"charged-LV {cnn_charged_feat}-d = {actual_context_dim}-d)"
        )
    else:
        print(
            f"  [flow] Architecture: CRTVelocityField "
            f"({cfg.flow_blocks} blocks × {cfg.flow_hidden} hidden)"
        )
        print(
            f"  [flow] Context enc : flat "
            f"({NEUTRAL_DIM} neutral + {CHARGED_DIM} charged-PU + "
            f"{CHARGED_DIM} charged-LV = {actual_context_dim}-d)"
        )
    print(
        f"  [flow] Target dim  : {N_TARGET} "
        f"({NEUTRAL_SIDE}×{NEUTRAL_SIDE} neutral-LV pixels)"
    )
    print(f"  [flow] log(1+pT)   : {use_log_pt}")
    print(f"  [flow] minibatch OT: {use_minibatch_ot}")
    print(f"  [flow] EMA decay   : {ema_decay if use_ema else 'disabled'}")
    print()

    best_val = float("inf")
    history = {"train": [], "val": []}
    patience = getattr(cfg, "flow_patience", 60)
    no_improve = 0

    print(
        f"  {'Epoch':>5}  {'Train':>12}  {'Val':>12}  {'Time':>8}"
    )
    print("  " + "-" * 46)

    for epoch in range(1, cfg.flow_epochs + 1):
        start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            neutral_lv, neutral_all, charged_pu, charged_lv = (
                _batch_to_device(batch, device)
            )

            context = ctx_enc(
                neutral_all,
                charged_pu,
                charged_lv,
            )

            x1 = tgt_prep.encode(neutral_lv)
            t, zt, ut = cfm.sample_training_pair(x1)

            optimizer.zero_grad()

            predicted_velocity = model(
                t,
                zt,
                context,
            )

            loss = nn.functional.mse_loss(
                predicted_velocity,
                ut,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            if ema is not None:
                ema.update(model)

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate using EMA weights when available for a more stable criterion.
        if ema is not None:
            _orig = ema.apply(model)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                neutral_lv, neutral_all, charged_pu, charged_lv = (
                    _batch_to_device(batch, device)
                )

                context = ctx_enc(
                    neutral_all,
                    charged_pu,
                    charged_lv,
                )

                x1 = tgt_prep.encode(neutral_lv)
                t, zt, ut = cfm.sample_training_pair(x1)

                predicted_velocity = model(
                    t,
                    zt,
                    context,
                )

                val_loss += nn.functional.mse_loss(
                    predicted_velocity,
                    ut,
                ).item()

        val_loss /= len(val_loader)

        if ema is not None:
            _EMA.restore(model, _orig)

        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < best_val
        marker = " *" if improved else ""
        elapsed = time.time() - start

        print(
            f"  {epoch:>5}  {train_loss:>12.6f}  "
            f"{val_loss:>12.6f}  {elapsed:>7.1f}s{marker}"
        )

        if improved:
            best_val = val_loss
            no_improve = 0

            torch.save(
                _checkpoint_payload(
                    model=model,
                    ctx_enc=ctx_enc,
                    tgt_prep=tgt_prep,
                    cfg=cfg,
                    dropout=dropout,
                    ema=ema,
                    context_dim=actual_context_dim,
                ),
                flow_ckpt,
            )
        else:
            no_improve += 1

            if patience > 0 and no_improve >= patience:
                print(
                    f"\n  [flow] Early stop at epoch {epoch} "
                    f"(no validation improvement for {patience} epochs)"
                )
                break

    checkpoint_path = Path(flow_ckpt)
    history_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_history.npz"
    )

    np.savez(
        history_path,
        train=np.asarray(history["train"], dtype=np.float32),
        val=np.asarray(history["val"], dtype=np.float32),
    )

    print(f"\n  [flow] History   -> {history_path}")

    _plot_loss_curve(
        history,
        save_path=str(plot_dir / "pileflow_loss.png"),
        title=(
            "Mixed-resolution image-only PileFlow training - "
            "flow-matching MSE"
        ),
    )

    print(f"  [flow] Best val  : {best_val:.6f}  ->  {flow_ckpt}")

    state = _load_checkpoint(flow_ckpt, device)
    _validate_checkpoint_contract(state)
    model.load_state_dict(state["model"])

    return model


def generate_and_save(
    npz_path: str,
    flow_ckpt: str,
    cfg,
    n_steps: int = 100,
    out_dir: str | None = None,
    eval_npz: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Load a PileFlow checkpoint and generate neutral-LV predictions.

    For every conditioning input, generate ``cfg.eval_samples`` independent
    decoded flow samples.

    Two prediction modes are retained:

        neutral_lv_pred_single
            The first generated sample, Y^(1).

        neutral_lv_pred_mean
            The pixelwise mean of all N decoded samples:

                mean_prediction = (1 / N) * sum_s prediction_s

    ``neutral_lv_pred`` remains an alias of ``neutral_lv_pred_mean`` so that
    existing observable-evaluation code continues to use the mean prediction.

    For ``eval_samples=1``:

        neutral_lv_pred_single
        == neutral_lv_pred_mean
        == neutral_lv_pred
    """
    if n_steps <= 0:
        raise ValueError(
            f"n_steps must be positive, got {n_steps}"
        )

    eval_samples = int(
        getattr(
            cfg,
            "eval_samples",
            1,
        )
    )

    if eval_samples <= 0:
        raise ValueError(
            f"eval_samples must be positive, got {eval_samples}"
        )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    out_dir = out_dir or os.path.join(
        cfg.outdir,
        "data",
    )

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    state = _load_checkpoint(
        flow_ckpt,
        device,
    )

    saved_cfg = _validate_checkpoint_contract(
        state
    )

    vf_type = saved_cfg.get("velocity_field_type", "crt")
    if vf_type == "set_transformer":
        model = PatchSetVelocityField(
            n_features=saved_cfg["n_target"],
            context_dim=saved_cfg["n_context"],
            d_model=saved_cfg.get("st_d_model", 256),
            n_heads=saved_cfg.get("st_n_heads", 4),
            n_layers=saved_cfg.get("st_n_layers", 6),
            n_inducing=saved_cfg.get("st_n_inducing", 0),
            time_emb_dim=saved_cfg.get("flow_time_emb", 64),
            dropout=0.0,
        ).to(device)
    elif vf_type == "unet":
        model = UNetVelocityField(
            n_features=saved_cfg["n_target"],
            context_dim=saved_cfg["n_context"],
            base_channels=saved_cfg.get("unet_base_channels", 64),
            time_emb_dim=saved_cfg.get("flow_time_emb", 64),
            dropout=0.0,
        ).to(device)
    elif vf_type == "transformer":
        model = TransformerVelocityField(
            n_features=saved_cfg["n_target"],
            context_dim=saved_cfg["n_context"],
            cnn_channels=saved_cfg.get("transformer_cnn_channels", 32),
            d_model=saved_cfg.get("transformer_d_model", 256),
            n_heads=saved_cfg.get("transformer_n_heads", 4),
            n_self_attn=saved_cfg.get("transformer_n_self_attn", 2),
            n_cross_attn=saved_cfg.get("transformer_n_cross_attn", 4),
            time_emb_dim=saved_cfg.get("flow_time_emb", 64),
            dropout=0.0,
        ).to(device)
    else:
        model = CRTVelocityField(
            n_features=saved_cfg["n_target"],
            context_dim=saved_cfg["n_context"],
            hidden_dim=saved_cfg["flow_hidden"],
            n_blocks=saved_cfg["flow_blocks"],
            time_emb_dim=saved_cfg["flow_time_emb"],
            dropout=0.0,
        ).to(device)

    model.load_state_dict(
        state["model"]
    )

    model.eval()

    enc_type = saved_cfg.get("ctx_enc_type", "flat")
    if enc_type == "cnn":
        ctx_enc = CNNContextEncoder(
            neutral_feat_dim=saved_cfg.get("cnn_neutral_feat", 128),
            charged_feat_dim=saved_cfg.get("cnn_charged_feat", 256),
        ).to(device)
    else:
        ctx_enc = ContextEncoder(
            neutral_dim=saved_cfg["neutral_dim"],
            charged_dim=saved_cfg["charged_dim"],
        ).to(device)

    ctx_enc.load_state_dict(
        state["ctx_enc"]
    )

    ctx_enc.eval()

    tgt_prep = TargetPreprocessor(
        n_img=saved_cfg["n_target"],
        use_log_pt=saved_cfg.get("use_log_pt", False),
    ).to(device)

    tgt_prep.load_state_dict(
        state["tgt_prep"]
    )

    tgt_prep.eval()

    # Apply EMA weights to model if available in checkpoint.
    ema_shadow = state.get("ema_shadow")
    if ema_shadow is not None:
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name].to(device))

    cfm = TargetCFM(
        sigma_min=saved_cfg["flow_sigma_min"],
    )

    pt_threshold = getattr(cfg, "pt_threshold", 0.0)

    # Use a dedicated eval set when provided; fall back to train npz otherwise.
    generation_npz = eval_npz if eval_npz is not None else npz_path
    if eval_npz is not None:
        print(f"  [generate] Eval set : {eval_npz}")
    else:
        print(
            "  [generate] Eval set : (none provided — using training npz; "
            "pass --eval-npz for a held-out set)"
        )

    dataset = PileFlowDataset(
        npz_path=generation_npz,
        max_n=getattr(
            cfg,
            "max_jets",
            None,
        ),
    )

    if len(dataset) == 0:
        raise ValueError(
            "Cannot generate PileFlow predictions for an empty dataset."
        )

    loader = DataLoader(
        dataset,
        batch_size=getattr(
            cfg,
            "eval_batch",
            getattr(
                cfg,
                "flow_batch",
                512,
            ),
        ),
        shuffle=False,
        num_workers=0,
    )

    all_pred_single = []
    all_pred_mean = []

    all_true = []
    all_neutral_all = []
    all_charged_pu = []
    all_charged_lv = []

    print(
        "  [generate] Running mixed-resolution image-only "
        "PileFlow ODE integration ..."
    )

    print(
        f"  [generate] Samples per jet: {eval_samples}"
    )

    print(
        "  [generate] Single prediction: first decoded sample"
    )

    print(
        "  [generate] Mean prediction: pixelwise mean "
        "after decoding"
    )

    print(
        "  [generate] Downstream neutral_lv_pred: "
        "mean prediction"
    )

    print(
        "  [generate] Decode: clamp pT >= 0, "
        "no positive pixel threshold"
    )

    with torch.no_grad():
        for batch in loader:
            neutral_lv, neutral_all, charged_pu, charged_lv = (
                _batch_to_device(
                    batch,
                    device,
                )
            )

            context = ctx_enc(
                neutral_all,
                charged_pu,
                charged_lv,
            )

            single_image = None
            mean_image = None

            for sample_index in range(eval_samples):
                z_generated = cfm.generate(
                    model=model,
                    context=context,
                    n_steps=n_steps,
                    device=device,
                )

                sample_image = tgt_prep.decode(
                    z_generated,
                    pt_threshold=pt_threshold if pt_threshold > 0.0 else None,
                )

                if not torch.isfinite(
                    sample_image
                ).all():
                    raise RuntimeError(
                        "PileFlow generated non-finite decoded pixels "
                        f"in evaluation sample {sample_index + 1}."
                    )

                if torch.any(
                    sample_image < 0.0
                ):
                    raise RuntimeError(
                        "PileFlow decoder returned negative-pT pixels "
                        f"in evaluation sample {sample_index + 1}."
                    )

                if sample_index == 0:
                    # Preserve the first stochastic prediction as the
                    # fixed single-sample diagnostic.
                    #
                    # clone() is important because the running mean will
                    # subsequently be updated.
                    single_image = sample_image.clone()
                    mean_image = sample_image.clone()

                else:
                    # Online mean:
                    #
                    # mean_s = mean_{s-1}
                    #          + (sample_s - mean_{s-1}) / s
                    #
                    # sample_index is zero-based, so the number of samples
                    # included after this update is sample_index + 1.
                    mean_image = mean_image + (
                        sample_image - mean_image
                    ) / float(sample_index + 1)

            if single_image is None or mean_image is None:
                raise RuntimeError(
                    "PileFlow evaluation produced no samples."
                )

            if not torch.isfinite(
                single_image
            ).all():
                raise RuntimeError(
                    "Single-sample PileFlow prediction contains "
                    "non-finite pixels."
                )

            if not torch.isfinite(
                mean_image
            ).all():
                raise RuntimeError(
                    "Mean PileFlow prediction contains "
                    "non-finite pixels."
                )

            if torch.any(
                single_image < 0.0
            ):
                raise RuntimeError(
                    "Single-sample PileFlow prediction contains "
                    "negative-pT pixels."
                )

            if torch.any(
                mean_image < 0.0
            ):
                raise RuntimeError(
                    "Mean PileFlow prediction contains "
                    "negative-pT pixels."
                )

            all_pred_single.append(
                single_image.cpu().numpy()
            )

            all_pred_mean.append(
                mean_image.cpu().numpy()
            )

            all_true.append(
                neutral_lv.cpu().numpy()
            )

            all_neutral_all.append(
                neutral_all.cpu().numpy()
            )

            all_charged_pu.append(
                charged_pu.cpu().numpy()
            )

            all_charged_lv.append(
                charged_lv.cpu().numpy()
            )

    prediction_single = np.concatenate(
        all_pred_single,
        axis=0,
    )

    prediction_mean = np.concatenate(
        all_pred_mean,
        axis=0,
    )

    results = {
        # Backward-compatible prediction used by observable_comparison.py.
        "neutral_lv_pred": prediction_mean,

        # Explicit prediction modes for image diagnostics.
        "neutral_lv_pred_single": prediction_single,
        "neutral_lv_pred_mean": prediction_mean,

        # Evaluation truth and conditioning images.
        "neutral_lv_true": np.concatenate(
            all_true,
            axis=0,
        ),
        "neutral_all_9x9": np.concatenate(
            all_neutral_all,
            axis=0,
        ),
        "charged_pu_36x36": np.concatenate(
            all_charged_pu,
            axis=0,
        ),
        "charged_lv_36x36": np.concatenate(
            all_charged_lv,
            axis=0,
        ),

        # Scalar metadata saved in the NPZ file.
        "eval_samples": np.asarray(
            eval_samples,
            dtype=np.int64,
        ),
    }

    expected_rows = len(dataset)

    n_generated = results[
        "neutral_lv_pred_mean"
    ].shape[0]

    if n_generated != expected_rows:
        raise RuntimeError(
            "Generated prediction row count does not match "
            "the input dataset: "
            f"generated={n_generated}, dataset={expected_rows}"
        )

    expected_shapes = {
        "neutral_lv_pred": (
            expected_rows,
            N_TARGET,
        ),
        "neutral_lv_pred_single": (
            expected_rows,
            N_TARGET,
        ),
        "neutral_lv_pred_mean": (
            expected_rows,
            N_TARGET,
        ),
        "neutral_lv_true": (
            expected_rows,
            N_TARGET,
        ),
        "neutral_all_9x9": (
            expected_rows,
            NEUTRAL_DIM,
        ),
        "charged_pu_36x36": (
            expected_rows,
            CHARGED_DIM,
        ),
        "charged_lv_36x36": (
            expected_rows,
            CHARGED_DIM,
        ),
    }

    for key, expected_shape in expected_shapes.items():
        array = results[key]

        if array.shape != expected_shape:
            raise RuntimeError(
                f"Generated output {key!r} has shape "
                f"{array.shape}; expected {expected_shape}."
            )

        if not np.isfinite(
            array
        ).all():
            raise RuntimeError(
                f"Generated output {key!r} contains "
                "non-finite values."
            )

    if np.any(
        results["neutral_lv_pred_single"] < 0.0
    ):
        raise RuntimeError(
            "Saved single-sample PileFlow predictions "
            "contain negative-pT pixels."
        )

    if np.any(
        results["neutral_lv_pred_mean"] < 0.0
    ):
        raise RuntimeError(
            "Saved mean PileFlow predictions contain "
            "negative-pT pixels."
        )

    # The backward-compatible prediction must be exactly the same array
    # numerically as the explicitly named mean prediction.
    if not np.array_equal(
        results["neutral_lv_pred"],
        results["neutral_lv_pred_mean"],
    ):
        raise RuntimeError(
            "neutral_lv_pred is not identical to "
            "neutral_lv_pred_mean."
        )

    if eval_samples == 1:
        if not np.array_equal(
            results["neutral_lv_pred_single"],
            results["neutral_lv_pred_mean"],
        ):
            raise RuntimeError(
                "For eval_samples=1, the single and mean "
                "predictions must be identical."
            )

    output_path = os.path.join(
        out_dir,
        "generated_jets.npz",
    )

    np.savez_compressed(
        output_path,
        **results,
    )

    print(
        f"  [generate] Generated predictions for "
        f"{n_generated:,} jets."
    )

    print(
        f"  [generate] Samples generated per jet: "
        f"{eval_samples}"
    )

    print(
        "  [generate] Saved single prediction key: "
        "neutral_lv_pred_single"
    )

    print(
        "  [generate] Saved mean prediction key: "
        "neutral_lv_pred_mean"
    )

    print(
        "  [generate] Observable comparison key: "
        "neutral_lv_pred -> mean"
    )

    print(
        f"  [generate] Saved -> {output_path}"
    )

    return results



def _plot_loss_curve(
    history: dict[str, list[float]],
    save_path: str,
    title: str = "Loss",
) -> None:
    """Save the train/validation loss curve."""
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  [flow] matplotlib not available; skipping loss curve."
        )
        return

    epochs = range(1, len(history["train"]) + 1)

    fig, axis = plt.subplots(figsize=(8, 4))

    axis.plot(
        epochs,
        history["train"],
        label="Train",
        linewidth=2,
    )

    axis.plot(
        epochs,
        history["val"],
        label="Validation",
        linewidth=2,
        linestyle="--",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  [flow] Loss curve -> {save_path}")


__all__ = [
    "FLOW_CONTRACT",
    "train_pileflow",
    "generate_and_save",
]