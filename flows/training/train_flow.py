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

FLOW_CONTRACT = "image-only-neutral9-charged36-v1"


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

    expected = {
        "n_context": N_CONTEXT,
        "n_target": N_TARGET,
        "n_scalars": N_SCALARS,
        "neutral_dim": NEUTRAL_DIM,
        "charged_dim": CHARGED_DIM,
    }

    actual = {
        "n_context": saved_cfg.get("n_context"),
        "n_target": saved_cfg.get("n_target"),
        "n_scalars": saved_cfg.get("n_scalars"),
        "neutral_dim": saved_cfg.get("neutral_dim"),
        "charged_dim": saved_cfg.get("charged_dim"),
    }

    if actual != expected:
        raise ValueError(
            "Checkpoint is incompatible with mixed-resolution PileFlow.\n"
            f"Expected: {expected}\n"
            f"Found:    {actual}\n"
            "Older 253-to-97 and 243-to-81 checkpoints must be retrained."
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
) -> CRTVelocityField:
    """Construct the mixed-resolution PileFlow velocity-field model."""
    return CRTVelocityField(
        n_features=N_TARGET,
        context_dim=N_CONTEXT,
        hidden_dim=cfg.flow_hidden,
        n_blocks=cfg.flow_blocks,
        time_emb_dim=cfg.flow_time_emb,
        dropout=dropout,
    ).to(device)


def _checkpoint_payload(
    model: CRTVelocityField,
    ctx_enc: ContextEncoder,
    tgt_prep: TargetPreprocessor,
    cfg,
    dropout: float,
) -> dict[str, Any]:
    """Build the checkpoint payload saved during training."""
    return {
        "contract": FLOW_CONTRACT,
        "model": model.state_dict(),
        "ctx_enc": ctx_enc.state_dict(),
        "tgt_prep": tgt_prep.state_dict(),
        "cfg": {
            "n_context": N_CONTEXT,
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
            "pt_threshold": None,
            "decode_postprocessing": "clamp-nonnegative-no-threshold",
        },
    }


def train_pileflow(
    npz_path: str,
    flow_ckpt: str,
    cfg,
) -> CRTVelocityField:
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

    ctx_enc = ContextEncoder(
        neutral_dim=NEUTRAL_DIM,
        charged_dim=CHARGED_DIM,
    )

    tgt_prep = TargetPreprocessor(n_img=N_TARGET)

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
    )

    cfm = TargetCFM(sigma_min=cfg.flow_sigma_min)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.flow_lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.flow_epochs,
    )

    print(f"  [flow] Parameters : {model.count_parameters():,}")
    print(
        f"  [flow] Context dim: {N_CONTEXT} "
        f"({NEUTRAL_DIM} neutral + {CHARGED_DIM} charged-PU + "
        f"{CHARGED_DIM} charged-LV)"
    )
    print(
        f"  [flow] Target  dim: {N_TARGET} "
        f"({NEUTRAL_SIDE}x{NEUTRAL_SIDE} neutral-LV pixels)"
    )
    print(
        f"  [flow] Architecture: {cfg.flow_blocks} blocks x "
        f"{cfg.flow_hidden} hidden"
    )
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
            train_loss += loss.item()

        train_loss /= len(train_loader)

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
    ).to(device)

    tgt_prep.load_state_dict(
        state["tgt_prep"]
    )

    tgt_prep.eval()

    cfm = TargetCFM(
        sigma_min=saved_cfg["flow_sigma_min"],
    )

    dataset = PileFlowDataset(
        npz_path=npz_path,
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
                    pt_threshold=None,
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