"""
flows/training/train_flow.py
============================

Training and generation utilities for the PileFlow model.

This module does not generate HEP events and does not build jet images.
It consumes generator outputs through `flows.data.dataset.PileFlowDataset`.

PileFlow conditions on a 253-dimensional context vector:

    7      generator-level scalar features
    3      jet-flavour one-hot labels
    81     neutral-all 9x9 image
    81     charged-pileup 9x9 image
    81     charged-LV 9x9 image

The flow generates a 97-dimensional target vector:

    81     neutral-LV 9x9 image
    16     reconstructed scalar jet observables

Outputs written under cfg.outdir:

    checkpoints/pileflow_best.pt
    checkpoints/pileflow_best_history.npz
    plots/pileflow_loss.png
    data/generated_jets.npz
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
    CRTVelocityField,
    ContextEncoder,
    IMG_DIM,
    N_CONTEXT,
    N_SCALARS,
    N_TARGET,
    TargetCFM,
    TargetPreprocessor,
)


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint with compatibility across torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _batch_to_device(batch, device: torch.device):
    """
    Move a PileFlowDataset batch to the requested device.
    """
    scalar_gen, flavour, neutral_lv, na9, cp9, cl9, scalars = [
        item.to(device) for item in batch
    ]

    return (
        scalar_gen,
        flavour.long(),
        neutral_lv,
        na9,
        cp9,
        cl9,
        scalars,
    )


def _make_model(cfg, dropout: float, device: torch.device) -> CRTVelocityField:
    """
    Construct the PileFlow velocity-field model.
    """
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
) -> dict:
    """
    Build the checkpoint payload saved during training.
    """
    return {
        "model": model.state_dict(),
        "ctx_enc": ctx_enc.state_dict(),
        "tgt_prep": tgt_prep.state_dict(),
        "cfg": {
            "n_context": N_CONTEXT,
            "n_target": N_TARGET,
            "n_img": IMG_DIM,
            "n_scalars": N_SCALARS,
            "flow_hidden": cfg.flow_hidden,
            "flow_blocks": cfg.flow_blocks,
            "flow_time_emb": cfg.flow_time_emb,
            "flow_sigma_min": cfg.flow_sigma_min,
            "flow_dropout": dropout,
        },
    }


def train_pileflow(
    npy_path: str,
    npz_path: str,
    flow_ckpt: str,
    cfg,
) -> CRTVelocityField:
    """
    Train PileFlow using generator `.npy` and `.npz` outputs.

    Parameters
    ----------
    npy_path:
        Path to generator jet-feature table, shape `(N, 25)`.

    npz_path:
        Path to generator image/constituent file.

    flow_ckpt:
        Path where the best PileFlow checkpoint should be saved.

    cfg:
        Config object with training hyperparameters and runtime options.

    Returns
    -------
    CRTVelocityField
        Trained velocity-field model with best-validation weights loaded.
    """
    if cfg.flow_epochs <= 0:
        raise ValueError("cfg.flow_epochs must be positive when training PileFlow.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    ckpt_dir = Path(flow_ckpt).expanduser().resolve().parent
    plot_dir = Path(cfg.outdir) / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset = PileFlowDataset(
        npy_path=npy_path,
        npz_path=npz_path,
        max_n=getattr(cfg, "max_jets", None),
    )

    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 jets to train/validate PileFlow.")

    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    print(f"  [flow] Train / Val: {n_train:,} / {n_val:,}")

    # Fit preprocessors on training data only.
    train_indices = train_ds.indices

    scalar_train = dataset.scalar_gen[train_indices]
    neutral_all_train = dataset.neutral_all_9x9[train_indices]
    charged_pu_train = dataset.charged_pu_9x9[train_indices]
    charged_lv_train = dataset.charged_lv_9x9[train_indices]
    neutral_lv_train = dataset.neutral_lv[train_indices]
    scalars_train = dataset.scalars[train_indices]

    ctx_enc = ContextEncoder()
    tgt_prep = TargetPreprocessor()

    ctx_enc.fit(
        scalar_train,
        neutral_all_train,
        charged_pu_train,
        charged_lv_train,
    )
    tgt_prep.fit(neutral_lv_train, scalars_train)

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
    model = _make_model(cfg=cfg, dropout=dropout, device=device)

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
    print(f"  [flow] Context dim: {N_CONTEXT}  (7 gen + 3 flavour + 3×81 images)")
    print(f"  [flow] Target  dim: {N_TARGET}   ({IMG_DIM} neutral LV + {N_SCALARS} scalars)")
    print(f"  [flow] Architecture: {cfg.flow_blocks} blocks × {cfg.flow_hidden} hidden")
    print()

    best_val = float("inf")
    history = {"train": [], "val": []}

    patience = getattr(cfg, "flow_patience", 60)
    no_improve = 0

    print(f"  {'Epoch':>5}  {'Train':>12}  {'Val':>12}  {'Time':>8}")
    print("  " + "-" * 46)

    for epoch in range(1, cfg.flow_epochs + 1):
        start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            scalar_gen, flavour, neutral_lv, na9, cp9, cl9, scalars = _batch_to_device(
                batch,
                device,
            )

            context = ctx_enc(scalar_gen, flavour, na9, cp9, cl9)
            x1 = tgt_prep.encode(neutral_lv, scalars)

            t, zt, ut = cfm.sample_training_pair(x1)

            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(t, zt, context), ut)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                scalar_gen, flavour, neutral_lv, na9, cp9, cl9, scalars = _batch_to_device(
                    batch,
                    device,
                )

                context = ctx_enc(scalar_gen, flavour, na9, cp9, cl9)
                x1 = tgt_prep.encode(neutral_lv, scalars)

                t, zt, ut = cfm.sample_training_pair(x1)
                val_loss += nn.functional.mse_loss(model(t, zt, context), ut).item()

        val_loss /= len(val_loader)

        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        improved = val_loss < best_val
        marker = " *" if improved else ""
        elapsed = time.time() - start

        print(
            f"  {epoch:>5}  "
            f"{train_loss:>12.6f}  "
            f"{val_loss:>12.6f}  "
            f"{elapsed:>7.1f}s{marker}"
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

    history_path = flow_ckpt.replace(".pt", "_history.npz")
    np.savez(
        history_path,
        train=np.asarray(history["train"], dtype=np.float32),
        val=np.asarray(history["val"], dtype=np.float32),
    )

    print(f"\n  [flow] History   -> {history_path}")

    _plot_loss_curve(
        history,
        save_path=str(plot_dir / "pileflow_loss.png"),
        title="PileFlow training — flow matching MSE loss",
    )

    print(f"  [flow] Best val  : {best_val:.6f}  ->  {flow_ckpt}")

    state = _load_checkpoint(flow_ckpt, device)
    model.load_state_dict(state["model"])

    return model


def generate_and_save(
    npy_path: str,
    npz_path: str,
    flow_ckpt: str,
    cfg,
    n_steps: int = 100,
    out_dir: str | None = None,
) -> dict:
    """
    Load a trained PileFlow checkpoint and generate mitigated jets.

    The output file is:

        generated_jets.npz

    with keys:

        neutral_lv_pred : (N, 81)
            PileFlow-generated neutral-LV image.

        neutral_lv_true : (N, 81)
            Generator truth neutral-LV image.

        scalar_obs : (N, 16)
            PileFlow-generated scalar jet observables.

        neutral_all_9x9 : (N, 81)
            Input neutral-all context image.

        charged_pu_9x9 : (N, 81)
            Input charged-pileup context image.

        charged_lv_9x9 : (N, 81)
            Input charged-LV context image.

    Returns
    -------
    dict
        Dictionary containing the same arrays saved to `generated_jets.npz`.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    device = torch.device(cfg.device)

    if out_dir is None:
        out_dir = os.path.join(cfg.outdir, "data")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    state = _load_checkpoint(flow_ckpt, device)
    saved_cfg = state["cfg"]

    model = CRTVelocityField(
        n_features=saved_cfg["n_target"],
        context_dim=saved_cfg["n_context"],
        hidden_dim=saved_cfg["flow_hidden"],
        n_blocks=saved_cfg["flow_blocks"],
        time_emb_dim=saved_cfg["flow_time_emb"],
        dropout=0.0,
    ).to(device)

    model.load_state_dict(state["model"])
    model.eval()

    ctx_enc = ContextEncoder().to(device)
    ctx_enc.load_state_dict(state["ctx_enc"])
    ctx_enc.eval()

    tgt_prep = TargetPreprocessor(
        n_img=saved_cfg["n_img"],
        n_scalars=saved_cfg["n_scalars"],
    ).to(device)
    tgt_prep.load_state_dict(state["tgt_prep"])
    tgt_prep.eval()

    cfm = TargetCFM(sigma_min=saved_cfg["flow_sigma_min"])

    dataset = PileFlowDataset(
        npy_path=npy_path,
        npz_path=npz_path,
        max_n=getattr(cfg, "max_jets", None),
    )

    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg, "eval_batch", getattr(cfg, "flow_batch", 512)),
        shuffle=False,
        num_workers=0,
    )

    all_pred = []
    all_true = []
    all_scalar = []
    all_na9 = []
    all_cp9 = []
    all_cl9 = []

    print("  [generate] Running PileFlow ODE integration ...")

    with torch.no_grad():
        for batch in loader:
            scalar_gen, flavour, neutral_lv, na9, cp9, cl9, _ = _batch_to_device(
                batch,
                device,
            )

            context = ctx_enc(scalar_gen, flavour, na9, cp9, cl9)
            z_gen = cfm.generate(
                model=model,
                context=context,
                n_steps=n_steps,
                device=device,
            )

            img, scalars = tgt_prep.decode(z_gen)

            all_pred.append(img.cpu().numpy())
            all_true.append(neutral_lv.cpu().numpy())
            all_scalar.append(scalars.cpu().numpy())
            all_na9.append(na9.cpu().numpy())
            all_cp9.append(cp9.cpu().numpy())
            all_cl9.append(cl9.cpu().numpy())

    results = {
        "neutral_lv_pred": np.concatenate(all_pred, axis=0),
        "neutral_lv_true": np.concatenate(all_true, axis=0),
        "scalar_obs": np.concatenate(all_scalar, axis=0),
        "neutral_all_9x9": np.concatenate(all_na9, axis=0),
        "charged_pu_9x9": np.concatenate(all_cp9, axis=0),
        "charged_lv_9x9": np.concatenate(all_cl9, axis=0),
    }

    n_generated = results["neutral_lv_pred"].shape[0]
    print(f"  [generate] Generated {n_generated:,} jets.")

    output_path = os.path.join(out_dir, "generated_jets.npz")
    np.savez_compressed(output_path, **results)

    print(f"  [generate] Saved -> {output_path}")

    return results


def _plot_loss_curve(
    history: dict,
    save_path: str,
    title: str = "Loss",
) -> None:
    """
    Save the train/validation loss curve.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:
        print("  [flow] matplotlib not available; skipping loss curve.")
        return

    epochs = range(1, len(history["train"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train"], label="Train", linewidth=2)
    ax.plot(epochs, history["val"], label="Validation", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  [flow] Loss curve -> {save_path}")


__all__ = [
    "train_pileflow",
    "generate_and_save",
]