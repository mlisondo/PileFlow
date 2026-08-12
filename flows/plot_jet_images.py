"""
flows/scripts/plot_jet_images.py
=================================

Plot average jet images from PileFlow's generated_jets.npz.

Produces two figures:

Figure A — Average neutral-LV images (9x9):
    True | Predicted (PileFlow) | Residual (Pred - True)

Figure B — Average conditioning images:
    Neutral-all 9x9 | Charged-PU 9x9 (sum-pooled) | Charged-LV 9x9 (sum-pooled)

Usage (from repo root):
    python flows/scripts/plot_jet_images.py \\
        --npz data/flows/exp1_eval/data/generated_jets.npz \\
        --out data/flows/exp1_eval/plots

Options:
    --npz       Path to generated_jets.npz (required)
    --out       Output directory for saved figures (default: same dir as npz)
    --max-jets  Cap the number of jets used for the averages (default: all)
    --dpi       Figure DPI (default: 150)
    --cmap      Colormap for pT images (default: viridis)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _sum_pool_36_to_9(flat36: np.ndarray) -> np.ndarray:
    """
    Sum-pool a batch of flattened 36x36 images to 9x9.

    Parameters
    ----------
    flat36 : (N, 1296)

    Returns
    -------
    flat9 : (N, 81)
    """
    n = flat36.shape[0]
    img = flat36.reshape(n, 36, 36)
    # Split into 9 blocks of 4 along each axis and sum.
    img = img.reshape(n, 9, 4, 9, 4).sum(axis=(2, 4))  # (N, 9, 9)
    return img.reshape(n, 81)


def _average_image(flat: np.ndarray, side: int) -> np.ndarray:
    """Return the mean 2-D image from a batch of flattened images."""
    return flat.mean(axis=0).reshape(side, side)


def _plot_row(
    axes,
    images: list[np.ndarray],
    titles: list[str],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    residual_idx: int | None = None,
) -> None:
    """Fill a row of axes with jet images."""
    import matplotlib.pyplot as plt

    for ax, img, title in zip(axes, images, titles):
        is_residual = residual_idx is not None and title == titles[residual_idx]
        if is_residual:
            vabs = np.abs(img).max()
            im = ax.imshow(
                img,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vabs,
                vmax=vabs,
                aspect="equal",
            )
        else:
            im = ax.imshow(
                img,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("η pixel")
        ax.set_ylabel("φ pixel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pT [GeV]")


def plot_neutral_lv(
    true: np.ndarray,
    pred: np.ndarray,
    out_dir: str,
    dpi: int = 150,
    cmap: str = "viridis",
) -> None:
    """
    Figure A: True vs Predicted neutral-LV average images + residual.

    Parameters
    ----------
    true : (N, 81)  flattened 9x9 true neutral-LV images
    pred : (N, 81)  flattened 9x9 predicted neutral-LV images
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_true = _average_image(true, 9)
    avg_pred = _average_image(pred, 9)
    avg_residual = avg_pred - avg_true

    vmax = max(avg_true.max(), avg_pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    _plot_row(
        axes,
        [avg_true, avg_pred, avg_residual],
        ["True neutral-LV", "PileFlow predicted", "Residual (Pred − True)"],
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        residual_idx=2,
    )

    fig.suptitle(
        f"Average neutral-LV jet images  (N = {len(true):,} jets)",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "avg_neutral_lv_images.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [images] Saved -> {out_path}")


def plot_context_channels(
    neutral_all: np.ndarray,
    charged_pu: np.ndarray,
    charged_lv: np.ndarray,
    out_dir: str,
    dpi: int = 150,
    cmap: str = "viridis",
) -> None:
    """
    Figure B: Average conditioning images (neutral-all 9x9,
    charged-PU and charged-LV sum-pooled to 9x9).

    Parameters
    ----------
    neutral_all : (N, 81)    flat 9x9
    charged_pu  : (N, 1296)  flat 36x36
    charged_lv  : (N, 1296)  flat 36x36
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_neutral_all = _average_image(neutral_all, 9)

    # Sum-pool charged images to 9x9 for display.
    avg_charged_pu = _average_image(_sum_pool_36_to_9(charged_pu), 9)
    avg_charged_lv = _average_image(_sum_pool_36_to_9(charged_lv), 9)

    vmax = max(
        avg_neutral_all.max(),
        avg_charged_pu.max(),
        avg_charged_lv.max(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    _plot_row(
        axes,
        [avg_neutral_all, avg_charged_pu, avg_charged_lv],
        [
            "Neutral-all (9×9)",
            "Charged-PU (36×36 → 9×9)",
            "Charged-LV (36×36 → 9×9)",
        ],
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    fig.suptitle(
        f"Average conditioning images  (N = {len(neutral_all):,} jets)",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "avg_context_images.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [images] Saved -> {out_path}")


def plot_pixel_distributions(
    true: np.ndarray,
    pred: np.ndarray,
    out_dir: str,
    dpi: int = 150,
) -> None:
    """
    Figure C: Histogram of pixel pT values (true vs predicted).

    Useful for checking whether PileFlow is generating the right pT scale
    and whether it over-spreads pT (high N95 issue).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Flatten all pixels from all jets, exclude exact zeros.
    true_px = true.ravel()
    pred_px = pred.ravel()
    true_nz = true_px[true_px > 0]
    pred_nz = pred_px[pred_px > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: full distribution (all pixels including zeros).
    axes[0].hist(
        true_px, bins=80, density=True, alpha=0.6, label="True", color="steelblue"
    )
    axes[0].hist(
        pred_px, bins=80, density=True, alpha=0.6, label="PileFlow", color="darkorange"
    )
    axes[0].set_xlabel("Pixel pT [GeV]")
    axes[0].set_ylabel("Density")
    axes[0].set_title("All pixels (incl. zeros)")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Right: non-zero pixels only.
    axes[1].hist(
        true_nz, bins=80, density=True, alpha=0.6, label="True", color="steelblue"
    )
    axes[1].hist(
        pred_nz, bins=80, density=True, alpha=0.6, label="PileFlow", color="darkorange"
    )
    axes[1].set_xlabel("Pixel pT [GeV]  (non-zero only)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Non-zero pixels")
    axes[1].legend()
    axes[1].set_yscale("log")

    fig.suptitle("Pixel pT distributions", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pixel_pt_distributions.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [images] Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average jet images from PileFlow generated_jets.npz"
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to generated_jets.npz produced by flows/runner.py",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for plots (default: same directory as --npz)",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Cap the number of jets used (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI (default: 150)",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for pT images (default: viridis)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"generated_jets.npz not found: {npz_path}")

    out_dir = args.out or str(npz_path.parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [images] Loading {npz_path} ...")

    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.files)
        print(f"  [images] Keys: {keys}")

        true = data["neutral_lv_true"].astype(np.float32)
        pred = data["neutral_lv_pred"].astype(np.float32)
        neutral_all = data["neutral_all_9x9"].astype(np.float32)
        charged_pu = data["charged_pu_36x36"].astype(np.float32)
        charged_lv = data["charged_lv_36x36"].astype(np.float32)

    n = len(true)
    if args.max_jets is not None:
        n = min(n, args.max_jets)
        true = true[:n]
        pred = pred[:n]
        neutral_all = neutral_all[:n]
        charged_pu = charged_pu[:n]
        charged_lv = charged_lv[:n]

    print(f"  [images] Jets: {n:,}")
    print(f"  [images] Output dir: {out_dir}")

    plot_neutral_lv(true, pred, out_dir, dpi=args.dpi, cmap=args.cmap)
    plot_context_channels(neutral_all, charged_pu, charged_lv, out_dir, dpi=args.dpi, cmap=args.cmap)
    plot_pixel_distributions(true, pred, out_dir, dpi=args.dpi)

    print("  [images] Done.")


if __name__ == "__main__":
    main()
