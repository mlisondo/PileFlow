# pumml/scripts/check_prediction.py
#
# Visual sanity check after training.
# Plots the 3 input channels, true target, and PUMML prediction
# side-by-side for a random selection of jets.
#
# Usage
# -----
#   python scripts/check_prediction.py \
#       --npz   ../../gen4e2e/data/run_XXX/antikt_R0.4/jets_XXX_images.npz \
#       --model checkpoints/pumml_model.pt \
#       --n     6 \
#       --out   plots/prediction_check.png

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

from utils.inference import load_model, predict_batch


def plot_predictions(args):
    data = np.load(args.npz, allow_pickle=False)
    N    = len(data["jet_pt"])

    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(N, size=min(args.n, N), replace=False)
    idxs = sorted(idxs)

    # stack inputs
    X = np.stack([
        data["ch_neutral_all"],
        data["ch_charged_pu"],
        data["ch_charged_lv"],
    ], axis=1).astype(np.float32)

    X_sel = X[idxs]
    y_sel = data["ch_neutral_lv"][idxs]

    model = load_model(args.model, device=args.device)
    pred  = predict_batch(model, X_sel, device=args.device)

    n_jets = len(idxs)
    # 5 columns: neutral_all | charged_pu | charged_lv | true | predicted
    fig, axes = plt.subplots(
        n_jets, 5, figsize=(15, 3 * n_jets),
        squeeze=False,
    )

    col_titles = [
        "Neutral total (input)",
        "Charged PU (input)",
        "Charged LV (input)",
        "Neutral LV (true)",
        "Neutral LV (PUMML)",
    ]
    cmaps = ["Reds", "Greens", "Blues", "Purples", "Purples"]

    def _show(ax, img, cmap, title):
        vmax = img.max() if img.max() > 0 else 1.0
        ax.imshow(
            img.T, origin="lower", cmap=cmap,
            norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax),
            aspect="auto",
        )
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for row, (global_idx, local_idx) in enumerate(zip(idxs, range(n_jets))):
        pt  = float(data["jet_pt"][global_idx])
        npu = int(data["n_pu"][global_idx])

        # input channels are 36x36; display the 9x9 neutral for fair comparison
        # use ch_neutral_all_raw if available, else downsample
        if "ch_neutral_all_raw" in data:
            neutral_disp = data["ch_neutral_all_raw"][global_idx]
        else:
            # downsample 36x36 -> 9x9 by summing 4x4 blocks
            ch = data["ch_neutral_all"][global_idx]
            neutral_disp = ch.reshape(9, 4, 9, 4).sum(axis=(1, 3))

        images = [
            neutral_disp,
            data["ch_charged_pu"][global_idx],
            data["ch_charged_lv"][global_idx],
            y_sel[local_idx],
            pred[local_idx],
        ]

        for col, (img, cmap, title) in enumerate(zip(images, cmaps, col_titles)):
            _show(axes[row, col], img, cmap, title if row == 0 else "")

        axes[row, 0].set_ylabel(
            f"Jet {global_idx}\npT={pt:.0f} GeV\nNPU={npu}",
            fontsize=8,
        )

    fig.suptitle("PUMML prediction check", fontsize=12, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[check] Saved -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Visual sanity check for PUMML predictions")
    p.add_argument("--npz",    required=True, help="Path to jet_images.npz")
    p.add_argument("--model",  required=True, help="Path to trained model .pt")
    p.add_argument("--n",      type=int, default=6, help="Number of jets to show")
    p.add_argument("--out",    default="plots/prediction_check.png")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed",   type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_predictions(args)
