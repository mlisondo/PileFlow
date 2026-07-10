# pumml/scripts/plot_average_images.py
#
# Reproduces PUMML paper Figure 2:
# Average jet images for all four channels across the full dataset.
#
# Usage
# -----
#   python scripts/plot_average_images.py \
#       --npz  ../../gen4e2e/data/run_XXX/antikt_R0.4/jets_XXX_images.npz \
#       --out  plots/average_images.png

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_figure2(args):
    data = np.load(args.npz, allow_pickle=False)

    N = len(data["jet_pt"])
    if args.max_images:
        N = min(N, args.max_images)

    mean_npu = float(data["n_pu"][:N].mean())

    # average images
    neutral_total_avg = data["ch_neutral_all"][:N].mean(axis=0)   # 36x36
    charged_pu_avg    = data["ch_charged_pu"][:N].mean(axis=0)    # 36x36
    charged_lv_avg    = data["ch_charged_lv"][:N].mean(axis=0)    # 36x36

    # use raw 9x9 for neutral LV if available
    if "clean_neutral_lv" in data:
        neutral_lv_avg = data["clean_neutral_lv"][:N].mean(axis=0)  # 9x9
    else:
        neutral_lv_avg = data["ch_neutral_lv"][:N].mean(axis=0)

    # extent for the image: window is +/-0.45 in eta and phi
    extent = [-0.45, 0.45, -0.45, 0.45]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.suptitle(
        f"Average leading-jet images\n"
        f"N={N:,} jets  |  mean NPU={mean_npu:.0f}",
        fontsize=12,
    )

    panels = [
        (axes[0, 0], neutral_total_avg, r"Neutral Total $p_T$",          "Reds"),
        (axes[0, 1], charged_pu_avg,    r"Charged Pileup $p_T$",         "Greens"),
        (axes[1, 0], charged_lv_avg,    r"Charged Leading Vertex $p_T$", "Blues"),
        (axes[1, 1], neutral_lv_avg,    r"Neutral Leading Vertex $p_T$", "Greys"),
    ]

    for ax, img, title, cmap in panels:
        vmax = float(np.percentile(img, 99.5))
        if vmax <= 0:
            vmax = 1.0
        im = ax.imshow(
            img.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"Pseudorapidity $\eta$", fontsize=9)
        ax.set_ylabel(r"Azimuthal Angle $\phi$", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_average_images] Saved -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Average jet images"
    )
    p.add_argument("--npz",        required=True, help="Path to jet_images.npz")
    p.add_argument("--out",        default="plots/average_images.png")
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap number of jets used for averaging")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    plot_figure2(args)
