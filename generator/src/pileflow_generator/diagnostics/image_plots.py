"""
PUMML image diagnostics for generated pileup image files.

This module loads the generator `.npz` image output and saves clean, pileup,
and clean-vs-pileup image panels.

It can be used as either:

    python -m pileflow_generator.diagnostics.image_plots --npz path/to/file.npz

or imported in a notebook.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def load_npz(path: str) -> dict:
    """
    Load a generator `.npz` image file as a plain dictionary.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"npz file not found: {path}")

    return dict(np.load(path))


def _show(ax, img: np.ndarray, cmap: str, title: str) -> None:
    """
    Render one jet image channel on a matplotlib axis.
    """
    vmax = img.max() if img.max() > 0 else 1.0

    ax.imshow(
        img.T,
        origin="lower",
        cmap=cmap,
        norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax),
        aspect="auto",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\Delta\eta$ pixel")
    ax.set_ylabel(r"$\Delta\phi$ pixel")


def _jet_label(imgs: dict, jet_idx: int) -> str:
    """
    Build a per-jet title from metadata stored in the `.npz`.
    """
    pt = imgs["jet_pt"][jet_idx]
    npu = imgs["n_pu"][jet_idx]
    return f"Jet {jet_idx}  |  pT={pt:.1f} GeV  |  N_PU={npu}"


def plot_clean(imgs: dict, jet_idx: int, save_path: str | None = None) -> None:
    """
    Plot clean no-pileup neutral images for one jet.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    _show(axes[0], imgs["clean_neutral_all"][jet_idx], "Reds", "clean neutral all (no PU)")
    _show(axes[1], imgs["clean_neutral_lv"][jet_idx], "Purples", "clean neutral LV (target)")
    fig.suptitle(_jet_label(imgs, jet_idx), fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_pileup(imgs: dict, jet_idx: int, save_path: str | None = None) -> None:
    """
    Plot the three PUMML input channels for one jet.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _show(axes[0], imgs["ch_charged_lv"][jet_idx], "Blues", "charged LV (PUMML input)")
    _show(axes[1], imgs["ch_charged_pu"][jet_idx], "Greens", "charged PU (PUMML input)")
    _show(axes[2], imgs["ch_neutral_all"][jet_idx], "Reds", "neutral all w/ PU (PUMML input)")
    fig.suptitle(_jet_label(imgs, jet_idx), fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_clean_vs_pileup(
    imgs: dict,
    jet_idx: int,
    save_path: str | None = None,
) -> None:
    """
    Plot clean neutral, contaminated neutral, and LV target neutral images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _show(axes[0], imgs["clean_neutral_all"][jet_idx], "Reds", "clean neutral all (no PU)")
    _show(axes[1], imgs["ch_neutral_all"][jet_idx], "Reds", "contaminated neutral all (with PU)")
    _show(axes[2], imgs["clean_neutral_lv"][jet_idx], "Purples", "clean neutral LV (target)")
    fig.suptitle(_jet_label(imgs, jet_idx), fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_mean_clean(imgs: dict, n_jets: int, save_path: str | None = None) -> None:
    """
    Plot mean clean no-pileup neutral images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    _show(axes[0], imgs["clean_neutral_all"].mean(axis=0), "Reds", "mean clean neutral all (no PU)")
    _show(axes[1], imgs["clean_neutral_lv"].mean(axis=0), "Purples", "mean clean neutral LV (target)")
    fig.suptitle(f"Mean over {n_jets} jets", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_mean_pileup(imgs: dict, n_jets: int, save_path: str | None = None) -> None:
    """
    Plot mean PUMML input channels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _show(axes[0], imgs["ch_charged_lv"].mean(axis=0), "Blues", "mean charged LV")
    _show(axes[1], imgs["ch_charged_pu"].mean(axis=0), "Greens", "mean charged PU")
    _show(axes[2], imgs["ch_neutral_all"].mean(axis=0), "Reds", "mean contaminated neutral all")
    fig.suptitle(f"Mean over {n_jets} jets", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_mean_clean_vs_pileup(
    imgs: dict,
    n_jets: int,
    save_path: str | None = None,
) -> None:
    """
    Plot mean clean, contaminated, and target neutral images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _show(axes[0], imgs["clean_neutral_all"].mean(axis=0), "Reds", "mean clean neutral all (no PU)")
    _show(axes[1], imgs["ch_neutral_all"].mean(axis=0), "Reds", "mean contaminated neutral all")
    _show(axes[2], imgs["clean_neutral_lv"].mean(axis=0), "Purples", "mean clean neutral LV (target)")
    fig.suptitle(f"Mean over {n_jets} jets", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def make_image_diagnostics(
    npz_path: str,
    out_dir: str = "jet_images",
    max_jets: int | None = None,
) -> None:
    """
    Save PUMML image diagnostics from a generator `.npz` file.
    """
    imgs = load_npz(npz_path)
    n_total = len(imgs["jet_pt"])
    n_save = min(n_total, max_jets) if max_jets is not None else n_total

    print(f"Loaded {n_total} jets from {npz_path}")

    clean_dir = os.path.join(out_dir, "clean")
    pileup_dir = os.path.join(out_dir, "pileup")
    compare_dir = os.path.join(out_dir, "clean_vs_pileup")

    for directory in [clean_dir, pileup_dir, compare_dir]:
        os.makedirs(directory, exist_ok=True)

    print("Saving mean images...")
    plot_mean_clean(imgs, n_total, save_path=os.path.join(clean_dir, "mean.png"))
    plot_mean_pileup(imgs, n_total, save_path=os.path.join(pileup_dir, "mean.png"))
    plot_mean_clean_vs_pileup(imgs, n_total, save_path=os.path.join(compare_dir, "mean.png"))
    print("  mean images saved.")

    print(f"Saving per-jet images (0-{n_save - 1})...")
    for i in range(n_save):
        plot_clean(imgs, i, save_path=os.path.join(clean_dir, f"jet_{i:04d}.png"))
        plot_pileup(imgs, i, save_path=os.path.join(pileup_dir, f"jet_{i:04d}.png"))
        plot_clean_vs_pileup(imgs, i, save_path=os.path.join(compare_dir, f"jet_{i:04d}.png"))

        if (i + 1) % 20 == 0 or i == n_save - 1:
            print(f"  {i + 1}/{n_save} jets done")

    print(f"\nImages saved to {out_dir}/")
    print(f"  {clean_dir}/")
    print(f"  {pileup_dir}/")
    print(f"  {compare_dir}/")


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Plot PUMML jet images from a PileFlow generator .npz file."
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to the generator pileup image .npz file.",
    )
    parser.add_argument(
        "--out",
        default="jet_images",
        help="Output directory for image panels.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Maximum number of per-jet panels to save.",
    )
    return parser


def main() -> None:
    """
    Command-line entry point.
    """
    args = build_parser().parse_args()
    make_image_diagnostics(
        npz_path=args.npz,
        out_dir=args.out,
        max_jets=args.max_jets,
    )


if __name__ == "__main__":
    main()