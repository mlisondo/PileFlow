#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


def downsample_36_to_9(x):
    """
    Sum each 4x4 block of a 36x36 image into one 9x9 calorimeter cell.

    Input:
        (N, 36, 36)

    Output:
        (N, 9, 9)
    """
    x = np.asarray(x)

    return (
        x.reshape(
            len(x),
            9, 4,
            9, 4,
        )
        .sum(axis=(2, 4))
    )


def robust_vmax(image, percentile=99.5):
    """
    Determine a robust upper color-scale bound.

    Using a percentile rather than the absolute maximum prevents
    one very bright central cell from washing out all surrounding
    jet structure.
    """
    positive = np.asarray(image)[
        np.asarray(image) > 0
    ]

    if len(positive) == 0:
        return 1.0

    vmax = float(
        np.percentile(
            positive,
            percentile,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(
            positive.max()
        )

    return max(vmax, 1.0e-12)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot PUMML-style average leading-jet "
            "input/target channel images."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Benchmark NPZ dataset.",
    )

    parser.add_argument(
        "--outdir",
        default="paper_channel_images",
    )

    parser.add_argument(
        "--split",
        choices=[
            "event",
            "image",
            "all",
        ],
        default="event",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.45,
        help=(
            "PowerNorm gamma used to reveal lower-intensity "
            "jet structure. Smaller values enhance tails more."
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile used for the color-scale maximum.",
    )

    parser.add_argument(
        "--colorbars",
        action="store_true",
        help="Add average-pT colorbars to each panel.",
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    d = np.load(
        args.data,
        allow_pickle=False,
    )


    # Select split
    if args.split == "event":

        if "event_test_idx" not in d:
            raise KeyError(
                "Dataset does not contain event_test_idx."
            )

        idx = np.asarray(
            d["event_test_idx"],
            dtype=np.int64,
        )

    elif args.split == "image":

        if "test_idx" not in d:
            raise KeyError(
                "Dataset does not contain test_idx."
            )

        idx = np.asarray(
            d["test_idx"],
            dtype=np.int64,
        )

    else:

        idx = np.arange(
            len(d["X"]),
            dtype=np.int64,
        )


    # Leading jets only
    if "jet_rank" not in d:
        raise KeyError(
            "Dataset must contain jet_rank."
        )

    rank = np.asarray(
        d["jet_rank"][idx]
    )

    idx = idx[
        rank == 0
    ]

    print(
        "Leading jets used:",
        len(idx),
    )


    # Load images
    X = np.asarray(
        d["X"][idx],
        dtype=np.float64,
    )

    if X.ndim != 4 or X.shape[1:] != (3, 36, 36):
        raise ValueError(
            f"Expected X shape (N,3,36,36), got {X.shape}"
        )

    # Channel convention:
    # X[:,0] = total neutral
    # X[:,1] = charged pileup
    # X[:,2] = charged leading vertex
    neutral_total36 = X[:, 0]
    charged_pu36 = X[:, 1]
    charged_lv36 = X[:, 2]

    neutral_lv9 = np.asarray(
        d["y_neutral_lv_9"][idx],
        dtype=np.float64,
    )


    # Neutral total is physically a 9x9 calorimeter image.
    # Convert the stored 36x36 representation back to 9x9.
    neutral_total9 = (
        downsample_36_to_9(
            neutral_total36
        )
    )


    # Average images
    avg_neutral_total = (
        neutral_total9.mean(axis=0)
    )

    avg_charged_pu = (
        charged_pu36.mean(axis=0)
    )

    avg_charged_lv = (
        charged_lv36.mean(axis=0)
    )

    avg_neutral_lv = (
        neutral_lv9.mean(axis=0)
    )

    print()
    print("Average integrated pT per leading jet:")
    print(
        f"  neutral total       : "
        f"{avg_neutral_total.sum():.3f} GeV"
    )
    print(
        f"  charged pileup      : "
        f"{avg_charged_pu.sum():.3f} GeV"
    )
    print(
        f"  charged leading vtx : "
        f"{avg_charged_lv.sum():.3f} GeV"
    )
    print(
        f"  neutral leading vtx : "
        f"{avg_neutral_lv.sum():.3f} GeV"
    )


    # Plot style
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 10.0),
    )

    # Images cover a 0.9 x 0.9 region centered on the jet.
    extent = [
        -0.45,
        0.45,
        -0.45,
        0.45,
    ]

    ticks = [
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
    ]

    panels = [
        (
            axes[0, 0],
            avg_neutral_total,
            r"Neutral Total $p_T$",
            "Reds",
        ),
        (
            axes[0, 1],
            avg_charged_pu,
            r"Charged Pileup $p_T$",
            "Greens",
        ),
        (
            axes[1, 0],
            avg_charged_lv,
            r"Charged Leading Vertex $p_T$",
            "Blues",
        ),
        (
            axes[1, 1],
            avg_neutral_lv,
            r"Neutral Leading Vertex $p_T$",
            "Greys",
        ),
    ]

    for ax, image, title, cmap in panels:

        vmax = robust_vmax(
            image,
            percentile=args.percentile,
        )

        norm = PowerNorm(
            gamma=args.gamma,
            vmin=0.0,
            vmax=vmax,
        )

        im = ax.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=extent,
            aspect="equal",
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            r"$\Delta\eta$"
        )

        ax.set_ylabel(
            r"$\Delta\phi$"
        )

        ax.set_xlim(
            -0.45,
            0.45,
        )

        ax.set_ylim(
            -0.45,
            0.45,
        )

        ax.set_xticks(
            ticks
        )

        ax.set_yticks(
            ticks
        )

        ax.tick_params(
            direction="in",
            top=True,
            right=True,
        )

        if args.colorbars:

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )

            cbar.set_label(
                r"Average $p_T$ [GeV]"
            )

    fig.tight_layout(
        h_pad=2.0,
        w_pad=2.0,
    )


    # Save
    png = (
        outdir
        / "average_leading_jet_channels.png"
    )

    pdf = (
        outdir
        / "average_leading_jet_channels.pdf"
    )

    fig.savefig(
        png,
        dpi=args.dpi,
    )

    fig.savefig(
        pdf,
    )

    plt.close(fig)


    # Save numerical averages
    np.savez_compressed(
        outdir
        / "average_leading_jet_channels.npz",

        neutral_total_9=
            avg_neutral_total,

        charged_pileup_36=
            avg_charged_pu,

        charged_leading_vertex_36=
            avg_charged_lv,

        neutral_leading_vertex_9=
            avg_neutral_lv,

        source_indices=
            idx,

        extent=np.asarray(
            extent,
            dtype=np.float64,
        ),

        gamma=np.asarray(
            args.gamma,
            dtype=np.float64,
        ),

        percentile=np.asarray(
            args.percentile,
            dtype=np.float64,
        ),
    )

    print()
    print("Saved:")
    print(" ", png)
    print(" ", pdf)
    print(
        " ",
        outdir
        / "average_leading_jet_channels.npz",
    )

if __name__ == "__main__":
    main()