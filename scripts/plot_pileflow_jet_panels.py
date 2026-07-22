#!/usr/bin/env python3
"""
Visualize detector-grid jet images and PileFlow neutral-LV predictions.

Each per-jet panel contains:

1. contaminated neutral input, 9x9;
2. charged-PU context, 36x36;
3. charged-LV context, 36x36;
4. true neutral-LV target, 9x9;
5. PileFlow neutral-LV prediction, 9x9;
6. prediction minus truth residual, 9x9.

The script also writes:
- mean_panel.png;
- jet_metrics.csv;
- one PNG per selected jet.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import numpy as np


ETA_RANGE = 0.45
PHI_RANGE = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-npz",
        required=True,
        help="Corrected generator evaluation NPZ.",
    )
    parser.add_argument(
        "--generated-npz",
        required=True,
        help="PileFlow data/generated_jets.npz.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for per-jet panels.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=50,
        help="Maximum number of individual panels to save.",
    )
    parser.add_argument(
        "--selection",
        choices=[
            "first",
            "worst-mae",
            "best-mae",
            "worst-total-pt",
        ],
        default="first",
        help="How to select the displayed jets.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
    )

    return parser.parse_args()


def image_batch(
    array: np.ndarray,
    shape: tuple[int, int],
    key: str,
) -> np.ndarray:
    """Return an image batch with shape (N, H, W)."""
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 3 and tuple(array.shape[1:]) == shape:
        return array

    flat_dim = shape[0] * shape[1]

    if array.ndim == 2 and array.shape[1] == flat_dim:
        return array.reshape(-1, *shape)

    raise ValueError(
        f"{key}: expected (N,{shape[0]},{shape[1]}) "
        f"or (N,{flat_dim}); got {array.shape}"
    )


def show_positive(
    fig,
    ax,
    image: np.ndarray,
    title: str,
    vmax: float | None = None,
) -> None:
    """Display a nonnegative pT image."""
    finite = image[np.isfinite(image)]

    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0

    vmax = max(float(vmax), 1.0e-6)

    rendered = ax.imshow(
        image.T,
        origin="lower",
        extent=[
            -ETA_RANGE,
            ETA_RANGE,
            -PHI_RANGE,
            PHI_RANGE,
        ],
        aspect="equal",
        cmap="magma",
        norm=PowerNorm(
            gamma=0.5,
            vmin=0.0,
            vmax=vmax,
        ),
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")
    fig.colorbar(rendered, ax=ax, fraction=0.046, pad=0.04)


def show_residual(
    fig,
    ax,
    residual: np.ndarray,
    title: str,
) -> None:
    """Display a signed prediction residual."""
    max_abs = float(np.nanmax(np.abs(residual)))
    max_abs = max(max_abs, 1.0e-6)

    rendered = ax.imshow(
        residual.T,
        origin="lower",
        extent=[
            -ETA_RANGE,
            ETA_RANGE,
            -PHI_RANGE,
            PHI_RANGE,
        ],
        aspect="equal",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0.0,
            vmax=max_abs,
        ),
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")
    fig.colorbar(rendered, ax=ax, fraction=0.046, pad=0.04)


def save_panel(
    output_path: Path,
    neutral_all: np.ndarray,
    charged_pu: np.ndarray,
    charged_lv: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    title: str,
    dpi: int,
) -> None:
    residual = prediction - truth

    neutral_vmax = max(
        float(np.nanmax(neutral_all)),
        float(np.nanmax(truth)),
        float(np.nanmax(prediction)),
        1.0e-6,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
    )

    show_positive(
        fig,
        axes[0, 0],
        neutral_all,
        "Contaminated neutral input (9x9)",
        neutral_vmax,
    )
    show_positive(
        fig,
        axes[0, 1],
        charged_pu,
        "Charged PU context (36x36)",
    )
    show_positive(
        fig,
        axes[0, 2],
        charged_lv,
        "Charged LV context (36x36)",
    )
    show_positive(
        fig,
        axes[1, 0],
        truth,
        "True neutral LV (9x9)",
        neutral_vmax,
    )
    show_positive(
        fig,
        axes[1, 1],
        prediction,
        "PileFlow neutral LV (9x9)",
        neutral_vmax,
    )
    show_residual(
        fig,
        axes[1, 2],
        residual,
        "PileFlow minus truth (9x9)",
    )

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz, allow_pickle=False) as source:
        neutral_all = image_batch(
            source["ch_neutral_all_raw"],
            (9, 9),
            "ch_neutral_all_raw",
        )
        charged_pu = image_batch(
            source["ch_charged_pu"],
            (36, 36),
            "ch_charged_pu",
        )
        charged_lv = image_batch(
            source["ch_charged_lv"],
            (36, 36),
            "ch_charged_lv",
        )
        source_truth = image_batch(
            source["ch_neutral_lv"],
            (9, 9),
            "ch_neutral_lv",
        )

        jet_pt = (
            np.asarray(source["jet_pt"], dtype=np.float32)
            if "jet_pt" in source.files
            else None
        )
        n_pu = (
            np.asarray(source["n_pu"], dtype=np.float32)
            if "n_pu" in source.files
            else None
        )

    with np.load(args.generated_npz, allow_pickle=False) as generated:
        prediction = image_batch(
            generated["neutral_lv_pred"],
            (9, 9),
            "neutral_lv_pred",
        )

        if "neutral_lv_true" in generated.files:
            truth = image_batch(
                generated["neutral_lv_true"],
                (9, 9),
                "neutral_lv_true",
            )
        else:
            truth = source_truth

    n_jets = min(
        len(neutral_all),
        len(charged_pu),
        len(charged_lv),
        len(truth),
        len(prediction),
    )

    neutral_all = neutral_all[:n_jets]
    charged_pu = charged_pu[:n_jets]
    charged_lv = charged_lv[:n_jets]
    truth = truth[:n_jets]
    prediction = prediction[:n_jets]

    if jet_pt is not None:
        jet_pt = jet_pt[:n_jets]

    if n_pu is not None:
        n_pu = n_pu[:n_jets]

    difference = prediction - truth

    pixel_mae = np.mean(
        np.abs(difference),
        axis=(1, 2),
    )
    pixel_rmse = np.sqrt(
        np.mean(
            difference**2,
            axis=(1, 2),
        )
    )

    truth_total = truth.sum(axis=(1, 2))
    prediction_total = prediction.sum(axis=(1, 2))
    neutral_all_total = neutral_all.sum(axis=(1, 2))
    total_pt_error = prediction_total - truth_total

    metrics_path = output_dir / "jet_metrics.csv"

    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "jet_index",
            "jet_pt",
            "n_pu",
            "neutral_all_total",
            "truth_neutral_lv_total",
            "prediction_neutral_lv_total",
            "prediction_minus_truth_total",
            "pixel_mae",
            "pixel_rmse",
        ])

        for index in range(n_jets):
            writer.writerow([
                index,
                (
                    float(jet_pt[index])
                    if jet_pt is not None
                    else ""
                ),
                (
                    float(n_pu[index])
                    if n_pu is not None
                    else ""
                ),
                float(neutral_all_total[index]),
                float(truth_total[index]),
                float(prediction_total[index]),
                float(total_pt_error[index]),
                float(pixel_mae[index]),
                float(pixel_rmse[index]),
            ])

    if args.selection == "first":
        ordering = np.arange(n_jets)
    elif args.selection == "worst-mae":
        ordering = np.argsort(pixel_mae)[::-1]
    elif args.selection == "best-mae":
        ordering = np.argsort(pixel_mae)
    else:
        ordering = np.argsort(np.abs(total_pt_error))[::-1]

    count = min(max(args.max_jets, 0), n_jets)
    selected = ordering[:count]

    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    for rank, index in enumerate(selected):
        pt_text = (
            f"{jet_pt[index]:.1f} GeV"
            if jet_pt is not None
            else "unknown"
        )

        npu_text = (
            f"{n_pu[index]:.0f}"
            if n_pu is not None
            else "unknown"
        )

        title = (
            f"Jet {index} | selection rank {rank + 1} | "
            f"jet pT={pt_text} | NPU={npu_text} | "
            f"truth neutral pT={truth_total[index]:.2f} GeV | "
            f"PileFlow={prediction_total[index]:.2f} GeV | "
            f"pixel MAE={pixel_mae[index]:.4f} GeV"
        )

        save_panel(
            output_path=panels_dir / f"jet_{index:06d}.png",
            neutral_all=neutral_all[index],
            charged_pu=charged_pu[index],
            charged_lv=charged_lv[index],
            truth=truth[index],
            prediction=prediction[index],
            title=title,
            dpi=args.dpi,
        )

    save_panel(
        output_path=output_dir / "mean_panel.png",
        neutral_all=neutral_all.mean(axis=0),
        charged_pu=charged_pu.mean(axis=0),
        charged_lv=charged_lv.mean(axis=0),
        truth=truth.mean(axis=0),
        prediction=prediction.mean(axis=0),
        title=f"Mean images over {n_jets:,} evaluation jets",
        dpi=args.dpi,
    )

    print(f"Evaluation jets : {n_jets:,}")
    print(f"Panels saved    : {count:,}")
    print(f"Selection       : {args.selection}")
    print(f"Metrics CSV     : {metrics_path}")
    print(f"Mean panel      : {output_dir / 'mean_panel.png'}")
    print(f"Individual PNGs : {panels_dir}")


if __name__ == "__main__":
    main()
