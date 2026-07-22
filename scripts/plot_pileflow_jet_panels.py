#!/usr/bin/env python3
"""
Visualize single-sample and N-sample-mean PileFlow predictions.

The generated NPZ must contain:

    neutral_lv_pred_single
        First stochastic PileFlow sample for each jet.

    neutral_lv_pred_mean
        Pixelwise mean of N independently generated samples for each jet.

The script creates four folders containing M panels:

    single_best_mae/
    single_worst_mae/
    mean_best_total_pt/
    mean_worst_total_pt/

It also creates paired_extremes/ with exactly four comparison plots:

    1. best single-sample MAE event;
    2. best N-sample-mean MAE event;
    3. worst single-sample MAE event;
    4. worst N-sample-mean MAE event.

Every panel uses a 2x4 layout:

Top row:
    contaminated neutral | charged PU | charged LV | truth neutral LV

Bottom row:
    single prediction | single residual | N-sample mean | mean residual
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
    parser = argparse.ArgumentParser(
        description=(
            "Create single-sample and N-sample-mean PileFlow jet panels."
        )
    )

    parser.add_argument(
        "--input-npz",
        required=True,
        help="Corrected generator evaluation NPZ.",
    )
    parser.add_argument(
        "--generated-npz",
        required=True,
        help=(
            "PileFlow data/generated_jets.npz containing the single and "
            "mean prediction keys."
        ),
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=12,
        help=(
            "Number M of panels written to each of the four main folders. "
            "Default: 12."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180.",
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


def require_equal_rows(arrays: dict[str, np.ndarray]) -> int:
    """Require all row-aligned arrays to contain the same number of jets."""
    lengths = {
        key: len(value)
        for key, value in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        details = ", ".join(
            f"{key}={length}"
            for key, length in lengths.items()
        )
        raise ValueError(
            "Input arrays are not row aligned: "
            f"{details}"
        )

    return next(iter(lengths.values()))


def show_positive(
    fig,
    axis,
    image: np.ndarray,
    title: str,
    vmax: float | None = None,
) -> None:
    """Display a nonnegative pT image."""
    finite = image[np.isfinite(image)]

    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0

    vmax = max(float(vmax), 1.0e-6)

    rendered = axis.imshow(
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

    axis.set_title(title, fontsize=9)
    axis.set_xlabel(r"$\Delta\eta$")
    axis.set_ylabel(r"$\Delta\phi$")
    fig.colorbar(
        rendered,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=r"$p_T$ [GeV]",
    )


def show_residual(
    fig,
    axis,
    residual: np.ndarray,
    title: str,
    max_abs: float,
) -> None:
    """Display a signed prediction-minus-truth residual."""
    max_abs = max(float(max_abs), 1.0e-6)

    rendered = axis.imshow(
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

    axis.set_title(title, fontsize=9)
    axis.set_xlabel(r"$\Delta\eta$")
    axis.set_ylabel(r"$\Delta\phi$")
    fig.colorbar(
        rendered,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=r"$\Delta p_T$ [GeV]",
    )


def save_comparison_panel(
    output_path: Path,
    neutral_all: np.ndarray,
    charged_pu: np.ndarray,
    charged_lv: np.ndarray,
    truth: np.ndarray,
    single_prediction: np.ndarray,
    mean_prediction: np.ndarray,
    title: str,
    eval_samples: int,
    dpi: int,
) -> None:
    """Save one 2x4 same-event comparison panel."""
    single_residual = single_prediction - truth
    mean_residual = mean_prediction - truth

    neutral_vmax = max(
        float(np.nanmax(neutral_all)),
        float(np.nanmax(truth)),
        float(np.nanmax(single_prediction)),
        float(np.nanmax(mean_prediction)),
        1.0e-6,
    )

    residual_max_abs = max(
        float(np.nanmax(np.abs(single_residual))),
        float(np.nanmax(np.abs(mean_residual))),
        1.0e-6,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(20, 9.5),
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
        axes[0, 3],
        truth,
        "True neutral LV (9x9)",
        neutral_vmax,
    )
    show_positive(
        fig,
        axes[1, 0],
        single_prediction,
        "Single PileFlow sample (sample 1)",
        neutral_vmax,
    )
    show_residual(
        fig,
        axes[1, 1],
        single_residual,
        "Single minus truth",
        residual_max_abs,
    )
    show_positive(
        fig,
        axes[1, 2],
        mean_prediction,
        f"PileFlow mean of N={eval_samples} samples",
        neutral_vmax,
    )
    show_residual(
        fig,
        axes[1, 3],
        mean_residual,
        "Mean minus truth",
        residual_max_abs,
    )

    fig.suptitle(
        title,
        fontsize=11,
    )
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def format_metadata(
    index: int,
    selection_label: str,
    selection_rank: int | None,
    jet_pt: np.ndarray | None,
    n_pu: np.ndarray | None,
    eval_samples: int,
    truth_total: np.ndarray,
    single_total: np.ndarray,
    mean_total: np.ndarray,
    single_mae: np.ndarray,
    mean_mae: np.ndarray,
) -> str:
    """Build a two-line title for one jet."""
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
    rank_text = (
        f" | selection rank {selection_rank}"
        if selection_rank is not None
        else ""
    )

    return (
        f"Jet {index} | {selection_label}{rank_text} | "
        f"jet pT={pt_text} | NPU={npu_text} | N={eval_samples}\n"
        f"truth neutral pT={truth_total[index]:.2f} GeV | "
        f"single={single_total[index]:.2f} GeV "
        f"(MAE={single_mae[index]:.4f} GeV) | "
        f"mean={mean_total[index]:.2f} GeV "
        f"(MAE={mean_mae[index]:.4f} GeV)"
    )


def write_metrics_csv(
    output_path: Path,
    n_jets: int,
    jet_pt: np.ndarray | None,
    n_pu: np.ndarray | None,
    neutral_all_total: np.ndarray,
    truth_total: np.ndarray,
    single_total: np.ndarray,
    mean_total: np.ndarray,
    single_total_error: np.ndarray,
    mean_total_error: np.ndarray,
    single_mae: np.ndarray,
    mean_mae: np.ndarray,
    single_rmse: np.ndarray,
    mean_rmse: np.ndarray,
) -> None:
    """Write per-jet diagnostics for both prediction modes."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "jet_index",
            "jet_pt",
            "n_pu",
            "neutral_all_total",
            "truth_neutral_lv_total",
            "single_neutral_lv_total",
            "mean_neutral_lv_total",
            "single_minus_truth_total",
            "mean_minus_truth_total",
            "single_abs_total_pt_error",
            "mean_abs_total_pt_error",
            "single_pixel_mae",
            "mean_pixel_mae",
            "single_pixel_rmse",
            "mean_pixel_rmse",
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
                float(single_total[index]),
                float(mean_total[index]),
                float(single_total_error[index]),
                float(mean_total_error[index]),
                float(abs(single_total_error[index])),
                float(abs(mean_total_error[index])),
                float(single_mae[index]),
                float(mean_mae[index]),
                float(single_rmse[index]),
                float(mean_rmse[index]),
            ])


def save_selection_folder(
    root_dir: Path,
    folder_name: str,
    ordering: np.ndarray,
    count: int,
    selection_label: str,
    neutral_all: np.ndarray,
    charged_pu: np.ndarray,
    charged_lv: np.ndarray,
    truth: np.ndarray,
    single_prediction: np.ndarray,
    mean_prediction: np.ndarray,
    jet_pt: np.ndarray | None,
    n_pu: np.ndarray | None,
    eval_samples: int,
    truth_total: np.ndarray,
    single_total: np.ndarray,
    mean_total: np.ndarray,
    single_mae: np.ndarray,
    mean_mae: np.ndarray,
    dpi: int,
) -> None:
    """Save M selected events using the common 2x4 layout."""
    folder = root_dir / folder_name
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = ordering[:count]
    summary_path = folder / "selected_jets.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "selection_rank",
            "jet_index",
            "single_pixel_mae",
            "mean_pixel_mae",
            "single_abs_total_pt_error",
            "mean_abs_total_pt_error",
        ])

        for rank, raw_index in enumerate(selected, start=1):
            index = int(raw_index)

            writer.writerow([
                rank,
                index,
                float(single_mae[index]),
                float(mean_mae[index]),
                float(abs(single_total[index] - truth_total[index])),
                float(abs(mean_total[index] - truth_total[index])),
            ])

            title = format_metadata(
                index=index,
                selection_label=selection_label,
                selection_rank=rank,
                jet_pt=jet_pt,
                n_pu=n_pu,
                eval_samples=eval_samples,
                truth_total=truth_total,
                single_total=single_total,
                mean_total=mean_total,
                single_mae=single_mae,
                mean_mae=mean_mae,
            )

            save_comparison_panel(
                output_path=(
                    folder
                    / f"rank_{rank:03d}_jet_{index:06d}.png"
                ),
                neutral_all=neutral_all[index],
                charged_pu=charged_pu[index],
                charged_lv=charged_lv[index],
                truth=truth[index],
                single_prediction=single_prediction[index],
                mean_prediction=mean_prediction[index],
                title=title,
                eval_samples=eval_samples,
                dpi=dpi,
            )


def save_paired_extremes(
    root_dir: Path,
    neutral_all: np.ndarray,
    charged_pu: np.ndarray,
    charged_lv: np.ndarray,
    truth: np.ndarray,
    single_prediction: np.ndarray,
    mean_prediction: np.ndarray,
    jet_pt: np.ndarray | None,
    n_pu: np.ndarray | None,
    eval_samples: int,
    truth_total: np.ndarray,
    single_total: np.ndarray,
    mean_total: np.ndarray,
    single_mae: np.ndarray,
    mean_mae: np.ndarray,
    dpi: int,
) -> None:
    """
    Save exactly four same-event comparisons.

    Single and mean extrema are both defined by pixel MAE so that this paired
    comparison uses the same performance criterion for both prediction modes.
    """
    folder = root_dir / "paired_extremes"
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = [
        (
            "01_best_single_event",
            int(np.argmin(single_mae)),
            "Best single-sample MAE event",
        ),
        (
            "02_best_mean_event",
            int(np.argmin(mean_mae)),
            "Best N-sample-mean MAE event",
        ),
        (
            "03_worst_single_event",
            int(np.argmax(single_mae)),
            "Worst single-sample MAE event",
        ),
        (
            "04_worst_mean_event",
            int(np.argmax(mean_mae)),
            "Worst N-sample-mean MAE event",
        ),
    ]

    summary_path = folder / "paired_extremes.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case",
            "jet_index",
            "single_pixel_mae",
            "mean_pixel_mae",
            "single_abs_total_pt_error",
            "mean_abs_total_pt_error",
        ])

        for file_prefix, index, label in cases:
            writer.writerow([
                label,
                index,
                float(single_mae[index]),
                float(mean_mae[index]),
                float(abs(single_total[index] - truth_total[index])),
                float(abs(mean_total[index] - truth_total[index])),
            ])

            title = format_metadata(
                index=index,
                selection_label=label,
                selection_rank=None,
                jet_pt=jet_pt,
                n_pu=n_pu,
                eval_samples=eval_samples,
                truth_total=truth_total,
                single_total=single_total,
                mean_total=mean_total,
                single_mae=single_mae,
                mean_mae=mean_mae,
            )

            save_comparison_panel(
                output_path=(
                    folder
                    / f"{file_prefix}_jet_{index:06d}.png"
                ),
                neutral_all=neutral_all[index],
                charged_pu=charged_pu[index],
                charged_lv=charged_lv[index],
                truth=truth[index],
                single_prediction=single_prediction[index],
                mean_prediction=mean_prediction[index],
                title=title,
                eval_samples=eval_samples,
                dpi=dpi,
            )


def main() -> None:
    args = parse_args()

    if args.max_jets < 0:
        raise ValueError(
            f"--max-jets must be nonnegative, got {args.max_jets}"
        )

    output_dir = Path(args.outdir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        args.input_npz,
        allow_pickle=False,
    ) as source:
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
            np.asarray(
                source["jet_pt"],
                dtype=np.float32,
            )
            if "jet_pt" in source.files
            else None
        )
        n_pu = (
            np.asarray(
                source["n_pu"],
                dtype=np.float32,
            )
            if "n_pu" in source.files
            else None
        )

    with np.load(
        args.generated_npz,
        allow_pickle=False,
    ) as generated:
        required_prediction_keys = [
            "neutral_lv_pred_single",
            "neutral_lv_pred_mean",
        ]
        missing = [
            key
            for key in required_prediction_keys
            if key not in generated.files
        ]

        if missing:
            raise KeyError(
                "The generated NPZ does not contain the new single/mean "
                f"prediction keys: {missing}. Rerun PileFlow generation "
                "after updating flows/training/train_flow.py."
            )

        single_prediction = image_batch(
            generated["neutral_lv_pred_single"],
            (9, 9),
            "neutral_lv_pred_single",
        )
        mean_prediction = image_batch(
            generated["neutral_lv_pred_mean"],
            (9, 9),
            "neutral_lv_pred_mean",
        )

        if "neutral_lv_true" in generated.files:
            truth = image_batch(
                generated["neutral_lv_true"],
                (9, 9),
                "neutral_lv_true",
            )
        else:
            truth = source_truth

        eval_samples = (
            int(
                np.asarray(
                    generated["eval_samples"]
                ).reshape(()).item()
            )
            if "eval_samples" in generated.files
            else 1
        )

        if "neutral_lv_pred" in generated.files:
            compatibility_prediction = image_batch(
                generated["neutral_lv_pred"],
                (9, 9),
                "neutral_lv_pred",
            )

            if not np.array_equal(
                compatibility_prediction,
                mean_prediction,
            ):
                raise ValueError(
                    "neutral_lv_pred is not identical to "
                    "neutral_lv_pred_mean."
                )

    aligned = {
        "neutral_all": neutral_all,
        "charged_pu": charged_pu,
        "charged_lv": charged_lv,
        "truth": truth,
        "single_prediction": single_prediction,
        "mean_prediction": mean_prediction,
    }

    n_jets = require_equal_rows(
        aligned
    )

    if jet_pt is not None and len(jet_pt) != n_jets:
        raise ValueError(
            f"jet_pt has {len(jet_pt)} rows; expected {n_jets}."
        )

    if n_pu is not None and len(n_pu) != n_jets:
        raise ValueError(
            f"n_pu has {len(n_pu)} rows; expected {n_jets}."
        )

    if eval_samples <= 0:
        raise ValueError(
            f"Saved eval_samples must be positive, got {eval_samples}."
        )

    single_difference = (
        single_prediction - truth
    )
    mean_difference = (
        mean_prediction - truth
    )

    single_mae = np.mean(
        np.abs(single_difference),
        axis=(1, 2),
    )
    mean_mae = np.mean(
        np.abs(mean_difference),
        axis=(1, 2),
    )

    single_rmse = np.sqrt(
        np.mean(
            single_difference**2,
            axis=(1, 2),
        )
    )
    mean_rmse = np.sqrt(
        np.mean(
            mean_difference**2,
            axis=(1, 2),
        )
    )

    truth_total = truth.sum(
        axis=(1, 2)
    )
    single_total = single_prediction.sum(
        axis=(1, 2)
    )
    mean_total = mean_prediction.sum(
        axis=(1, 2)
    )
    neutral_all_total = neutral_all.sum(
        axis=(1, 2)
    )

    single_total_error = (
        single_total - truth_total
    )
    mean_total_error = (
        mean_total - truth_total
    )

    metrics_path = (
        output_dir / "jet_metrics.csv"
    )

    write_metrics_csv(
        output_path=metrics_path,
        n_jets=n_jets,
        jet_pt=jet_pt,
        n_pu=n_pu,
        neutral_all_total=neutral_all_total,
        truth_total=truth_total,
        single_total=single_total,
        mean_total=mean_total,
        single_total_error=single_total_error,
        mean_total_error=mean_total_error,
        single_mae=single_mae,
        mean_mae=mean_mae,
        single_rmse=single_rmse,
        mean_rmse=mean_rmse,
    )

    count = min(
        args.max_jets,
        n_jets,
    )

    selections = [
        (
            "single_best_mae",
            np.argsort(single_mae),
            "Selected by lowest single-sample pixel MAE",
        ),
        (
            "single_worst_mae",
            np.argsort(single_mae)[::-1],
            "Selected by highest single-sample pixel MAE",
        ),
        (
            "mean_best_total_pt",
            np.argsort(
                np.abs(mean_total_error)
            ),
            "Selected by lowest mean absolute total-pT error",
        ),
        (
            "mean_worst_total_pt",
            np.argsort(
                np.abs(mean_total_error)
            )[::-1],
            "Selected by highest mean absolute total-pT error",
        ),
    ]

    for folder_name, ordering, label in selections:
        save_selection_folder(
            root_dir=output_dir,
            folder_name=folder_name,
            ordering=ordering,
            count=count,
            selection_label=label,
            neutral_all=neutral_all,
            charged_pu=charged_pu,
            charged_lv=charged_lv,
            truth=truth,
            single_prediction=single_prediction,
            mean_prediction=mean_prediction,
            jet_pt=jet_pt,
            n_pu=n_pu,
            eval_samples=eval_samples,
            truth_total=truth_total,
            single_total=single_total,
            mean_total=mean_total,
            single_mae=single_mae,
            mean_mae=mean_mae,
            dpi=args.dpi,
        )

    save_paired_extremes(
        root_dir=output_dir,
        neutral_all=neutral_all,
        charged_pu=charged_pu,
        charged_lv=charged_lv,
        truth=truth,
        single_prediction=single_prediction,
        mean_prediction=mean_prediction,
        jet_pt=jet_pt,
        n_pu=n_pu,
        eval_samples=eval_samples,
        truth_total=truth_total,
        single_total=single_total,
        mean_total=mean_total,
        single_mae=single_mae,
        mean_mae=mean_mae,
        dpi=args.dpi,
    )

    save_comparison_panel(
        output_path=(
            output_dir / "mean_panel.png"
        ),
        neutral_all=neutral_all.mean(axis=0),
        charged_pu=charged_pu.mean(axis=0),
        charged_lv=charged_lv.mean(axis=0),
        truth=truth.mean(axis=0),
        single_prediction=single_prediction.mean(axis=0),
        mean_prediction=mean_prediction.mean(axis=0),
        title=(
            f"Mean images over {n_jets:,} evaluation jets | "
            f"N={eval_samples}"
        ),
        eval_samples=eval_samples,
        dpi=args.dpi,
    )

    print(f"Evaluation jets        : {n_jets:,}")
    print(f"Samples in mean (N)    : {eval_samples}")
    print(f"Panels per main folder : {count}")
    print(f"Metrics CSV            : {metrics_path}")
    print(f"Mean panel             : {output_dir / 'mean_panel.png'}")
    print("Main folders:")
    print(f"  {output_dir / 'single_best_mae'}")
    print(f"  {output_dir / 'single_worst_mae'}")
    print(f"  {output_dir / 'mean_best_total_pt'}")
    print(f"  {output_dir / 'mean_worst_total_pt'}")
    print(
        "Paired comparisons     : "
        f"{output_dir / 'paired_extremes'} "
        "(exactly 4 PNGs)"
    )


if __name__ == "__main__":
    main()
