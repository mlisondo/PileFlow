#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Plot configuration
# ============================================================

CHARGED_LV_COLOR = "#5B1A9A"       # purple
CHARGED_PU_COLOR = "#00A51A"       # green

NEUTRAL_LOW_COLOR = "#FFE600"      # yellow
NEUTRAL_MID_COLOR = "#FF9800"      # orange
NEUTRAL_HIGH_COLOR = "#E00000"     # red

PANE_COLOR = (0.985, 0.985, 0.985, 1.0)
PANE_EDGE = "0.70"

DISPLAY_SIZE = 9


# ============================================================
# Utilities
# ============================================================

def downsample_36_to_9(x: np.ndarray) -> np.ndarray:
    """
    Sum 4x4 blocks.

    (36,36)      -> (9,9)
    (N,36,36)    -> (N,9,9)

    Used ONLY for qualitative visualization so charged and
    neutral components can be drawn on the same 9x9 display grid.
    """

    x = np.asarray(x)

    if x.ndim == 2:

        if x.shape != (36, 36):
            raise ValueError(
                f"Expected (36,36), got {x.shape}"
            )

        return (
            x.reshape(
                9, 4,
                9, 4,
            )
            .sum(axis=(1, 3))
        )

    if x.ndim == 3:

        if x.shape[1:] != (36, 36):
            raise ValueError(
                f"Expected (N,36,36), got {x.shape}"
            )

        return (
            x.reshape(
                len(x),
                9, 4,
                9, 4,
            )
            .sum(axis=(2, 4))
        )

    raise ValueError(
        f"Unsupported shape: {x.shape}"
    )


def load_npz(path: str | Path):

    return np.load(
        path,
        allow_pickle=False,
    )


def get_split_indices(
    d,
    split: str,
) -> np.ndarray:

    if split == "event":

        if "event_test_idx" not in d:
            raise KeyError(
                "Dataset does not contain event_test_idx"
            )

        return np.asarray(
            d["event_test_idx"],
            dtype=np.int64,
        )

    if split == "image":

        if "test_idx" not in d:
            raise KeyError(
                "Dataset does not contain test_idx"
            )

        return np.asarray(
            d["test_idx"],
            dtype=np.int64,
        )

    if split == "all":

        return np.arange(
            len(d["X"]),
            dtype=np.int64,
        )

    raise ValueError(
        f"Unknown split: {split}"
    )


def leading_only_indices(
    d,
    idx: np.ndarray,
) -> np.ndarray:

    rank = np.asarray(
        d["jet_rank"][idx],
        dtype=np.int64,
    )

    return idx[
        rank == 0
    ]


# ============================================================
# N95 helper for representative-jet selection
# ============================================================

def n95_single(image9: np.ndarray) -> int:

    x = np.clip(
        np.asarray(
            image9,
            dtype=np.float64,
        ).ravel(),
        0.0,
        None,
    )

    total = float(
        x.sum()
    )

    if total <= 0.0:
        return 0

    x = np.sort(
        x
    )[::-1]

    return int(
        np.searchsorted(
            np.cumsum(x),
            0.95 * total,
        )
        + 1
    )


# ============================================================
# Representative jet ranking
# ============================================================

def rank_representative_jets(
    leading_idx: np.ndarray,
    X_all: np.ndarray,
    truth_neutral_all: np.ndarray,
    max_candidates: int = 30,
):
    """
    Rank leading jets for qualitative display.

    Favors jets with:
      * non-trivial neutral morphology,
      * several active neutral cells,
      * substantial neutral pT,
      * substantial charged-LV activity,
      * visible charged pileup.

    The aim is NOT to select the best-performing events, but jets
    that are visually informative for comparing reconstruction methods.
    """

    candidates = []

    for src in leading_idx:

        src = int(
            src
        )

        X = X_all[
            src
        ]

        charged_pu36 = (
            X[1]
        )

        charged_lv36 = (
            X[2]
        )

        truth9 = np.asarray(
            truth_neutral_all[src],
            dtype=np.float64,
        )

        neutral_pt = float(
            truth9.sum()
        )

        charged_lv_pt = float(
            charged_lv36.sum()
        )

        charged_pu_pt = float(
            charged_pu36.sum()
        )

        n95 = n95_single(
            truth9
        )

        active_025 = int(
            np.sum(
                truth9 > 0.25
            )
        )

        active_050 = int(
            np.sum(
                truth9 > 0.50
            )
        )

        active_100 = int(
            np.sum(
                truth9 > 1.00
            )
        )

        max_neutral = float(
            truth9.max()
        )

        # ----------------------------------------------------
        # Reject visually trivial jets.
        # ----------------------------------------------------

        if n95 < 3:
            continue

        if active_050 < 3:
            continue

        if neutral_pt < 25.0:
            continue

        if charged_lv_pt < 25.0:
            continue

        if charged_pu_pt < 15.0:
            continue

        # ----------------------------------------------------
        # Avoid extremely diffuse/pathological examples.
        # We want representative jets, not bizarre outliers.
        # ----------------------------------------------------

        if n95 > 14:
            continue

        # ----------------------------------------------------
        # Score.
        #
        # Moderate morphology is favored, rather than simply
        # maximizing N95.
        # ----------------------------------------------------

        morphology_bonus = (
            12.0
            - 1.2
            * abs(
                n95 - 6
            )
        )

        cell_bonus = (
            2.0
            * min(
                active_050,
                10,
            )
        )

        energy_bonus = (
            0.06
            * min(
                neutral_pt,
                150.0,
            )
        )

        lv_bonus = (
            0.015
            * min(
                charged_lv_pt,
                300.0,
            )
        )

        pu_bonus = (
            0.04
            * min(
                charged_pu_pt,
                100.0,
            )
        )

        score = (
            morphology_bonus
            + cell_bonus
            + energy_bonus
            + lv_bonus
            + pu_bonus
        )

        candidates.append(
            {
                "score":
                    float(score),

                "source_index":
                    src,

                "n95":
                    int(n95),

                "active_025":
                    active_025,

                "active_050":
                    active_050,

                "active_100":
                    active_100,

                "neutral_pt":
                    neutral_pt,

                "charged_lv_pt":
                    charged_lv_pt,

                "charged_pu_pt":
                    charged_pu_pt,

                "max_neutral":
                    max_neutral,
            }
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[
        :max_candidates
    ]


def print_representative_candidates(
    candidates,
):

    print()
    print("=" * 118)
    print(
        "REPRESENTATIVE LEADING-JET CANDIDATES"
    )
    print("=" * 118)

    print(
        f"{'rank':>4} "
        f"{'source':>8} "
        f"{'score':>8} "
        f"{'N95':>5} "
        f"{'>0.5':>6} "
        f"{'>1.0':>6} "
        f"{'neutral pT':>12} "
        f"{'charged LV':>12} "
        f"{'charged PU':>12} "
        f"{'max neutral':>12}"
    )

    print("-" * 118)

    for rank, c in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"{rank:4d} "
            f"{c['source_index']:8d} "
            f"{c['score']:8.2f} "
            f"{c['n95']:5d} "
            f"{c['active_050']:6d} "
            f"{c['active_100']:6d} "
            f"{c['neutral_pt']:12.2f} "
            f"{c['charged_lv_pt']:12.2f} "
            f"{c['charged_pu_pt']:12.2f} "
            f"{c['max_neutral']:12.2f}"
        )

    print("=" * 118)


# ============================================================
# Final jet selection
# ============================================================

def choose_rows(
    source_indices: np.ndarray,
    num_jets: int,
    seed: int,
    requested_source_indices: list[int] | None = None,
    selection: str = "representative",
    representative_candidates=None,
) -> np.ndarray:
    """
    Return POSITIONS inside source_indices.

    Priority:
      1) explicit --source-indices
      2) representative ranking
      3) random selection
    """

    # --------------------------------------------------------
    # Explicit user-selected source indices
    # --------------------------------------------------------

    if (
        requested_source_indices is not None
        and len(requested_source_indices) > 0
    ):

        mapping = {
            int(src): i
            for i, src
            in enumerate(
                source_indices.tolist()
            )
        }

        positions = []
        missing = []

        for src in requested_source_indices:

            src = int(
                src
            )

            if src in mapping:

                positions.append(
                    mapping[src]
                )

            else:

                missing.append(
                    src
                )

        if missing:

            raise ValueError(
                "Could not find requested source indices "
                "among leading jets.\n"
                f"Missing: {missing}"
            )

        return np.asarray(
            positions,
            dtype=np.int64,
        )

    # --------------------------------------------------------
    # Automatically choose representative jets
    # --------------------------------------------------------

    if selection == "representative":

        if not representative_candidates:

            raise RuntimeError(
                "Representative selection requested, "
                "but no candidate jets were found."
            )

        chosen_sources = [
            int(
                c["source_index"]
            )
            for c in representative_candidates[
                :num_jets
            ]
        ]

        mapping = {
            int(src): i
            for i, src
            in enumerate(
                source_indices.tolist()
            )
        }

        return np.asarray(
            [
                mapping[src]
                for src in chosen_sources
            ],
            dtype=np.int64,
        )

    # --------------------------------------------------------
    # Random selection
    # --------------------------------------------------------

    if num_jets > len(source_indices):

        raise ValueError(
            f"Requested {num_jets} jets, but only "
            f"{len(source_indices)} are available."
        )

    rng = np.random.default_rng(
        seed
    )

    chosen = rng.choice(
        len(source_indices),
        size=num_jets,
        replace=False,
    )

    return np.asarray(
        chosen,
        dtype=np.int64,
    )


# ============================================================
# PUMML
# ============================================================

def load_pumml_model(
    module_name: str,
    class_name: str,
    checkpoint_path: str,
    device: str,
):

    module = importlib.import_module(
        module_name
    )

    if not hasattr(
        module,
        class_name,
    ):

        raise AttributeError(
            f"Module '{module_name}' "
            f"does not define '{class_name}'"
        )

    model_cls = getattr(
        module,
        class_name,
    )

    model = model_cls()

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(
        ckpt,
        dict,
    ):

        if "model_state_dict" in ckpt:

            state = ckpt[
                "model_state_dict"
            ]

        elif "model_state" in ckpt:

            state = ckpt[
                "model_state"
            ]

        elif "state_dict" in ckpt:

            state = ckpt[
                "state_dict"
            ]

        else:

            is_raw_state = all(
                isinstance(k, str)
                and torch.is_tensor(v)
                for k, v
                in ckpt.items()
            )

            if is_raw_state:
                state = ckpt

            else:
                raise KeyError(
                    "Could not find model weights in "
                    "PUMML checkpoint.\n"
                    f"Checkpoint keys: {list(ckpt.keys())}"
                )

    else:

        state = ckpt

    model.load_state_dict(
        state
    )

    model = model.to(
        device
    )

    model.eval()

    nparams = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Loaded PUMML successfully "
        f"({nparams:,} trainable parameters)"
    )

    return model


def run_pumml(
    model,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:

    preds = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size,
        ):

            stop = min(
                start + batch_size,
                len(X),
            )

            xb = (
                torch
                .from_numpy(
                    X[start:stop]
                )
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            yb = model(
                xb
            )

            if (
                yb.ndim == 4
                and yb.shape[1] == 1
            ):

                yb = yb[:, 0]

            elif yb.ndim != 3:

                raise ValueError(
                    "Unexpected PUMML output shape: "
                    f"{tuple(yb.shape)}"
                )

            preds.append(
                yb
                .detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        preds,
        axis=0,
    ).astype(
        np.float32
    )


# ============================================================
# PileFlow
# ============================================================

def load_pileflow_predictions(
    pileflow_path: str | Path,
    desired_source_indices: np.ndarray,
    estimator: str = "sample",
) -> np.ndarray:
    """PileFlow neutral images for the requested jets.

    estimator 'sample' (default) shows one stochastic draw, i.e. the same
    PileFlow output that compare_methods.py evaluates; 'mean' / 'median'
    show the per-pixel posterior summaries (smoother than any single jet).
    """
    key = {"sample": "neutral_lv_pred", "mean": "neutral_lv_pred_mean",
           "median": "neutral_lv_pred_median"}[estimator]
    with np.load(pileflow_path, allow_pickle=False) as pf:
        if key not in pf.files or "source_indices" not in pf.files:
            raise KeyError(f"{pileflow_path} lacks {key!r} or 'source_indices'")
        pred = np.asarray(pf[key], dtype=np.float32)
        src = np.asarray(pf["source_indices"], dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(src)}
    missing = [int(i) for i in desired_source_indices if int(i) not in pos]
    if missing:
        raise ValueError(f"PileFlow predictions miss source indices {missing}")
    return pred[[pos[int(i)] for i in desired_source_indices]]


# ============================================================
# Color maps
# ============================================================

def make_neutral_cmap():

    return LinearSegmentedColormap.from_list(
        "paper_neutral",
        [
            NEUTRAL_LOW_COLOR,
            NEUTRAL_MID_COLOR,
            NEUTRAL_HIGH_COLOR,
        ],
    )


def make_charged_lv_cmap():

    return LinearSegmentedColormap.from_list(
        "charged_lv",
        [
            "#A99BD9",
            "#7342B6",
            CHARGED_LV_COLOR,
        ],
    )


def make_charged_pu_cmap():

    return LinearSegmentedColormap.from_list(
        "charged_pu",
        [
            "#70D66A",
            "#21B836",
            CHARGED_PU_COLOR,
        ],
    )


# ============================================================
# Cylinder rendering
# ============================================================

def draw_cylinder(
    ax,
    xc: float,
    yc: float,
    radius: float,
    height: float,
    color,
    alpha: float = 1.0,
    n_theta: int = 20,
):

    if (
        not np.isfinite(height)
        or height <= 0.0
    ):
        return

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        n_theta,
    )

    z = np.asarray(
        [0.0, height]
    )

    theta_grid, z_grid = np.meshgrid(
        theta,
        z,
    )

    x_grid = (
        xc
        + radius
        * np.cos(
            theta_grid
        )
    )

    y_grid = (
        yc
        + radius
        * np.sin(
            theta_grid
        )
    )

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=alpha,
    )

    radial = np.linspace(
        0.0,
        radius,
        8,
    )

    rr, tt = np.meshgrid(
        radial,
        theta,
    )

    x_top = (
        xc
        + rr
        * np.cos(tt)
    )

    y_top = (
        yc
        + rr
        * np.sin(tt)
    )

    z_top = np.full_like(
        x_top,
        height,
    )

    ax.plot_surface(
        x_top,
        y_top,
        z_top,
        color=color,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=alpha,
    )


def plot_component_cylinders(
    ax,
    image9: np.ndarray,
    cmap,
    vmax: float,
    radius: float,
    threshold: float,
    alpha: float = 1.0,
):

    image9 = np.asarray(
        image9,
        dtype=np.float64,
    )

    norm = Normalize(
        vmin=0.0,
        vmax=max(
            float(vmax),
            1.0e-12,
        ),
    )

    for iy in range(
        DISPLAY_SIZE
    ):

        for ix in range(
            DISPLAY_SIZE
        ):

            value = float(
                image9[
                    iy,
                    ix,
                ]
            )

            if (
                not np.isfinite(value)
                or value <= threshold
            ):
                continue

            xc = ix + 0.5
            yc = iy + 0.5

            color = cmap(
                norm(value)
            )

            draw_cylinder(
                ax=ax,
                xc=xc,
                yc=yc,
                radius=radius,
                height=value,
                color=color,
                alpha=alpha,
            )


# ============================================================
# Axis styling
# ============================================================

def style_3d_axis(
    ax,
    zmax: float,
):

    ax.set_xlim(
        0,
        DISPLAY_SIZE,
    )

    ax.set_ylim(
        0,
        DISPLAY_SIZE,
    )

    ax.set_zlim(
        0,
        max(
            zmax,
            1.0,
        ),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.view_init(
        elev=29,
        azim=-61,
    )

    ax.set_box_aspect(
        (
            1.0,
            1.0,
            0.72,
        )
    )

    ax.xaxis.pane.set_facecolor(
        PANE_COLOR
    )

    ax.yaxis.pane.set_facecolor(
        PANE_COLOR
    )

    ax.zaxis.pane.set_facecolor(
        (
            1.0,
            1.0,
            1.0,
            0.0,
        )
    )

    ax.xaxis.pane.set_edgecolor(
        PANE_EDGE
    )

    ax.yaxis.pane.set_edgecolor(
        PANE_EDGE
    )

    ax.zaxis.pane.set_edgecolor(
        "white"
    )

    ax.grid(
        False
    )


# ============================================================
# Individual panel
# ============================================================

def plot_one_panel(
    ax,
    neutral9: np.ndarray | None,
    charged_lv9: np.ndarray | None,
    charged_pu9: np.ndarray | None,
    zmax: float,
    title: str | None = None,
):

    neutral_cmap = (
        make_neutral_cmap()
    )

    charged_lv_cmap = (
        make_charged_lv_cmap()
    )

    charged_pu_cmap = (
        make_charged_pu_cmap()
    )

    if charged_pu9 is not None:

        pu_max = max(
            float(
                np.max(
                    charged_pu9
                )
            ),
            1.0e-12,
        )

        plot_component_cylinders(
            ax=ax,
            image9=charged_pu9,
            cmap=charged_pu_cmap,
            vmax=pu_max,
            radius=0.18,
            threshold=0.01,
            alpha=0.92,
        )

    if charged_lv9 is not None:

        lv_max = max(
            float(
                np.max(
                    charged_lv9
                )
            ),
            1.0e-12,
        )

        plot_component_cylinders(
            ax=ax,
            image9=charged_lv9,
            cmap=charged_lv_cmap,
            vmax=lv_max,
            radius=0.22,
            threshold=0.01,
            alpha=0.98,
        )

    if neutral9 is not None:

        neutral_max = max(
            float(
                np.max(
                    neutral9
                )
            ),
            1.0e-12,
        )

        plot_component_cylinders(
            ax=ax,
            image9=neutral9,
            cmap=neutral_cmap,
            vmax=neutral_max,
            radius=0.34,
            threshold=0.01,
            alpha=0.98,
        )

    style_3d_axis(
        ax,
        zmax=zmax,
    )

    if title is not None:

        ax.set_title(
            title,
            fontsize=19,
            pad=6,
        )


# ============================================================
# Legend
# ============================================================

def add_figure_legend(
    fig,
):

    legend_handles = [

        Patch(
            facecolor=CHARGED_LV_COLOR,
            edgecolor="none",
            label="Charged leading vertex",
        ),

        Patch(
            facecolor=CHARGED_PU_COLOR,
            edgecolor="none",
            label="Charged pileup",
        ),

        Patch(
            facecolor=NEUTRAL_LOW_COLOR,
            edgecolor="none",
            label=r"Neutral: lower $p_T$",
        ),

        Patch(
            facecolor=NEUTRAL_MID_COLOR,
            edgecolor="none",
            label=r"Neutral: intermediate $p_T$",
        ),

        Patch(
            facecolor=NEUTRAL_HIGH_COLOR,
            edgecolor="none",
            label=r"Neutral: higher $p_T$",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        ncol=5,
        frameon=False,
        fontsize=12,
        handlelength=1.4,
        columnspacing=1.8,
    )


# ============================================================
# Per-row z scale
# ============================================================

def determine_row_zmax(
    row_panels,
    percentile: float = 99.5,
):

    values = []

    for panel in row_panels:

        for key in [
            "neutral9",
            "charged_lv9",
            "charged_pu9",
        ]:

            arr = panel[
                key
            ]

            if arr is None:
                continue

            arr = np.asarray(
                arr,
                dtype=np.float64,
            )

            positive = arr[
                np.isfinite(arr)
                & (arr > 0)
            ]

            if len(positive):

                values.append(
                    positive
                )

    if not values:

        return 1.0

    values = np.concatenate(
        values
    )

    q = float(
        np.percentile(
            values,
            percentile,
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    return (
        max(
            q,
            maximum,
            1.0,
        )
        * 1.08
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PUMML-paper-style qualitative jet visualization "
            "for Truth / Pileup / PUMML / PUPPI / PileFlow."
        )
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--pumml-model", required=True)
    parser.add_argument("--pumml-module", default="pumml.pumml_arch")
    parser.add_argument("--pumml-class", default="PUMML")
    parser.add_argument("--pileflow", required=True)
    parser.add_argument("--outdir", default="paper_jet_images")
    parser.add_argument("--split", choices=["event", "image", "all",], default="event")
    parser.add_argument("--num-jets", type=int, default=3)
    parser.add_argument("--source-indices", nargs="*", type=int, default=None)

    parser.add_argument(
        "--selection",
        choices=[
            "representative",
            "random",
        ],
        default="representative",
        help=(
            "How to choose jets when --source-indices is not given. "
            "Default: representative."
        ),
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=30,
        help=(
            "Number of representative candidates printed "
            "before the final selection."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--pileflow-threshold",
        type=float,
        default=0.0,
        help="Use the value chosen on validation data by select_threshold.py.",
    )

    parser.add_argument(
        "--cell-threshold",
        type=float,
        default=0.0,
        help="Neutral-cell cut in GeV on every method (e.g. 0.1); 0 = no cut.",
    )

    parser.add_argument(
        "--threshold-truth",
        action="store_true",
    )

    parser.add_argument(
        "--pileflow-estimator",
        choices=["sample", "mean", "median"],
        default="sample",
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(parents=True, exist_ok=True)


    # Dataset
    d = load_npz(args.data)

    split_idx = get_split_indices(d, args.split)

    leading_idx = leading_only_indices(d, split_idx)


    # Representative candidate scan
    X_all = np.asarray(d["X"], dtype=np.float32)

    truth_neutral_all = np.asarray(d["y_neutral_lv_9"], dtype=np.float32)

    representative_candidates = (
        rank_representative_jets(
            leading_idx=leading_idx,
            X_all=X_all,
            truth_neutral_all=truth_neutral_all,
            max_candidates=args.candidate_pool,
        )
    )

    print_representative_candidates(
        representative_candidates
    )


    # Final selection
    chosen_positions = choose_rows(
        source_indices=leading_idx,
        num_jets=args.num_jets,
        seed=args.seed,
        requested_source_indices=args.source_indices,
        selection=args.selection,
        representative_candidates=representative_candidates,
    )

    chosen_source_idx = (
        leading_idx[
            chosen_positions
        ]
    )

    print()
    print("SELECTED JETS FOR QUALITATIVE PANEL")

    print("=" * 72)

    for j, src in enumerate(
        chosen_source_idx.tolist(),
        start=1,
    ):

        event_id = int(
            np.asarray(
                d["event_id"][src]
            ).item()
        )

        rank = int(
            np.asarray(
                d["jet_rank"][src]
            ).item()
        )

        truth9 = truth_neutral_all[
            src
        ]

        print(
            f"Jet {j}: "
            f"source_index={src:<8d} "
            f"event_id={event_id} "
            f"jet_rank={rank} "
            f"N95={n95_single(truth9)} "
            f"neutral_pT={truth9.sum():.2f} GeV"
        )


    # Input / truth
    X = np.asarray(
        d["X"][
            chosen_source_idx
        ],
        dtype=np.float32,
    )

    total_neutral36 = (
        X[:, 0]
    )

    charged_pu36 = (
        X[:, 1]
    )

    charged_lv36 = (
        X[:, 2]
    )

    truth_neutral9 = np.asarray(
        d["y_neutral_lv_9"][
            chosen_source_idx
        ],
        dtype=np.float32,
    )

    puppi_neutral9 = np.asarray(
        d["puppi_neutral_9"][
            chosen_source_idx
        ],
        dtype=np.float32,
    )

    charged_pu9 = (
        downsample_36_to_9(
            charged_pu36
        )
    )

    charged_lv9 = (
        downsample_36_to_9(
            charged_lv36
        )
    )

    total_neutral9 = (
        downsample_36_to_9(
            total_neutral36
        )
    )


    # PUMML
    print()
    print(
        "Loading PUMML..."
    )

    pumml_model = load_pumml_model(
        module_name=args.pumml_module,
        class_name=args.pumml_class,
        checkpoint_path=args.pumml_model,
        device=args.device,
    )

    pumml_pred9 = run_pumml(
        model=pumml_model,
        X=X,
        batch_size=args.batch_size,
        device=args.device,
    )


    # PileFlow
    print(
        "Loading PileFlow predictions..."
    )

    pileflow_pred9 = (
        load_pileflow_predictions(
            pileflow_path=args.pileflow,
            desired_source_indices=chosen_source_idx,
            estimator=args.pileflow_estimator,
        )
    )

    pileflow_pred9 = np.clip(
        pileflow_pred9,
        0.0,
        None,
    )

    if (
        args.pileflow_threshold
        > 0.0
    ):

        pileflow_pred9 = np.where(
            pileflow_pred9
            >= args.pileflow_threshold,
            pileflow_pred9,
            0.0,
        )

    # Common neutral-cell cut on every reconstruction (same as compare_methods --cut-mode on)
    if args.cell_threshold > 0.0:
        cut = lambda a: np.where(np.clip(a, 0.0, None) >= args.cell_threshold, a, 0.0)
        pumml_pred9, puppi_neutral9, pileflow_pred9 = cut(pumml_pred9), cut(puppi_neutral9), cut(pileflow_pred9)
        total_neutral9 = cut(total_neutral9)
        if args.threshold_truth:
            truth_neutral9 = cut(truth_neutral9)
        print(f"Applied a {1000 * args.cell_threshold:.0f} MeV neutral-cell cut to all reconstructions")


    # Panel representations
    column_titles = [
        "Leading Vertex",
        "with Pileup",
        "PUMML",
        "PUPPI",
        "PileFlow",
    ]

    panel_specs = []

    for i in range(
        len(
            chosen_source_idx
        )
    ):

        panel_specs.append(
            [
                {
                    "neutral9":
                        truth_neutral9[i],

                    "charged_lv9":
                        charged_lv9[i],

                    "charged_pu9":
                        None,
                },

                {
                    "neutral9":
                        total_neutral9[i],

                    "charged_lv9":
                        charged_lv9[i],

                    "charged_pu9":
                        charged_pu9[i],
                },

                {
                    "neutral9":
                        pumml_pred9[i],

                    "charged_lv9":
                        charged_lv9[i],

                    "charged_pu9":
                        None,
                },

                {
                    "neutral9":
                        puppi_neutral9[i],

                    "charged_lv9":
                        charged_lv9[i],

                    "charged_pu9":
                        None,
                },

                {
                    "neutral9":
                        pileflow_pred9[i],

                    "charged_lv9":
                        charged_lv9[i],

                    "charged_pu9":
                        None,
                },
            ]
        )


    # Row-specific z scale
    row_zmax = [
        determine_row_zmax(
            row
        )
        for row
        in panel_specs
    ]

    print()
    print(
        "Vertical scales:"
    )

    for i, value in enumerate(
        row_zmax,
        start=1,
    ):

        print(
            f"  Jet {i}: "
            f"zmax = {value:.2f} GeV"
        )


    # Figure
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    nrows = len(
        chosen_source_idx
    )

    ncols = 5

    fig = plt.figure(
        figsize=(
            17.5,
            3.65 * nrows + 0.8,
        )
    )

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        left=0.055,
        right=0.995,
        bottom=0.11,
        top=0.94,
        wspace=-0.02,
        hspace=0.02,
    )

    axes = []

    for r in range(
        nrows
    ):

        row_axes = []

        for c in range(
            ncols
        ):

            ax = fig.add_subplot(
                gs[r, c],
                projection="3d",
            )

            row_axes.append(
                ax
            )

        axes.append(
            row_axes
        )


    # Draw
    for r in range(
        nrows
    ):

        for c in range(
            ncols
        ):

            title = (
                column_titles[c]
                if r == 0
                else None
            )

            plot_one_panel(
                ax=axes[r][c],

                neutral9=
                    panel_specs[r][c][
                        "neutral9"
                    ],

                charged_lv9=
                    panel_specs[r][c][
                        "charged_lv9"
                    ],

                charged_pu9=
                    panel_specs[r][c][
                        "charged_pu9"
                    ],

                zmax=row_zmax[r],

                title=title,
            )


    # Row labels
    for r in range(
        nrows
    ):

        bbox = (
            axes[r][0]
            .get_position()
        )

        ycenter = (
            0.5
            * (
                bbox.y0
                + bbox.y1
            )
        )

        fig.text(
            0.020,
            ycenter,
            f"Jet {r + 1}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=18,
        )

    # Legend
    add_figure_legend(fig)


    # Save
    png = (
        outdir
        / "qualitative_jets_figure3_style.png"
    )

    pdf = (
        outdir
        / "qualitative_jets_figure3_style.pdf"
    )

    txt = (
        outdir
        / "selected_jets.txt"
    )

    fig.savefig(
        png,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # Store selection
    with open(
        txt,
        "w",
    ) as f:

        f.write("Selected jets for qualitative panel\n")
        f.write("=" * 72+ "\n")
        f.write(f"selection = {args.selection}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f"pileflow_threshold = " f"{args.pileflow_threshold} GeV\n\n")

        for j, src in enumerate(
            chosen_source_idx.tolist(),
            start=1,
        ):

            event_id = int(
                np.asarray(
                    d["event_id"][src]
                ).item()
            )

            rank = int(
                np.asarray(
                    d["jet_rank"][src]
                ).item()
            )

            truth9 = truth_neutral_all[src]

            f.write(
                f"Jet {j}: "
                f"source_index={src}  "
                f"event_id={event_id}  "
                f"jet_rank={rank}  "
                f"N95={n95_single(truth9)}  "
                f"neutral_pt={truth9.sum():.6f}  "
                f"zmax={row_zmax[j - 1]:.6f}\n"
)
    print()
    print("=" * 72)
    print("Saved publication figure")
    print("=" * 72)
    print(f"  {png}")
    print(f"  {pdf}")
    print(f"  {txt}")


if __name__ == "__main__":
    main()