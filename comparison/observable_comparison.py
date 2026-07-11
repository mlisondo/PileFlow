"""
comparison/observable_comparison.py
===================================

PUMML-inspired observable evaluation on the PileFlow ppjj benchmark.

All methods are evaluated using the same detector-image representation:

  True:
      neutral-LV image + charged-LV image

  w/ Pileup:
      neutral-all image + charged-LV image + charged-PU image

  PUPPI:
      PUPPI neutral image + PUPPI charged image

  PUMML:
      predicted neutral-LV image + charged-LV image

  PileFlow:
      predicted neutral-LV image + charged-LV image

Observables:
  jet_mass | dijet_mass | jet_pt | neutral_n95 | ecf2_log | ecf3_log

This is not an exact numerical reproduction of the PUMML paper benchmark.
The dataset, hard process, pileup distribution, and event selection differ.
"""

from __future__ import annotations

import os
import io
import csv
import math
import contextlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Observable configuration for the PUMML-inspired comparison.
OBS_KEYS = [
    "jet_mass",
    "dijet_mass",
    "jet_pt",
    "neutral_n95",
    "ecf2_log",
    "ecf3_log",
]

# Observables evaluated only for the stored leading jet in each event.
LEADING_KEYS = [
    "jet_mass",
    "jet_pt",
    "neutral_n95",
    "ecf2_log",
    "ecf3_log",
]

# Table 2 contains percent-error IQRs. Neutral N95 is excluded because
# its reconstruction error is an absolute cell-count difference.
IQR_KEYS = [
    "jet_mass",
    "dijet_mass",
    "jet_pt",
    "ecf2_log",
    "ecf3_log",
]

_AUX_KEYS = [
    "_reco_px",
    "_reco_py",
    "_reco_pz",
    "_reco_e",
]

OBS_LABELS = {
    "jet_mass":    r"Jet Mass [GeV]",
    "dijet_mass":  r"Dijet Mass [GeV]",
    "jet_pt":      r"Jet $p_T$ [GeV]",
    "neutral_n95": r"Neutral $N_{95}$",
    "ecf2_log":    r"$\ln\,\mathrm{ECF}^{(\beta{=}4)}_{N{=}2}$",
    "ecf3_log":    r"$\ln\,\mathrm{ECF}^{(\beta{=}4)}_{N{=}3}$",
}
OBS_RANGES = {
    "jet_mass":    (0,   100),
    "dijet_mass":  (0,  1000),
    "jet_pt":      (0,   600),
    "neutral_n95": (0,    81),
}
OBS_SHORT = {
    "jet_mass":    "Jet mass",
    "dijet_mass":  "Dijet mass",
    "jet_pt":      "Jet pT",
    "neutral_n95": "Neutral N95",
    "ecf2_log":    "ln ECF2(β=4)",
    "ecf3_log":    "ln ECF3(β=4)",
}

METHODS = ["true", "pileup", "puppi", "pumml", "pileflow"]
COLORS  = {
    "true":      "#2ca02c",
    "pileup":    "#d62728",
    "puppi":     "#ff7f0e",
    "pumml":     "#000000",
    "pileflow":  "#377eb8",
}
LABELS  = {
    "true":      "True",
    "pileup":    "w. Pileup",
    "puppi":     "PUPPI",
    "pumml":     "PUMML",
    "pileflow":  "PileFlow",
}

# Jet image grid constants
ETA_RANGE = 0.45
PHI_RANGE = 0.45
JET_R     = 0.4
ECF_BETA  = 4.0
# Exact ECF calculation uses every detector cell.
# Set to an integer only for explicitly labelled debugging runs.
ECF_MAX_CONST: int | None = None

# Common cell threshold applied identically to every method.
# Keep at zero initially.
CELL_PT_MIN = 0.0



# FastJet helpers
def _try_import_fj():
    try:
        import fastjet as fj
        return fj
    except ImportError:
        return None

def _cluster_jet(pjs, R=JET_R):
    fj      = _try_import_fj()
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    cs      = fj.ClusterSequence(pjs, jet_def)
    jets    = fj.sorted_by_pt(cs.inclusive_jets(ptmin=5.0))
    return cs, jets


def _jet_mass(jet):
    m = jet.m()
    return float(m) if m >= 0 else 0.0


def _delta_phi(phi1: float, phi2: float) -> float:
    """
    Wrapped azimuthal difference in [-pi, pi).
    """
    return float(
        (phi1 - phi2 + np.pi) % (2.0 * np.pi) - np.pi
    )


def _apply_cell_threshold(
    image: np.ndarray,
    threshold: float = CELL_PT_MIN,
) -> np.ndarray:
    """
    Apply the same nonnegative pT threshold to every method.
    """
    result = np.asarray(
        image,
        dtype=np.float32,
    ).copy()

    result[~np.isfinite(result)] = 0.0
    result = np.clip(result, 0.0, None)

    if threshold > 0.0:
        result[result < threshold] = 0.0

    return result


def _neutral_n95_from_image(
    neutral_image: np.ndarray,
    threshold: float = CELL_PT_MIN,
) -> float:
    """
    Number of neutral calorimeter cells containing 95% of neutral pT.
    """
    values = _apply_cell_threshold(
        neutral_image,
        threshold=threshold,
    ).reshape(-1)

    values = values[values > 0.0]

    if values.size == 0:
        return 0.0

    values = np.sort(values)[::-1]
    total = float(values.sum())

    if total <= 0.0:
        return 0.0

    cumulative = np.cumsum(values)

    return float(
        np.searchsorted(
            cumulative,
            0.95 * total,
            side="left",
        )
        + 1
    )


def _constituent_kinematics(
    jet,
    max_const: int | None = ECF_MAX_CONST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract constituent pT, eta and phi arrays.
    """
    constituents = jet.constituents()

    pts = np.asarray(
        [constituent.pt() for constituent in constituents],
        dtype=np.float64,
    )
    etas = np.asarray(
        [constituent.eta() for constituent in constituents],
        dtype=np.float64,
    )
    phis = np.asarray(
        [constituent.phi() for constituent in constituents],
        dtype=np.float64,
    )

    if max_const is not None and len(pts) > max_const:
        order = np.argsort(pts)[::-1][:max_const]
        pts = pts[order]
        etas = etas[order]
        phis = phis[order]

    return pts, etas, phis


def _delta_r_beta_matrix(
    etas: np.ndarray,
    phis: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Matrix with entries DeltaR_ij**beta.
    """
    deta = etas[:, None] - etas[None, :]

    dphi = phis[:, None] - phis[None, :]
    dphi = (
        dphi + np.pi
    ) % (2.0 * np.pi) - np.pi

    dr_squared = deta**2 + dphi**2
    dr_beta = dr_squared ** (beta / 2.0)

    np.fill_diagonal(
        dr_beta,
        0.0,
    )

    return dr_beta


def _ecf2(
    jet,
    beta: float = ECF_BETA,
    max_const: int | None = ECF_MAX_CONST,
) -> float:
    """
    ECF(2, beta) = sum_i<j pT_i pT_j DeltaR_ij**beta.
    """
    pts, etas, phis = _constituent_kinematics(
        jet,
        max_const=max_const,
    )

    if len(pts) < 2:
        return 0.0

    dr_beta = _delta_r_beta_matrix(
        etas,
        phis,
        beta,
    )

    i, j = np.triu_indices(
        len(pts),
        k=1,
    )

    return float(
        np.sum(
            pts[i]
            * pts[j]
            * dr_beta[i, j]
        )
    )


def _ecf3(
    jet,
    beta: float = ECF_BETA,
    max_const: int | None = ECF_MAX_CONST,
) -> float:
    """
    ECF(3, beta) = sum_i<j<k pT_i pT_j pT_k
                   * (DeltaR_ij DeltaR_ik DeltaR_jk)**beta.
    """
    pts, etas, phis = _constituent_kinematics(
        jet,
        max_const=max_const,
    )

    if len(pts) < 3:
        return 0.0

    dr_beta = _delta_r_beta_matrix(
        etas,
        phis,
        beta,
    )

    weighted_distance = (
        np.sqrt(
            pts[:, None]
            * pts[None, :]
        )
        * dr_beta
    )

    total = (
        np.trace(
            weighted_distance
            @ weighted_distance
            @ weighted_distance
        )
        / 6.0
    )

    return float(
        max(total, 0.0)
    )


def _image_to_pjs(
    image: np.ndarray,
    jet_eta: float,
    jet_phi: float,
):
    """
    Convert a square detector pT image into massless PseudoJets.

    Supports both the neutral 9x9 grid and charged 36x36 grid.
    """
    fj = _try_import_fj()

    image = _apply_cell_threshold(
        image,
        threshold=CELL_PT_MIN,
    )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D detector image, got {image.shape}"
        )

    n_eta, n_phi = image.shape

    eta_step = 2.0 * ETA_RANGE / n_eta
    phi_step = 2.0 * PHI_RANGE / n_phi

    pseudojets = []

    for eta_index in range(n_eta):
        for phi_index in range(n_phi):
            pt = float(
                image[eta_index, phi_index]
            )

            if pt <= 0.0:
                continue

            eta = (
                jet_eta
                - ETA_RANGE
                + (eta_index + 0.5) * eta_step
            )

            phi = (
                jet_phi
                - PHI_RANGE
                + (phi_index + 0.5) * phi_step
            )

            pseudojets.append(
                fj.PseudoJet(
                    pt * math.cos(phi),
                    pt * math.sin(phi),
                    pt * math.sinh(eta),
                    pt * math.cosh(eta),
                )
            )

    return pseudojets


def _detector_images_to_pjs(
    neutral_image: np.ndarray,
    charged_image: np.ndarray,
    jet_eta: float,
    jet_phi: float,
):
    """
    Construct one reconstructed jet input from neutral and charged cells.
    """
    pseudojets = _image_to_pjs(
        neutral_image,
        jet_eta,
        jet_phi,
    )

    pseudojets += _image_to_pjs(
        charged_image,
        jet_eta,
        jet_phi,
    )

    return pseudojets

# Observable computation (NaN-indexed arrays for aligned jet comparison)
def _obs_from_pjs(
    pseudojets,
    neutral_image: np.ndarray,
    jet_eta: float,
    jet_phi: float,
):
    """
    Cluster detector cells and compute observables for the matched jet.
    """
    if not pseudojets:
        return None

    cluster_sequence, jets = _cluster_jet(
        pseudojets,
    )

    if not jets:
        return None

    best = min(
        jets,
        key=lambda jet: math.hypot(
            jet.eta() - jet_eta,
            _delta_phi(
                jet.phi(),
                jet_phi,
            ),
        ),
    )

    ecf2 = _ecf2(
        best,
        max_const=ECF_MAX_CONST,
    )

    ecf3 = _ecf3(
        best,
        max_const=ECF_MAX_CONST,
    )

    result = {
        "jet_mass": _jet_mass(best),
        "dijet_mass": np.nan,
        "jet_pt": float(best.pt()),
        "neutral_n95": _neutral_n95_from_image(
            neutral_image,
            threshold=CELL_PT_MIN,
        ),
        "ecf2_log": (
            float(math.log(ecf2))
            if ecf2 > 0.0
            else np.nan
        ),
        "ecf3_log": (
            float(math.log(ecf3))
            if ecf3 > 0.0
            else np.nan
        ),
        "_reco_px": float(best.px()),
        "_reco_py": float(best.py()),
        "_reco_pz": float(best.pz()),
        "_reco_e": float(best.e()),
    }

    del cluster_sequence

    return result

def _compute_obs(
    pseudojet_getter,
    neutral_image_getter,
    jet_etas: np.ndarray,
    jet_phis: np.ndarray,
    n_jets: int,
    label: str,
):
    """
    Compute row-aligned observable arrays for one method.
    """
    store = {
        key: np.full(
            n_jets,
            np.nan,
            dtype=np.float64,
        )
        for key in OBS_KEYS + _AUX_KEYS
    }

    n_fail = 0

    for index in range(n_jets):
        if index % 1000 == 0:
            print(
                f"    [{label}] {index}/{n_jets}"
            )

        pseudojets = pseudojet_getter(index)
        neutral_image = neutral_image_getter(index)

        observables = _obs_from_pjs(
            pseudojets,
            neutral_image,
            float(jet_etas[index]),
            float(jet_phis[index]),
        )

        if observables is None:
            n_fail += 1
            continue

        for key in OBS_KEYS + _AUX_KEYS:
            store[key][index] = observables.get(
                key,
                np.nan,
            )

    if n_fail:
        print(
            f"    [{label}] {n_fail}/{n_jets} "
            "jets with no reconstruction — NaN"
        )

    return store

def _fill_dijet_masses(
    store: dict,
    event_ids: np.ndarray,
    jet_ranks: np.ndarray,
) -> None:
    """
    Compute one dijet mass per event from stored rank-0 and rank-1 jets.

    The value is stored only in the rank-0 row so each event contributes
    exactly once to the dijet distribution.
    """
    store["dijet_mass"][:] = np.nan

    for event_id in np.unique(event_ids):
        event_rows = np.where(
            event_ids == event_id
        )[0]

        leading_rows = event_rows[
            jet_ranks[event_rows] == 0
        ]
        subleading_rows = event_rows[
            jet_ranks[event_rows] == 1
        ]

        if (
            len(leading_rows) != 1
            or len(subleading_rows) != 1
        ):
            continue

        leading = int(leading_rows[0])
        subleading = int(subleading_rows[0])

        required = [
            store["_reco_px"][leading],
            store["_reco_py"][leading],
            store["_reco_pz"][leading],
            store["_reco_e"][leading],
            store["_reco_px"][subleading],
            store["_reco_py"][subleading],
            store["_reco_pz"][subleading],
            store["_reco_e"][subleading],
        ]

        if not np.isfinite(required).all():
            continue

        px = (
            store["_reco_px"][leading]
            + store["_reco_px"][subleading]
        )
        py = (
            store["_reco_py"][leading]
            + store["_reco_py"][subleading]
        )
        pz = (
            store["_reco_pz"][leading]
            + store["_reco_pz"][subleading]
        )
        energy = (
            store["_reco_e"][leading]
            + store["_reco_e"][subleading]
        )

        mass_squared = (
            energy**2
            - px**2
            - py**2
            - pz**2
        )

        store["dijet_mass"][leading] = math.sqrt(
            max(
                mass_squared,
                0.0,
            )
        )


def _keep_leading_jet_observables_only(
    store: dict,
    jet_ranks: np.ndarray,
) -> None:
    """
    Remove non-leading rows from single-jet observable distributions.
    """
    nonleading = jet_ranks != 0

    for key in LEADING_KEYS:
        store[key][nonleading] = np.nan


# External PUMML inference helper
class _PUMMLNet(torch.nn.Module):
    """
    Minimal copy of the PUMML CNN architecture for loading an external
    checkpoint.  Matches pumml_in_server/src/models/pumml_model.py exactly.

    Input : (N, 3, 36, 36)
    Output: (N, 1,  9,  9)
    """
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ZeroPad2d(2),
            torch.nn.Conv2d(3,  10, 6, stride=2, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.ZeroPad2d(2),
            torch.nn.Conv2d(10, 10, 6, stride=2, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10,  1, 1, stride=1, padding=0, bias=True),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


def _pumml_predict(npz_path: str, pumml_ckpt: str | None, device: str) -> np.ndarray:
    """
    Run PUMML inference on all jets in npz_path.
    Returns (N, 9, 9) float32 numpy array.
    """
    dev  = torch.device(device)
    data = np.load(npz_path, allow_pickle=False)
    X    = np.stack([
        data["ch_neutral_all"].astype(np.float32),
        data["ch_charged_pu"].astype(np.float32),
        data["ch_charged_lv"].astype(np.float32),
    ], axis=1)   # (N, 3, 36, 36)

    model = _PUMMLNet().to(dev).eval()
    try:
        state = torch.load(
            pumml_ckpt,
            map_location=dev,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            pumml_ckpt,
            map_location=dev,
        )
    # Handle various checkpoint formats from pumml_in_server
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out = model(torch.from_numpy(X[i:i+256]).to(dev))
            preds.append(out.squeeze(1).clamp(min=0.0).cpu().numpy())
    result = np.concatenate(preds, axis=0)
    print(f"  [pumml] Predicted {len(result):,} jets  shape={result.shape}")
    return result   # (N, 9, 9)

def _prepare_pileflow_predictions(results: dict) -> np.ndarray:
    """
    Validate and reshape image-only PileFlow predictions.

    Accepted input shapes:
        (N, 81)
        (N, 9, 9)

    Returns
    -------
    np.ndarray
        Predictions with shape (N, 9, 9).
    """
    if "neutral_lv_pred" not in results:
        raise KeyError(
            "PileFlow results must contain 'neutral_lv_pred'. "
            f"Available keys: {sorted(results.keys())}"
        )

    predictions = np.asarray(
        results["neutral_lv_pred"],
        dtype=np.float32,
    )

    if predictions.ndim == 2 and predictions.shape[1] == 81:
        predictions = predictions.reshape(-1, 9, 9)

    elif predictions.ndim == 3 and predictions.shape[1:] == (9, 9):
        pass

    else:
        raise ValueError(
            "Expected PileFlow neutral_lv_pred shape "
            f"(N, 81) or (N, 9, 9), got {predictions.shape}"
        )

    if not np.isfinite(predictions).all():
        n_bad = int(
            predictions.size
            - np.count_nonzero(np.isfinite(predictions))
        )
        raise ValueError(
            f"PileFlow predictions contain {n_bad} non-finite values."
        )

    return predictions

# Main entry point
def run_comparison(
    npz_path: str,
    npy_path: str | None,
    pumml_ckpt: str | None,
    results: dict,
    cfg,
    mean_npu: float | None = None,
):
    """
    Run PUMML-inspired detector-level observable evaluation.

    Parameters
    ----------
    npz_path:
        Detector-image dataset. It must contain image channels, event_id,
        jet_rank, and detector-level PUPPI images.

    npy_path:
        Retained for runner compatibility. It is not used for event pairing
        or model evaluation.

    pumml_ckpt:
        Optional external PUMML checkpoint.

    results:
        Image-only PileFlow output containing neutral_lv_pred with shape
        (N, 81) or (N, 9, 9).

    cfg:
        Runtime configuration providing outdir and device.

    mean_npu:
        Optional mean pileup value for plot titles.
    """
    plots_dir = os.path.join(cfg.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Loss curve — re-plot from history file if it exists
    hist_path = os.path.join(cfg.outdir, "checkpoints", "pileflow_best_history.npz")
    if os.path.isfile(hist_path):
        h = np.load(hist_path)
        make_loss_curve(
            {"train": h["train"].tolist(), "val": h["val"].tolist()},
            save_path=os.path.join(plots_dir, "pileflow_loss.png"),
            title="Image-only PileFlow training — flow-matching MSE",
        )

    fj = _try_import_fj()
    if fj is None:
        print("  [compare] WARNING: fastjet not found — skipping clustering plots.")
        _fallback_image_plots(npz_path, results, cfg, plots_dir)
        return

    data = np.load(npz_path, allow_pickle=False)

    pileflow_predictions = _prepare_pileflow_predictions(results)

    n_data = len(data["jet_eta"])
    n_pred = len(pileflow_predictions)

    if n_data != n_pred:
        raise ValueError(
            "Generator rows and PileFlow prediction rows must match exactly. "
            f"generator={n_data:,}, pileflow={n_pred:,}"
        )

    N = n_data

    if mean_npu is None:
        mean_npu = float(data["n_pu"].mean()) if "n_pu" in data else 50.0

    jet_etas = data["jet_eta"][:N].astype(np.float32)
    jet_phis = data["jet_phi"][:N].astype(np.float32)

    # The corrected evaluation requires explicit row metadata and
    # detector-level PUPPI images.
    required_keys = [
        "event_id",
        "jet_rank",
        "jet_eta",
        "jet_phi",
        "ch_neutral_lv",
        "ch_neutral_all_raw",
        "ch_charged_lv",
        "ch_charged_pu",
        "puppi_neutral_9x9",
        "puppi_charged_36x36",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing_keys:
        raise KeyError(
            "Correct detector-level comparison requires regenerated "
            f"test data. Missing NPZ keys: {missing_keys}"
        )

    event_ids = data["event_id"][:N].astype(
        np.int64,
    )
    jet_ranks = data["jet_rank"][:N].astype(
        np.int64,
    )

    if len(event_ids) != N or len(jet_ranks) != N:
        raise ValueError(
            "event_id and jet_rank must align one-to-one with jet rows."
        )

    print(
        f"  [compare] {len(np.unique(event_ids)):,} real events"
    )

    # The old .npy table is retained only for runner compatibility.
    # It is not used for event pairing or model evaluation.
    _ = npy_path

    # Run PUMML inference.
    pumml_imgs = None

    if pumml_ckpt and os.path.isfile(pumml_ckpt):
        print(
            f"  [compare] Running PUMML inference from "
            f"{pumml_ckpt} ..."
        )

        pumml_imgs = _pumml_predict(
            npz_path,
            pumml_ckpt,
            cfg.device,
        )[:N]

    else:
        print(
            f"  [compare] WARNING: PUMML checkpoint not found at "
            f"'{pumml_ckpt}' — skipping PUMML"
        )

    pileflow_imgs = pileflow_predictions[:N]

    # Apply the same evaluation threshold to every neutral and charged image.
    neutral_images = {
        "true": _apply_cell_threshold(
            data["ch_neutral_lv"][:N]
        ),
        "pileup": _apply_cell_threshold(
            data["ch_neutral_all_raw"][:N]
        ),
        "puppi": _apply_cell_threshold(
            data["puppi_neutral_9x9"][:N]
        ),
        "pileflow": _apply_cell_threshold(
            pileflow_imgs
        ),
    }

    charged_images = {
        "true": _apply_cell_threshold(
            data["ch_charged_lv"][:N]
        ),
        "pileup": _apply_cell_threshold(
            data["ch_charged_lv"][:N]
            + data["ch_charged_pu"][:N]
        ),
        "puppi": _apply_cell_threshold(
            data["puppi_charged_36x36"][:N]
        ),
        "pileflow": _apply_cell_threshold(
            data["ch_charged_lv"][:N]
        ),
    }

    if pumml_imgs is not None:
        neutral_images["pumml"] = _apply_cell_threshold(
            pumml_imgs
        )
        charged_images["pumml"] = _apply_cell_threshold(
            data["ch_charged_lv"][:N]
        )

    method_order = [
        "true",
        "pileup",
        "puppi",
        "pumml",
        "pileflow",
    ]

    active_methods = [
        method
        for method in method_order
        if method in neutral_images
    ]

    store = {}

    for method in active_methods:
        print(
            f"  [compare] Computing {LABELS[method]} observables ..."
        )

        def pseudojet_getter(
            index: int,
            method_name: str = method,
        ):
            return _detector_images_to_pjs(
                neutral_images[method_name][index],
                charged_images[method_name][index],
                float(jet_etas[index]),
                float(jet_phis[index]),
            )

        def neutral_image_getter(
            index: int,
            method_name: str = method,
        ):
            return neutral_images[method_name][index]

        store[method] = _compute_obs(
            pseudojet_getter,
            neutral_image_getter,
            jet_etas,
            jet_phis,
            N,
            method,
        )

    print(
        "  [compare] Computing event-level dijet masses ..."
    )


    for method_store in store.values():
        _fill_dijet_masses(
            method_store,
            event_ids,
            jet_ranks,
        )

        _keep_leading_jet_observables_only(
            method_store,
            jet_ranks,
        )

    # Produce plots and tables once, after every method is finalized.
    make_figure4(
        store,
        plots_dir,
        mean_npu,
    )

    make_figure5(
        store,
        plots_dir,
        mean_npu,
    )

    make_tables(
        store,
        plots_dir,
    )

    make_wasserstein_table(
        store,
        plots_dir,
    )

    print(f"  [compare] All plots saved to {plots_dir}/")

# Figure 4 — normalised distributions
def make_figure4(store: dict, plots_dir: str, mean_npu: float = 50.0):
    """Six-panel normalised observable distributions (PUMML paper Figure 4)."""
    active = [m for m in METHODS if m in store]
    N      = int(np.sum(np.isfinite(store["true"]["jet_mass"])))

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        "Pileup mitigation — observable distributions\n"
        f"$\\langle N_{{PU}}\\rangle = {mean_npu:.0f}$   |   N jets ≈ {N:,}",
        fontsize=12,
    )

    for ax, key in zip(axes.ravel(), OBS_KEYS):
        all_v = np.concatenate([
            store[m][key][np.isfinite(store[m][key])] for m in active
        ])
        if len(all_v) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        fixed  = OBS_RANGES.get(key)
        lo, hi = fixed if fixed else (
            float(np.percentile(all_v, 0.5)), float(np.percentile(all_v, 99.5))
        )
        if lo >= hi:
            continue
        edges   = np.linspace(lo, hi, 51)
        centres = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]

        for m in active:
            v = store[m][key]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            counts, _ = np.histogram(v, bins=edges)
            norm = counts.sum() * width
            density = counts / norm if norm > 0 else counts.astype(float)
            lw = 2.0 if m == "true" else 1.5
            if m == "true":
                ax.fill_between(centres, density, alpha=0.10, color=COLORS[m], step="mid")
            ax.step(centres, density, where="mid",
                    color=COLORS[m], lw=lw, label=LABELS[m])

        ax.set_xlabel(OBS_LABELS[key], fontsize=10)
        ax.set_ylabel("Normalised", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, plots_dir, "figure4_distributions")


def _centered_reconstruction_error(
    truth: np.ndarray,
    prediction: np.ndarray,
    key: str,
) -> np.ndarray:
    """
    Return the paper-style centered reconstruction error.

    Neutral N95:
        prediction - truth, in cells.

    Other observables:
        100 * (prediction - truth) / abs(truth), in percent.
    """
    mask = (
        np.isfinite(truth)
        & np.isfinite(prediction)
    )

    if key == "neutral_n95":
        error = (
            prediction[mask]
            - truth[mask]
        )

    else:
        mask &= np.abs(truth) > 1e-6

        error = (
            100.0
            * (
                prediction[mask]
                - truth[mask]
            )
            / np.abs(truth[mask])
        )

    if error.size:
        error = (
            error
            - np.median(error)
        )

    return error


# Figure 5 — percent-error distributions
def make_figure5(
    store: dict,
    plots_dir: str,
    mean_npu: float = 50.0,
):
    """
    Centered reconstruction-error distributions.

    Continuous observables use percent error.
    Neutral N95 uses an absolute cell-count difference.
    """
    compare = [
        method
        for method in [
            "pileup",
            "puppi",
            "pumml",
            "pileflow",
        ]
        if method in store
    ]

    n_leading = int(
        np.sum(
            np.isfinite(
                store["true"]["jet_mass"]
            )
        )
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 12),
    )

    fig.suptitle(
        "Centered reconstruction errors\n"
        f"$\\langle N_{{PU}}\\rangle = {mean_npu:.0f}$"
        f"   |   N leading jets ≈ {n_leading:,}",
        fontsize=12,
    )

    for axis, key in zip(
        axes.ravel(),
        OBS_KEYS,
    ):
        truth = store["true"][key]

        method_errors = {
            method: _centered_reconstruction_error(
                truth,
                store[method][key],
                key,
            )
            for method in compare
        }

        combined = [
            errors
            for errors in method_errors.values()
            if errors.size
        ]

        if not combined:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
            )
            continue

        all_errors = np.concatenate(
            combined
        )

        low = float(
            np.percentile(
                all_errors,
                1.0,
            )
        )
        high = float(
            np.percentile(
                all_errors,
                99.0,
            )
        )

        if key != "neutral_n95":
            low = max(low, -500.0)
            high = min(high, 500.0)

        if low >= high:
            low, high = (
                (-5.0, 5.0)
                if key == "neutral_n95"
                else (-100.0, 100.0)
            )

        edges = np.linspace(
            low,
            high,
            61,
        )

        width = edges[1] - edges[0]
        centers = 0.5 * (
            edges[:-1]
            + edges[1:]
        )

        for method in compare:
            errors = method_errors[method]

            if not errors.size:
                continue

            counts, _ = np.histogram(
                errors,
                bins=edges,
            )

            normalization = (
                counts.sum()
                * width
            )

            density = (
                counts / normalization
                if normalization > 0
                else counts.astype(float)
            )

            axis.step(
                centers,
                density,
                where="mid",
                color=COLORS[method],
                linewidth=1.8,
                label=LABELS[method],
            )

        axis.axvline(
            0.0,
            color="gray",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
        )

        if key == "neutral_n95":
            axis.set_xlabel(
                r"$N_{95}^{\rm reco}-N_{95}^{\rm true}$ "
                "[neutral cells], centered",
                fontsize=9,
            )
        else:
            axis.set_xlabel(
                f"Centered % error [{OBS_LABELS[key]}]",
                fontsize=9,
            )

        axis.set_ylabel(
            "Normalised",
            fontsize=9,
        )
        axis.legend(
            fontsize=7,
            frameon=False,
        )
        axis.tick_params(
            labelsize=8,
        )
        axis.set_xlim(
            low,
            high,
        )
        axis.set_ylim(
            bottom=0.0,
        )

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    _save_fig(
        fig,
        plots_dir,
        "figure5_reconstruction_errors",
    )

def make_wasserstein_table(store: dict, plots_dir: str):
    """
    Compute raw and truth-IQR-normalized Wasserstein-1 distances.

    Lower values indicate better agreement with the truth distribution.
    """
    compare = [
        method
        for method in ["pileup", "puppi", "pumml", "pileflow"]
        if method in store
    ]

    raw_table = {method: {} for method in compare}
    normalized_table = {method: {} for method in compare}

    for key in OBS_KEYS:
        truth = store["true"][key]
        truth = truth[np.isfinite(truth)]

        if len(truth) < 2:
            for method in compare:
                raw_table[method][key] = np.nan
                normalized_table[method][key] = np.nan
            continue

        truth_q25, truth_q75 = np.percentile(truth, [25, 75])
        truth_iqr = float(truth_q75 - truth_q25)

        for method in compare:
            prediction = store[method][key]
            prediction = prediction[np.isfinite(prediction)]

            if len(prediction) < 2:
                raw_table[method][key] = np.nan
                normalized_table[method][key] = np.nan
                continue

            distance = float(
                wasserstein_distance(
                    truth,
                    prediction,
                )
            )

            raw_table[method][key] = distance

            if truth_iqr > 1e-12:
                normalized_table[method][key] = distance / truth_iqr
            else:
                normalized_table[method][key] = np.nan

    txt_path = os.path.join(
        plots_dir,
        "table_wasserstein.txt",
    )

    csv_path = os.path.join(
        plots_dir,
        "table_wasserstein.csv",
    )

    lines = []

    lines.append(
        "Wasserstein-1 distance — raw observable units "
        "(lower = better)"
    )
    lines.append("-" * 78)

    header = f"{'Observable':<18}" + "".join(
        f"{LABELS[method]:>15}"
        for method in compare
    )

    lines.append(header)
    lines.append("-" * 78)

    for key in OBS_KEYS:
        row = f"{OBS_SHORT[key]:<18}"

        for method in compare:
            value = raw_table[method][key]

            row += (
                f"{value:>15.4f}"
                if np.isfinite(value)
                else f"{'N/A':>15}"
            )

        lines.append(row)

    lines.append("")
    lines.append(
        "Wasserstein-1 distance normalized by truth IQR "
        "(lower = better)"
    )
    lines.append("-" * 78)
    lines.append(header)
    lines.append("-" * 78)

    for key in OBS_KEYS:
        row = f"{OBS_SHORT[key]:<18}"

        for method in compare:
            value = normalized_table[method][key]

            row += (
                f"{value:>15.4f}"
                if np.isfinite(value)
                else f"{'N/A':>15}"
            )

        lines.append(row)

    text = "\n".join(lines) + "\n"

    print(text)

    with open(txt_path, "w") as output:
        output.write(text)

    with open(csv_path, "w", newline="") as output:
        writer = csv.writer(output)

        writer.writerow(
            ["Raw Wasserstein-1 distance"]
            + [LABELS[method] for method in compare]
        )

        for key in OBS_KEYS:
            writer.writerow(
                [OBS_SHORT[key]]
                + [
                    raw_table[method][key]
                    for method in compare
                ]
            )

        writer.writerow([])

        writer.writerow(
            ["Wasserstein-1 / truth IQR"]
            + [LABELS[method] for method in compare]
        )

        for key in OBS_KEYS:
            writer.writerow(
                [OBS_SHORT[key]]
                + [
                    normalized_table[method][key]
                    for method in compare
                ]
            )

    print(f"  [tables] Wasserstein TXT -> {txt_path}")
    print(f"  [tables] Wasserstein CSV -> {csv_path}")

    return raw_table, normalized_table

# Tables 1 & 2
def make_tables(
    store: dict,
    plots_dir: str,
):
    """
    Produce:

      Table 1:
          Pearson correlation coefficient, multiplied by 100.

      Table 2:
          IQR of continuous-observable percent errors.

      Additional diagnostic:
          IQR of neutral-N95 absolute differences, in cells.
    """
    from scipy.stats import pearsonr

    compare = [
        method
        for method in [
            "pileup",
            "puppi",
            "pumml",
            "pileflow",
        ]
        if method in store
    ]

    truth = store["true"]

    table1 = {
        method: {}
        for method in compare
    }

    table2 = {
        method: {}
        for method in compare
    }

    n95_difference_iqr = {}

    for method in compare:
        for key in OBS_KEYS:
            true_values = truth[key]
            predicted_values = store[method][key]

            mask = (
                np.isfinite(true_values)
                & np.isfinite(predicted_values)
            )

            if mask.sum() < 5:
                table1[method][key] = np.nan
                continue

            correlation, _ = pearsonr(
                true_values[mask],
                predicted_values[mask],
            )

            table1[method][key] = float(
                100.0 * correlation
            )

        for key in IQR_KEYS:
            true_values = truth[key]
            predicted_values = store[method][key]

            mask = (
                np.isfinite(true_values)
                & np.isfinite(predicted_values)
                & (np.abs(true_values) > 1e-6)
            )

            if mask.sum() < 5:
                table2[method][key] = np.nan
                continue

            error = (
                100.0
                * (
                    predicted_values[mask]
                    - true_values[mask]
                )
                / np.abs(true_values[mask])
            )

            q25, q75 = np.percentile(
                error,
                [25, 75],
            )

            table2[method][key] = float(
                q75 - q25
            )

        true_n95 = truth["neutral_n95"]
        predicted_n95 = store[method]["neutral_n95"]

        n95_mask = (
            np.isfinite(true_n95)
            & np.isfinite(predicted_n95)
        )

        if n95_mask.sum() < 5:
            n95_difference_iqr[method] = np.nan

        else:
            difference = (
                predicted_n95[n95_mask]
                - true_n95[n95_mask]
            )

            q25, q75 = np.percentile(
                difference,
                [25, 75],
            )

            n95_difference_iqr[method] = float(
                q75 - q25
            )

    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        _print_table(
            "Table 1 — Pearson correlation coefficient (%)",
            table1,
            compare,
            OBS_KEYS,
            "% (higher = better)",
        )

        _print_table(
            "Table 2 — IQR of percent-error distribution (%)",
            table2,
            compare,
            IQR_KEYS,
            "% (lower = better)",
        )

        print(
            "\nNeutral N95 difference IQR [cells]"
        )
        print("-" * 76)

        header = (
            f"{'Observable':<18}"
            + "".join(
                f"{LABELS[method]:>14}"
                for method in compare
            )
        )

        print(header)
        print("-" * 76)

        row = f"{'Neutral N95':<18}"

        for method in compare:
            value = n95_difference_iqr[method]

            row += (
                f"{value:>14.2f}"
                if np.isfinite(value)
                else f"{'N/A':>14}"
            )

        print(row)
        print("-" * 76)

    text = output.getvalue()
    print(text)

    txt_path = os.path.join(
        plots_dir,
        "tables_1_2.txt",
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)

    csv_path = os.path.join(
        plots_dir,
        "tables_1_2.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["Table 1 - Pearson r (%)"]
            + [
                LABELS[method]
                for method in compare
            ]
        )

        for key in OBS_KEYS:
            writer.writerow(
                [OBS_SHORT[key]]
                + [
                    table1[method][key]
                    for method in compare
                ]
            )

        writer.writerow([])

        writer.writerow(
            ["Table 2 - Percent-error IQR (%)"]
            + [
                LABELS[method]
                for method in compare
            ]
        )

        for key in IQR_KEYS:
            writer.writerow(
                [OBS_SHORT[key]]
                + [
                    table2[method][key]
                    for method in compare
                ]
            )

        writer.writerow([])

        writer.writerow(
            ["Neutral N95 difference IQR (cells)"]
            + [
                LABELS[method]
                for method in compare
            ]
        )

        writer.writerow(
            ["Neutral N95"]
            + [
                n95_difference_iqr[method]
                for method in compare
            ]
        )

    print(
        f"  [tables] TXT -> {txt_path}"
    )
    print(
        f"  [tables] CSV -> {csv_path}"
    )

    return (
        table1,
        table2,
        n95_difference_iqr,
    )

def _print_table(
    title: str,
    table: dict,
    compare: list[str],
    keys: list[str],
    unit: str,
):
    column_width = 14

    header = (
        f"{'Observable':<18}"
        + "".join(
            f"{LABELS[method]:>{column_width}}"
            for method in compare
        )
    )

    separator = "-" * max(
        len(header),
        60,
    )

    print(f"\n{title}")
    print(separator)
    print(header)
    print(separator)

    for key in keys:
        row = f"{OBS_SHORT[key]:<18}"

        for method in compare:
            value = table[method][key]

            row += (
                f"{value:>{column_width}.1f}"
                if np.isfinite(value)
                else f"{'N/A':>{column_width}}"
            )

        print(row)

    print(separator)
    print(f"  unit: {unit}\n")


# Loss curve
def make_loss_curve(history: dict, save_path: str, title: str = "Loss"):
    """Plot train/val MSE loss vs epoch and save PNG."""
    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train"], label="Train",      color="#1f77b4", lw=2)
    ax.plot(epochs, history["val"],   label="Validation", color="#ff7f0e", lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (flow matching)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plots] Loss curve -> {save_path}")



# Helpers
def _save_fig(fig, plots_dir, name):
    for ext in ("png", "pdf"):
        path = os.path.join(plots_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  [plots] Saved -> {path}")
    plt.close(fig)

# Fallback — image-level plots when FastJet is unavailable
def _fallback_image_plots(npz_path, results, cfg, plots_dir):
    data        = np.load(npz_path, allow_pickle=False)
    true_imgs   = data["ch_neutral_lv"].astype(np.float32)
    if "ch_neutral_all_raw" in data.files:
        pileup_imgs = data["ch_neutral_all_raw"].astype(np.float32)
    else:
        pileup_imgs = (
            data["ch_neutral_all"]
            .reshape(-1, 9, 4, 9, 4)
            .sum(axis=(2, 4))
            .astype(np.float32)
        )
    pred_imgs   = results["neutral_lv_pred"].reshape(-1, 9, 9).astype(np.float32)

    obs_names  = ["total_pt", "n_active", "max_pt"]
    obs_labels = ["Total neutral pT [GeV]", "Active pixels (pT>0.1)", "Max pixel pT [GeV]"]

    def _stats(imgs):
        return {
            "total_pt": imgs.sum(axis=(1,2)),
            "n_active": (imgs > 0.1).sum(axis=(1,2)).astype(float),
            "max_pt":   imgs.max(axis=(1,2)),
        }

    store = {
        "True":     _stats(true_imgs),
        "w/ Pileup": _stats(pileup_imgs),
        "PileFlow": _stats(pred_imgs),
    }
    colors_fb = {"True": "#2ca02c", "w/ Pileup": "#d62728", "PileFlow": "#377eb8"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, obs, label in zip(axes, obs_names, obs_labels):
        for name, s in store.items():
            vals = s[obs]
            lo   = float(np.percentile(vals, 0.5))
            hi   = float(np.percentile(vals, 99.5))
            ax.hist(vals, bins=50, range=(lo, hi), histtype="step",
                    density=True, label=name, color=colors_fb[name], lw=1.5)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Normalised", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("PileFlow image-level comparison (no FastJet)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(plots_dir, "image_level_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plots] Image-level comparison -> {out}")
