"""
comparison/observable_comparison.py
========================================
Full comparison engine — reproduces the PUMML paper plots for all five methods:

  True      : clean jets (no pileup)
  w/ Pileup : all particles (hard + pileup)
  PUPPI     : PUPPI-weighted particles
  PUMML     : external PUMML CNN checkpoint (loaded for comparison only)
  PileFlow  : PileFlow-generated neutral LV image

Six observables (PUMML paper Table 1 / Figures 4-5):
  jet_mass   | dijet_mass | jet_pt | neutral_n95 | ecf2_log | ecf3_log

Outputs (all saved to cfg.outdir/plots/):
  figure4_distributions.{png,pdf}   — normalised observable distributions
  figure5_percent_errors.{png,pdf}  — per-jet percent-error distributions
  tables_1_2.txt / .csv             — Pearson r (%) and IQR (%)
  pileflow_loss.png                 — train/val loss curve (re-plotted if history given)

FastJet is required for jet clustering.  If not installed the script falls
back to image-level comparisons with a warning.
"""

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

# Observable configuration (matches PUMML paper)
OBS_KEYS   = ["jet_mass", "dijet_mass", "jet_pt", "neutral_n95", "ecf2_log", "ecf3_log"]
_AUX_KEYS  = ["_reco_px", "_reco_py", "_reco_pz", "_reco_e"]

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
    "neutral_n95": (0,    50),
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
MAX_CONST = 50



# FastJet helpers
def _try_import_fj():
    try:
        import fastjet as fj
        return fj
    except ImportError:
        return None


def _pseudojet_list(px, py, pz, e):
    fj = _try_import_fj()
    return [fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(e[i]))
            for i in range(len(px))]


def _cluster_jet(pjs, R=JET_R):
    fj      = _try_import_fj()
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    cs      = fj.ClusterSequence(pjs, jet_def)
    jets    = fj.sorted_by_pt(cs.inclusive_jets(ptmin=5.0))
    return cs, jets


def _jet_mass(jet):
    m = jet.m()
    return float(m) if m >= 0 else 0.0


def _n95(jet):
    consts = jet.constituents()
    if not consts:
        return 0
    pts = np.sort(np.array([c.pt() for c in consts]))[::-1]
    tot = pts.sum()
    if tot <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(pts) / tot, 0.95) + 1)


def _ecf2(jet, beta=ECF_BETA, max_c=MAX_CONST):
    consts = jet.constituents()
    if len(consts) < 2:
        return 0.0
    pts  = np.array([c.pt()  for c in consts])
    etas = np.array([c.eta() for c in consts])
    phis = np.array([c.phi() for c in consts])
    if len(pts) > max_c:
        idx = np.argsort(pts)[::-1][:max_c]
        pts, etas, phis = pts[idx], etas[idx], phis[idx]
    ii, jj = np.triu_indices(len(pts), k=1)
    deta   = etas[ii] - etas[jj]
    dphi   = (phis[ii] - phis[jj] + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(pts[ii] * pts[jj] * (deta**2 + dphi**2)**(beta / 2)))


def _ecf3(jet, beta=ECF_BETA, max_c=MAX_CONST):
    consts = jet.constituents()
    if len(consts) < 3:
        return 0.0
    pts  = np.array([c.pt()  for c in consts])
    etas = np.array([c.eta() for c in consts])
    phis = np.array([c.phi() for c in consts])
    if len(pts) > max_c:
        idx = np.argsort(pts)[::-1][:max_c]
        pts, etas, phis = pts[idx], etas[idx], phis[idx]
    n   = len(pts)
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dij = math.sqrt((etas[i]-etas[j])**2 + ((phis[i]-phis[j]+math.pi)%(2*math.pi)-math.pi)**2)
                dik = math.sqrt((etas[i]-etas[k])**2 + ((phis[i]-phis[k]+math.pi)%(2*math.pi)-math.pi)**2)
                djk = math.sqrt((etas[j]-etas[k])**2 + ((phis[j]-phis[k]+math.pi)%(2*math.pi)-math.pi)**2)
                val += pts[i] * pts[j] * pts[k] * min(dij, dik, djk)**beta
    return float(val)


def _image_to_pjs(img9x9, jet_eta, jet_phi):
    # 9x9 neutral pT image -> list of PseudoJets 
    fj   = _try_import_fj()
    pjs  = []
    step = 2 * ETA_RANGE / 9
    for i in range(9):
        for j in range(9):
            pt = float(img9x9[i, j])
            if pt <= 0:
                continue
            eta = jet_eta + (-ETA_RANGE + (i + 0.5) * step)
            phi = jet_phi + (-PHI_RANGE + (j + 0.5) * step)
            pjs.append(fj.PseudoJet(pt*math.cos(phi), pt*math.sin(phi),
                                    pt*math.sinh(eta),  pt*math.cosh(eta)))
    return pjs


def _charged_lv_pjs(ch_lv_36, jet_eta, jet_phi):
    # 36x36 charged-LV pT image -> list of PseudoJets
    fj   = _try_import_fj()
    pjs  = []
    step = 2 * ETA_RANGE / 36
    for i in range(36):
        for j in range(36):
            pt = float(ch_lv_36[i, j])
            if pt <= 0:
                continue
            eta = jet_eta + (-ETA_RANGE + (i + 0.5) * step)
            phi = jet_phi + (-PHI_RANGE + (j + 0.5) * step)
            pjs.append(fj.PseudoJet(pt*math.cos(phi), pt*math.sin(phi),
                                    pt*math.sinh(eta),  pt*math.cosh(eta)))
    return pjs



# Observable computation (NaN-indexed arrays for aligned jet comparison)
def _obs_from_pjs(pjs, jet_eta, jet_phi):
    """Cluster pjs with anti-kT, compute observables for the nearest jet."""
    if not pjs:
        return None
    cs, jets = _cluster_jet(pjs)
    if not jets:
        return None
    best  = min(jets, key=lambda j: math.hypot(j.eta()-jet_eta, j.phi()-jet_phi))
    e2    = _ecf2(best)
    e3    = _ecf3(best)
    result = {
        "jet_mass":    _jet_mass(best),
        "dijet_mass":  np.nan,
        "jet_pt":      float(best.pt()),
        "neutral_n95": float(_n95(best)),
        "ecf2_log":    float(math.log(e2)) if e2 > 0 else np.nan,
        "ecf3_log":    float(math.log(e3)) if e3 > 0 else np.nan,
        "_reco_px":    float(best.px()),
        "_reco_py":    float(best.py()),
        "_reco_pz":    float(best.pz()),
        "_reco_e":     float(best.e()),
    }
    del cs
    return result


def _compute_obs(pj_getter, jet_etas, jet_phis, N, label):
    # Fixed-size NaN-array store — index i always = same physical jet
    store  = {k: np.full(N, np.nan) for k in OBS_KEYS + _AUX_KEYS}
    n_fail = 0
    for i in range(N):
        if i % 1000 == 0:
            print(f"    [{label}] {i}/{N}")
        pjs = pj_getter(i)
        obs = _obs_from_pjs(pjs, float(jet_etas[i]), float(jet_phis[i]))
        if obs is None:
            n_fail += 1
            continue
        for k in OBS_KEYS + _AUX_KEYS:
            store[k][i] = obs.get(k, np.nan)
    if n_fail:
        print(f"    [{label}] {n_fail}/{N} jets with no reconstruction — NaN")
    return store


def _fill_dijet_masses(store, event_ids):
    # Pair the two leading-pT jets per event and fill dijet_mass in-place
    reco_pt = np.sqrt(store["_reco_px"]**2 + store["_reco_py"]**2)
    _, inv  = np.unique(event_ids, return_inverse=True)
    for ui in range(inv.max() + 1):
        idx   = np.where(inv == ui)[0]
        valid = idx[np.isfinite(reco_pt[idx])]
        if len(valid) < 2:
            continue
        order  = np.argsort(reco_pt[valid])[::-1][:2]
        i1, i2 = valid[order[0]], valid[order[1]]
        px = store["_reco_px"][i1] + store["_reco_px"][i2]
        py = store["_reco_py"][i1] + store["_reco_py"][i2]
        pz = store["_reco_pz"][i1] + store["_reco_pz"][i2]
        e  = store["_reco_e"][i1]  + store["_reco_e"][i2]
        m2 = e**2 - px**2 - py**2 - pz**2
        dj = float(math.sqrt(max(m2, 0.0)))
        store["dijet_mass"][i1] = dj
        store["dijet_mass"][i2] = dj


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
    state = torch.load(pumml_ckpt, map_location=dev, weights_only=True)
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



# Main entry point
def run_comparison(
    npz_path:    str,
    npy_path:    str,
    pumml_ckpt:  str,
    results:     dict,
    cfg,
    mean_npu:    float = None,
):
    """
    Compute observables for all 5 methods and produce all paper plots + tables.

    Parameters
    ----------
    npz_path   : jets_pileup.npz (image channels + constituent lists)
    npy_path   : jets.npy (N, ≥25 columns from gen4e2e)
    pumml_ckpt : path to external PUMML checkpoint (pumml_in_server/pumml_model.pt)
    results    : dict from generate_and_save() — must have "neutral_lv_pred" (N,81)
    cfg        : Config (uses cfg.outdir, cfg.device)
    mean_npu   : mean pileup vertices for plot titles (read from npz if None)
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
            title="PileFlow training — flow matching MSE loss",
        )

    fj = _try_import_fj()
    if fj is None:
        print("  [compare] WARNING: fastjet not found — skipping clustering plots.")
        _fallback_image_plots(npz_path, results, cfg, plots_dir)
        return

    data = np.load(npz_path, allow_pickle=False)

    n_data = len(data["jet_eta"])
    n_pred = len(results["neutral_lv_pred"])
    N = min(n_data, n_pred)

    if n_data != n_pred:
        print(
            "  [compare] WARNING: generator rows and PileFlow prediction rows differ. "
            f"Using first {N:,} aligned rows. "
            f"generator={n_data:,}, pileflow={n_pred:,}"
        )

    if mean_npu is None:
        mean_npu = float(data["n_pu"].mean()) if "n_pu" in data else 50.0

    jet_etas = data["jet_eta"][:N].astype(np.float32)
    jet_phis = data["jet_phi"][:N].astype(np.float32)

    # Event IDs for dijet pairing
    feats = np.load(npy_path)[:N]
    if feats.shape[1] >= 26:
        event_ids = feats[:N, 25].astype(int)
        print(f"  [compare] {len(np.unique(event_ids)):,} unique events from column 25")
    else:
        event_ids = np.arange(N, dtype=int) // 2
        print("  [compare] No event_id column — using consecutive jet pairs (pp→jj)")

    # Constituent arrays
    true_px = data["true_px"][:N]
    true_py = data["true_py"][:N]
    true_pz = data["true_pz"][:N]
    true_e  = data["true_e"][:N]

    pu_px = data["pileup_px"][:N]
    pu_py = data["pileup_py"][:N]
    pu_pz = data["pileup_pz"][:N]
    pu_e  = data["pileup_e"][:N]

    pp_px = data["puppi_px"][:N]
    pp_py = data["puppi_py"][:N]
    pp_pz = data["puppi_pz"][:N]
    pp_e  = data["puppi_e"][:N]

    ch_lv_36 = data["ch_charged_lv"][:N].astype(np.float32)

    # PseudoJet getters
    def _pj_true(i):
        mask = true_e[i] > 0
        return _pseudojet_list(true_px[i][mask], true_py[i][mask],
                               true_pz[i][mask], true_e[i][mask])

    def _pj_pileup(i):
        mask = pu_e[i] > 0
        return _pseudojet_list(pu_px[i][mask], pu_py[i][mask],
                               pu_pz[i][mask], pu_e[i][mask])

    def _pj_puppi(i):
        mask = pp_e[i] > 0
        return _pseudojet_list(pp_px[i][mask], pp_py[i][mask],
                               pp_pz[i][mask], pp_e[i][mask])

    # Run PUMML inference from external checkpoint
    pumml_imgs = None
    if pumml_ckpt and os.path.isfile(pumml_ckpt):
        print(f"  [compare] Running PUMML inference from {pumml_ckpt} ...")
        pumml_imgs = _pumml_predict(npz_path, pumml_ckpt, cfg.device)
    else:
        print(f"  [compare] WARNING: PUMML checkpoint not found at '{pumml_ckpt}' — skipping PUMML column")

    def _pj_pumml(i):
        pjs  = _image_to_pjs(pumml_imgs[i], jet_etas[i], jet_phis[i])
        pjs += _charged_lv_pjs(ch_lv_36[i], jet_etas[i], jet_phis[i])
        return pjs

    # PileFlow generated images
    pileflow_imgs = results["neutral_lv_pred"][:N].reshape(N, 9, 9).astype(np.float32)

    def _pj_pileflow(i):
        pjs  = _image_to_pjs(pileflow_imgs[i], jet_etas[i], jet_phis[i])
        pjs += _charged_lv_pjs(ch_lv_36[i], jet_etas[i], jet_phis[i])
        return pjs

    # Compute observables for all methods
    print("  [compare] Computing True observables ...")
    obs_true   = _compute_obs(_pj_true,     jet_etas, jet_phis, N, "true")
    print("  [compare] Computing Pileup observables ...")
    obs_pileup = _compute_obs(_pj_pileup,   jet_etas, jet_phis, N, "pileup")
    print("  [compare] Computing PUPPI observables ...")
    obs_puppi  = _compute_obs(_pj_puppi,    jet_etas, jet_phis, N, "puppi")
    print("  [compare] Computing PileFlow observables ...")
    obs_pileflow = _compute_obs(_pj_pileflow, jet_etas, jet_phis, N, "pileflow")

    store = {
        "true":     obs_true,
        "pileup":   obs_pileup,
        "puppi":    obs_puppi,
        "pileflow": obs_pileflow,
    }
    if pumml_imgs is not None:
        print("  [compare] Computing PUMML observables ...")
        obs_pumml = _compute_obs(_pj_pumml, jet_etas, jet_phis, N, "pumml")
        store["pumml"] = obs_pumml

    # Fill dijet masses (requires cross-jet pairing)
    print("  [compare] Computing dijet masses ...")
    for obs in store.values():
        _fill_dijet_masses(obs, event_ids)

    # Produce all plots + tables
    make_figure4(store, plots_dir, mean_npu)
    make_figure5(store, plots_dir, mean_npu)
    make_tables(store, plots_dir)

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



# Figure 5 — percent-error distributions
def make_figure5(store: dict, plots_dir: str, mean_npu: float = 50.0):
    """Per-jet percent-error distributions (PUMML paper Figure 5)."""
    compare = [m for m in ["pileup", "puppi", "pumml", "pileflow"] if m in store]
    N       = int(np.sum(np.isfinite(store["true"]["jet_mass"])))
    eps     = 1e-6

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        r"Percent error  $\frac{\hat{x} - x_{\rm true}}{|x_{\rm true}|} \times 100$"
        "\n"
        f"$\\langle N_{{PU}}\\rangle = {mean_npu:.0f}$   |   N jets ≈ {N:,}",
        fontsize=12,
    )

    for ax, key in zip(axes.ravel(), OBS_KEYS):
        true = store["true"][key]
        errs_all = []
        for m in compare:
            p    = store[m][key]
            mask = np.isfinite(true) & np.isfinite(p) & (np.abs(true) > eps)
            if mask.sum() > 0:
                errs_all.extend((100.0*(p[mask]-true[mask])/np.abs(true[mask])).tolist())

        if not errs_all:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        arr = np.array(errs_all)
        lo  = max(float(np.percentile(arr, 1)), -500)
        hi  = min(float(np.percentile(arr, 99)),  500)
        if lo >= hi:
            lo, hi = -100, 100
        edges = np.linspace(lo, hi, 61)
        width = edges[1] - edges[0]

        for m in compare:
            p    = store[m][key]
            mask = np.isfinite(true) & np.isfinite(p) & (np.abs(true) > eps)
            if mask.sum() == 0:
                continue
            err = 100.0*(p[mask]-true[mask])/np.abs(true[mask])
            counts, _ = np.histogram(err, bins=edges)
            norm = counts.sum() * width
            density = counts / norm if norm > 0 else counts.astype(float)
            ax.step(0.5*(edges[:-1]+edges[1:]), density, where="mid",
                    color=COLORS[m], lw=1.8, label=LABELS[m])

        ax.axvline(0, color="gray", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlabel(f"% error  [{OBS_LABELS[key]}]", fontsize=9)
        ax.set_ylabel("Normalised", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, plots_dir, "figure5_percent_errors")

# Tables 1 & 2
def make_tables(store: dict, plots_dir: str):
    """
    Pearson r (%) and IQR (%) tables — PUMML paper Tables 1 & 2.

    Saves tables_1_2.txt (plain text) and tables_1_2.csv.
    Prints both tables to stdout.
    """
    from scipy.stats import pearsonr

    compare = [m for m in ["pileup", "puppi", "pumml", "pileflow"] if m in store]
    true    = store["true"]
    eps     = 1e-6

    table1 = {m: {} for m in compare}
    table2 = {m: {} for m in compare}

    for m in compare:
        for k in OBS_KEYS:
            t    = true[k]
            p    = store[m][k]
            both = np.isfinite(t) & np.isfinite(p)
            if both.sum() < 5:
                table1[m][k] = np.nan
                table2[m][k] = np.nan
                continue
            r, _  = pearsonr(t[both], p[both])
            table1[m][k] = float(r * 100.0)
            mask = both & (np.abs(t) > eps)
            if mask.sum() < 5:
                table2[m][k] = np.nan
            else:
                err = 100.0*(p[mask]-t[mask])/np.abs(t[mask])
                q25, q75 = np.percentile(err, [25, 75])
                table2[m][k] = float(q75 - q25)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_table("Table 1 — Pearson correlation coefficient (%)",
                     table1, compare, "% (higher = better)")
        _print_table("Table 2 — IQR of percent-error distribution (%)",
                     table2, compare, "% (lower = better)")
    text = buf.getvalue()
    print(text)

    txt_path = os.path.join(plots_dir, "tables_1_2.txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"  [tables] TXT -> {txt_path}")

    csv_path = os.path.join(plots_dir, "tables_1_2.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Table 1 - Pearson r (%)"] + [LABELS[m] for m in compare])
        for k in OBS_KEYS:
            w.writerow([OBS_SHORT[k]] + [
                f"{table1[m][k]:.1f}" if np.isfinite(table1[m][k]) else "N/A"
                for m in compare
            ])
        w.writerow([])
        w.writerow(["Table 2 - IQR (%)"] + [LABELS[m] for m in compare])
        for k in OBS_KEYS:
            w.writerow([OBS_SHORT[k]] + [
                f"{table2[m][k]:.1f}" if np.isfinite(table2[m][k]) else "N/A"
                for m in compare
            ])
    print(f"  [tables] CSV -> {csv_path}")

    return table1, table2


def _print_table(title, table, compare, unit):
    col_w  = 14
    header = f"{'Observable':<18}" + "".join(f"{LABELS[m]:>{col_w}}" for m in compare)
    sep    = "-" * max(len(header), 60)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for k in OBS_KEYS:
        row = f"{OBS_SHORT[k]:<18}"
        for m in compare:
            v = table[m][k]
            row += f"{'N/A':>{col_w}}" if np.isnan(v) else f"{v:>{col_w}.1f}"
        print(row)
    print(sep)
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
