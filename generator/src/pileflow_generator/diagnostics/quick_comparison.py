#!/usr/bin/env python3
# quick_comparison.py
#
# Quick comparison plot from a jet_images.npz produced by gen4e2e.
# Shows three curves per panel:
#   True     — LV-only jet (no pileup)
#   w/ Pileup — jet with pileup overlaid
#   PUPPI    — PUPPI-mitigated jet
#
# No trained models needed — everything is already in the .npz file.
#
# Usage
# -----
#   python quick_comparison.py \
#       --npz path/to/jet_images.npz \
#       --out plots/ \
#       --max-jets 5000
#
# Dependencies: numpy, matplotlib, fastjet

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# constants 

ETA_RANGE  = 0.45
PHI_RANGE  = 0.45
N_NEUTRAL  = 9
JET_R      = 0.4
ECF_BETA   = 4.0

_COLORS = {
    "true":   "#2ca02c",
    "pileup": "#d62728",
    "puppi":  "#ff7f0e",
}
_LABELS = {
    "true":   "True (no pileup)",
    "pileup": "w. Pileup",
    "puppi":  "PUPPI",
}


# jet building from constituent arrays 

def _build_jet(px, py, pz, e, n):
    """Cluster constituent arrays into a FastJet jet. Returns (jet, cs) or (None, None)."""
    try:
        import fastjet as fj
    except ImportError:
        raise ImportError("fastjet not found. Install the FastJet Python bindings.")

    pjs = []
    for i in range(int(n)):
        if float(e[i]) <= 0.0:
            continue
        pjs.append(fj.PseudoJet(float(px[i]), float(py[i]),
                                 float(pz[i]), float(e[i])))
    if not pjs:
        return None, None
    jet_def = fj.JetDefinition(fj.antikt_algorithm, JET_R)
    cs      = fj.ClusterSequence(pjs, jet_def)
    try:
        jets = list(fj.sorted_by_pt(cs.inclusive_jets()))
    except Exception:
        jets = sorted(cs.inclusive_jets(), key=lambda j: -j.pt())
    if not jets:
        return None, None
    return jets[0], cs


# observables 

def _n95(neutral_image):
    """Number of cells containing 95% of the neutral pT."""
    flat  = neutral_image.flatten()
    total = float(flat.sum())
    if total <= 0:
        return 0
    cumsum = np.cumsum(np.sort(flat)[::-1])
    idx    = int(np.searchsorted(cumsum, 0.95 * total, side="left"))
    return min(idx + 1, len(flat))


def _ecf2(jet, beta=ECF_BETA):
    """Log of the 2-point Energy Correlation Function."""
    consts = jet.constituents()
    if len(consts) < 2:
        return None
    pts  = np.array([c.pt()  for c in consts], dtype=np.float64)
    etas = np.array([c.eta() for c in consts], dtype=np.float64)
    phis = np.array([c.phi() for c in consts], dtype=np.float64)
    mask = pts > 1e-6
    pts, etas, phis = pts[mask], etas[mask], phis[mask]
    if len(pts) < 2:
        return None
    jpt = float(jet.pt())
    if jpt <= 0:
        return None
    pts  = pts / jpt          # use momentum fractions z_i = pT_i / pT_jet
    deta = etas[:, None] - etas[None, :]
    dphi = phis[:, None] - phis[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dr   = np.sqrt(deta**2 + dphi**2)
    ii, jj = np.triu_indices(len(pts), k=1)
    total  = float(np.sum(pts[ii] * pts[jj] * dr[ii, jj]**beta))
    return float(np.log(total)) if total > 0 else None


def _neutral_image_from_puppi(px, py, pz, e, n, jet_eta, jet_phi):
    """Project PUPPI constituent arrays onto a 9x9 neutral pT grid."""
    grid = np.zeros((N_NEUTRAL, N_NEUTRAL), dtype=np.float32)
    cell = 2 * ETA_RANGE / N_NEUTRAL
    for i in range(int(n)):
        ei = float(e[i])
        if ei <= 0.0:
            continue
        pt_i = float(np.sqrt(px[i]**2 + py[i]**2))
        if pt_i <= 0.0:
            continue
        p_mag = float(np.sqrt(px[i]**2 + py[i]**2 + pz[i]**2))
        if p_mag == 0 or p_mag == abs(float(pz[i])):
            eta_i = float(np.sign(pz[i]) * 1e9)
        else:
            eta_i = float(0.5 * np.log((p_mag + pz[i]) / (p_mag - pz[i])))
        phi_i = float(np.arctan2(py[i], px[i]))
        deta  = eta_i - jet_eta
        dphi  = (phi_i - jet_phi + np.pi) % (2 * np.pi) - np.pi
        if abs(deta) >= ETA_RANGE or abs(dphi) >= PHI_RANGE:
            continue
        ieta = int(np.clip(int((deta + ETA_RANGE) / cell), 0, N_NEUTRAL - 1))
        iphi = int(np.clip(int((dphi + PHI_RANGE) / cell), 0, N_NEUTRAL - 1))
        grid[ieta, iphi] += pt_i
    return grid


# main collection loop 

def collect(npz_path, max_jets=5000):
    data  = np.load(npz_path, allow_pickle=False)
    total = len(data["jet_pt"])
    N     = min(max_jets, total)
    print(f"[plot] Loading {N} / {total} jets from {npz_path}")

    jet_eta_arr = data["jet_eta"][:N]
    jet_phi_arr = data["jet_phi"][:N]
    mean_npu    = float(data["n_pu"][:N].mean())

    # constituent arrays
    true_px = data["true_px"][:N];  true_py = data["true_py"][:N]
    true_pz = data["true_pz"][:N];  true_e  = data["true_e"][:N]
    true_n  = data["true_n"][:N]

    pile_px = data["pileup_px"][:N]; pile_py = data["pileup_py"][:N]
    pile_pz = data["pileup_pz"][:N]; pile_e  = data["pileup_e"][:N]
    pile_n  = data["pileup_n"][:N]

    pupp_px = data["puppi_px"][:N];  pupp_py = data["puppi_py"][:N]
    pupp_pz = data["puppi_pz"][:N];  pupp_e  = data["puppi_e"][:N]
    pupp_n  = data["puppi_n"][:N]

    # neutral images for N95
    true_neutral = data["clean_neutral_lv"][:N]       # (N, 9, 9)
    pile_neutral = data["ch_neutral_all_raw"][:N]      # (N, 9, 9)

    obs_keys = ["jet_mass", "jet_pt", "neutral_n95", "ecf2_log"]
    store    = {m: {k: [] for k in obs_keys} for m in ("true", "pileup", "puppi")}

    for i in range(N):
        jet_eta = float(jet_eta_arr[i])
        jet_phi = float(jet_phi_arr[i])

        # True 
        j, cs = _build_jet(true_px[i], true_py[i], true_pz[i], true_e[i], true_n[i])
        if j is not None:
            store["true"]["jet_mass"].append(float(j.m()))
            store["true"]["jet_pt"].append(float(j.pt()))
            v2 = _ecf2(j)
            if v2 is not None:
                store["true"]["ecf2_log"].append(v2)
            del cs
        store["true"]["neutral_n95"].append(_n95(true_neutral[i]))

        # w/ Pileup 
        j, cs = _build_jet(pile_px[i], pile_py[i], pile_pz[i], pile_e[i], pile_n[i])
        if j is not None:
            store["pileup"]["jet_mass"].append(float(j.m()))
            store["pileup"]["jet_pt"].append(float(j.pt()))
            v2 = _ecf2(j)
            if v2 is not None:
                store["pileup"]["ecf2_log"].append(v2)
            del cs
        store["pileup"]["neutral_n95"].append(_n95(pile_neutral[i]))

        # PUPPI 
        j, cs = _build_jet(pupp_px[i], pupp_py[i], pupp_pz[i], pupp_e[i], pupp_n[i])
        pupp_neutral = _neutral_image_from_puppi(
            pupp_px[i], pupp_py[i], pupp_pz[i], pupp_e[i], pupp_n[i],
            jet_eta, jet_phi)
        if j is not None:
            store["puppi"]["jet_mass"].append(float(j.m()))
            store["puppi"]["jet_pt"].append(float(j.pt()))
            v2 = _ecf2(j)
            if v2 is not None:
                store["puppi"]["ecf2_log"].append(v2)
            del cs
        store["puppi"]["neutral_n95"].append(_n95(pupp_neutral))

        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  {i+1}/{N} jets processed")

    return store, mean_npu, N


# plotting 

def _panel(ax, store, key, xlabel, bins=50, x_range=None):
    all_vals = []
    for m in ("true", "pileup", "puppi"):
        all_vals.extend([v for v in store[m][key] if np.isfinite(v)])
    if not all_vals:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
        return
    all_vals = np.array(all_vals)
    if x_range is None:
        lo = float(np.percentile(all_vals, 0.5))
        hi = float(np.percentile(all_vals, 99.5))
    else:
        lo, hi = x_range
    if lo >= hi:
        lo, hi = float(all_vals.min()), float(all_vals.max())
    edges = np.linspace(lo, hi, bins + 1)

    for m in ("true", "pileup", "puppi"):
        vals = np.array([v for v in store[m][key] if np.isfinite(v)])
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=edges)
        norm = counts.sum() * (edges[1] - edges[0])
        if norm > 0:
            counts = counts / norm
        centres = 0.5 * (edges[:-1] + edges[1:])
        if m == "true":
            ax.fill_between(centres, counts, alpha=0.15,
                            color=_COLORS[m], step="mid")
        ax.step(centres, counts, where="mid",
                color=_COLORS[m], linewidth=1.8,
                label=_LABELS[m])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Normalised", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)


def make_plot(store, mean_npu, n_jets, out_dir, process="pp → jj"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Pileup mitigation:  True  |  w. Pileup  |  PUPPI\n"
        f"Process: {process}    "
        f"⟨N_PU⟩ = {mean_npu:.0f}    N jets = {n_jets:,}",
        fontsize=12, y=1.01,
    )

    panels = [
        (axes[0, 0], "jet_mass",    "Jet Mass (GeV)",               50, (0, 100)),
        (axes[0, 1], "jet_pt",      r"Jet $p_T$ (GeV)",             50, (0, 600)),
        (axes[1, 0], "neutral_n95", r"Neutral $N_{95}$",            50, (0,  50)),
        (axes[1, 1], "ecf2_log",
         r"$\ln\,\mathrm{ECF}^{(\beta=4)}_{N=2}$",                  50, None),
    ]

    for ax, key, xlabel, bins, x_range in panels:
        _panel(ax, store, key, xlabel, bins=bins, x_range=x_range)

    plt.tight_layout()

    out_png = os.path.join(out_dir, "quick_comparison.png")
    out_pdf = os.path.join(out_dir, "quick_comparison.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] Saved → {out_png}")
    print(f"[plot] Saved → {out_pdf}")
    return out_png


# CLI 

def build_parser():
    p = argparse.ArgumentParser(
        description="Plot True vs w/Pileup vs PUPPI from a jet_images.npz file."
    )
    p.add_argument("--npz",      required=True,
                   help="Path to jet_images.npz produced by gen4e2e")
    p.add_argument("--out",      default="plots",
                   help="Output directory for the plots (default: plots/)")
    p.add_argument("--process",  default="pp → jj",
                   help="Process label for the plot title")
    p.add_argument("--max-jets", type=int, default=5000,
                   help="Maximum number of jets to process (default: 5000)")
    return p


def main():
    args  = build_parser().parse_args()
    store, mean_npu, n_jets = collect(args.npz, max_jets=args.max_jets)
    make_plot(store, mean_npu, n_jets, args.out, process=args.process)
    print("\n[plot] Done.")
    for m in ("true", "pileup", "puppi"):
        print(f"  {m:8s}: mass={len(store[m]['jet_mass']):,}  "
              f"ecf2={len(store[m]['ecf2_log']):,}  "
              f"n95={len(store[m]['neutral_n95']):,}")


if __name__ == "__main__":
    main()