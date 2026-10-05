"""Jet observables of the PUMML paper, shared by EVERY evaluation script.

Detector representation (PUMML, Sec. 3: "a detector discretization is
applied to the true and reconstructed events"):

  * charged particles -> 36x36 cells of 0.025 x 0.025,
  * neutral particles -> 9x9 calorimeter cells of 0.1 x 0.1,

each non-empty cell being treated as a massless pseudo-particle at the cell
centre, in GLOBAL (eta, phi) = image centre + local offset. Neutral cells are
NOT spread over 16 fine pixels (that would add artificial mass/ECF).

Observables (PUMML Tables 1-2):
  jet_mass, jet_pt           leading jet, from the 4-vector sum
  dijet_mass                 rank-0 + rank-1 jets of the same hard event
  neutral_n95                # neutral CELLS holding 95% of the neutral pT
  ln_ecf2, ln_ecf3           log normalised ECFs with beta = 4,
                             e_N = sum_{i<..} z_i..z_k prod(Delta R)^beta,
                             z_i = pT_i / sum pT.  log(0) -> NaN.
"""
from __future__ import annotations

import numpy as np

IMAGE_HALF = 0.45
N_FINE, N_COARSE = 36, 9
BETA = 4.0

OBS_ORDER = ("jet_mass", "dijet_mass", "jet_pt", "neutral_n95", "ln_ecf2", "ln_ecf3")
OBS_LABELS = {
    "jet_mass": "Jet mass [GeV]",
    "dijet_mass": "Dijet mass [GeV]",
    "jet_pt": r"Jet $p_T$ [GeV]",
    "neutral_n95": r"Neutral $N_{95}$",
    "ln_ecf2": r"$\ln\,\mathrm{ECF}_{N=2}^{(\beta=4)}$",
    "ln_ecf3": r"$\ln\,\mathrm{ECF}_{N=3}^{(\beta=4)}$",
}
OBS_PLAIN = {"jet_mass": "Jet mass", "dijet_mass": "Dijet mass", "jet_pt": "Jet pT",
             "neutral_n95": "Neutral N95", "ln_ecf2": "ln ECF2 (b=4)", "ln_ecf3": "ln ECF3 (b=4)"}
# PUMML paper, Tables 1 and 2 (<NPU> = 140). Correlations and IQRs in percent.
PAPER_CORRELATION = {
    "with_pileup": {"jet_mass": 65.5, "dijet_mass": 85.5, "jet_pt": 94.4, "neutral_n95": 36.2,
                    "ln_ecf2": 60.4, "ln_ecf3": 41.6},
    "pumml": {"jet_mass": 97.4, "dijet_mass": 99.5, "jet_pt": 99.7, "neutral_n95": 75.3,
              "ln_ecf2": 90.5, "ln_ecf3": 77.2},
    "puppi": {"jet_mass": 94.0, "dijet_mass": 95.8, "jet_pt": 98.0, "neutral_n95": 70.4,
              "ln_ecf2": 83.3, "ln_ecf3": 69.1},
}
PAPER_IQR = {
    "pumml": {"jet_mass": 13.0, "dijet_mass": 2.02, "jet_pt": 2.26, "ln_ecf2": 5.63, "ln_ecf3": 8.48},
    "puppi": {"jet_mass": 28.7, "dijet_mass": 2.95, "jet_pt": 3.40, "ln_ecf2": 8.82, "ln_ecf3": 10.7},
}


def cell_centers(n: int, half: float = IMAGE_HALF) -> np.ndarray:
    w = 2.0 * half / n
    return -half + w * (np.arange(n) + 0.5)


FINE_DETA, FINE_DPHI = np.meshgrid(cell_centers(N_FINE), cell_centers(N_FINE), indexing="ij")
COARSE_DETA, COARSE_DPHI = np.meshgrid(cell_centers(N_COARSE), cell_centers(N_COARSE), indexing="ij")


def wrap_phi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# 4-vectors
def four_vectors(charged36, neutral9, center_eta, center_phi) -> dict:
    """Vectorised jet 4-momenta. charged36 (N,36,36), neutral9 (N,9,9)."""
    c = np.clip(np.asarray(charged36, dtype=np.float64), 0.0, None)
    n = np.clip(np.asarray(neutral9, dtype=np.float64), 0.0, None)
    eta0 = np.asarray(center_eta, dtype=np.float64)[:, None, None]
    phi0 = np.asarray(center_phi, dtype=np.float64)[:, None, None]
    out = {"px": 0.0, "py": 0.0, "pz": 0.0, "e": 0.0}
    for pt, deta, dphi in ((c, FINE_DETA, FINE_DPHI), (n, COARSE_DETA, COARSE_DPHI)):
        eta, phi = eta0 + deta, phi0 + dphi
        out["px"] = out["px"] + (pt * np.cos(phi)).sum((1, 2))
        out["py"] = out["py"] + (pt * np.sin(phi)).sum((1, 2))
        out["pz"] = out["pz"] + (pt * np.sinh(eta)).sum((1, 2))
        out["e"] = out["e"] + (pt * np.cosh(eta)).sum((1, 2))
    out["pt"] = np.hypot(out["px"], out["py"])
    out["mass"] = _mass(out["e"], out["px"], out["py"], out["pz"])
    return out


def _mass(e, px, py, pz):
    return np.sqrt(np.clip(e * e - px * px - py * py - pz * pz, 0.0, None))


def dijet_mass(event_id, jet_rank, fv: dict):
    """Invariant mass of rank-0 + rank-1 jets per event.

    Returns (event_ids, masses) sorted by event id; events missing either jet
    are skipped. Identical inputs give identical event lists for all methods.
    """
    event_id = np.asarray(event_id)
    jet_rank = np.asarray(jet_rank)
    i0 = {int(e): i for i, (e, r) in enumerate(zip(event_id, jet_rank)) if r == 0}
    i1 = {int(e): i for i, (e, r) in enumerate(zip(event_id, jet_rank)) if r == 1}
    events = np.array(sorted(set(i0) & set(i1)), dtype=np.int64)
    a = np.array([i0[e] for e in events], dtype=np.int64)
    b = np.array([i1[e] for e in events], dtype=np.int64)
    s = {k: fv[k][a] + fv[k][b] for k in ("e", "px", "py", "pz")} if len(events) else None
    m = _mass(s["e"], s["px"], s["py"], s["pz"]) if s else np.zeros(0)
    return events, m


# N95
def n95(neutral9) -> np.ndarray:
    """Number of neutral cells containing 95% of the neutral pT (0 if empty)."""
    x = np.clip(np.asarray(neutral9, dtype=np.float64), 0.0, None).reshape(len(neutral9), -1)
    x = -np.sort(-x, axis=1)
    cum = np.cumsum(x, axis=1)
    total = cum[:, -1:]
    count = (cum < 0.95 * total - 1e-12 * total).sum(axis=1) + 1
    return np.where(total[:, 0] > 0, count, 0).astype(np.float64)


# ECF
def jet_particles(charged36, neutral9, center_eta, center_phi, eps=1e-12):
    """Pseudo-particles (pt, eta, phi) of ONE jet in global coordinates."""
    c = np.asarray(charged36, dtype=np.float64)
    n = np.asarray(neutral9, dtype=np.float64)
    cm, nm = c > eps, n > eps
    pt = np.concatenate([c[cm], n[nm]])
    eta = np.concatenate([center_eta + FINE_DETA[cm], center_eta + COARSE_DETA[nm]])
    phi = wrap_phi(np.concatenate([center_phi + FINE_DPHI[cm], center_phi + COARSE_DPHI[nm]]))
    return pt, eta, phi


def normalized_ecfs(pt, eta, phi, beta: float = BETA):
    """(e2, e3) with z_i = pT_i / sum pT, O(n^3) via matrix products.

    e2 = 1/2 sum_ij z_i z_j D_ij,   e3 = 1/6 sum_ij z_i z_j D_ij (D Z D)_ij,
    D_ij = Delta R_ij^beta (D_ii = 0 removes repeated indices exactly).
    """
    pt = np.asarray(pt, dtype=np.float64)
    if len(pt) < 2 or pt.sum() <= 0:
        return 0.0, 0.0
    z = pt / pt.sum()
    deta = eta[:, None] - eta[None, :]
    dphi = wrap_phi(phi[:, None] - phi[None, :])
    dr2 = deta * deta + dphi * dphi
    d = dr2 * dr2 if beta == 4.0 else np.power(dr2, beta / 2.0)
    zd = z[:, None] * d * z[None, :]
    e2 = 0.5 * zd.sum()
    e3 = (zd * (d @ (z[:, None] * d))).sum() / 6.0 if len(pt) >= 3 else 0.0
    return float(e2), float(e3)


def log_ecfs(charged36, neutral9, center_eta, center_phi):
    n = len(neutral9)
    out = np.full((n, 2), np.nan)
    for i in range(n):
        e2, e3 = normalized_ecfs(*jet_particles(charged36[i], neutral9[i], center_eta[i], center_phi[i]))
        with np.errstate(divide="ignore"):
            out[i] = [np.log(e2) if e2 > 0 else np.nan, np.log(e3) if e3 > 0 else np.nan]
    return out[:, 0], out[:, 1]


# summary
def compute_observables(charged36, neutral9, center_eta, center_phi, event_id, jet_rank) -> dict:
    """All six PUMML observables for one representation of the eval jets.

    Single-jet observables use leading jets (rank 0) only, in input order;
    dijet_mass is one entry per complete event. Also returns 'npu'-ready
    index arrays: 'leading_idx' (positions of the leading jets) and
    'dijet_events'.
    """
    jet_rank = np.asarray(jet_rank)
    lead = np.where(jet_rank == 0)[0]
    fv = four_vectors(charged36, neutral9, center_eta, center_phi)
    events, mjj = dijet_mass(event_id, jet_rank, fv)
    e2, e3 = log_ecfs(np.asarray(charged36)[lead], np.asarray(neutral9)[lead],
                      np.asarray(center_eta)[lead], np.asarray(center_phi)[lead])
    return {
        "jet_mass": fv["mass"][lead], "jet_pt": fv["pt"][lead],
        "neutral_n95": n95(np.asarray(neutral9)[lead]),
        "ln_ecf2": e2, "ln_ecf3": e3, "dijet_mass": mjj,
        "leading_idx": lead, "dijet_events": events,
    }
