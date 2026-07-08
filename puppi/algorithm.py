from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as scipy_chi2

from .config import PUPPIConfig
from .particles import Particle


def _compute_eta_phi(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized eta and phi computation.
    """
    p = np.sqrt(px**2 + py**2 + pz**2)

    safe = (p > 0) & (p != np.abs(pz))

    eta = np.where(
        safe,
        0.5 * np.log((p + pz) / np.where(safe, p - pz, 1.0)),
        np.sign(pz) * 1e9,
    )

    phi = np.arctan2(py, px)

    return eta.astype(np.float64), phi.astype(np.float64)


def _delta_r_matrix(
    eta1: np.ndarray,
    phi1: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:
    """
    Compute the full Delta-R matrix between two particle sets.
    """
    deta = eta1[:, None] - eta2[None, :]
    dphi = phi1[:, None] - phi2[None, :]

    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi

    return np.sqrt(deta**2 + dphi**2)


def _compute_alpha_vectorised(
    target_eta: np.ndarray,
    target_phi: np.ndarray,
    nb_eta: np.ndarray,
    nb_phi: np.ndarray,
    nb_pt: np.ndarray,
    R0: float = 0.3,
    Rmin: float = 0.02,
) -> np.ndarray:
    """
    Compute PUPPI local-shape alpha for many particles at once.
    """
    if len(target_eta) == 0 or len(nb_eta) == 0:
        return np.full(len(target_eta), -np.inf, dtype=np.float64)

    dr = _delta_r_matrix(target_eta, target_phi, nb_eta, nb_phi)

    mask = (dr >= Rmin) & (dr <= R0)

    dr_safe = np.where(mask, dr, 1.0)
    contrib = np.where(mask, nb_pt[None, :] / dr_safe, 0.0)

    total = contrib.sum(axis=1)

    alpha = np.where(total > 0, np.log(total), -np.inf)

    return alpha.astype(np.float64)


def _left_rms(values: np.ndarray, median: float) -> float:
    """
    Compute left-side RMS around the median.
    """
    left = values[values < median]

    if len(left) == 0:
        return 1.0

    return float(np.sqrt(np.mean((left - median) ** 2)))


def characterise_pileup(alpha_pu: np.ndarray) -> tuple[float, float]:
    """
    Compute event-level PUPPI pileup parameters.
    """
    if len(alpha_pu) == 0:
        return 0.0, 1.0

    finite = alpha_pu[np.isfinite(alpha_pu)]

    if len(finite) == 0:
        return 0.0, 1.0

    alpha_bar = float(np.median(finite))
    sigma = _left_rms(finite, alpha_bar)

    if sigma <= 0:
        sigma = 1.0

    return alpha_bar, sigma


def run_puppi(
    particles: list[Particle],
    n_pu: int = 140,
    R0: float = 0.3,
    Rmin: float = 0.02,
    w_cut: float = 0.1,
    eta_tracker: float = 2.5,
) -> list[Particle]:
    """
    Run the simplified vectorized PUPPI baseline on one event.
    """
    if not particles:
        return []

    px = np.array([p.px for p in particles], dtype=np.float64)
    py = np.array([p.py for p in particles], dtype=np.float64)
    pz = np.array([p.pz for p in particles], dtype=np.float64)
    chg = np.array([p.charge for p in particles], dtype=np.float64)
    lv = np.array([p.is_lv for p in particles], dtype=bool)

    pt = np.sqrt(px**2 + py**2)
    eta, phi = _compute_eta_phi(px, py, pz)

    is_charged = np.abs(chg) > 0
    is_charged_lv = is_charged & lv
    is_charged_pu = is_charged & ~lv
    is_neutral = ~is_charged
    is_central = np.abs(eta) <= eta_tracker
    is_forward = ~is_central

    pt_cut_central = 0.1 + 0.007 * n_pu
    pt_cut_forward = 0.2 + 0.011 * n_pu

    lv_idx = np.where(is_charged_lv)[0]
    pu_idx = np.where(is_charged_pu)[0]

    if len(lv_idx) > 0 and len(pu_idx) > 0:
        alpha_pu = _compute_alpha_vectorised(
            target_eta=eta[pu_idx],
            target_phi=phi[pu_idx],
            nb_eta=eta[lv_idx],
            nb_phi=phi[lv_idx],
            nb_pt=pt[lv_idx],
            R0=R0,
            Rmin=Rmin,
        )
    else:
        alpha_pu = np.array([], dtype=np.float64)

    alpha_bar, sigma = characterise_pileup(alpha_pu)

    neutral_central_idx = np.where(is_neutral & is_central)[0]
    neutral_forward_idx = np.where(is_neutral & is_forward)[0]

    if len(neutral_central_idx) > 0 and len(lv_idx) > 0:
        alpha_nc = _compute_alpha_vectorised(
            target_eta=eta[neutral_central_idx],
            target_phi=phi[neutral_central_idx],
            nb_eta=eta[lv_idx],
            nb_phi=phi[lv_idx],
            nb_pt=pt[lv_idx],
            R0=R0,
            Rmin=Rmin,
        )
    else:
        alpha_nc = np.full(len(neutral_central_idx), -np.inf, dtype=np.float64)

    if len(neutral_forward_idx) > 0:
        alpha_nf = _compute_alpha_vectorised(
            target_eta=eta[neutral_forward_idx],
            target_phi=phi[neutral_forward_idx],
            nb_eta=eta,
            nb_phi=phi,
            nb_pt=pt,
            R0=R0,
            Rmin=Rmin,
        )
    else:
        alpha_nf = np.full(len(neutral_forward_idx), -np.inf, dtype=np.float64)

    def _weights(alpha_arr: np.ndarray) -> np.ndarray:
        above = alpha_arr > alpha_bar

        chi2_values = np.where(
            above,
            ((alpha_arr - alpha_bar) / sigma) ** 2,
            0.0,
        )

        weights = np.where(
            above,
            scipy_chi2.cdf(chi2_values, df=1),
            0.0,
        )

        return weights.astype(np.float64)

    w_nc = _weights(alpha_nc)
    w_nf = _weights(alpha_nf)

    result: list[Particle] = []

    for i in lv_idx:
        p = particles[i]
        p.weight = 1.0
        result.append(p)

    for local_i, global_i in enumerate(neutral_central_idx):
        w = float(w_nc[local_i])

        if w < w_cut:
            continue

        pt_rescaled = float(pt[global_i]) * w

        if pt_rescaled < pt_cut_central:
            continue

        p = particles[global_i]
        p.weight = w
        result.append(p.rescaled())

    for local_i, global_i in enumerate(neutral_forward_idx):
        w = float(w_nf[local_i])

        if w < w_cut:
            continue

        pt_rescaled = float(pt[global_i]) * w

        if pt_rescaled < pt_cut_forward:
            continue

        p = particles[global_i]
        p.weight = w
        result.append(p.rescaled())

    return result


def run_puppi_with_config(
    particles: list[Particle],
    n_pu: int,
    config: PUPPIConfig,
) -> list[Particle]:
    """
    Run PUPPI using a PUPPIConfig object.
    """
    return run_puppi(
        particles=particles,
        n_pu=n_pu,
        R0=config.R0,
        Rmin=config.Rmin,
        w_cut=config.w_cut,
        eta_tracker=config.eta_tracker,
    )