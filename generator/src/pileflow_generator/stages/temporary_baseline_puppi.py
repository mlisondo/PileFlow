"""
Temporary PUPPI baseline used by the migrated generator.

The old generator wrote PUPPI-mitigated constituents directly into the
PUMML-style `.npz` output using these keys:

    puppi_px
    puppi_py
    puppi_pz
    puppi_e
    puppi_n

Long term, this code can move to the top-level PileFlow/puppi package.  


Reference
---------
Bertolini, Harris, Low, Tran, arXiv:1407.6013.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import chi2 as scipy_chi2


@dataclass
class Particle:
    """
    Minimal four-momentum carrier with PUPPI weight and truth tags.

    Parameters
    ----------
    px, py, pz, e:
        Four-momentum components.
    charge:
        Particle electric charge.
    is_lv:
        True for leading-vertex particles, False for pileup particles.
    weight:
        PUPPI weight. Defaults to 1.
    """

    px: float
    py: float
    pz: float
    e: float
    charge: float
    is_lv: bool
    weight: float = 1.0

    @property
    def pt(self) -> float:
        """Transverse momentum."""
        return float(np.sqrt(self.px**2 + self.py**2))

    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
        if p == 0 or p == abs(self.pz):
            return float(np.sign(self.pz) * 1e9)
        return float(0.5 * np.log((p + self.pz) / (p - self.pz)))

    @property
    def phi(self) -> float:
        """Azimuthal angle."""
        return float(np.arctan2(self.py, self.px))

    @property
    def mass(self) -> float:
        """Invariant mass."""
        m2 = self.e**2 - self.px**2 - self.py**2 - self.pz**2
        return float(np.sqrt(max(m2, 0.0)))

    def rescaled(self) -> "Particle":
        """
        Return a new particle with four-momentum scaled by the PUPPI weight.

        The charge and LV/PU truth tag are copied unchanged.
        """
        w = self.weight
        return Particle(
            px=self.px * w,
            py=self.py * w,
            pz=self.pz * w,
            e=self.e * w,
            charge=self.charge,
            is_lv=self.is_lv,
            weight=self.weight,
        )


def unpack_particles(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    e: np.ndarray,
    charge: np.ndarray,
    is_lv: np.ndarray,
    n: int,
) -> list[Particle]:
    """
    Convert zero-padded constituent arrays into a list of Particle objects.

    Parameters
    ----------
    px, py, pz, e:
        Padded four-momentum arrays.
    charge:
        Padded charge array.
    is_lv:
        Padded leading-vertex truth array. Values > 0.5 are treated as True.
    n:
        Number of valid particles before padding.

    Returns
    -------
    list[Particle]
        Unpadded particle list.
    """
    particles: list[Particle] = []

    for i in range(int(n)):
        particles.append(
            Particle(
                px=float(px[i]),
                py=float(py[i]),
                pz=float(pz[i]),
                e=float(e[i]),
                charge=float(charge[i]),
                is_lv=bool(is_lv[i] > 0.5),
            )
        )

    return particles


def _compute_eta_phi(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized eta and phi computation.

    Parameters
    ----------
    px, py, pz:
        Momentum component arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(eta, phi)`` arrays with dtype float64.
    """
    p = np.sqrt(px**2 + py**2 + pz**2)

    # Guard against p = 0 and exactly longitudinal particles.
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

    Parameters
    ----------
    eta1, phi1:
        Coordinates for the first particle set, shape ``(N,)``.
    eta2, phi2:
        Coordinates for the second particle set, shape ``(M,)``.

    Returns
    -------
    np.ndarray
        Delta-R matrix with shape ``(N, M)``.
    """
    deta = eta1[:, None] - eta2[None, :]
    dphi = phi1[:, None] - phi2[None, :]

    # Wrap Delta phi into [-pi, pi].
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
    Compute PUPPI local shape alpha for many particles at once.

    The local shape is

        alpha_i = log(sum_j pT_j / DeltaR_ij)

    where the sum only includes neighbours satisfying

        Rmin <= DeltaR_ij <= R0.

    Parameters
    ----------
    target_eta, target_phi:
        Particles for which alpha is computed.
    nb_eta, nb_phi, nb_pt:
        Neighbour particles entering the alpha sum.
    R0:
        Cone size.
    Rmin:
        Collinear regulator / minimum angular distance.

    Returns
    -------
    np.ndarray
        Alpha array. Particles with no neighbours receive ``-np.inf``.
    """
    if len(target_eta) == 0 or len(nb_eta) == 0:
        return np.full(len(target_eta), -np.inf, dtype=np.float64)

    dr = _delta_r_matrix(target_eta, target_phi, nb_eta, nb_phi)

    mask = (dr >= Rmin) & (dr <= R0)

    # Outside the cone, use denominator 1.0 and zero contribution.
    dr_safe = np.where(mask, dr, 1.0)
    contrib = np.where(mask, nb_pt[None, :] / dr_safe, 0.0)

    total = contrib.sum(axis=1)

    alpha = np.where(total > 0, np.log(total), -np.inf)

    return alpha.astype(np.float64)


def _left_rms(values: np.ndarray, median: float) -> float:
    """
    Compute left-side RMS around the median.

    This is used to characterize the pileup alpha distribution while reducing
    sensitivity to right-side tails.
    """
    left = values[values < median]

    if len(left) == 0:
        return 1.0

    return float(np.sqrt(np.mean((left - median) ** 2)))


def characterise_pileup(alpha_pu: np.ndarray) -> tuple[float, float]:
    """
    Compute event-level PUPPI pileup parameters.

    Parameters
    ----------
    alpha_pu:
        Alpha values for charged pileup particles.

    Returns
    -------
    tuple[float, float]
        ``(alpha_bar, sigma)``, where ``alpha_bar`` is the median and
        ``sigma`` is the left-side RMS.
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
    Run the vectorized PUPPI baseline on one event.

    Parameters
    ----------
    particles:
        Full event particle list containing leading-vertex and pileup particles.
    n_pu:
        Number of pileup interactions. Used in the PUPPI pT threshold.
    R0:
        Cone size for alpha computation.
    Rmin:
        Minimum Delta-R cutoff.
    w_cut:
        Minimum PUPPI weight required to keep a neutral particle.
    eta_tracker:
        Tracker acceptance boundary.

    Returns
    -------
    list[Particle]
        PUPPI-mitigated particles.

    Notes
    -----
    Behavior:
        - Charged LV particles are kept with weight 1.
        - Charged PU particles are discarded.
        - Neutral particles receive PUPPI weights.
        - Neutrals failing weight or pT cuts are discarded.
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

    # PUPPI paper-inspired pT cuts.
    pt_cut_central = 0.1 + 0.007 * n_pu
    pt_cut_forward = 0.2 + 0.011 * n_pu

    lv_idx = np.where(is_charged_lv)[0]
    pu_idx = np.where(is_charged_pu)[0]

    # Step 1: characterize pileup using charged PU particles.
    # Central alpha uses charged LV neighbours.
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

    # Step 2: compute alpha for neutral particles.
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

    # Step 3: convert alpha values into PUPPI weights.
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

    # Step 4: build output particle list.
    result: list[Particle] = []

    # Charged LV particles are kept.
    for i in lv_idx:
        p = particles[i]
        p.weight = 1.0
        result.append(p)

    # Charged PU particles are discarded.

    # Central neutrals.
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

    # Forward neutrals.
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


def run_puppi_on_dataset(
    npz_path: str,
    n_pu_override: Optional[int] = None,
    R0: float = 0.3,
    Rmin: float = 0.02,
    w_cut: float = 0.1,
) -> list[list[Particle]]:
    """
    Load an image `.npz` file and run PUPPI on each stored event. This is kept mostly for standalone testing.

    Parameters
    ----------
    npz_path:
        Path to an `.npz` file containing full-event arrays.
    n_pu_override:
        Optional pileup override. If None, use each event's stored ``n_pu``.
    R0, Rmin, w_cut:
        PUPPI algorithm parameters.

    Returns
    -------
    list[list[Particle]]
        One mitigated particle list per stored jet/event.
    """
    data = np.load(npz_path, allow_pickle=False)

    if "full_px" not in data:
        raise ValueError(
            "The provided .npz file does not contain full_px arrays. "
            "PUPPI probably already ran inside the generator. "
            "Use puppi_px/puppi_py/puppi_pz/puppi_e/puppi_n directly."
        )

    full_px = data["full_px"]
    full_py = data["full_py"]
    full_pz = data["full_pz"]
    full_e = data["full_e"]
    full_charge = data["full_charge"]
    full_is_lv = data["full_is_lv"]
    full_n = data["full_n"]
    n_pu_arr = data["n_pu"]

    n_jets = len(full_n)
    print(f"[PUPPI] Running on {n_jets} jets from {npz_path}")

    results: list[list[Particle]] = []

    for i in range(n_jets):
        n_pu = int(n_pu_override if n_pu_override is not None else n_pu_arr[i])

        particles = unpack_particles(
            full_px[i],
            full_py[i],
            full_pz[i],
            full_e[i],
            full_charge[i],
            full_is_lv[i],
            full_n[i],
        )

        mitigated = run_puppi(
            particles,
            n_pu=n_pu,
            R0=R0,
            Rmin=Rmin,
            w_cut=w_cut,
        )

        results.append(mitigated)

        if (i + 1) % 1000 == 0 or (i + 1) == n_jets:
            print(f"  {i + 1}/{n_jets} jets done")

    return results