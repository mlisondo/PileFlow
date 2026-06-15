"""
Detector-response smearing helpers.
"""

import numpy as np

from pileflow_generator.physics.kinematics import wrap_phi

def apply_detector_smearing(
    rng,
    pt_true: float,
    eta_true: float,
    phi_true: float,
    m_true: float,
) -> tuple[float, float, float, float]:
    """
    Apply simple detector-like smearing to jet kinematics.

    Parameters
    ----------
    rng:
        NumPy random generator used for reproducibility.
    pt_true:
        Generator-level transverse momentum.
    eta_true:
        Generator-level pseudorapidity.
    phi_true:
        Generator-level azimuthal angle.
    m_true:
        Generator-level invariant mass.

    Returns
    -------
    tuple[float, float, float, float]
        (reco_pt, reco_eta, reco_phi, reco_mass).
    """
    a = 1.0
    b = 0.05

    sigma_pt = pt_true * np.sqrt((a / np.sqrt(max(pt_true, 1.0))) ** 2 + b**2)

    reco_pt = max(0.0, rng.normal(pt_true, sigma_pt))
    reco_eta = float(rng.normal(eta_true, 0.01))
    reco_phi = wrap_phi(rng.normal(phi_true, 0.01))
    reco_m = max(0.0, rng.normal(m_true, 0.05 * max(m_true, 0.1)))

    return float(reco_pt), reco_eta, reco_phi, float(reco_m)