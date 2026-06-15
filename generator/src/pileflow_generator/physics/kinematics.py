"""
Kinematic helper functions to encode angular and eta-phi geometry.
"""

import numpy as np


def wrap_phi(phi: float) -> float:
    """
    Wrap an azimuthal angle into (-pi, pi].

    Parameters
    ----------
    phi:
        Input azimuthal angle in radians.

    Returns
    -------
    float
        Wrapped azimuthal angle in (-pi, pi].
    """
    while phi > np.pi:
        phi -= 2.0 * np.pi
    while phi <= -np.pi:
        phi += 2.0 * np.pi
    return float(phi)


def delta_phi(phi1: float, phi2: float) -> float:
    """
    Compute the absolute angular separation between two azimuthal angles.

    Parameters
    ----------
    phi1, phi2:
        Azimuthal angles in radians.

    Returns
    -------
    float
        Absolute angular separation in [0, pi].
    """
    dphi = abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2.0 * np.pi - dphi
    return float(dphi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """
    Compute the collider distance Delta R in eta-phi space.

    Parameters
    ----------
    eta1, phi1:
        First eta-phi point.
    eta2, phi2:
        Second eta-phi point.

    Returns
    -------
    float
        Delta R distance.
    """
    return float(np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2))