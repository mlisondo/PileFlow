"""
Jet feature-construction helpers.

This module contains functions that compute feature-table quantities from
already reconstructed jets or constituent summaries.

These functions are not general kinematics. They are part of the generator's
feature-building stage.
"""

import numpy as np


def jet_quality_id(fracs: dict) -> int:
    """
    Compute a simple CMS-like jet ID proxy.

    This is not an official CMS jet ID. It is a proxy.

    Parameters
    ----------
    fracs:
        Dictionary containing at least:

        - ``chf``: charged hadron fraction
        - ``nhf``: neutral hadron fraction
        - ``nef``: neutral electromagnetic fraction
        - ``ncharged``: number of charged constituents

    Returns
    -------
    int
        Jet quality code:

        - ``0`` = fail
        - ``1`` = loose
        - ``3`` = tight
    """
    chf = fracs["chf"]
    nhf = fracs["nhf"]
    nef = fracs["nef"]
    ncharged = fracs["ncharged"]

    loose = (nhf < 0.99) and (nef < 0.99) and (ncharged > 0)
    tight = loose and (nhf < 0.90) and (nef < 0.90) and (chf > 0.0)

    if tight:
        return 3
    if loose:
        return 1
    return 0


def quark_gluon_likelihood(fracs: dict) -> float:
    """
    Compute a simple quark-gluon likelihood proxy.

    This is not a calibrated quark/gluon discriminator. It is only a
    handcrafted feature.

    The proxy encodes:

    - high constituent count -> more gluon-like -> lower score
    - high charged hadron fraction -> more quark-like -> higher score

    Parameters
    ----------
    fracs:
        Dictionary containing at least:

        - ``n_const``: total constituent count
        - ``chf``: charged hadron fraction

    Returns
    -------
    float
        Proxy score in [0, 1].
    """
    n_const = min(fracs["n_const"], 60)
    qgl = (1.0 - (n_const / 60.0)) * 0.5 + fracs["chf"] * 0.5

    return float(np.clip(qgl, 0.0, 1.0))