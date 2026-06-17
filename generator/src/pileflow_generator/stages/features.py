"""
Jet feature-construction helpers.

This module contains functions that compute feature-table quantities from
already reconstructed jets or constituent summaries.

These functions are not general kinematics. They are part of the generator's
feature-building stage.
"""

from __future__ import annotations

from typing import Sequence

from pileflow_generator.physics.pdg import (
    B_HADRON_IDS_ABS,
    C_HADRON_IDS_ABS,
    LONG_LIVED_IDS_ABS,
)
from pileflow_generator.physics.smearing import apply_detector_smearing
from pileflow_generator.schemas.jet_features import N_FEATURES

import numpy as np

# utils/helpers.py
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

# utils/helpers.py
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






















# reconstructor.py
def compute_fractions(const_info: Sequence[tuple]) -> dict:
    """
    Compute constituent-based energy fractions and counters.

    Parameters
    ----------
    const_info:
        Sequence of ``(PseudoJet constituent, abs_pid, is_charged)`` entries.

    Returns
    -------
    dict
        Constituent-derived quantities used by the 25-feature jet table.
    """
    e_total = sum(cj.e() for cj, _pid, _is_charged in const_info)

    if e_total <= 0.0:
        e_total = 1e-9

    e_chf = 0.0
    e_nef = 0.0
    e_nhf = 0.0
    e_cef = 0.0

    n_charged = 0
    n_neutral = 0
    has_b = False
    has_c = False
    n_sv = 0
    muon_pt = 0.0

    for cj, pid, is_charged in const_info:
        energy = cj.e()
        apid = abs(int(pid))

        if is_charged:
            n_charged += 1

            if apid == 11:
                # electron / positron
                e_cef += energy
            elif apid == 13:
                # muon / antimuon
                muon_pt = max(muon_pt, cj.pt())
            else:
                e_chf += energy
        else:
            n_neutral += 1

            if apid == 22:
                # photon
                e_nef += energy
            else:
                e_nhf += energy

        if apid in B_HADRON_IDS_ABS:
            has_b = True

        if apid in C_HADRON_IDS_ABS:
            has_c = True

        if apid in LONG_LIVED_IDS_ABS:
            n_sv += 1

    return {
        "nef": float(np.clip(e_nef / e_total, 0.0, 1.0)),
        "nhf": float(np.clip(e_nhf / e_total, 0.0, 1.0)),
        "cef": float(np.clip(e_cef / e_total, 0.0, 1.0)),
        "chf": float(np.clip(e_chf / e_total, 0.0, 1.0)),
        "ncharged": int(n_charged),
        "nneutral": int(n_neutral),
        "n_const": int(len(const_info)),
        "has_b": bool(has_b),
        "has_c": bool(has_c),
        "n_sv": int(n_sv),
        "muon_pt": float(muon_pt),
    }

# reconstructor.py
def compute_btag(rng: np.random.Generator, fracs: dict, flavour: int) -> float:
    """
    Compute the old b-tag proxy score.

    This is not a calibrated experimental b-tagger. It is a generator-level
    proxy used to preserve the old 25-feature table behavior.
    """
    flavour = abs(int(flavour))

    if fracs["has_b"] or flavour == 5:
        return float(np.clip(rng.normal(0.85, 0.10), 0.0, 1.0))

    if fracs["has_c"] or flavour == 4:
        return float(np.clip(rng.normal(0.25, 0.10), 0.0, 1.0))

    return float(np.clip(rng.exponential(0.05), 0.0, 0.30))

# reconstructor.py
def compute_ctag(rng: np.random.Generator, fracs: dict, flavour: int) -> float:
    """
    Compute the old c-tag proxy score.

    This is not a calibrated experimental c-tagger. It is a generator-level
    proxy used to preserve the old 25-feature table behavior.
    """
    flavour = abs(int(flavour))

    if fracs["has_c"] or flavour == 4:
        return float(np.clip(rng.normal(0.80, 0.12), 0.0, 1.0))

    if fracs["has_b"] or flavour == 5:
        return float(np.clip(rng.normal(0.15, 0.08), 0.0, 1.0))

    return float(np.clip(rng.exponential(0.04), 0.0, 0.25))

# reconstructor.py
def build_feature_row(
    *,
    rng: np.random.Generator,
    jet,
    fracs: dict,
    flavour: int,
    jet_R: float,
    jet_area: float,
) -> np.ndarray:
    """
    Build one 25-column jet feature row.

    Parameters
    ----------
    rng:
        NumPy random generator used for smearing and tag proxies.

    jet:
        FastJet PseudoJet.

    fracs:
        Output of ``compute_fractions``.

    flavour:
        Absolute matched parton PDG ID. Use 0 for unmatched.

    jet_R:
        Jet radius parameter.

    jet_area:
        Active jet area. Use 0 if unavailable.

    Returns
    -------
    np.ndarray
        Shape ``(25,)`` and dtype ``float32``.
    """
    flavour_abs = abs(int(flavour))

    reco_pt, reco_eta, reco_phi, reco_m = apply_detector_smearing(
        rng,
        jet.pt(),
        jet.eta(),
        jet.phi(),
        jet.m(),
    )

    jet_id = jet_quality_id(fracs)
    qgl = quark_gluon_likelihood(fracs)
    btag = compute_btag(rng, fracs, flavour_abs)
    ctag = compute_ctag(rng, fracs, flavour_abs)

    features = np.array(
        [
            jet.pt(),                     #  0 pt_gen
            jet.eta(),                    #  1 eta_gen
            jet.phi(),                    #  2 phi_gen
            jet.m(),                      #  3 m_gen
            float(flavour_abs),           #  4 flavour
            btag,                         #  5 btag
            reco_pt,                      #  6 recoPt
            reco_phi,                     #  7 recoPhi
            reco_eta,                     #  8 recoEta
            fracs["muon_pt"],             #  9 muon_pT
            float(fracs["n_const"]),      # 10 recoNConst
            fracs["nef"],                 # 11 nef
            fracs["nhf"],                 # 12 nhf
            fracs["cef"],                 # 13 cef
            fracs["chf"],                 # 14 chf
            qgl,                          # 15 qgl
            float(jet_id),                # 16 jetId
            float(fracs["ncharged"]),     # 17 ncharged
            float(fracs["nneutral"]),     # 18 nneutral
            ctag,                         # 19 ctag
            float(fracs["n_sv"]),         # 20 nSV
            reco_m,                       # 21 recoMass
            float(jet_R),                 # 22 jetR
            1.0,                          # 23 algoCode = anti-kT
            float(jet_area),              # 24 jetArea
        ],
        dtype=np.float32,
    )

    if features.shape != (N_FEATURES,):
        raise RuntimeError(
            f"Feature row has shape {features.shape}; expected ({N_FEATURES},)."
        )

    return features