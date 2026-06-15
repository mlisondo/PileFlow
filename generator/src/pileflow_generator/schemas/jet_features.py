"""
Jet feature schema for the generator output table.

This module defines the column contract for the `.npy` jet table.

The output table has shape:

    (N_jets, N_FEATURES)

where each row is one reconstructed jet and each column is defined by
FEATURE_NAMES.
"""

# Dataset feature definition.
FEATURE_NAMES = [
    "pt_gen",        #  0  generator-level jet transverse momentum (GeV)
    "eta_gen",       #  1  generator-level pseudorapidity eta
    "phi_gen",       #  2  generator-level azimuthal angle phi
    "m_gen",         #  3  generator-level jet invariant mass (GeV)
    "flavour",       #  4  absolute PDG ID of the matched parton
    "btag",          #  5  b-tagging discriminant proxy in [0, 1]
    "recoPt",        #  6  smeared detector-like jet pT (GeV)
    "recoPhi",       #  7  smeared azimuthal angle phi
    "recoEta",       #  8  smeared pseudorapidity eta
    "muon_pT",       #  9  max muon pT among jet constituents; 0 if none
    "recoNConst",    # 10  number of visible final-state particles inside the jet
    "nef",           # 11  neutral electromagnetic energy fraction
    "nhf",           # 12  neutral hadronic energy fraction
    "cef",           # 13  charged electromagnetic energy fraction
    "chf",           # 14  charged hadronic energy fraction
    "qgl",           # 15  quark-gluon likelihood proxy in [0, 1]
    "jetId",         # 16  jet quality ID integer
    "ncharged",      # 17  raw count of charged constituents
    "nneutral",      # 18  raw count of neutral constituents
    "ctag",          # 19  c-tagging discriminant proxy in [0, 1]
    "nSV",           # 20  number of secondary-vertex proxy candidates
    "recoMass",      # 21  smeared jet invariant mass (GeV)
    "jetR",          # 22  jet radius parameter R
    "algoCode",      # 23  integer code for the clustering algorithm
    "jetArea",       # 24  active jet area; 0 if unavailable
]

N_FEATURES = len(FEATURE_NAMES)

# anti-kT with R = 0.4 only
ALGO_NAME_TO_CODE = {
    "antikt": 1,
}

ALGO_CODE_TO_NAME = {v: k for k, v in ALGO_NAME_TO_CODE.items()}