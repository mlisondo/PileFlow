"""
PDG ID constants used by the generator.

The workflow stores and compares absolute PDG IDs in several places, so these sets contain positive IDs only.

For more information on PDG IDs : https://pdg.lbl.gov/2024/reviews/rpp2024-rev-monte-carlo-numbering.pdf
"""

# Used by feature extraction to decide whether a jet contains a B hadron.
B_HADRON_IDS_ABS = {
    511,   # B0
    521,   # B+
    531,   # Bs0
    541,   # Bc+
    5122,  # Lambda_b0
    5132,  # Xi_b-
    5232,  # Xi_b0
    5332,  # Omega_b-
}

# Used by feature extraction to decide whether a jet contains a charm hadron.
C_HADRON_IDS_ABS = {
    411,   # D+
    421,   # D0
    431,   # Ds+
    4122,  # Lambda_c+
    4132,  # Xi_c0
    4232,  # Xi_c+
    4332,  # Omega_c0
}

# Long-lived particles used as a rough secondary-vertex proxy.
LONG_LIVED_IDS_ABS = {
    310,   # K_S0
    130,   # K_L0
    3122,  # Lambda0
    3112,  # Sigma-
    3222,  # Sigma+
    3312,  # Xi-
    3334,  # Omega-
    421,   # D0
    411,   # D+
}

# Used when extracting partons from the Pythia event record for flavour matching.
QUARK_GLUON_IDS_ABS = {
    1,   # d
    2,   # u
    3,   # s
    4,   # c
    5,   # b
    6,   # t
    21,  # gluon
}

# Pythia status codes used to identify relevant partons for matching.
RELEVANT_STATUS_ABS = {
    23,  # outgoing hard-process parton
    33,  # parton from initial-state radiation
    43,  # parton from final-state radiation off an initial-state leg
    51,  # first-generation FSR splitting
    52,  # first-generation FSR splitting
    53,  # first-generation FSR splitting
    59,  # final parton before hadronisation transition
    62,  # last copy of a parton before it hadronises
}

# Neutrinos are removed from the particle list before jet clustering because they are invisible to the detector.
NEUTRINO_IDS_ABS = {
    12,  # electron neutrino
    14,  # muon neutrino
    16,  # tau neutrino
}