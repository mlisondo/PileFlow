"""
Schema constants for PUMML/PUPPI image-array outputs.

This module defines the expected `.npz` contract produced by the generator.

The `.npz` file contains:
    - PUMML input images
    - PUMML target images
    - clean no-pileup reference images
    - true / pileup / PUPPI constituent arrays

Do not change these keys casually. Downstream diagnostics, PUMML training,
and comparison scripts expect this contract.
"""

from __future__ import annotations


# Maximum number of constituents saved per jet.
#
# Constituent arrays are zero-padded to shape:
#
#     (N_jets, MAX_CONST)
#
MAX_CONST = 500


# Image grid sizes.
N_PIXELS_CHARGED = 36
N_PIXELS_NEUTRAL = 9


# Default image window.
#
# The full window is 0.9 x 0.9 in Delta eta and Delta phi.
# Therefore the half-width is 0.45.
ETA_RANGE = 0.45
PHI_RANGE = 0.45


# Charged-particle tracking threshold used by the image builder.
PT_CHARGED_CUT = 0.5


# PUMML image keys.
IMAGE_KEYS = [
    "ch_charged_lv",
    "ch_charged_pu",
    "ch_neutral_all",
    "ch_neutral_all_raw",
    "ch_neutral_lv",
    "clean_neutral_lv",
    "clean_neutral_all",
]


# Per-jet metadata keys stored in the `.npz`.
JET_METADATA_KEYS = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "n_pu",
]


# Constituent-array prefixes.
CONSTITUENT_PREFIXES = [
    "true",
    "pileup",
    "puppi",
]


# Constituent components.
CONSTITUENT_COMPONENTS = [
    "px",
    "py",
    "pz",
    "e",
    "n",
]


CONSTITUENT_KEYS = [
    f"{prefix}_{component}"
    for prefix in CONSTITUENT_PREFIXES
    for component in CONSTITUENT_COMPONENTS
]


REQUIRED_NPZ_KEYS = (
    IMAGE_KEYS
    + JET_METADATA_KEYS
    + CONSTITUENT_KEYS
)


IMAGE_SHAPES = {
    "ch_charged_lv": ("N", N_PIXELS_CHARGED, N_PIXELS_CHARGED),
    "ch_charged_pu": ("N", N_PIXELS_CHARGED, N_PIXELS_CHARGED),
    "ch_neutral_all": ("N", N_PIXELS_CHARGED, N_PIXELS_CHARGED),
    "ch_neutral_all_raw": ("N", N_PIXELS_NEUTRAL, N_PIXELS_NEUTRAL),
    "ch_neutral_lv": ("N", N_PIXELS_NEUTRAL, N_PIXELS_NEUTRAL),
    "clean_neutral_lv": ("N", N_PIXELS_NEUTRAL, N_PIXELS_NEUTRAL),
    "clean_neutral_all": ("N", N_PIXELS_NEUTRAL, N_PIXELS_NEUTRAL),
}


CONSTITUENT_SHAPES = {
    "true_px": ("N", MAX_CONST),
    "true_py": ("N", MAX_CONST),
    "true_pz": ("N", MAX_CONST),
    "true_e": ("N", MAX_CONST),
    "true_n": ("N",),
    "pileup_px": ("N", MAX_CONST),
    "pileup_py": ("N", MAX_CONST),
    "pileup_pz": ("N", MAX_CONST),
    "pileup_e": ("N", MAX_CONST),
    "pileup_n": ("N",),
    "puppi_px": ("N", MAX_CONST),
    "puppi_py": ("N", MAX_CONST),
    "puppi_pz": ("N", MAX_CONST),
    "puppi_e": ("N", MAX_CONST),
    "puppi_n": ("N",),
}


def empty_image_arrays(n_charged: int = N_PIXELS_CHARGED, n_neutral: int = N_PIXELS_NEUTRAL) -> dict:
    """
    Return empty arrays with the correct `.npz` schema.

    This is used when no jets pass the image-building cuts.
    """
    import numpy as np

    return {
        "ch_charged_lv": np.empty((0, n_charged, n_charged), dtype=np.float32),
        "ch_charged_pu": np.empty((0, n_charged, n_charged), dtype=np.float32),
        "ch_neutral_all": np.empty((0, n_charged, n_charged), dtype=np.float32),
        "ch_neutral_all_raw": np.empty((0, n_neutral, n_neutral), dtype=np.float32),
        "ch_neutral_lv": np.empty((0, n_neutral, n_neutral), dtype=np.float32),
        "clean_neutral_lv": np.empty((0, n_neutral, n_neutral), dtype=np.float32),
        "clean_neutral_all": np.empty((0, n_neutral, n_neutral), dtype=np.float32),
        "jet_pt": np.empty(0, dtype=np.float32),
        "jet_eta": np.empty(0, dtype=np.float32),
        "jet_phi": np.empty(0, dtype=np.float32),
        "n_pu": np.empty(0, dtype=np.int32),
        "true_px": np.empty((0, MAX_CONST), dtype=np.float32),
        "true_py": np.empty((0, MAX_CONST), dtype=np.float32),
        "true_pz": np.empty((0, MAX_CONST), dtype=np.float32),
        "true_e": np.empty((0, MAX_CONST), dtype=np.float32),
        "true_n": np.empty(0, dtype=np.int32),
        "pileup_px": np.empty((0, MAX_CONST), dtype=np.float32),
        "pileup_py": np.empty((0, MAX_CONST), dtype=np.float32),
        "pileup_pz": np.empty((0, MAX_CONST), dtype=np.float32),
        "pileup_e": np.empty((0, MAX_CONST), dtype=np.float32),
        "pileup_n": np.empty(0, dtype=np.int32),
        "puppi_px": np.empty((0, MAX_CONST), dtype=np.float32),
        "puppi_py": np.empty((0, MAX_CONST), dtype=np.float32),
        "puppi_pz": np.empty((0, MAX_CONST), dtype=np.float32),
        "puppi_e": np.empty((0, MAX_CONST), dtype=np.float32),
        "puppi_n": np.empty(0, dtype=np.int32),
    }