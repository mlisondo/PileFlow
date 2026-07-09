"""
Dataset adapter for PileFlow.

Reads generator outputs:
    jets_*.npy
    jets_*_pileup_images.npz

and returns tensors in the format expected by the PileFlow model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


GEN_SCALAR_IDX = [0, 1, 2, 3, 9, 22, 24]
GEN_FLAVOUR_IDX = 4

SCALAR_TARGET_COLS = [
    5,   # btag
    6,   # recoPt
    7,   # recoPhi
    8,   # recoEta
    10,  # recoNConst
    11,  # nef
    12,  # nhf
    13,  # cef
    14,  # chf
    15,  # qgl
    16,  # jetId
    17,  # ncharged
    18,  # nneutral
    19,  # ctag
    20,  # nSV
    21,  # recoMass
]


REQUIRED_NPY_COLUMNS = 25

REQUIRED_NPZ_KEYS = [
    "ch_neutral_lv",
    "ch_neutral_all_raw",
    "ch_charged_pu",
    "ch_charged_lv",
]


def sum_pool_36_to_9(img36: np.ndarray) -> np.ndarray:
    """
    Sum-pool (N, 36, 36) to flattened (N, 81).

    This preserves total pT when converting from the fine charged grid
    to the 9x9 PileFlow context grid.
    """
    if img36.ndim != 3 or img36.shape[1:] != (36, 36):
        raise ValueError(f"Expected image shape (N, 36, 36), got {img36.shape}")

    n = img36.shape[0]
    return img36.reshape(n, 9, 4, 9, 4).sum(axis=(2, 4)).reshape(n, 81)


def flatten_9x9(img9: np.ndarray, key: str) -> np.ndarray:
    """
    Flatten (N, 9, 9) to (N, 81).
    """
    if img9.ndim != 3 or img9.shape[1:] != (9, 9):
        raise ValueError(f"Expected {key} shape (N, 9, 9), got {img9.shape}")

    return img9.reshape(img9.shape[0], 81)


class PileFlowDataset(Dataset):
    """
    Dataset for PileFlow training and generation.

    Each item returns:
        scalar_gen      (7,)
        flavour         scalar
        neutral_lv      (81,)
        neutral_all_9x9 (81,)
        charged_pu_9x9  (81,)
        charged_lv_9x9  (81,)
        scalars         (16,)
    """

    def __init__(
        self,
        npy_path: str,
        npz_path: str,
        max_n: int | None = None,
    ):
        feats = np.load(npy_path).astype(np.float32)
        data = np.load(npz_path, allow_pickle=False)

        if feats.ndim != 2 or feats.shape[1] < REQUIRED_NPY_COLUMNS:
            raise ValueError(
                f"Expected .npy shape (N, >=25), got {feats.shape}"
            )

        missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"Missing required .npz keys: {missing}")

        neutral_lv = data["ch_neutral_lv"].astype(np.float32)
        neutral_all_raw = data["ch_neutral_all_raw"].astype(np.float32)
        charged_pu = data["ch_charged_pu"].astype(np.float32)
        charged_lv = data["ch_charged_lv"].astype(np.float32)

        lengths = {
            "jets.npy": len(feats),
            "ch_neutral_lv": len(neutral_lv),
            "ch_neutral_all_raw": len(neutral_all_raw),
            "ch_charged_pu": len(charged_pu),
            "ch_charged_lv": len(charged_lv),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Generator .npy/.npz row-count mismatch. "
                "PileFlow requires one-to-one aligned rows. "
                f"Lengths: {lengths}"
            )

        n = len(feats)

        if max_n is not None:
            n = min(n, int(max_n))

        if n <= 0:
            raise ValueError("No jets available after loading generator outputs.")

        self.scalar_gen = torch.from_numpy(feats[:n, GEN_SCALAR_IDX])
        self.flavour = torch.from_numpy(feats[:n, GEN_FLAVOUR_IDX].astype(np.int64))

        self.neutral_lv = torch.from_numpy(
            flatten_9x9(neutral_lv[:n], "ch_neutral_lv")
        )

        self.neutral_all_9x9 = torch.from_numpy(
            flatten_9x9(neutral_all_raw[:n], "ch_neutral_all_raw")
        )

        self.charged_pu_9x9 = torch.from_numpy(
            sum_pool_36_to_9(charged_pu[:n])
        )

        self.charged_lv_9x9 = torch.from_numpy(
            sum_pool_36_to_9(charged_lv[:n])
        )

        self.scalars = torch.from_numpy(feats[:n, SCALAR_TARGET_COLS])
        self.N = n

        print(f"  [dataset] Loaded {self.N:,} jets")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        return (
            self.scalar_gen[i],
            self.flavour[i],
            self.neutral_lv[i],
            self.neutral_all_9x9[i],
            self.charged_pu_9x9[i],
            self.charged_lv_9x9[i],
            self.scalars[i],
        )
    
__all__ = [
    "PileFlowDataset",
    "sum_pool_36_to_9",
    "flatten_9x9",
    "GEN_SCALAR_IDX",
    "GEN_FLAVOUR_IDX",
    "SCALAR_TARGET_COLS",
]