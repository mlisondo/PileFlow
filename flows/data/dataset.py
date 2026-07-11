"""
Dataset adapter for image-only PileFlow.

Reads generator output:
    jets_*_pileup_images.npz

and returns tensors in the format expected by the image-only PileFlow model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


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
    Dataset for image-only PileFlow training and generation.

    Each item returns:
        neutral_lv      (81,)  target
        neutral_all_9x9 (81,)  input
        charged_pu_9x9  (81,)  input
        charged_lv_9x9  (81,)  input
    """

    def __init__(
        self,
        npz_path: str,
        max_n: int | None = None,
    ):
        data = np.load(npz_path, allow_pickle=False)

        missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
        if missing:
            raise KeyError(f"Missing required .npz keys: {missing}")

        neutral_lv = data["ch_neutral_lv"].astype(np.float32)
        neutral_all_raw = data["ch_neutral_all_raw"].astype(np.float32)
        charged_pu = data["ch_charged_pu"].astype(np.float32)
        charged_lv = data["ch_charged_lv"].astype(np.float32)

        lengths = {
            "ch_neutral_lv": len(neutral_lv),
            "ch_neutral_all_raw": len(neutral_all_raw),
            "ch_charged_pu": len(charged_pu),
            "ch_charged_lv": len(charged_lv),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Generator .npz arrays have mismatched row counts. "
                "PileFlow requires one-to-one aligned rows. "
                f"Lengths: {lengths}"
            )

        n = len(neutral_lv)

        if max_n is not None:
            n = min(n, int(max_n))

        if n <= 0:
            raise ValueError("No jets available after loading generator outputs.")

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

        self.N = n

        print(f"  [dataset] Loaded {self.N:,} jets")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        return (
            self.neutral_lv[i],
            self.neutral_all_9x9[i],
            self.charged_pu_9x9[i],
            self.charged_lv_9x9[i],
        )


__all__ = [
    "PileFlowDataset",
    "sum_pool_36_to_9",
    "flatten_9x9",
]