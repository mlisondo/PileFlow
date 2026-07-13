"""
Dataset adapter for image-only PileFlow.

Reads generator output:
    jets_*_pileup_images.npz

The neutral input and target remain on the 9x9 grid.

The charged input images retain their native 36x36 resolution and are
flattened directly without pooling.
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


def flatten_9x9(img9: np.ndarray, key: str) -> np.ndarray:
    """
    Flatten an image batch from (N, 9, 9) to (N, 81).

    Parameters
    ----------
    img9:
        Batch of 9x9 images.

    key:
        Name of the source array. Used in validation error messages.

    Returns
    -------
    np.ndarray
        Flattened array with shape (N, 81).
    """
    if img9.ndim != 3 or img9.shape[1:] != (9, 9):
        raise ValueError(
            f"Expected {key} shape (N, 9, 9), got {img9.shape}"
        )

    return img9.reshape(img9.shape[0], 81)


def flatten_36x36(img36: np.ndarray, key: str) -> np.ndarray:
    """
    Flatten an image batch from (N, 36, 36) to (N, 1296).

    No pooling or spatial reduction is applied. Every charged-image pixel
    is passed to the model.

    Parameters
    ----------
    img36:
        Batch of native-resolution 36x36 charged images.

    key:
        Name of the source array. Used in validation error messages.

    Returns
    -------
    np.ndarray
        Flattened array with shape (N, 1296).
    """
    if img36.ndim != 3 or img36.shape[1:] != (36, 36):
        raise ValueError(
            f"Expected {key} shape (N, 36, 36), got {img36.shape}"
        )

    return img36.reshape(img36.shape[0], 36 * 36)


class PileFlowDataset(Dataset):
    """
    Dataset for mixed-resolution image-only PileFlow.

    Each item returns
    -----------------
    neutral_lv:
        Shape (81,). Target neutral leading-vertex image, flattened from 9x9.

    neutral_all_9x9:
        Shape (81,). Contaminated neutral input image, flattened from 9x9.

    charged_pu_36x36:
        Shape (1296,). Charged pileup input image, flattened directly
        from its native 36x36 grid.

    charged_lv_36x36:
        Shape (1296,). Charged leading-vertex input image, flattened
        directly from its native 36x36 grid.
    """

    def __init__(
        self,
        npz_path: str,
        max_n: int | None = None,
    ):
        with np.load(npz_path, allow_pickle=False) as data:
            missing = [
                key
                for key in REQUIRED_NPZ_KEYS
                if key not in data.files
            ]

            if missing:
                raise KeyError(
                    f"Missing required .npz keys: {missing}"
                )

            neutral_lv = data["ch_neutral_lv"].astype(
                np.float32,
                copy=True,
            )
            neutral_all_raw = data["ch_neutral_all_raw"].astype(
                np.float32,
                copy=True,
            )
            charged_pu = data["ch_charged_pu"].astype(
                np.float32,
                copy=True,
            )
            charged_lv = data["ch_charged_lv"].astype(
                np.float32,
                copy=True,
            )

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
            raise ValueError(
                "No jets available after loading generator outputs."
            )

        # Target: neutral LV image, 9x9 -> 81.
        self.neutral_lv = torch.from_numpy(
            flatten_9x9(
                neutral_lv[:n],
                "ch_neutral_lv",
            )
        )

        # Neutral context: contaminated neutral image, 9x9 -> 81.
        self.neutral_all_9x9 = torch.from_numpy(
            flatten_9x9(
                neutral_all_raw[:n],
                "ch_neutral_all_raw",
            )
        )

        # Charged context: retain every native 36x36 pixel.
        self.charged_pu_36x36 = torch.from_numpy(
            flatten_36x36(
                charged_pu[:n],
                "ch_charged_pu",
            )
        )

        self.charged_lv_36x36 = torch.from_numpy(
            flatten_36x36(
                charged_lv[:n],
                "ch_charged_lv",
            )
        )

        self.N = n

        print(f"  [dataset] Loaded {self.N:,} jets")
        print(
            "  [dataset] Shapes: "
            "target=81, neutral context=81, "
            "charged PU context=1296, charged LV context=1296"
        )
        print(
            "  [dataset] Total context dimension: "
            f"{81 + 1296 + 1296}"
        )

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        return (
            self.neutral_lv[i],
            self.neutral_all_9x9[i],
            self.charged_pu_36x36[i],
            self.charged_lv_36x36[i],
        )


__all__ = [
    "PileFlowDataset",
    "flatten_9x9",
    "flatten_36x36",
]