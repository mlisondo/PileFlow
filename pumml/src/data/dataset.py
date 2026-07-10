# pumml/src/data/dataset.py
#
# Dataset that reads jet_images.npz produced by gen4e2e.
#
# Input channels stacked as (3, 36, 36):
#   channel 0 (RED)   : ch_neutral_all    — all neutral pT (upsampled 9->36)
#   channel 1 (GREEN) : ch_charged_pu     — charged pileup pT (36x36)
#   channel 2 (BLUE)  : ch_charged_lv     — charged LV pT (36x36)
#
# Target: ch_neutral_lv (9x9) — neutral LV pT (ground truth)
#
# No normalisation is applied, matching the paper:
# "No image normalisation or standardisation was applied to the jet images,
# allowing the network to make use of the overall transverse momentum scale."

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from typing import Tuple, Optional


class PUMMLDataset(Dataset):
    """
    Dataset for PUMML training and evaluation.

    Reads jet_images.npz produced by gen4e2e/pumml_jet_images.py.

    Parameters
    ----------
    npz_path : str
        Path to jet_images.npz.
    max_images : int or None
        Cap the dataset at this many images (useful for quick tests).
        None = use all available images.
    """

    def __init__(self, npz_path: str, max_images: Optional[int] = None):
        data = np.load(npz_path, allow_pickle=False)

        # Input channels
        ch_neutral_all = data["ch_neutral_all"]   # (N, 36, 36)
        ch_charged_pu  = data["ch_charged_pu"]    # (N, 36, 36)
        ch_charged_lv  = data["ch_charged_lv"]    # (N, 36, 36)

        # Target
        ch_neutral_lv  = data["ch_neutral_lv"]    # (N,  9,  9)

        N = len(ch_neutral_lv)
        if max_images is not None:
            N = min(N, max_images)

        # Stack inputs: (N, 3, 36, 36)
        # Order: [neutral_total, charged_pu, charged_lv]
        X = np.stack([
            ch_neutral_all[:N],   # channel 0  RED
            ch_charged_pu[:N],    # channel 1  GREEN
            ch_charged_lv[:N],    # channel 2  BLUE
        ], axis=1).astype(np.float32)

        # Target: (N, 9, 9)
        y = ch_neutral_lv[:N].astype(np.float32)

        self.X = torch.from_numpy(X)               # (N, 3, 36, 36)
        self.y = torch.from_numpy(y).unsqueeze(1)  # (N, 1,  9,  9)

        # Store metadata for bookkeeping
        self.jet_pt  = data["jet_pt"][:N]
        self.jet_eta = data["jet_eta"][:N]
        self.n_pu    = data["n_pu"][:N]
        self.n_total = N

        print(
            f"[PUMMLDataset] Loaded {N} jets from {npz_path}\n"
            f"  X shape : {tuple(self.X.shape)}\n"
            f"  y shape : {tuple(self.y.shape)}\n"
            f"  mean NPU: {self.n_pu.mean():.1f}"
        )

    def __len__(self) -> int:
        return self.n_total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_train_val_split(
    dataset: PUMMLDataset,
    train_frac: float = 0.9,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split a PUMMLDataset into train and validation subsets.

    Paper: 90% train / 10% test split on 56k images.

    Parameters
    ----------
    dataset    : PUMMLDataset instance
    train_frac : fraction of data for training (default 0.9)
    seed       : random seed for reproducibility

    Returns
    -------
    (train_dataset, val_dataset)
    """
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val   = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    print(
        f"[make_train_val_split] "
        f"train={n_train:,}  val={n_val:,}  "
        f"(split {train_frac:.0%}/{1-train_frac:.0%})"
    )
    return train_ds, val_ds
