"""Dataset, splits and a fast in-memory batch loader for PileFlow.

NPZ format (shared with PUMML / PUPPI, produced by generator/gen4pf.py):

    X               (N, 3, 36, 36)  [total neutral (upsampled 9x9 / 16), charged PU, charged LV]
    y_neutral_lv_9  (N, 9, 9)       neutral leading-vertex target
    event_id, jet_rank              hard-event bookkeeping (rank 0 = leading jet)
    event_train_idx, event_test_idx event-grouped 90/10 split (leakage free)

All images are transverse momenta in GeV; axis 0 is eta and axis 1 is phi.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED_CHANNELS = ["total_neutral", "charged_pileup", "charged_leading_vertex"]

OPTIONAL_KEYS = (
    "npu", "image_center_eta", "image_center_phi",
    "puppi_36", "puppi_charged_36", "puppi_neutral_9", "puppi_neutral_36",
    "no_pileup_36", "with_pileup_36",
    "truth_jet_pt", "truth_jet_eta", "truth_jet_phi", "truth_jet_mass",
    "reco_jet_pt", "reco_jet_eta", "reco_jet_phi", "reco_jet_mass",
)


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    eval: np.ndarray

    def as_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "eval": self.eval}

    @classmethod
    def from_dict(cls, d: dict) -> "SplitIndices":
        return cls(**{k: np.asarray(d[k], dtype=np.int64) for k in ("train", "val", "eval")})


def neutral_36_to_9(x: torch.Tensor) -> torch.Tensor:
    """Exactly invert the PUMML 9x9 -> 36x36 upsampling (sum each 4x4 block).

    Accepts (..., 36, 36) and returns (..., 9, 9).
    """
    lead = x.shape[:-2]
    y = F.avg_pool2d(x.reshape(-1, 1, 36, 36), kernel_size=4, stride=4) * 16.0
    return y.reshape(*lead, 9, 9)


class PileFlowImages(torch.utils.data.Dataset):
    """Holds every jet in memory as float32 tensors."""

    def __init__(self, path: str | Path, max_jets: int | None = None, verbose: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {self.path}")
        if self.path.stat().st_size == 0:
            raise ValueError(f"Dataset file is empty (0 bytes): {self.path}")

        with np.load(self.path, allow_pickle=False) as f:
            files = set(f.files)
            missing = {"X", "y_neutral_lv_9"} - files
            if missing:
                raise KeyError(f"Missing required keys {sorted(missing)}; available: {sorted(files)}")

            n_total = len(f["y_neutral_lv_9"])
            n = n_total if max_jets is None else min(int(max_jets), n_total)
            X = np.asarray(f["X"][:n], dtype=np.float32)
            y = np.asarray(f["y_neutral_lv_9"][:n], dtype=np.float32)
            if X.shape[1:] != (3, 36, 36):
                raise ValueError(f"Expected X of shape (N,3,36,36), got {X.shape}")
            if y.shape[1:] != (9, 9):
                raise ValueError(f"Expected y_neutral_lv_9 of shape (N,9,9), got {y.shape}")

            if "channel_names" in files:
                names = [str(c) for c in f["channel_names"].tolist()]
                if names != EXPECTED_CHANNELS:
                    raise ValueError(f"Unexpected channel order {names}; expected {EXPECTED_CHANNELS}")
            elif verbose:
                print(f"WARNING: channel_names not stored; assuming {EXPECTED_CHANNELS}")
            self.channel_names = list(EXPECTED_CHANNELS)

            self.event_id = np.asarray(f["event_id"][:n], dtype=np.int64) if "event_id" in files else None
            self.jet_rank = np.asarray(f["jet_rank"][:n], dtype=np.int64) if "jet_rank" in files else None
            self.extra = {k: np.asarray(f[k][:n]) for k in OPTIONAL_KEYS if k in files}

            # Predefined event-grouped split. With --max-jets we keep only the
            # indices that survive truncation, so train and eval stay disjoint
            # and the eval set is always a subset of the official test set.
            self.has_predefined_event_split = "event_train_idx" in files and "event_test_idx" in files
            if self.has_predefined_event_split:
                tr = np.asarray(f["event_train_idx"], dtype=np.int64)
                te = np.asarray(f["event_test_idx"], dtype=np.int64)
                self.train_pool_idx = np.sort(tr[tr < n])
                self.eval_idx = np.sort(te[te < n])
                if len(self.train_pool_idx) == 0 or len(self.eval_idx) == 0:
                    raise ValueError("max_jets is too small: a predefined split became empty")
            else:
                self.train_pool_idx = self.eval_idx = None
        self.n_total_in_file = n_total

        self.total_neutral_36 = torch.from_numpy(X[:, 0:1].copy())
        self.charged_pu_36 = torch.from_numpy(X[:, 1:2].copy())
        self.charged_lv_36 = torch.from_numpy(X[:, 2:3].copy())
        self.total_neutral_9 = neutral_36_to_9(self.total_neutral_36)
        self.target_neutral_lv_9 = torch.from_numpy(y).unsqueeze(1)
        del X

        self._sanity_checks()
        if verbose:
            self._summary()

    # checks
    def _sanity_checks(self):
        for name in ("total_neutral_36", "charged_pu_36", "charged_lv_36", "target_neutral_lv_9"):
            t = getattr(self, name)
            if not torch.isfinite(t).all():
                raise ValueError(f"Non-finite values in {name}")
            if float(t.min()) < -1e-6:
                raise ValueError(f"Negative pT pixel in {name}: {float(t.min())}")
        # The neutral channel must be block constant (9x9 upsampled by 4x4 / 16).
        up = self.total_neutral_9.repeat_interleave(4, -2).repeat_interleave(4, -1) / 16.0
        err = float((up - self.total_neutral_36).abs().max())
        if err > 1e-3:
            raise ValueError(f"Neutral channel is not a 4x4 block upsampling of a 9x9 image (max dev {err:.3g} GeV)")

    def _summary(self):
        print("\n" + "=" * 64)
        print(f"PileFlow dataset: {self.path}")
        print(f"  jets used / in file : {len(self)} / {self.n_total_in_file}")
        print(f"  neutral 9x9, charged 36x36, target 9x9 : "
              f"{tuple(self.total_neutral_9.shape)}, {tuple(self.charged_lv_36.shape)}, "
              f"{tuple(self.target_neutral_lv_9.shape)}")
        if self.has_predefined_event_split:
            print(f"  predefined event split: dev={len(self.train_pool_idx)} eval={len(self.eval_idx)}")
        else:
            print("  no predefined split: one will be generated")
        print("=" * 64)

    # access
    def __len__(self):
        return int(self.target_neutral_lv_9.shape[0])

    def get_batch(self, idx) -> dict:
        idx = torch.as_tensor(np.asarray(idx, dtype=np.int64))
        batch = {
            "neutral_all_9": self.total_neutral_9[idx],
            "charged_pu_36": self.charged_pu_36[idx],
            "charged_lv_36": self.charged_lv_36[idx],
            "target": self.target_neutral_lv_9[idx],
            "index": idx,
        }
        if self.event_id is not None:
            batch["event_id"] = torch.from_numpy(self.event_id[idx.numpy()])
        if self.jet_rank is not None:
            batch["jet_rank"] = torch.from_numpy(self.jet_rank[idx.numpy()])
        return batch

    def __getitem__(self, i):
        return {k: v[0] for k, v in self.get_batch([int(i)]).items()}


class BatchLoader:
    """Iterates over contiguous in-memory batches (much faster than per-item Subset)."""

    def __init__(self, dataset: PileFlowImages, indices, batch_size: int,
                 shuffle: bool = False, seed: int = 0):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(self.indices) if self.shuffle else self.indices
        for start in range(0, len(order), self.batch_size):
            yield self.dataset.get_batch(order[start:start + self.batch_size])


# splits
def _grouped_split(indices: np.ndarray, groups: np.ndarray, fraction: float, seed: int):
    """Move `fraction` of the groups (hard events) in `indices` to the second output."""
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups[indices]))
    n_out = int(np.clip(round(fraction * len(unique)), 1, len(unique) - 1))
    out = np.isin(groups[indices], unique[:n_out])
    return np.sort(indices[~out]), np.sort(indices[out])


def make_splits(dataset: PileFlowImages, seed: int = 42, eval_fraction: float = 0.10,
                val_fraction_of_train: float = 0.10) -> SplitIndices:
    """Build train/val/eval splits with no hard-event shared between them.

    1. If the file stores event_train_idx/event_test_idx they are used as-is
       (the eval set is then identical to the one used for PUMML).
    2. The development pool is split into train/val by event_id.
    3. Without stored indices an event-grouped (or, lacking event_id,
       row-level) split is generated.
    """
    if not 0.0 < val_fraction_of_train < 1.0 or not 0.0 < eval_fraction < 1.0:
        raise ValueError("split fractions must lie in (0, 1)")
    n = len(dataset)
    groups = dataset.event_id if dataset.event_id is not None else np.arange(n)

    if dataset.has_predefined_event_split:
        dev, ev = dataset.train_pool_idx, dataset.eval_idx
    else:
        dev, ev = _grouped_split(np.arange(n), groups, eval_fraction, seed)
    tr, va = _grouped_split(dev, groups, val_fraction_of_train, seed + 1)
    splits = SplitIndices(train=tr, val=va, eval=ev)
    check_splits(splits, dataset.event_id)
    return splits


def check_splits(splits: SplitIndices, event_id: np.ndarray | None):
    parts = splits.as_dict()
    names = list(parts)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = parts[names[i]], parts[names[j]]
            if np.intersect1d(a, b).size:
                raise RuntimeError(f"{names[i]} and {names[j]} indices overlap")
            if event_id is not None and np.intersect1d(event_id[a], event_id[b]).size:
                raise RuntimeError(f"Event leakage between {names[i]} and {names[j]}")


def describe_split_mode(dataset: PileFlowImages) -> str:
    if dataset.has_predefined_event_split:
        return "predefined event-level (event_train_idx / event_test_idx)"
    return "generated event-level" if dataset.event_id is not None else "generated row-level"
