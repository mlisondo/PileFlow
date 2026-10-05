"""Invertible per-field preprocessing for PileFlow jet images.

Every field (target, total neutral, charged pileup, charged LV) is mapped as

    x  ->  (f(x) - mean) / std,      f = log1p(x / pt_scale)  or identity,

with a single scalar mean/std per field fitted on the TRAINING split only.
All statistics are stored as buffers so they travel with ``state_dict()``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FIELD_NAMES = ("target", "neutral_all", "charged_pu", "charged_lv")

# Accept historical names so older scripts keep working.
_ALIASES = {
    "target": "target", "neutral_lv": "target", "neutral_lv_9": "target",
    "neutral_all": "neutral_all", "neutral_all_9": "neutral_all",
    "total_neutral": "neutral_all", "total_neutral_9": "neutral_all",
    "charged_pu": "charged_pu", "charged_pu_36": "charged_pu", "charged_pileup": "charged_pu",
    "charged_lv": "charged_lv", "charged_lv_36": "charged_lv",
    "charged_leading_vertex": "charged_lv",
}

# Dataset attribute holding the raw (physical GeV) tensor for every field.
_DATASET_ATTR = {
    "target": "target_neutral_lv_9",
    "neutral_all": "total_neutral_9",
    "charged_pu": "charged_pu_36",
    "charged_lv": "charged_lv_36",
}


def canonical_field(name: str) -> str:
    try:
        return _ALIASES[name]
    except KeyError:
        raise KeyError(f"Unknown preprocessing field {name!r}; known: {sorted(_ALIASES)}") from None


class ImagePreprocessor(nn.Module):
    def __init__(self, mode: str = "log1p", pt_scale: float = 1.0, eps: float = 1e-6,
                 target_mode: str = "log1p", zero_width: float = 0.1, frac_scale: float = 2.0):
        super().__init__()
        if mode not in {"log1p", "linear"}:
            raise ValueError("mode must be 'log1p' or 'linear'")
        if target_mode not in {"log1p", "fraction"}:
            raise ValueError("target_mode must be 'log1p' or 'fraction'")
        if pt_scale <= 0 or eps <= 0:
            raise ValueError("pt_scale and eps must be positive")
        self.mode = mode
        self.pt_scale = float(pt_scale)
        self.eps = float(eps)
        # Target parameterisation (see encode_target):
        #   log1p    : standardised log1p(pT) of the neutral-LV image (draft behaviour)
        #   fraction : neutral-LV fraction of the total neutral pT in each cell, with the
        #              exact zeros dequantised into [-zero_width, 0)
        self.target_mode = target_mode
        self.zero_width = float(zero_width)
        self.frac_scale = float(frac_scale)
        for name in FIELD_NAMES:
            self.register_buffer(f"{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"{name}_std", torch.tensor(1.0))
        # Stored as a buffer so a reloaded preprocessor knows it was fitted.
        self.register_buffer("fitted_flag", torch.tensor(False))

    # config
    @property
    def fitted(self) -> bool:
        return bool(self.fitted_flag.item())

    def get_config(self) -> dict:
        return {"mode": self.mode, "pt_scale": self.pt_scale, "eps": self.eps,
                "target_mode": self.target_mode, "zero_width": self.zero_width, "frac_scale": self.frac_scale}

    @classmethod
    def from_config(cls, config: dict, state_dict: dict | None = None) -> "ImagePreprocessor":
        pre = cls(**config)
        if state_dict is not None:
            state_dict = dict(state_dict)
            # Checkpoints written before `fitted_flag` existed were always fitted.
            state_dict.setdefault("fitted_flag", torch.tensor(True))
            pre.load_state_dict(state_dict, strict=True)
        return pre

    # transform
    def _base(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x / self.pt_scale) if self.mode == "log1p" else x

    def _inverse_base(self, y: torch.Tensor) -> torch.Tensor:
        x = self.pt_scale * torch.expm1(y) if self.mode == "log1p" else y
        return torch.clamp(x, min=0.0)

    def _stats(self, name: str):
        name = canonical_field(name)
        return getattr(self, f"{name}_mean"), getattr(self, f"{name}_std")

    def transform(self, name: str, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(name)
        return (self._base(x) - mean) / std

    def inverse_transform(self, name: str, y: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(name)
        return self._inverse_base(y * std + mean)

    def inverse_target(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse_transform("target", y)

    # target parameterisation
    ZERO_EPS = 1e-6

    def encode_target(self, y_gev: torch.Tensor, total_gev: torch.Tensor,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """Map the physical neutral-LV image to the flow's data space.

        fraction mode: f = y / total in [0, 1] per 9x9 cell (the neutral LV pT can never
        exceed the total neutral pT). About 89% of the cells are exactly empty; these
        are dequantised uniformly into [-zero_width, 0) (as FlowSim does for integer
        targets), so the flow sees a continuous density and decode() maps everything
        below 0 back to an exact zero. Cells without any neutral pT are empty by
        construction and are forced to zero at decoding time.
        """
        if self.target_mode == "log1p":
            return self.transform("target", y_gev)
        total = torch.clamp(total_gev, min=0.0)
        occupied_total = total > self.ZERO_EPS
        f = torch.where(occupied_total, torch.clamp(y_gev, min=0.0) / total.clamp_min(self.ZERO_EPS),
                        torch.zeros_like(y_gev)).clamp(0.0, 1.0)
        empty = f <= 0.0
        u = torch.rand(y_gev.shape, device=y_gev.device, dtype=y_gev.dtype, generator=generator)
        x = torch.where(empty, -self.zero_width * u, f)
        return self.frac_scale * x

    def decode_target(self, x: torch.Tensor, total_gev: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "log1p":
            return self.inverse_target(x)
        f = torch.clamp(x / self.frac_scale, 0.0, 1.0)
        total = torch.clamp(total_gev, min=0.0)
        return torch.where(total > self.ZERO_EPS, f * total, torch.zeros_like(total))

    # fit
    @torch.no_grad()
    def fit(self, dataset, train_indices, chunk: int = 4096, verbose: bool = True):
        """Fit mean/std of f(x) on the training jets only (float64, chunked)."""
        idx = np.asarray(train_indices, dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("Cannot fit the preprocessor on an empty training split")
        if verbose:
            print(f"\nFitting preprocessing on {len(idx)} training jets "
                  f"(mode={self.mode}, pt_scale={self.pt_scale})")
        for name in FIELD_NAMES:
            data = getattr(dataset, _DATASET_ATTR[name])
            s1 = s2 = 0.0
            count = 0
            for start in range(0, len(idx), chunk):
                sel = torch.as_tensor(idx[start:start + chunk])
                y = self._base(data[sel].double())
                s1 += float(y.sum())
                s2 += float((y * y).sum())
                count += y.numel()
            mean = s1 / count
            std = max(np.sqrt(max(s2 / count - mean * mean, 0.0)), self.eps)
            getattr(self, f"{name}_mean").fill_(mean)
            getattr(self, f"{name}_std").fill_(std)
            if verbose:
                print(f"  {name:12s} mean={mean: .6g} std={std: .6g}")
        self.fitted_flag.fill_(True)
        return self
