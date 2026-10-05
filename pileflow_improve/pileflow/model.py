"""Velocity-field networks for PileFlow.

All models map (t, z_t, neutral_all_9, charged_pu_36, charged_lv_36) to a
velocity of shape (N, 1, 9, 9) and are built from a JSON-serialisable config
with :func:`build_model`, so a checkpoint fully describes its architecture.

Presets
-------
spatial        The original draft architecture (kept bit-for-bit so existing
               checkpoints load). ~505k parameters.
spatial_v2     Recommended. Exact 4x4 charged->neutral cell alignment
               (PixelUnshuffle), FiLM time conditioning and a time embedding
               with useful frequency range. ~0.5M parameters.
spatial_v3     spatial_v2 + fixed (d_eta, d_phi, Delta R) coordinate channels.
spatial_small  Same design as spatial_v2 at PUMML-like size (~20k parameters)
               for a parameter-matched comparison with PUMML (4,711).
flat           Flattened CRT-style MLP (FlowSim-like), kept as an ablation.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


# Presets
PRESETS: dict[str, dict] = {
    "spatial": dict(kind="spatial", base_channels=64, charged_channels=16, n_blocks=6,
                    time_dim=64, time_scale=1.0, encoder="conv", film=False),
    "spatial_v2": dict(kind="spatial", base_channels=64, charged_channels=16, n_blocks=6,
                       time_dim=64, time_scale=1000.0, encoder="unshuffle", film=True),
    "spatial_v3": dict(kind="spatial", base_channels=64, charged_channels=16, n_blocks=6,
                       time_dim=64, time_scale=1000.0, encoder="unshuffle", film=True, coords=True),
    "spatial_small": dict(kind="spatial", base_channels=16, charged_channels=8, n_blocks=2,
                          time_dim=32, time_scale=1000.0, encoder="unshuffle", film=True),
    "flat": dict(kind="flat", hidden=256, n_blocks=4, time_dim=64, time_scale=1.0),
}


def model_config_from_preset(name: str, **overrides) -> dict:
    if name not in PRESETS:
        raise KeyError(f"Unknown architecture preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = copy.deepcopy(PRESETS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["preset"] = name
    return cfg


def build_model(config: dict) -> nn.Module:
    cfg = {k: v for k, v in config.items() if k not in {"preset", "kind"}}
    kind = config.get("kind", "spatial")
    if kind == "spatial":
        return SpatialPileFlowVelocity(**cfg)
    if kind == "flat":
        return FlatCRTVelocity(**cfg)
    raise ValueError(f"Unknown model kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Building blocks
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal features of `scale * t`.

    Frequencies span [1e-4, 1]. With t in [0, 1] and scale=1 every feature is
    almost linear in t (the draft setting); scale=1000 (as in DDPM-style
    embeddings) spreads the features over the full frequency range.
    """

    def __init__(self, dim: int = 64, scale: float = 1.0):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim
        self.scale = float(scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype)
                          / max(half - 1, 1))
        angles = (self.scale * t)[:, None] * freqs[None]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConvChargedEncoder(nn.Module):
    """Draft encoder: two stride-2 3x3 convs, 36x36 -> 9x9.

    Note: output cell j sees fine pixels 4j-3 ... 4j+3 (before the last 3x3
    conv), i.e. it is shifted by ~1.5 fine pixels w.r.t. neutral cell j.
    """

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(8, out_channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UnshuffleChargedEncoder(nn.Module):
    """Lossless, exactly aligned encoder: each 4x4 block of fine charged
    pixels becomes the 16-channel feature vector of the matching 9x9 neutral
    cell (PixelUnshuffle), followed by 1x1 and 3x3 convolutions.
    The sum over the 16 sub-pixels (the cell pT) is passed explicitly."""

    def __init__(self, out_channels: int = 16):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(4)
        self.net = nn.Sequential(
            nn.Conv2d(17, out_channels, 1), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU(),
        )

    def forward(self, x):
        cells = self.unshuffle(x)                       # (N, 16, 9, 9)
        return self.net(torch.cat([cells, cells.sum(1, keepdim=True)], dim=1))


def _groups(channels: int) -> int:
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class SpatialResBlock(nn.Module):
    """Pre-norm residual block; time enters additively (draft) or via FiLM."""

    def __init__(self, channels: int, time_dim: int, film: bool = False):
        super().__init__()
        g = _groups(channels) if film else (8 if channels % 8 == 0 else 1)
        self.film = film
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, 2 * channels if film else channels)
        self.act = nn.GELU()
        if film:
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        if self.film:
            scale, shift = self.time_proj(temb)[:, :, None, None].chunk(2, dim=1)
            h = self.norm2(h) * (1.0 + scale) + shift
        else:
            h = self.norm2(h + self.time_proj(temb)[:, :, None, None])
        return x + self.conv2(self.act(h))



# Velocity fields
class SpatialPileFlowVelocity(nn.Module):
    """Convolutional velocity field on the native 9x9 neutral grid.

    The two 36x36 charged images are encoded to 9x9 feature maps aligned with
    the neutral calorimeter cells and concatenated with z_t and the total
    neutral image, preserving the local eta-phi structure exploited by PUMML.
    """

    def __init__(self, base_channels: int = 64, charged_channels: int = 16, n_blocks: int = 6,
                 time_dim: int = 64, time_scale: float = 1.0, encoder: str = "conv",
                 film: bool = False, coords: bool = False):
        super().__init__()
        # Optional fixed (d_eta, d_phi, Delta R) channels: jet images are centred on the jet
        # axis, so absolute position carries physics (core vs. periphery) that plain
        # translation-equivariant convolutions only see through the zero padding.
        self.coords = coords
        if coords:
            c = (torch.arange(9, dtype=torch.float32) - 4.0) / 4.0
            ee, pp = torch.meshgrid(c, c, indexing="ij")
            self.register_buffer("coord_map", torch.stack([ee, pp, torch.sqrt(ee ** 2 + pp ** 2)])[None])
        enc = {"conv": ConvChargedEncoder, "unshuffle": UnshuffleChargedEncoder}[encoder]
        self.charged_pu_encoder = enc(charged_channels)
        self.charged_lv_encoder = enc(charged_channels)
        self.time_emb = SinusoidalTimeEmbedding(time_dim, scale=time_scale)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv2d(2 + 2 * charged_channels + (3 if coords else 0), base_channels, 3, padding=1)
        self.blocks = nn.ModuleList(SpatialResBlock(base_channels, time_dim, film)
                                    for _ in range(n_blocks))
        g = _groups(base_channels) if film else (8 if base_channels % 8 == 0 else 1)
        self.out_norm = nn.GroupNorm(g, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, t, z, neutral_all, charged_pu, charged_lv):
        parts = [z, neutral_all, self.charged_pu_encoder(charged_pu), self.charged_lv_encoder(charged_lv)]
        if self.coords:
            parts.append(self.coord_map.expand(z.shape[0], -1, -1, -1))
        x = torch.cat(parts, dim=1)
        h = self.stem(x)
        temb = self.time_mlp(self.time_emb(t))
        for block in self.blocks:
            h = block(h, temb)
        return self.out(self.act(self.out_norm(h)))

    def count_parameters(self):
        return count_parameters(self)


class FlatResBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden + cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()

    def forward(self, h, cond):
        return self.norm(h + self.fc2(self.act(self.fc1(torch.cat([h, cond], dim=1)))))


class FlatCRTVelocity(nn.Module):
    """Flattened ResNet velocity field (FlowSim CRT-like ablation)."""

    def __init__(self, hidden: int = 256, n_blocks: int = 4, time_dim: int = 64,
                 time_scale: float = 1.0):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim, scale=time_scale)
        self.input = nn.Linear(81, hidden)
        cond_dim = 81 + 2 * 36 * 36 + time_dim
        self.blocks = nn.ModuleList(FlatResBlock(hidden, cond_dim) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, 81)

    def forward(self, t, z, neutral_all, charged_pu, charged_lv):
        n = z.shape[0]
        cond = torch.cat([self.time_emb(t), neutral_all.reshape(n, -1),
                          charged_pu.reshape(n, -1), charged_lv.reshape(n, -1)], dim=1)
        h = self.input(z.reshape(n, 81))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).reshape(n, 1, 9, 9)

    def count_parameters(self):
        return count_parameters(self)
