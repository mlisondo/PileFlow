"""
flows/models/pileflow.py
================================
PileFlow: end-to-end pileup mitigation via Target Conditional Flow Matching.

Extends FlowSim (Vaselli et al. arXiv:2402.13684) to the pileup-mitigation
problem.  Given pileup-contaminated jet images and generator-level context,
the flow jointly generates:

  (a) the pileup-mitigated neutral LV image  (9x9 -> 81 dims)
  (b) reconstructed scalar jet observables   (16 dims)

Context vector Y (253 dims total):
  [0:7]     7  gen scalar features     (pt_gen, eta, phi, m, muon_pT, jetR, jetArea)
  [7:10]    3  flavour one-hot         (light/gluon=0, c=1, b=2)
  [10:91]  81  ch_neutral_all @ 9x9   (total neutral pT incl. pileup)
  [91:172] 81  ch_charged_pu  @ 9x9   (charged pileup pT)
  [172:253]81  ch_charged_lv  @ 9x9   (charged LV pT)

Target vector X (97 dims total):
  [0:81]   81  neutral_lv @ 9x9, standardised   (what the flow generates)
  [81:97]  16  scalar jet observables, standardised

References:
  Vaselli et al. arXiv:2402.13684v2  (FlowSim)
  Lipman et al.  arXiv:2210.02747   (Flow Matching)
  Komiske et al. arXiv:1707.08600   (PUMML)
"""

import math
import torch
import torch.nn as nn

# Dimensionalities
IMG_DIM      = 81    # 9×9 neutral-LV image, flattened
N_SCALARS    = 16    # scalar observables target
N_TARGET     = IMG_DIM + N_SCALARS          # 97
N_GEN_SCALAR = 7                            # pt_gen, eta, phi, m, muon_pT, jetR, jetArea
N_FLAVOUR    = 3                            # one-hot: light/gluon, c, b
N_IMAGES     = 3                            # neutral_all + charged_pu + charged_lv
N_CONTEXT    = N_GEN_SCALAR + N_FLAVOUR + N_IMAGES * IMG_DIM  # 253



# Velocity field components
class SinusoidalTimeEmb(nn.Module):
    """Fixed sinusoidal time embedding — same as DDPM / FlowSim."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """
    Residual block: h_new = LayerNorm(h + Dropout(fc2(SiLU(fc1([h | cond])))))

    The conditioning vector (time_emb + context) is injected at every block.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(hidden_dim + cond_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.act     = nn.SiLU()
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.dropout(self.fc2(self.act(self.fc1(
            torch.cat([h, cond], dim=-1)
        )))))


class CRTVelocityField(nn.Module):
    """
    Continuous ResNet Target (CRT) velocity field  v_θ(z_t, t, Y).

    Learns the velocity field for the flow-matching ODE:
        dz/dt = v_θ(z_t, t, Y)

    Parameters
    ----------
    n_features   : target dimension (97 = 81 img + 16 scalars)
    context_dim  : context dimension (253)
    hidden_dim   : ResNet hidden width (default 512)
    n_blocks     : number of residual blocks (default 8)
    time_emb_dim : sinusoidal time embedding dimension (default 64)
    dropout      : dropout in ResBlocks (set 0 at inference)
    """

    def __init__(
        self,
        n_features:   int,
        context_dim:  int,
        hidden_dim:   int   = 512,
        n_blocks:     int   = 8,
        time_emb_dim: int   = 64,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.context_dim = context_dim

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        cond_dim      = time_emb_dim + context_dim

        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.blocks     = nn.ModuleList([
            ResBlock(hidden_dim, cond_dim, dropout) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        t:       torch.Tensor,   # (N,)   time in [0, 1]
        z:       torch.Tensor,   # (N, n_features)
        context: torch.Tensor,   # (N, context_dim)
    ) -> torch.Tensor:
        # Returns predicted velocity (N, n_features)
        t_emb = self.time_emb(t)                    # (N, time_emb_dim)
        cond  = torch.cat([t_emb, context], dim=-1) # (N, cond_dim)
        h     = self.input_proj(z)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Flow matching loss + ODE integration
class TargetCFM(nn.Module):
    """
    Target Conditional Flow Matching (TCFM) — Lipman et al. / FlowSim.

    Interpolation path:
        z_t = t * x1 + [1 - (1 - sigma_min) * t] * x0
        u_t = x1 - (1 - sigma_min) * x0         (target velocity)
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min

    def sample_training_pair(self, x1: torch.Tensor):
        """
        Sample (t, z_t, u_t) for one training step.

        x1 : (N, D) standardised target samples
        Returns t (N,), z_t (N,D), u_t (N,D)
        """
        N, D = x1.shape
        t    = torch.rand(N, device=x1.device)
        x0   = torch.randn_like(x1)
        z_t  = t[:, None] * x1 + (1 - (1 - self.sigma_min) * t[:, None]) * x0
        u_t  = x1 - (1 - self.sigma_min) * x0
        return t, z_t, u_t

    @torch.no_grad()
    def generate(
        self,
        model:   CRTVelocityField,
        context: torch.Tensor,
        n_steps: int = 100,
        device:  str = "cpu",
    ) -> torch.Tensor:
        """Euler integration z_0 ~ N(0,I) → z_1.  Returns (N, n_features)."""
        model.eval()
        N  = context.shape[0]
        z  = torch.randn(N, model.n_features, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_batch = torch.full((N,), i * dt, device=device)
            z = z + model(t_batch, z, context) * dt
        return z



# Context encoder  (assembles the 253-dim conditioning vector)
class ContextEncoder(nn.Module):
    """
    Assemble the 253-dim context vector Y for PileFlow.

    Inputs
    ------
    scalar_gen     : (N, 7)   [pt_gen, eta, phi, m, muon_pT, jetR, jetArea]
    flavour        : (N,) int  PDG code {1=light, 4=c, 5=b, 21=gluon}
    ch_neutral_all : (N, 81)  total neutral pT @ 9×9, flattened
    ch_charged_pu  : (N, 81)  charged pileup pT @ 9×9, flattened
    ch_charged_lv  : (N, 81)  charged LV pT @ 9×9, flattened

    Output: (N, 253)

    Call .fit() on training data before training to set standardisation stats.
    """

    FLAVOUR_MAP = {
        1: 0,
        2: 0,
        3: 0,
        21: 0,
        4: 1,
        5: 2,
    }   # PDG → class index

    def __init__(
        self,
        scalar_dim:  int = N_GEN_SCALAR,
        n_flavours:  int = N_FLAVOUR,
        img_dim:     int = IMG_DIM,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.n_flavours = n_flavours
        self.img_dim    = img_dim

        # Per-feature standardisation stats — fitted from training data
        self.register_buffer("scalar_mean",      torch.zeros(scalar_dim))
        self.register_buffer("scalar_std",       torch.ones(scalar_dim))
        self.register_buffer("neutral_all_mean", torch.zeros(img_dim))
        self.register_buffer("neutral_all_std",  torch.ones(img_dim))
        self.register_buffer("charged_pu_mean",  torch.zeros(img_dim))
        self.register_buffer("charged_pu_std",   torch.ones(img_dim))
        self.register_buffer("charged_lv_mean",  torch.zeros(img_dim))
        self.register_buffer("charged_lv_std",   torch.ones(img_dim))

    def fit(
        self,
        scalar_gen:     torch.Tensor,   # (N, 7)
        ch_neutral_all: torch.Tensor,   # (N, 81)
        ch_charged_pu:  torch.Tensor,   # (N, 81)
        ch_charged_lv:  torch.Tensor,   # (N, 81)
    ):
        """Compute per-feature mean/std from training set only (no val leakage)."""
        def _stats(t):
            return t.mean(dim=0), t.std(dim=0, unbiased=False).clamp(min=1e-6)

        self.scalar_mean,      self.scalar_std      = _stats(scalar_gen)
        self.neutral_all_mean, self.neutral_all_std = _stats(ch_neutral_all)
        self.charged_pu_mean,  self.charged_pu_std  = _stats(ch_charged_pu)
        self.charged_lv_mean,  self.charged_lv_std  = _stats(ch_charged_lv)

    def forward(
        self,
        scalar_gen:     torch.Tensor,   # (N, 7)
        flavour:        torch.Tensor,   # (N,) int PDG codes
        ch_neutral_all: torch.Tensor,   # (N, 81)
        ch_charged_pu:  torch.Tensor,   # (N, 81)
        ch_charged_lv:  torch.Tensor,   # (N, 81)
    ) -> torch.Tensor:
        """Returns (N, 253) context vector."""
        # Standardise continuous scalars
        s  = (scalar_gen     - self.scalar_mean)      / self.scalar_std
        na = (ch_neutral_all - self.neutral_all_mean) / self.neutral_all_std
        cp = (ch_charged_pu  - self.charged_pu_mean)  / self.charged_pu_std
        cl = (ch_charged_lv  - self.charged_lv_mean)  / self.charged_lv_std

        # One-hot encode jet flavour → (N, 3)
        flv_idx = torch.zeros_like(flavour, dtype=torch.long)
        for pdg, idx in self.FLAVOUR_MAP.items():
            flv_idx[flavour == pdg] = idx
        flv_ohe = torch.zeros(flavour.shape[0], self.n_flavours,
                              device=flavour.device, dtype=s.dtype)
        flv_ohe.scatter_(1, flv_idx.unsqueeze(1), 1.0)

        return torch.cat([s, flv_ohe, na, cp, cl], dim=-1)   # (N, 253)



# Target preprocessor  (standardise / unstandardise the 97-dim target)
class TargetPreprocessor(nn.Module):
    """
    Standardise the 97-dim target vector for flow-matching training.

    Target layout:
      [0:81]   81-dim neutral LV pT image (9x9 flattened)
      [81:97]  16-dim scalar observables

    Call .fit() on training data first, then .encode() / .decode().
    """
    def __init__(self, n_img: int = IMG_DIM, n_scalars: int = N_SCALARS):
        super().__init__()
        self.n_img     = n_img
        self.n_scalars = n_scalars

        # Pixel-wise stats for the neutral LV image
        self.register_buffer("img_mean",    torch.zeros(n_img))
        self.register_buffer("img_std",     torch.ones(n_img))
        # Feature-wise stats for scalar observables
        self.register_buffer("scalar_mean", torch.zeros(n_scalars))
        self.register_buffer("scalar_std",  torch.ones(n_scalars))
        # Threshold below which pixels are zeroed at decode time
        self.pt_threshold = 0.05   # GeV

    def fit(
        self,
        neutral_lv: torch.Tensor,   # (N, 81)
        scalars:    torch.Tensor,   # (N, 16)
    ):
        """Compute standardisation stats from training data."""
        def _stats(t):
            return t.mean(dim=0), t.std(dim=0, unbiased=False).clamp(min=1e-6)

        self.img_mean,    self.img_std    = _stats(neutral_lv)
        self.scalar_mean, self.scalar_std = _stats(scalars)

    def encode(
        self,
        neutral_lv: torch.Tensor,   # (N, 81)
        scalars:    torch.Tensor,   # (N, 16)
    ) -> torch.Tensor:
        #Returns (N, 97) standardised target
        img_z    = (neutral_lv - self.img_mean)    / self.img_std
        scalar_z = (scalars    - self.scalar_mean) / self.scalar_std
        return torch.cat([img_z, scalar_z], dim=-1)

    def decode(
        self,
        Z: torch.Tensor,   # (N, 97)  flow output in standardised space
    ):
        """
        Unstandardise flow output.

        Returns
        -------
        img     : (N, 81) predicted neutral LV pT, clamped >= 0 with threshold
        scalars : (N, 16) predicted scalar observables
        """
        img_z    = Z[:, :self.n_img]
        scalar_z = Z[:, self.n_img:]

        img     = img_z    * self.img_std    + self.img_mean
        scalars = scalar_z * self.scalar_std + self.scalar_mean

        # Physical constraint: pT >= 0; zero out sub-threshold pixels
        img = img.clamp(min=0.0)
        img = img * (img >= self.pt_threshold).float()

        return img, scalars
