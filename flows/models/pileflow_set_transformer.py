"""
flows/models/pileflow_set_transformer.py
=========================================

Patch-token set-transformer velocity field for PileFlow.

Token design
------------
81 tokens — one per output pixel in the 9×9 neutral-LV image.

Each token aggregates ALL channel information for its spatial position (r, c):

    z_t[r, c]                 1 scalar   — noisy target at this pixel
    neutral_all[r, c]         1 scalar   — contaminated neutral image pixel
    charged_pu[4r:4r+4, 4c:4c+4]  16 scalars  — 4×4 non-overlapping patch
    charged_lv[4r:4r+4, 4c:4c+4]  16 scalars  — same for leading-vertex charged
    η = (r − 4) / 4           1 scalar   — explicit row coordinate in [−1, 1]
    φ = (c − 4) / 4           1 scalar   — explicit column coordinate in [−1, 1]
    ─────────────────────────────────────
    total raw features per token: 36

The 4×4 patch extraction uses torch.Tensor.unfold with size=4, step=4, which
tiles the 36×36 charged images into 9×9 = 81 non-overlapping 4×4 blocks.

The explicit (η, φ) values give the model direct access to the spatial
coordinate of each token without forcing it to discover this from position
embeddings alone.  Sinusoidal 2D position encodings are added on top for
multi-frequency spatial representations.

Time conditioning
-----------------
Time is embedded via a sinusoidal embedding followed by a two-layer MLP,
then added to every token before the attention layers.  This is the standard
approach for flow/diffusion transformers.

Attention blocks
----------------
SAB  (Self-Attention Block)
    Full global self-attention.  Complexity O(N²) where N=81 — only 6561
    pairs per head, completely negligible.  Uses F.scaled_dot_product_attention
    for MPS acceleration (avoids the slow nn.MultiheadAttention batch_first path).

ISAB (Induced Self-Attention Block, Lee et al. 2019)
    Reduces attention to O(N·m) via m learnable inducing points.
    Step 1: inducing points I attend to input X  →  H of shape (m, d)
    Step 2: input X attends to H                 →  output of shape (N, d)
    For N=81 and m=16, saves 81² / (2·81·16) ≈ 2.5× in attention FLOPs.
    More useful when scaling up to larger token grids (e.g. 36×36 = 1296).
    Enable with n_inducing > 0.

Architecture
------------
    raw tokens  : (N, 81, 36)  from patch extraction + explicit coordinates
    projection  : Linear(36, d_model)  →  (N, 81, d_model)
    pos encoding: add sinusoidal 2D PE  (shape: (1, 81, d_model))
    time cond.  : add MLP(SinEmb(t))   (shape: (N, 1, d_model) broadcast)
    attn layers : n_layers × SAB or ISAB
    output      : LayerNorm → Linear(d_model, 1) → squeeze → (N, 81)

References
----------
Lee et al.     arXiv:1810.00825  (Set Transformer / ISAB)
Vaswani et al. arXiv:1706.03762  (Transformer / scaled dot-product attention)
Vaselli et al. arXiv:2402.13684  (FlowSim / Conditional Flow Matching)
Komiske et al. arXiv:1707.08600  (PUMML / pileup mitigation)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pileflow import (
    CHARGED_SIDE,
    N_CONTEXT,
    N_TARGET,
    NEUTRAL_SIDE,
    SinusoidalTimeEmb,
)

# Grid constants 
_GRID = NEUTRAL_SIDE        # 9
_N_TOKENS = _GRID * _GRID   # 81
_PATCH = CHARGED_SIDE // _GRID   # 4  (36 / 9)

# Raw features per token: z_t(1) + neutral(1) + pu_patch(16) + lv_patch(16) + η(1) + φ(1)
_RAW_DIM = 1 + 1 + _PATCH * _PATCH + _PATCH * _PATCH + 2   # = 36


# Positional encoding 

def _make_sinusoidal_2d(n_rows: int, n_cols: int, d_model: int) -> torch.Tensor:
    """
    Fixed sinusoidal 2D position encoding for an n_rows × n_cols grid.

    For each pixel (r, c) the encoding uses d_model/4 frequency bands:

        PE[r*n_cols+c, 4k]   = sin(η / f_k)
        PE[r*n_cols+c, 4k+1] = cos(η / f_k)
        PE[r*n_cols+c, 4k+2] = sin(φ / f_k)
        PE[r*n_cols+c, 4k+3] = cos(φ / f_k)

    where η, φ ∈ [−1, 1] are the normalised row/col coordinates and
    f_k = 10000^(2k / (d_model/4)).

    Returns
    -------
    Tensor of shape (n_rows*n_cols, d_model)
    """
    if d_model % 4 != 0:
        raise ValueError(
            f"d_model must be divisible by 4 for 2D sinusoidal encoding, "
            f"got {d_model}"
        )
    d_quarter = d_model // 4
    pe = torch.zeros(n_rows * n_cols, d_model)

    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            eta = (r - n_rows // 2) / max(n_rows // 2, 1)
            phi = (c - n_cols // 2) / max(n_cols // 2, 1)
            for k in range(d_quarter):
                freq = 10000.0 ** (2.0 * k / max(d_quarter, 1))
                pe[idx, 4 * k]     = math.sin(eta / freq)
                pe[idx, 4 * k + 1] = math.cos(eta / freq)
                pe[idx, 4 * k + 2] = math.sin(phi / freq)
                pe[idx, 4 * k + 3] = math.cos(phi / freq)

    return pe  # (n_rows*n_cols, d_model)


def _make_explicit_coords(n_rows: int, n_cols: int) -> torch.Tensor:
    """
    Explicit (η, φ) coordinates for each pixel on an n_rows × n_cols grid.

    Returns
    -------
    Tensor of shape (n_rows*n_cols, 2) with values in [−1, 1].
    """
    coords = torch.zeros(n_rows * n_cols, 2)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            coords[idx, 0] = (r - n_rows // 2) / max(n_rows // 2, 1)
            coords[idx, 1] = (c - n_cols // 2) / max(n_cols // 2, 1)
    return coords  # (81, 2)


# Attention blocks 

class _SAB(nn.Module):
    """
    Self-Attention Block with full global attention.

    Complexity: O(N²) where N = number of tokens (81 here).
    Uses F.scaled_dot_product_attention for MPS acceleration.
    Pre-LayerNorm residual layout (more stable than post-norm).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, S, D) → (N, S, D)"""
        N, S, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # Pre-norm → fused QKV → multi-head split
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (N, H, S, Dh)

        # Scaled dot-product attention (Flash / MPS-accelerated)
        a = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        a = a.transpose(1, 2).contiguous().view(N, S, D)
        x = x + self.drop(self.out_proj(a))

        # Feed-forward
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class _ISAB(nn.Module):
    """
    Induced Self-Attention Block (Lee et al., arXiv:1810.00825).

    Reduces attention complexity from O(N²) to O(N·m) using m learnable
    inducing points.

    Forward pass:
        H   = CrossAttn(query=I, key=X, value=X)   →  (N, m, D)
        out = CrossAttn(query=X, key=H, value=H)   →  (N, S, D)

    where I are the m learnable inducing points shared across the batch.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_inducing = n_inducing

        # Learnable inducing points: (1, m, d_model) broadcast over batch
        self.inducing = nn.Parameter(
            torch.randn(1, n_inducing, d_model) * (d_model ** -0.5)
        )

        # Cross-attention 1: inducing → input   (I queries X)
        self.norm_i  = nn.LayerNorm(d_model)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.q1  = nn.Linear(d_model, d_model, bias=False)
        self.kv1 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o1  = nn.Linear(d_model, d_model)

        # Cross-attention 2: input → inducing   (X queries H)
        self.norm_h  = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)
        self.q2  = nn.Linear(d_model, d_model, bias=False)
        self.kv2 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o2  = nn.Linear(d_model, d_model)

        # Feed-forward
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout

    def _cross_attn(
        self,
        query_src: torch.Tensor,   # (N, Sq, D) — queries
        kv_src: torch.Tensor,      # (N, Sk, D) — keys and values
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        out_proj: nn.Linear,
        norm_q: nn.LayerNorm,
        norm_kv: nn.LayerNorm,
    ) -> torch.Tensor:
        N, Sq, D = query_src.shape
        Sk = kv_src.shape[1]
        H, Dh = self.n_heads, self.head_dim

        q  = q_proj(norm_q(query_src)).reshape(N, Sq, H, Dh).transpose(1, 2)
        kv = kv_proj(norm_kv(kv_src)).reshape(N, Sk, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        a = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return out_proj(a.transpose(1, 2).contiguous().view(N, Sq, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, S, D) → (N, S, D)"""
        N = x.shape[0]
        I = self.inducing.expand(N, -1, -1)  # (N, m, D)

        # Step 1: inducing → input
        H = I + self.drop(self._cross_attn(
            I, x, self.q1, self.kv1, self.o1, self.norm_i, self.norm_x1
        ))

        # Step 2: input → inducing
        out = x + self.drop(self._cross_attn(
            x, H, self.q2, self.kv2, self.o2, self.norm_x2, self.norm_h
        ))

        # Feed-forward
        out = out + self.drop(self.ffn(self.norm_ffn(out)))
        return out


# Main model 

class PatchSetVelocityField(nn.Module):
    """
    Patch-token set-transformer velocity field  v_θ(z_t, t, context).

    Parameters
    ----------
    n_features : int
        Output dimension.  Must equal N_TARGET = 81.
    context_dim : int
        Flat context dimension (N_CONTEXT = 2673 = 81 + 1296 + 1296).
    d_model : int
        Transformer hidden dimension.  Must be divisible by 4 (for 2D
        sinusoidal encoding) and by n_heads.
    n_heads : int
        Number of attention heads per block.
    n_layers : int
        Number of SAB or ISAB blocks.
    n_inducing : int
        Number of inducing points per ISAB block.  Use 0 to disable ISAB
        and fall back to full SAB (global attention, O(N²)).
    time_emb_dim : int
        Width of the sinusoidal time embedding before the MLP projection.
    dropout : float
        Dropout rate applied inside attention blocks and FFNs.
    """

    def __init__(
        self,
        n_features: int = N_TARGET,
        context_dim: int = N_CONTEXT,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        n_inducing: int = 0,
        time_emb_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % 4 != 0:
            raise ValueError(
                f"d_model must be divisible by 4 for 2D sinusoidal encoding, "
                f"got {d_model}"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.n_features = n_features      # required by TargetCFM.generate
        self.context_dim = context_dim    # required by TargetCFM.generate
        self._n_inducing = n_inducing

        # Time embedding 
        T = time_emb_dim * 4
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, T),
            nn.SiLU(),
            nn.Linear(T, d_model),
        )

        # Token projection 
        # Raw features: 36 = 1(z_t) + 1(neutral) + 16(pu) + 16(lv) + 2(η,φ)
        self.token_proj = nn.Linear(_RAW_DIM, d_model)

        # Fixed buffers: sinusoidal 2D PE + explicit (η, φ) 
        # Sinusoidal 2D positional encoding: (1, 81, d_model)
        pe_2d = _make_sinusoidal_2d(_GRID, _GRID, d_model)  # (81, d_model)
        self.register_buffer("pos_enc", pe_2d.unsqueeze(0))  # (1, 81, d_model)

        # Explicit (η, φ) coordinates: (1, 81, 2), appended to raw features
        coords = _make_explicit_coords(_GRID, _GRID)          # (81, 2)
        self.register_buffer("coords", coords.unsqueeze(0))   # (1, 81, 2)

        # Attention layers 
        if n_inducing > 0:
            self.layers = nn.ModuleList([
                _ISAB(d_model, n_heads, n_inducing, dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                _SAB(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])

        # Output head 
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    # Context splitting 

    def _split_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split flat 2673-dim context into three image tensors.

        Returns
        -------
        neutral : (N, 1,  9,  9)
        ch_pu   : (N, 1, 36, 36)
        ch_lv   : (N, 1, 36, 36)
        """
        N = context.shape[0]
        neutral = context[:, :81].reshape(N, 1, _GRID, _GRID)
        ch_pu   = context[:, 81:1377].reshape(N, 1, CHARGED_SIDE, CHARGED_SIDE)
        ch_lv   = context[:, 1377:2673].reshape(N, 1, CHARGED_SIDE, CHARGED_SIDE)
        return neutral, ch_pu, ch_lv

    # Patch extraction 

    @staticmethod
    def _extract_patches(img: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Extract non-overlapping patches from a square image.

        Parameters
        ----------
        img : (N, 1, H, H)  where H = patch_size × grid_size
        patch_size : int    size of each square patch

        Returns
        -------
        (N, grid_size², patch_size²)   — one patch per token, flattened
        """
        # unfold: (N, 1, H, H) → (N, 1, n_patches_h, n_patches_w, p, p)
        patches = (
            img
            .unfold(2, patch_size, patch_size)   # (N, 1, G, W, p)
            .unfold(3, patch_size, patch_size)   # (N, 1, G, G, p, p)
        )
        N = img.shape[0]
        G = img.shape[2] // patch_size
        return patches.reshape(N, G * G, patch_size * patch_size)

    #  Forward 

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t       : (N,)     — flow time in [0, 1]
        z       : (N, 81)  — noisy target at time t
        context : (N, 2673) — flat standardised context

        Returns
        -------
        velocity : (N, 81)
        """
        N = z.shape[0]

        # 1. Split context into images 
        neutral, ch_pu, ch_lv = self._split_context(context)

        # 2. Per-token patch features 
        # z_t and neutral: already 81 scalars, one per token
        zt_px  = z.unsqueeze(-1)                           # (N, 81, 1)
        neut_px = neutral.reshape(N, _N_TOKENS, 1)         # (N, 81, 1)

        # 4×4 patches from each charged channel
        pu_patches = self._extract_patches(ch_pu, _PATCH)  # (N, 81, 16)
        lv_patches = self._extract_patches(ch_lv, _PATCH)  # (N, 81, 16)

        # Explicit (η, φ) coordinates broadcast over batch
        coords = self.coords.expand(N, -1, -1)             # (N, 81, 2)

        # Concatenate all raw features: (N, 81, 36)
        raw = torch.cat([zt_px, neut_px, pu_patches, lv_patches, coords], dim=-1)

        # 3. Project to d_model + sinusoidal 2D position encoding 
        tokens = self.token_proj(raw)                       # (N, 81, d_model)
        tokens = tokens + self.pos_enc                      # broadcast (1, 81, d_model)

        # 4. Time conditioning: add to every token 
        t_emb = self.time_proj(self.time_emb(t))           # (N, d_model)
        tokens = tokens + t_emb.unsqueeze(1)               # (N, 81, d_model)

        #  5. Attention layers 
        for layer in self.layers:
            tokens = layer(tokens)

        # 6. Output head 
        # (N, 81, d_model) → (N, 81, 1) → (N, 81)
        out = self.out_proj(self.out_norm(tokens))
        return out.squeeze(-1)  # (N, 81)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["PatchSetVelocityField"]
