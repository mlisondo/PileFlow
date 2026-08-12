"""
flows/models/pileflow_attention.py
===================================

Transformer-based velocity field for PileFlow.

Replaces the flat residual MLP of CRTVelocityField with a cross-attention
architecture that explicitly models spatial relationships between input image
pixels and output target pixels.

Architecture
------------
Stage 1  CNN feature extraction (per channel → 9×9 feature maps)
         Neutral  9×9  input: three 3×3 conv layers, no stride.
         Charged 36×36 input: two stride-2 convs → 9×9, then one 3×3.

Stage 2  Context tokens: flatten + channel-type embedding + 2D position
         encoding + self-attention (captures cross-channel correlations).
         Total context tokens: 81 × 3 channels = 243.

Stage 3  Target tokens: z_t pixels + 2D position encoding + time embedding.
         81 tokens (one per output pixel).

Stage 4  Cross-attention: each target pixel (query) attends to all 243
         context tokens (key/value). Repeated for n_cross_attn layers.

Stage 5  Per-token MLP + LayerNorm → velocity scalar per pixel → (N, 81).

Implementation note
-------------------
Attention blocks use torch.nn.functional.scaled_dot_product_attention
(introduced in PyTorch 2.0) instead of nn.MultiheadAttention.  On Apple MPS
this activates the Metal Performance Shaders kernel and avoids the slow
batch_first transpose path, giving 3-5× faster per-epoch time.

Context split (from the flat 2673-dim standardised context vector)
-------------------------------------------------------------------
  neutral_all : context[    0:   81] → (N,  9,  9)
  charged_pu  : context[   81: 1377] → (N, 36, 36)
  charged_lv  : context[ 1377: 2673] → (N, 36, 36)

References
----------
Vaselli et al.  arXiv:2402.13684v2  (FlowSim / flow matching)
Lipman et al.   arXiv:2210.02747   (Conditional Flow Matching)
Komiske et al.  arXiv:1707.08600   (PUMML / pileup mitigation)
Lee et al.      arXiv:1810.00825   (Set Transformer)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pileflow import (
    CHARGED_DIM,
    CHARGED_SIDE,
    N_CONTEXT,
    N_TARGET,
    NEUTRAL_DIM,
    NEUTRAL_SIDE,
    SinusoidalTimeEmb,
)

_GRID_SIZE = NEUTRAL_SIDE            # 9  — all channels mapped to 9×9
_N_TOKENS = _GRID_SIZE * _GRID_SIZE  # 81 tokens per channel
_N_CHANNELS = 3                      # neutral-all, charged-PU, charged-LV
_N_CTX_TOKENS = _N_TOKENS * _N_CHANNELS  # 243 context tokens total


# ─────────────────────────────────────────────────────────────────────────────
# Position encoding
# ─────────────────────────────────────────────────────────────────────────────

class _LearnablePos2D(nn.Module):
    """Learnable 2-D position embedding for an H×W pixel grid."""

    def __init__(self, n_pixels: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_pixels, d_model)
        self.register_buffer("ids", torch.arange(n_pixels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, n_pixels, d_model) — adds position embedding in-place."""
        return x + self.emb(self.ids).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Attention blocks — uses F.scaled_dot_product_attention for MPS speed
# ─────────────────────────────────────────────────────────────────────────────

class _SelfAttnBlock(nn.Module):
    """
    Pre-LayerNorm self-attention block.

    Uses F.scaled_dot_product_attention (PyTorch ≥ 2.0) which activates
    Metal Performance Shaders on Apple MPS for significantly faster execution.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) % n_heads ({n_heads}) != 0")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(d_model)
        # fused QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

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

        # Pre-norm → fused QKV → split heads
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, S, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                 # each: (N, H, S, Dh)

        # Flash / MPS-accelerated attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, S, D)
        attn_out = self.out_proj(attn_out)

        x = x + self.drop(attn_out)
        return x + self.drop(self.ffn(self.norm2(x)))


class _CrossAttnBlock(nn.Module):
    """
    Pre-LayerNorm cross-attention block.

    The query tensor (target pixels) attends to the context tensor
    (context tokens from all three input image channels).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) % n_heads ({n_heads}) != 0")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        query   : (N, Sq, D)   — target pixel tokens
        context : (N, Sk, D)   — context tokens (243)
        returns : (N, Sq, D)
        """
        N, Sq, D = query.shape
        Sk = context.shape[1]
        H, Dh = self.n_heads, self.head_dim

        # Queries from target tokens
        q = self.q_proj(self.norm_q(query))           # (N, Sq, D)
        q = q.reshape(N, Sq, H, Dh).transpose(1, 2)  # (N, H, Sq, Dh)

        # Keys + values from context tokens (fused projection)
        kv = self.kv_proj(self.norm_kv(context))      # (N, Sk, 2D)
        kv = kv.reshape(N, Sk, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)                           # each: (N, H, Sk, Dh)

        # Flash / MPS-accelerated cross-attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, Sq, D)
        attn_out = self.out_proj(attn_out)

        query = query + self.drop(attn_out)
        return query + self.drop(self.ffn(self.norm2(query)))


# ─────────────────────────────────────────────────────────────────────────────
# CNN helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_channel_cnn(
    in_side: int,
    out_side: int,
    cnn_channels: int,
) -> nn.Sequential:
    """
    Build a small CNN that maps (N, 1, in_side, in_side) →
    (N, cnn_channels, out_side, out_side).

    Supported:
        in_side == out_side           three 3×3 convs, no stride.
        in_side == 4 × out_side       two stride-2 convs, then one 3×3.
    """
    C = cnn_channels
    if in_side == out_side:
        return nn.Sequential(
            nn.Conv2d(1, C // 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C // 2, C, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1),       nn.SiLU(),
        )
    elif in_side == 4 * out_side:
        return nn.Sequential(
            nn.Conv2d(1, C // 2, 3, padding=1, stride=2), nn.SiLU(),  # ÷2
            nn.Conv2d(C // 2, C, 3, padding=1, stride=2), nn.SiLU(),  # ÷2
            nn.Conv2d(C, C, 3, padding=1),                nn.SiLU(),
        )
    else:
        raise ValueError(
            f"Unsupported CNN sizes: {in_side} → {out_side}. "
            "Expected same size or 4× downsampling."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TransformerVelocityField
# ─────────────────────────────────────────────────────────────────────────────

class TransformerVelocityField(nn.Module):
    """
    Cross-attention velocity field v_θ(z_t, t, context).

    Drop-in replacement for CRTVelocityField.  The interface
    (t, z, context) and output shape (N, n_features) are identical.

    Parameters
    ----------
    n_features : int
        Flow-state / target dimensionality (81 for 9×9 image).
    context_dim : int
        Flat context input dimension (2673 for mixed-resolution PileFlow).
    cnn_channels : int
        Feature channels produced by the per-channel CNN before tokenisation.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.  Must evenly divide d_model.
    n_self_attn : int
        Self-attention layers applied to the 243 context tokens.
    n_cross_attn : int
        Cross-attention layers (81 target queries × 243 context keys).
    time_emb_dim : int
        Dimension of the sinusoidal time embedding.
    dropout : float
        Dropout probability in attention and FFN layers.
    """

    def __init__(
        self,
        n_features: int = N_TARGET,
        context_dim: int = N_CONTEXT,
        cnn_channels: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        n_self_attn: int = 2,
        n_cross_attn: int = 4,
        time_emb_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_features != N_TARGET:
            raise ValueError(
                f"TransformerVelocityField requires n_features={N_TARGET}, got {n_features}"
            )
        if context_dim != N_CONTEXT:
            raise ValueError(
                f"TransformerVelocityField requires context_dim={N_CONTEXT}, got {context_dim}"
            )
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        if cnn_channels < 2 or cnn_channels % 2 != 0:
            raise ValueError(
                f"cnn_channels must be a positive even integer, got {cnn_channels}"
            )

        self.n_features = n_features
        self.context_dim = context_dim
        self.d_model = d_model
        self.cnn_channels = cnn_channels

        # ── Time embedding ──────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Per-channel CNN (all channels → 9×9 feature maps) 
        self.neutral_cnn = _make_channel_cnn(NEUTRAL_SIDE, _GRID_SIZE, cnn_channels)
        self.charged_pu_cnn = _make_channel_cnn(CHARGED_SIDE, _GRID_SIZE, cnn_channels)
        self.charged_lv_cnn = _make_channel_cnn(CHARGED_SIDE, _GRID_SIZE, cnn_channels)

        # Context token embedding
        self.ctx_proj = nn.Linear(cnn_channels, d_model)
        # Channel-type embedding distinguishes neutral / PU / LV tokens
        self.channel_emb = nn.Embedding(_N_CHANNELS, d_model)
        # Shared 2-D learnable position encoding (81 positions)
        self.ctx_pos = _LearnablePos2D(_N_TOKENS, d_model)

        # Context self-attention 
        self.ctx_self_attn = nn.ModuleList(
            [_SelfAttnBlock(d_model, n_heads, dropout) for _ in range(n_self_attn)]
        )

        # Target token embedding 
        # Project the flow-state scalar z_t per pixel to d_model
        self.tgt_proj = nn.Linear(1, d_model)
        # Separate 2-D learnable position encoding for target pixels
        self.tgt_pos = _LearnablePos2D(_N_TOKENS, d_model)

        # Cross-attention layers 
        self.cross_attn = nn.ModuleList(
            [_CrossAttnBlock(d_model, n_heads, dropout) for _ in range(n_cross_attn)]
        )

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    # Helpers

    def _split_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split 2673-dim context → three 2-D image tensors."""
        n = context.shape[0]
        neutral = context[:, :NEUTRAL_DIM].view(n, 1, NEUTRAL_SIDE, NEUTRAL_SIDE)
        pu = context[:, NEUTRAL_DIM : NEUTRAL_DIM + CHARGED_DIM].view(
            n, 1, CHARGED_SIDE, CHARGED_SIDE
        )
        lv = context[:, NEUTRAL_DIM + CHARGED_DIM :].view(
            n, 1, CHARGED_SIDE, CHARGED_SIDE
        )
        return neutral, pu, lv

    # Forward 

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t       : (N,)
        z       : (N, 81)
        context : (N, 2673)

        Returns
        -------
        velocity : (N, 81)
        """
        # Time embedding 
        t_emb = self.time_proj(self.time_emb(t))          # (N, d_model)

        # Stage 1: CNN feature maps 
        neutral_img, pu_img, lv_img = self._split_context(context)

        neutral_feat = self.neutral_cnn(neutral_img)       # (N, C, 9, 9)
        pu_feat = self.charged_pu_cnn(pu_img)              # (N, C, 9, 9)
        lv_feat = self.charged_lv_cnn(lv_img)             # (N, C, 9, 9)

        # Stage 2: Context tokens 
        # (N, C, 9, 9) → (N, 81, C) → project → (N, 81, d_model)
        def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
            return self.ctx_proj(feat.flatten(2).transpose(1, 2))

        neutral_tok = _to_tokens(neutral_feat)
        pu_tok = _to_tokens(pu_feat)
        lv_tok = _to_tokens(lv_feat)

        # Add channel-type embeddings
        ch_embs = self.channel_emb.weight                  # (3, d_model)
        neutral_tok = neutral_tok + ch_embs[0]
        pu_tok      = pu_tok      + ch_embs[1]
        lv_tok      = lv_tok      + ch_embs[2]

        # Add shared 2-D position encoding
        neutral_tok = self.ctx_pos(neutral_tok)
        pu_tok      = self.ctx_pos(pu_tok)
        lv_tok      = self.ctx_pos(lv_tok)

        # Concatenate → (N, 243, d_model)
        ctx_tokens = torch.cat([neutral_tok, pu_tok, lv_tok], dim=1)

        # Stage 3: Context self-attention 
        for layer in self.ctx_self_attn:
            ctx_tokens = layer(ctx_tokens)

        #  Stage 4: Target tokens 
        # z: (N, 81) → (N, 81, 1) → (N, 81, d_model)
        tgt_tokens = self.tgt_proj(z.unsqueeze(-1))
        tgt_tokens = self.tgt_pos(tgt_tokens)
        tgt_tokens = tgt_tokens + t_emb.unsqueeze(1)      # broadcast over seq

        #  Stage 5: Cross-attention 
        for layer in self.cross_attn:
            tgt_tokens = layer(tgt_tokens, ctx_tokens)

        #  Stage 6: Output 
        # (N, 81, d_model) → (N, 81, 1) → (N, 81)
        velocity = self.output_proj(
            self.output_norm(tgt_tokens)
        ).squeeze(-1)

        return velocity

    def count_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["TransformerVelocityField"]
