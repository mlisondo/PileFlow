"""
flows/models/pileflow.py
========================

PileFlow: pileup mitigation using Target Conditional Flow Matching.

The image-only model is conditioned exclusively on three jet-image channels:

    1. ch_neutral_all_raw : contaminated neutral pT image, 9x9
    2. ch_charged_pu      : charged pileup pT image, 36x36
    3. ch_charged_lv      : charged leading-vertex pT image, 36x36

The neutral image is flattened from 9x9 to 81 pixels.
Each charged image is flattened directly from 36x36 to 1296 pixels.

Raw context:
    [0:81]       ch_neutral_all_raw
    [81:1377]    ch_charged_pu
    [1377:2673]  ch_charged_lv

Learned same-resolution local context:
    [2673:2754]  neutral local CNN map
    [2754:4050]  charged-PU local CNN map
    [4050:5346]  charged-LV local CNN map

Total context dimension:
    5346

Target vector X:
    [0:81]       ch_neutral_lv

The flow therefore learns:

    v_theta(z_t, t, Y): R^81 x R x R^5346 -> R^81

References:
    Vaselli et al. arXiv:2402.13684v2
    Lipman et al. arXiv:2210.02747
    Komiske et al. arXiv:1707.08600
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# Mixed-resolution image dimensionalities
NEUTRAL_SIDE = 9
CHARGED_SIDE = 36
NEUTRAL_DIM = NEUTRAL_SIDE * NEUTRAL_SIDE
CHARGED_DIM = CHARGED_SIDE * CHARGED_SIDE

# Backward-compatible alias for code that uses IMG_DIM as the target dimension.
IMG_DIM = NEUTRAL_DIM

N_IMAGES = 3
N_SCALARS = 0
N_TARGET = NEUTRAL_DIM

# Original flattened image context:
#
#   neutral all :  9x9   = 81
#   charged PU  : 36x36  = 1296
#   charged LV  : 36x36  = 1296
#
#   total raw context = 2673
RAW_CONTEXT_DIM = NEUTRAL_DIM + 2 * CHARGED_DIM

# Each CNN branch returns one same-resolution learned local feature per
# original image pixel:

#   total local context = 2673
LOCAL_CONTEXT_DIM = NEUTRAL_DIM + 2 * CHARGED_DIM

# Raw pixels plus learned same-resolution local descriptors.
N_CONTEXT = RAW_CONTEXT_DIM + LOCAL_CONTEXT_DIM


class SinusoidalTimeEmb(nn.Module):
    """Fixed sinusoidal time embedding, as used in DDPM and FlowSim."""

    def __init__(self, dim: int):
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"time embedding dimension must be a positive even integer, got {dim}"
            )

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2

        freqs = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """
    Pre-normalized residual block with repeated time and image conditioning.

    The update is:

        h_normalized = LayerNorm(h)

        residual = Dropout(
            fc2(
                SiLU(
                    fc1([h_normalized, condition])
                )
            )
        )

        h_new = h + residual

    LayerNorm is applied only to the input of the residual transformation.
    The identity path remains unchanged, allowing the hidden-state magnitude
    to carry information through the network.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

        if cond_dim <= 0:
            raise ValueError(
                f"cond_dim must be positive, got {cond_dim}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1, "
                f"got {dropout}"
            )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.fc1 = nn.Linear(
            hidden_dim + cond_dim,
            hidden_dim,
        )

        self.act = nn.SiLU()

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one pre-normalized residual update.

        Parameters
        ----------
        h:
            Hidden state with shape ``(N, hidden_dim)``.

        cond:
            Concatenated time and image context with shape
            ``(N, cond_dim)``.

        Returns
        -------
        torch.Tensor
            Updated hidden state with the same shape as ``h``.
        """
        if h.ndim != 2:
            raise ValueError(
                "Expected h to have shape (N, hidden_dim), "
                f"got {tuple(h.shape)}"
            )

        if cond.ndim != 2:
            raise ValueError(
                "Expected cond to have shape (N, cond_dim), "
                f"got {tuple(cond.shape)}"
            )

        if h.shape[0] != cond.shape[0]:
            raise ValueError(
                "Hidden state and condition have different batch sizes: "
                f"{h.shape[0]} and {cond.shape[0]}"
            )

        # Normalize only the input to the learned residual transformation.
        # The original h remains untouched along the identity connection.
        h_normalized = self.norm(
            h
        )

        residual = self.fc1(
            torch.cat(
                [
                    h_normalized,
                    cond,
                ],
                dim=-1,
            )
        )

        residual = self.act(
            residual
        )

        residual = self.fc2(
            residual
        )

        residual = self.dropout(
            residual
        )

        output = h + residual

        if output.shape != h.shape:
            raise RuntimeError(
                "Residual block changed the hidden-state shape: "
                f"input={tuple(h.shape)}, "
                f"output={tuple(output.shape)}"
            )

        if not torch.isfinite(output).all():
            raise RuntimeError(
                "Residual block produced non-finite values."
            )

        return output

class CRTVelocityField(nn.Module):
    """
    Continuous ResNet Target velocity field:

        v_theta(z_t, t, Y)

    Parameters
    ----------
    n_features:
        Flow-state and target dimensionality. PileFlow uses 81.

    context_dim:
        Conditioning-vector dimensionality. Local-context PileFlow uses 5346:
        2673 original flattened pixels plus 2673 same-resolution learned
        local CNN features.

    hidden_dim:
        Width of the residual network.

    n_blocks:
        Number of residual blocks.

    time_emb_dim:
        Dimension of the sinusoidal time embedding.

    dropout:
        Dropout probability inside residual blocks.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 8,
        time_emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")

        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks}")

        self.n_features = n_features
        self.context_dim = context_dim

        self.time_emb = SinusoidalTimeEmb(time_emb_dim)
        cond_dim = time_emb_dim + context_dim

        self.input_proj = nn.Linear(n_features, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the flow velocity.

        Shapes
        ------
        t:
            (N,)

        z:
            (N, n_features)

        context:
            (N, context_dim)

        returns:
            (N, n_features)
        """
        if z.ndim != 2 or z.shape[1] != self.n_features:
            raise ValueError(
                f"Expected z shape (N, {self.n_features}), got {tuple(z.shape)}"
            )

        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(
                f"Expected context shape (N, {self.context_dim}), "
                f"got {tuple(context.shape)}"
            )

        if context.shape[0] != z.shape[0]:
            raise ValueError(
                "Flow state and context have different batch sizes: "
                f"{z.shape[0]} and {context.shape[0]}"
            )

        if t.ndim != 1 or t.shape[0] != z.shape[0]:
            raise ValueError(
                f"Expected t shape ({z.shape[0]},), got {tuple(t.shape)}"
            )

        t_emb = self.time_emb(t)
        cond = torch.cat([t_emb, context], dim=-1)
        h = self.input_proj(z)

        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(h)

    def count_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class TargetCFM(nn.Module):
    """
    Target Conditional Flow Matching.

    Interpolation path:

        z_t = t*x1 + [1 - (1 - sigma_min)*t]*x0

    Analytical target velocity:

        u_t = x1 - (1 - sigma_min)*x0
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()

        if not 0.0 <= sigma_min < 1.0:
            raise ValueError(
                f"sigma_min must satisfy 0 <= sigma_min < 1, got {sigma_min}"
            )

        self.sigma_min = sigma_min

    def sample_training_pair(
        self,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a flow-matching training tuple.

        Parameters
        ----------
        x1:
            Standardized target image with shape (N, 81).

        Returns
        -------
        t:
            Integration times, shape (N,).

        z_t:
            Interpolated flow states, shape (N, 81).

        u_t:
            Target velocities, shape (N, 81).
        """
        if x1.ndim != 2:
            raise ValueError(
                f"Expected x1 to have shape (N, D), got {tuple(x1.shape)}"
            )

        n = x1.shape[0]
        t = torch.rand(n, device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1)
        scale = 1.0 - (1.0 - self.sigma_min) * t[:, None]
        z_t = t[:, None] * x1 + scale * x0
        u_t = x1 - (1.0 - self.sigma_min) * x0
        return t, z_t, u_t

    @torch.no_grad()
    def generate(
        self,
        model: CRTVelocityField,
        context: torch.Tensor,
        n_steps: int = 100,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """
        Integrate the flow ODE using Euler steps.

        Starts from:

            z_0 ~ Normal(0, I)

        and returns:

            z_1 with shape (N, 81)
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        if context.ndim != 2 or context.shape[1] != model.context_dim:
            raise ValueError(
                f"Expected context shape (N, {model.context_dim}), "
                f"got {tuple(context.shape)}"
            )

        model.eval()
        n = context.shape[0]

        z = torch.randn(
            n,
            model.n_features,
            device=device,
            dtype=context.dtype,
        )

        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_batch = torch.full(
                (n,),
                i * dt,
                device=device,
                dtype=context.dtype,
            )
            z = z + model(t_batch, z, context) * dt

        return z

class LocalImageEncoder(nn.Module):
    """
    Extract one learned local descriptor at every input pixel.

    The encoder preserves the original spatial resolution:

        (N, 1, H, W) -> (N, 1, H, W)

    There is no pooling, striding, interpolation, or downsampling.

    The two 3x3 convolutions allow each output pixel to use information from
    a local neighborhood. The final 1x1 convolution compresses the hidden
    channels into one learned local feature per original pixel.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        if hidden_channels <= 0:
            raise ValueError(
                "hidden_channels must be positive, "
                f"got {hidden_channels}"
            )

        if not 0.0 < negative_slope < 1.0:
            raise ValueError(
                "negative_slope must lie in (0, 1), "
                f"got {negative_slope}"
            )

        self.hidden_channels = hidden_channels
        self.negative_slope = negative_slope

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=False,
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=False,
            ),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one image channel without changing its spatial resolution.

        Parameters
        ----------
        image:
            Tensor with shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Learned local feature map with shape ``(N, 1, H, W)``.
        """
        if image.ndim != 4:
            raise ValueError(
                "Expected image to have four dimensions "
                "(N, 1, H, W), "
                f"got {tuple(image.shape)}"
            )

        if image.shape[1] != 1:
            raise ValueError(
                "Expected exactly one input image channel, "
                f"got shape {tuple(image.shape)}"
            )

        output = self.network(image)

        if output.shape != image.shape:
            raise RuntimeError(
                "LocalImageEncoder changed the image shape: "
                f"input={tuple(image.shape)}, "
                f"output={tuple(output.shape)}"
            )

        return output


class ContextEncoder(nn.Module):
    """
    Build the raw-plus-local mixed-resolution context vector.

    Inputs
    ------
    ch_neutral_all:
        Contaminated neutral image, flattened to shape ``(N, 81)``.

    ch_charged_pu:
        Charged-pileup image, flattened to shape ``(N, 1296)``.

    ch_charged_lv:
        Charged leading-vertex image, flattened to shape ``(N, 1296)``.

    Processing
    ----------
    1. Fit and apply per-pixel standardization.
    2. Preserve the original standardized flattened pixels.
    3. Reshape each channel to its native image grid.
    4. Apply a separate same-resolution CNN to each channel.
    5. Flatten the three local CNN maps.
    6. Concatenate raw and local features.

    Context layout
    --------------
    Raw context:

        neutral raw       : 81
        charged-PU raw    : 1296
        charged-LV raw    : 1296

    Local CNN context:

        neutral local     : 81
        charged-PU local  : 1296
        charged-LV local  : 1296

    Total:

        2673 raw + 2673 local = 5346

    Call ``fit`` using training rows only before training.
    """

    def __init__(
        self,
        neutral_dim: int = NEUTRAL_DIM,
        charged_dim: int = CHARGED_DIM,
        local_hidden_channels: int = 8,
        local_negative_slope: float = 0.1,
    ):
        super().__init__()

        if neutral_dim <= 0:
            raise ValueError(
                f"neutral_dim must be positive, got {neutral_dim}"
            )

        if charged_dim <= 0:
            raise ValueError(
                f"charged_dim must be positive, got {charged_dim}"
            )

        neutral_side = math.isqrt(neutral_dim)
        charged_side = math.isqrt(charged_dim)

        if neutral_side * neutral_side != neutral_dim:
            raise ValueError(
                "neutral_dim must describe a square image, "
                f"got {neutral_dim}"
            )

        if charged_side * charged_side != charged_dim:
            raise ValueError(
                "charged_dim must describe a square image, "
                f"got {charged_dim}"
            )

        if local_hidden_channels <= 0:
            raise ValueError(
                "local_hidden_channels must be positive, "
                f"got {local_hidden_channels}"
            )

        if not 0.0 < local_negative_slope < 1.0:
            raise ValueError(
                "local_negative_slope must lie in (0, 1), "
                f"got {local_negative_slope}"
            )

        self.neutral_dim = neutral_dim
        self.charged_dim = charged_dim

        self.neutral_side = neutral_side
        self.charged_side = charged_side

        self.local_hidden_channels = local_hidden_channels
        self.local_negative_slope = local_negative_slope

        self.raw_context_dim = neutral_dim + 2 * charged_dim
        self.local_context_dim = neutral_dim + 2 * charged_dim
        self.context_dim = (
            self.raw_context_dim
            + self.local_context_dim
        )

        # Separate CNN weights are used because the three channels have
        # physically different meanings.
        self.neutral_local_encoder = LocalImageEncoder(
            hidden_channels=local_hidden_channels,
            negative_slope=local_negative_slope,
        )

        self.charged_pu_local_encoder = LocalImageEncoder(
            hidden_channels=local_hidden_channels,
            negative_slope=local_negative_slope,
        )

        self.charged_lv_local_encoder = LocalImageEncoder(
            hidden_channels=local_hidden_channels,
            negative_slope=local_negative_slope,
        )

        self.register_buffer(
            "neutral_all_mean",
            torch.zeros(neutral_dim),
        )
        self.register_buffer(
            "neutral_all_std",
            torch.ones(neutral_dim),
        )

        self.register_buffer(
            "charged_pu_mean",
            torch.zeros(charged_dim),
        )
        self.register_buffer(
            "charged_pu_std",
            torch.ones(charged_dim),
        )

        self.register_buffer(
            "charged_lv_mean",
            torch.zeros(charged_dim),
        )
        self.register_buffer(
            "charged_lv_std",
            torch.ones(charged_dim),
        )

    @staticmethod
    def _validate_image(
        image: torch.Tensor,
        name: str,
        expected_dim: int,
    ) -> None:
        if image.ndim != 2 or image.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {name} shape (N, {expected_dim}), "
                f"got {tuple(image.shape)}"
            )

    @staticmethod
    def _stats(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = tensor.mean(dim=0)

        std = tensor.std(
            dim=0,
            unbiased=False,
        ).clamp(min=1.0e-6)

        return mean, std

    @staticmethod
    def _reshape_flat_image(
        image: torch.Tensor,
        side: int,
        name: str,
    ) -> torch.Tensor:
        """
        Reshape ``(N, side*side)`` into ``(N, 1, side, side)``.
        """
        expected_dim = side * side

        if image.ndim != 2 or image.shape[1] != expected_dim:
            raise ValueError(
                f"Expected {name} shape (N, {expected_dim}), "
                f"got {tuple(image.shape)}"
            )

        return image.reshape(
            image.shape[0],
            1,
            side,
            side,
        )

    @torch.no_grad()
    def fit(
        self,
        ch_neutral_all: torch.Tensor,
        ch_charged_pu: torch.Tensor,
        ch_charged_lv: torch.Tensor,
    ) -> None:
        """
        Fit per-pixel normalization statistics using training rows only.

        Only the deterministic normalization buffers are fitted here.
        The CNN parameters are learned through gradient descent during
        PileFlow training.
        """
        self._validate_image(
            ch_neutral_all,
            "ch_neutral_all",
            self.neutral_dim,
        )

        self._validate_image(
            ch_charged_pu,
            "ch_charged_pu",
            self.charged_dim,
        )

        self._validate_image(
            ch_charged_lv,
            "ch_charged_lv",
            self.charged_dim,
        )

        if not (
            ch_neutral_all.shape[0]
            == ch_charged_pu.shape[0]
            == ch_charged_lv.shape[0]
        ):
            raise ValueError(
                "Context image channels have different row counts."
            )

        neutral_mean, neutral_std = self._stats(
            ch_neutral_all
        )

        charged_pu_mean, charged_pu_std = self._stats(
            ch_charged_pu
        )

        charged_lv_mean, charged_lv_std = self._stats(
            ch_charged_lv
        )

        self.neutral_all_mean.copy_(
            neutral_mean
        )
        self.neutral_all_std.copy_(
            neutral_std
        )

        self.charged_pu_mean.copy_(
            charged_pu_mean
        )
        self.charged_pu_std.copy_(
            charged_pu_std
        )

        self.charged_lv_mean.copy_(
            charged_lv_mean
        )
        self.charged_lv_std.copy_(
            charged_lv_std
        )

    def forward(
        self,
        ch_neutral_all: torch.Tensor,
        ch_charged_pu: torch.Tensor,
        ch_charged_lv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the standardized raw-plus-local context vector.

        Returns
        -------
        torch.Tensor
            Context with shape ``(N, 5346)`` for the default image sizes.
        """
        self._validate_image(
            ch_neutral_all,
            "ch_neutral_all",
            self.neutral_dim,
        )

        self._validate_image(
            ch_charged_pu,
            "ch_charged_pu",
            self.charged_dim,
        )

        self._validate_image(
            ch_charged_lv,
            "ch_charged_lv",
            self.charged_dim,
        )

        if not (
            ch_neutral_all.shape[0]
            == ch_charged_pu.shape[0]
            == ch_charged_lv.shape[0]
        ):
            raise ValueError(
                "Context image channels have different row counts."
            )

        # Standardized original flattened pixels.
        neutral_all_z = (
            ch_neutral_all
            - self.neutral_all_mean
        ) / self.neutral_all_std

        charged_pu_z = (
            ch_charged_pu
            - self.charged_pu_mean
        ) / self.charged_pu_std

        charged_lv_z = (
            ch_charged_lv
            - self.charged_lv_mean
        ) / self.charged_lv_std

        # Restore native spatial layouts solely for the local CNN branches.
        # No pixels are discarded or downsampled.
        neutral_image = self._reshape_flat_image(
            neutral_all_z,
            side=self.neutral_side,
            name="standardized neutral image",
        )

        charged_pu_image = self._reshape_flat_image(
            charged_pu_z,
            side=self.charged_side,
            name="standardized charged-PU image",
        )

        charged_lv_image = self._reshape_flat_image(
            charged_lv_z,
            side=self.charged_side,
            name="standardized charged-LV image",
        )

        # Same-resolution learned local descriptors.
        neutral_local = self.neutral_local_encoder(
            neutral_image
        ).flatten(start_dim=1)

        charged_pu_local = self.charged_pu_local_encoder(
            charged_pu_image
        ).flatten(start_dim=1)

        charged_lv_local = self.charged_lv_local_encoder(
            charged_lv_image
        ).flatten(start_dim=1)

        context = torch.cat(
            [
                # Original information.
                neutral_all_z,
                charged_pu_z,
                charged_lv_z,

                # Added explicit local information.
                neutral_local,
                charged_pu_local,
                charged_lv_local,
            ],
            dim=-1,
        )

        if context.shape[1] != self.context_dim:
            raise RuntimeError(
                "Encoded context has the wrong dimension: "
                f"expected {self.context_dim}, "
                f"got {context.shape[1]}"
            )

        if not torch.isfinite(context).all():
            raise RuntimeError(
                "Encoded context contains non-finite values."
            )

        return context

    def count_parameters(self) -> int:
        """
        Return the number of trainable local-CNN parameters.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class TargetPreprocessor(nn.Module):
    """
    Standardize and decode the 81-dimensional neutral-LV target image.

    Target:
        ch_neutral_lv, flattened from 9x9 to 81 pixels.

    Call ``fit`` using training rows only before training.

    Decoding restores physical units and clamps negative transverse momentum
    to zero. An optional pixel threshold may be supplied explicitly, but no
    threshold is applied by default.
    """

    def __init__(
        self,
        n_img: int = N_TARGET,
    ):
        super().__init__()

        if n_img <= 0:
            raise ValueError(f"n_img must be positive, got {n_img}")

        self.n_img = n_img

        self.register_buffer(
            "img_mean",
            torch.zeros(n_img),
        )
        self.register_buffer(
            "img_std",
            torch.ones(n_img),
        )

    def _validate_image(
        self,
        image: torch.Tensor,
        name: str,
    ) -> None:
        if image.ndim != 2 or image.shape[1] != self.n_img:
            raise ValueError(
                f"Expected {name} shape (N, {self.n_img}), "
                f"got {tuple(image.shape)}"
            )

    @torch.no_grad()
    def fit(
        self,
        neutral_lv: torch.Tensor,
    ) -> None:
        """Fit per-pixel target statistics using training rows only."""
        self._validate_image(neutral_lv, "neutral_lv")

        self.img_mean = neutral_lv.mean(dim=0)
        self.img_std = neutral_lv.std(
            dim=0,
            unbiased=False,
        ).clamp(min=1e-6)

    def encode(
        self,
        neutral_lv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standardize the target image.

        Parameters
        ----------
        neutral_lv:
            Physical neutral-LV pT image with shape (N, 81).

        Returns
        -------
        torch.Tensor
            Standardized target image with shape (N, 81).
        """
        self._validate_image(neutral_lv, "neutral_lv")
        return (neutral_lv - self.img_mean) / self.img_std

    def decode(
        self,
        z: torch.Tensor,
        pt_threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Decode a generated standardized image.

        Parameters
        ----------
        z:
            Flow output in standardized space with shape (N, 81).

        pt_threshold:
            Optional nonnegative pixel-pT threshold in GeV.

            None or 0.0:
                Return the raw nonnegative decoded image.

            Positive value:
                Set decoded pixels below this threshold to zero.

        Returns
        -------
        torch.Tensor
            Predicted neutral-LV pT image with shape (N, 81).
        """
        self._validate_image(z, "flow output")

        if pt_threshold is not None and pt_threshold < 0.0:
            raise ValueError(
                "pt_threshold must be nonnegative or None, "
                f"got {pt_threshold}"
            )

        image = z * self.img_std + self.img_mean
        image = image.clamp(min=0.0)

        if pt_threshold is not None and pt_threshold > 0.0:
            image = torch.where(
                image >= pt_threshold,
                image,
                torch.zeros_like(image),
            )

        return image


__all__ = [
    "NEUTRAL_SIDE",
    "CHARGED_SIDE",
    "NEUTRAL_DIM",
    "CHARGED_DIM",
    "IMG_DIM",
    "N_IMAGES",
    "N_SCALARS",
    "N_TARGET",
    "RAW_CONTEXT_DIM",
    "LOCAL_CONTEXT_DIM",
    "N_CONTEXT",
    "SinusoidalTimeEmb",
    "ResBlock",
    "CRTVelocityField",
    "TargetCFM",
    "LocalImageEncoder",
    "ContextEncoder",
    "TargetPreprocessor",
]