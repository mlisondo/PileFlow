"""
flows/config.py
===============

Central configuration for the mixed-resolution image-only PileFlow pipeline.

Pipeline stages:
  Stage 1 - Load generated jet images from jets_pileup_images.npz
  Stage 2 - Train PileFlow using only three image channels
  Stage 3 - Generate the neutral-LV 9x9 image
  Stage 4 - Compare and plot predictions

PileFlow input channels:
  1. ch_neutral_all_raw - contaminated neutral image, 9x9
  2. ch_charged_pu      - charged pileup image, native 36x36
  3. ch_charged_lv      - charged leading-vertex image, native 36x36

PileFlow context:
  81 + 1296 + 1296 = 2673 features

PileFlow target:
  ch_neutral_lv - neutral leading-vertex image, 9x9

Evaluation sampling:
  During evaluation, PileFlow may generate multiple independent samples for
  each conditioning input. The samples are averaged pixel-by-pixel before
  computing images, jet observables, plots, and metrics.

  eval_samples = 1 preserves the original single-sample behavior.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # I/O
    outdir: str = "output"
    process_name: str = "ppjj"

    # Stage toggles
    skip_gen: bool = False
    skip_flow: bool = False
    skip_eval: bool = False

    # Precomputed paths
    data_npz: Optional[str] = None       # training data
    eval_npz: Optional[str] = None       # held-out evaluation data (if None, falls back to data_npz)
    eval_npy: Optional[str] = None       # optional .npy table for eval comparison plots
    flow_ckpt: Optional[str] = None
    pumml_ckpt: Optional[str] = None

    # Debugging / dataset limits
    max_jets: Optional[int] = None

    # PileFlow training hyperparameters
    flow_epochs: int = 800
    flow_batch: int = 512
    flow_lr: float = 1e-4
    flow_hidden: int = 512
    flow_blocks: int = 8
    flow_time_emb: int = 64
    flow_sigma_min: float = 1e-4
    flow_dropout: float = 0.1
    flow_patience: int = 120  # increased from 60

    # Minor improvements (pass --no-use-X to disable individually)
    use_log_pt: bool = False      # log(1+pT) normalization of target image (off: hurts mass/pT)
    use_minibatch_ot: bool = False # minibatch OT noise coupling (requires scipy)
    use_cnn_context: bool = False  # CNN context encoder (superseded by use_transformer)
    cnn_neutral_feat: int = 128    # CNN output features for 9x9 neutral channel
    cnn_charged_feat: int = 256    # CNN output features for each 36x36 charged channel
    use_ema: bool = True          # exponential moving average of weights
    ema_decay: float = 0.999      # EMA decay rate (only used when use_ema=True)
    pt_threshold: float = 0.0     # pixel pT threshold at decode in GeV; 0 = off

    # Patch-token set-transformer velocity field
    use_set_transformer: bool = False    # 81 patch tokens, global SAB / ISAB
    st_d_model: int = 256                # hidden dimension (must be div by 4 and n_heads)
    st_n_heads: int = 4                  # attention heads
    st_n_layers: int = 6                 # number of SAB or ISAB blocks
    st_n_inducing: int = 0               # inducing points per ISAB; 0 = use full SAB

    # U-Net velocity field (drastic change — convolutional encoder-decoder)
    use_unet: bool = False               # replace flat ResNet MLP with U-Net
    unet_base_channels: int = 64         # base channel count C (encoder: C→2C→4C, bottleneck: 8C)

    # Transformer velocity field (drastic change — cross-attention architecture)
    use_transformer: bool = False        # replace flat ResNet MLP with Transformer
    transformer_d_model: int = 256       # hidden dimension for all transformer layers
    transformer_n_heads: int = 4         # attention heads (must divide d_model)
    transformer_n_self_attn: int = 2     # self-attention layers over context tokens
    transformer_n_cross_attn: int = 4    # cross-attention layers (target × context)
    transformer_cnn_channels: int = 32   # CNN feature channels (all channels → 9×9)

    # Generation / evaluation
    eval_batch: int = 512
    ode_steps: int = 100

    # Number of independent conditional-flow samples generated per jet during
    # evaluation. The decoded images are averaged pixel-by-pixel before any
    # downstream observable or metric is computed.
    #
    # This setting is evaluation-only and does not affect training.
    eval_samples: int = 8  # increased from 1

    # Image-only model contract
    image_channel_keys: tuple[str, str, str] = (
        "ch_neutral_all_raw",
        "ch_charged_pu",
        "ch_charged_lv",
    )
    target_key: str = "ch_neutral_lv"

    # image_size is retained as the neutral and target grid size.
    image_size: int = 9
    charged_image_size: int = 36

    @property
    def neutral_dim(self) -> int:
        """Flattened neutral-image dimension: 9x9 = 81."""
        return self.image_size**2

    @property
    def charged_dim(self) -> int:
        """Flattened charged-image dimension: 36x36 = 1296."""
        return self.charged_image_size**2

    @property
    def n_image_channels(self) -> int:
        return len(self.image_channel_keys)

    @property
    def context_dim(self) -> int:
        """
        Mixed-resolution context dimension:

            neutral all:  9x9   = 81
            charged PU:  36x36  = 1296
            charged LV:  36x36  = 1296

            total = 2673
        """
        return self.neutral_dim + 2 * self.charged_dim

    @property
    def n_target(self) -> int:
        """One flattened neutral-LV 9x9 target image: 81."""
        return self.neutral_dim

    @property
    def n_scalars(self) -> int:
        return 0

    @property
    def channel_shapes(self) -> dict[str, tuple[int, int]]:
        """Expected spatial shape of each conditioning image."""
        return {
            "ch_neutral_all_raw": (
                self.image_size,
                self.image_size,
            ),
            "ch_charged_pu": (
                self.charged_image_size,
                self.charged_image_size,
            ),
            "ch_charged_lv": (
                self.charged_image_size,
                self.charged_image_size,
            ),
        }

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(
                f"image_size must be positive, got {self.image_size}"
            )

        if self.charged_image_size <= 0:
            raise ValueError(
                "charged_image_size must be positive, "
                f"got {self.charged_image_size}"
            )

        if len(self.image_channel_keys) != 3:
            raise ValueError(
                "PileFlow requires exactly three image channels, "
                f"got {len(self.image_channel_keys)}"
            )

        if self.flow_epochs <= 0:
            raise ValueError(
                f"flow_epochs must be positive, got {self.flow_epochs}"
            )

        if self.flow_batch <= 0:
            raise ValueError(
                f"flow_batch must be positive, got {self.flow_batch}"
            )

        if self.eval_batch <= 0:
            raise ValueError(
                f"eval_batch must be positive, got {self.eval_batch}"
            )

        if self.eval_samples <= 0:
            raise ValueError(
                "eval_samples must be a positive integer, "
                f"got {self.eval_samples}"
            )

        if self.flow_hidden <= 0:
            raise ValueError(
                f"flow_hidden must be positive, got {self.flow_hidden}"
            )

        if self.flow_blocks <= 0:
            raise ValueError(
                f"flow_blocks must be positive, got {self.flow_blocks}"
            )

        if self.flow_time_emb <= 0 or self.flow_time_emb % 2 != 0:
            raise ValueError(
                "flow_time_emb must be a positive even integer, "
                f"got {self.flow_time_emb}"
            )

        if not 0.0 <= self.flow_sigma_min < 1.0:
            raise ValueError(
                "flow_sigma_min must satisfy 0 <= flow_sigma_min < 1, "
                f"got {self.flow_sigma_min}"
            )

        if not 0.0 <= self.flow_dropout < 1.0:
            raise ValueError(
                "flow_dropout must satisfy 0 <= flow_dropout < 1, "
                f"got {self.flow_dropout}"
            )

        if self.cnn_neutral_feat <= 0:
            raise ValueError(
                f"cnn_neutral_feat must be positive, got {self.cnn_neutral_feat}"
            )

        if self.cnn_charged_feat <= 0:
            raise ValueError(
                f"cnn_charged_feat must be positive, got {self.cnn_charged_feat}"
            )

        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                "ema_decay must satisfy 0 <= ema_decay < 1, "
                f"got {self.ema_decay}"
            )

        if self.pt_threshold < 0.0:
            raise ValueError(
                f"pt_threshold must be nonnegative, got {self.pt_threshold}"
            )

        if self.ode_steps <= 0:
            raise ValueError(
                f"ode_steps must be positive, got {self.ode_steps}"
            )

        if self.transformer_d_model <= 0:
            raise ValueError(
                f"transformer_d_model must be positive, got {self.transformer_d_model}"
            )

        if self.transformer_n_heads <= 0:
            raise ValueError(
                f"transformer_n_heads must be positive, got {self.transformer_n_heads}"
            )

        if self.transformer_d_model % self.transformer_n_heads != 0:
            raise ValueError(
                f"transformer_d_model ({self.transformer_d_model}) must be divisible "
                f"by transformer_n_heads ({self.transformer_n_heads})"
            )

        if self.transformer_n_self_attn < 0:
            raise ValueError(
                f"transformer_n_self_attn must be nonnegative, got {self.transformer_n_self_attn}"
            )

        if self.transformer_n_cross_attn <= 0:
            raise ValueError(
                f"transformer_n_cross_attn must be positive, got {self.transformer_n_cross_attn}"
            )

        if self.transformer_cnn_channels < 2 or self.transformer_cnn_channels % 2 != 0:
            raise ValueError(
                "transformer_cnn_channels must be a positive even integer, "
                f"got {self.transformer_cnn_channels}"
            )

        if self.unet_base_channels <= 0:
            raise ValueError(
                f"unet_base_channels must be positive, got {self.unet_base_channels}"
            )

        if self.st_d_model % 4 != 0:
            raise ValueError(
                f"st_d_model must be divisible by 4, got {self.st_d_model}"
            )

        if self.st_d_model % self.st_n_heads != 0:
            raise ValueError(
                f"st_d_model ({self.st_d_model}) must be divisible by "
                f"st_n_heads ({self.st_n_heads})"
            )

        if self.st_n_layers <= 0:
            raise ValueError(
                f"st_n_layers must be positive, got {self.st_n_layers}"
            )

        if self.st_n_inducing < 0:
            raise ValueError(
                f"st_n_inducing must be nonnegative (0 = SAB), got {self.st_n_inducing}"
            )

        n_active = sum([self.use_set_transformer, self.use_unet, self.use_transformer])
        if n_active > 1:
            raise ValueError(
                "use_set_transformer, use_unet, and use_transformer are mutually exclusive; "
                "enable at most one."
            )

    # Generic
    device: str = "cpu"
    seed: int = 42