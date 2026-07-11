"""
flows/config.py
===============

Central configuration for the image-only PileFlow pipeline.

Pipeline stages:
  Stage 1 — Load generated jet images from jets_pileup_images.npz
  Stage 2 — Train PileFlow using only three image channels
  Stage 3 — Generate the neutral-LV 9x9 image
  Stage 4 — Compare and plot predictions

PileFlow input channels:
  1. ch_neutral_all_raw  — contaminated neutral image, 9x9
  2. ch_charged_pu       — charged pileup image, pooled 36x36 -> 9x9
  3. ch_charged_lv       — charged leading-vertex image, pooled 36x36 -> 9x9

PileFlow target:
  ch_neutral_lv          — neutral leading-vertex image, 9x9
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
    #
    # Image-only PileFlow no longer requires the generator scalar .npy table.
    data_npz: Optional[str] = None
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
    flow_patience: int = 60

    # Generation / evaluation
    eval_batch: int = 512

    # Image-only model contract
    image_channel_keys: tuple[str, str, str] = (
        "ch_neutral_all_raw",
        "ch_charged_pu",
        "ch_charged_lv",
    )
    target_key: str = "ch_neutral_lv"

    image_size: int = 9
    charged_image_size: int = 36

    @property
    def charged_pool_factor(self) -> int:
        """
        Pool each 36x36 charged image into a 9x9 image.

        Each output pixel contains the sum of one 4x4 block, preserving
        the total transverse momentum.
        """
        return self.charged_image_size // self.image_size

    @property
    def n_image_channels(self) -> int:
        return len(self.image_channel_keys)

    @property
    def context_dim(self) -> int:
        """
        Three flattened 9x9 input images:

            3 * 9 * 9 = 243
        """
        return self.n_image_channels * self.image_size**2

    @property
    def n_target(self) -> int:
        """
        One flattened neutral-LV 9x9 target image:

            9 * 9 = 81
        """
        return self.image_size**2

    @property
    def n_scalars(self) -> int:
        return 0

    # Generic
    device: str = "cpu"
    seed: int = 42