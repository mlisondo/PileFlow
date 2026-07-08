"""
flows/config.py
===============

Central configuration for the PileFlow model pipeline.

Pipeline stages:
  Stage 1 — Load generator data  (jets.npy + jets_pileup_images.npz)
  Stage 2 — Train PileFlow flow model
  Stage 3 — Generate mitigated jets  (generated_jets.npz)
  Stage 4 — Compare + plot

External tools:
  pumml_ckpt : optional trained PUMML checkpoint used only for comparison.
"""

from dataclasses import dataclass, field
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
    data_npy: Optional[str] = None
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

    # Generator feature indices
    GEN_SCALAR_IDX: list = field(default_factory=lambda: [0, 1, 2, 3, 9, 22, 24])
    GEN_FLAVOUR_IDX: int = 4

    @property
    def n_target(self) -> int:
        return 97

    @property
    def context_dim(self) -> int:
        return len(self.GEN_SCALAR_IDX) + 3 + 3 * 81

    # Generic
    device: str = "cpu"
    seed: int = 42