"""
Configuration dataclasses for the PileFlow generator.

Two simple dataclasses:
    - JetConfig stores the fixed anti-kT jet setup.
    - WorkflowConfig stores the full run configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _default_mg5_path() -> str:
    """
    Return the default MadGraph executable path.

    The environment variable MG5_PATH takes priority. If it is not set,
    fall back to a conventional user-local path.
    """
    return os.environ.get(
        "MG5_PATH",
        os.path.expanduser("~/softwares/mg5amcnlo/bin/mg5_aMC"), # THIS IS USER SPECIFIC. IT MUST BE HARDCODED
    )


@dataclass
class JetConfig:
    """
    Jet clustering configuration. Only anti-kT with R=0.4.
    """

    R: float = 0.4
    algo_name: str = "antikt"
    algo_code: int = 1


@dataclass
class WorkflowConfig:
    """
    Full workflow configuration.

    This object stores the settings needed by the generator pipeline:

        - MadGraph settings
        - LHE input path
        - output paths
        - Pythia event count
        - jet cuts
        - pileup settings
        - image settings
        - plotting options
        - RNG seeds
    """

    # MadGraph executable path.
    mg5_path: str = field(default_factory=_default_mg5_path)

    # Input/output.
    lhe_file: str | None = None
    process_name: str | None = None
    output_dir: str = "data"
    work_dir: str = field(default_factory=os.getcwd)

    # Pythia / workflow processing controls.
    n_events: int = 1000
    jet_pt_min: float = 15.0
    min_hard_parton_pt: float = 0.0

    # MG5 automatic mode controls.
    use_mg5_auto: bool = False
    mg5_process_command: str = "generate p p > j j"
    mg5_nevents: int = 1000

    # Optional minimum jet pT cut written into MadGraph's run_card as ptj.
    mg5_ptj: float | None = None

    # Extra run-card edits written as "set key value" lines in the MG5 script.
    mg5_run_card_edits: dict[str, Any] = field(default_factory=dict)

    # Pileup.
    n_pu: int = 50

    # PUMML image settings.
    image_pt_min: float = 15.0

    # Plot and output options.
    save_figures: bool = False
    save_feynman_diagrams: bool = False
    max_event_figures: int = 30
    max_scatter_points_global: int = 20000

    # RNG seeds.
    rng_seed: int = 42
    pythia_seed: int = 42

    # Execution style.
    interactive: bool = False

    # Fixed jet config.
    jet_config: JetConfig = field(default_factory=JetConfig)