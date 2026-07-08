from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PUPPIConfig:
    """
    Configuration for the simplified PUPPI baseline.
    """

    R0: float = 0.3
    Rmin: float = 0.02
    w_cut: float = 0.1
    eta_tracker: float = 2.5
    max_const: int = 500