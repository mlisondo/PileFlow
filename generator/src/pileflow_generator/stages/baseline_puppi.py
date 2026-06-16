"""
Compatibility alias for the temporary PUPPI baseline implementation.

The real implementation currently lives in:

    temporary_baseline_puppi.py

This wrapper lets code import either:

    pileflow_generator.stages.temporary_baseline_puppi

or:

    pileflow_generator.stages.baseline_puppi

without changing behavior.
"""

from .temporary_baseline_puppi import (
    Particle,
    unpack_particles,
    characterise_pileup,
    run_puppi,
    run_puppi_on_dataset,
)

__all__ = [
    "Particle",
    "unpack_particles",
    "characterise_pileup",
    "run_puppi",
    "run_puppi_on_dataset",
]