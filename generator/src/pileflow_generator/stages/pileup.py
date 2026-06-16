"""
Pileup-overlay stage.

This module handles:
    - stable visible particle filtering,
    - conversion into TaggedParticle objects,
    - generation of minimum-bias pileup events,
    - leading-vertex / pileup truth tagging.

It does not build jet images.
It does not run PUPPI.
It does not pack constituent arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pythia8

from pileflow_generator.physics.pdg import NEUTRINO_IDS_ABS


@dataclass
class TaggedParticle:
    """
    Minimal particle container carrying truth-origin information.

    Parameters
    ----------
    px, py, pz, e:
        Four-momentum components.
    is_lv:
        True for leading-vertex hard-scatter particles.
        False for pileup particles.
    charge:
        Electric charge.
    pdg_id:
        PDG particle ID.
    """

    px: float
    py: float
    pz: float
    e: float
    is_lv: bool
    charge: float
    pdg_id: int

    @property
    def pt(self) -> float:
        """Transverse momentum."""
        return float(np.sqrt(self.px**2 + self.py**2))

    @property
    def eta(self) -> float:
        """Pseudorapidity."""
        p = np.sqrt(self.px**2 + self.py**2 + self.pz**2)

        if p == 0 or p == abs(self.pz):
            return float(np.sign(self.pz) * 1e9)

        return float(0.5 * np.log((p + self.pz) / (p - self.pz)))

    @property
    def phi(self) -> float:
        """Azimuthal angle."""
        return float(np.arctan2(self.py, self.px))


def _is_stable_visible(p: pythia8.Particle) -> bool:
    """
    Return True if a Pythia particle is final-state and detector-visible.

    Neutrinos are removed because they are invisible to the detector-level jet
    reconstruction used by this toy generator.
    """
    if not p.isFinal():
        return False

    if p.idAbs() in NEUTRINO_IDS_ABS:
        return False

    return True


class PileupOverlay:
    """
    Generate minimum-bias pileup with a separate Pythia8 instance.

    The hard-scatter particles are tagged with ``is_lv=True``.
    The overlaid pileup particles are tagged with ``is_lv=False``.
    """

    def __init__(self, pythia_seed: int = 1, beam_energy: float = 13000.0):
        self._pythia_pu = pythia8.Pythia()

        self._pythia_pu.readString("Random:setSeed = on")
        self._pythia_pu.readString(f"Random:seed = {pythia_seed}")
        self._pythia_pu.readString(f"Beams:eCM = {beam_energy}")
        self._pythia_pu.readString("SoftQCD:inelastic = on")
        self._pythia_pu.readString("Print:quiet = on")
        self._pythia_pu.readString("Next:numberShowInfo = 0")
        self._pythia_pu.readString("Next:numberShowProcess = 0")
        self._pythia_pu.readString("Next:numberShowEvent = 0")

        if not self._pythia_pu.init():
            raise RuntimeError("PileupOverlay: Pythia8 minimum-bias initialization failed.")

    def _extract_stable(self, event: pythia8.Event, is_lv: bool) -> list[TaggedParticle]:
        """
        Extract stable visible particles from a Pythia event.
        """
        particles: list[TaggedParticle] = []

        for i in range(event.size()):
            p = event[i]

            if not _is_stable_visible(p):
                continue

            particles.append(
                TaggedParticle(
                    px=float(p.px()),
                    py=float(p.py()),
                    pz=float(p.pz()),
                    e=float(p.e()),
                    is_lv=bool(is_lv),
                    charge=float(p.charge()),
                    pdg_id=int(p.id()),
                )
            )

        return particles

    def overlay(self, hard_event: pythia8.Event, n_pu: int) -> list[TaggedParticle]:
        """
        Return hard-scatter particles plus ``n_pu`` minimum-bias pileup events.
        """
        all_particles = self._extract_stable(hard_event, is_lv=True)
        all_particles.extend(self._get_pu_particles(n_pu))
        return all_particles

    def _get_pu_particles(self, n_pu: int) -> list[TaggedParticle]:
        """
        Generate and return pileup particles only.
        """
        import warnings

        pu_particles: list[TaggedParticle] = []

        if n_pu <= 0:
            return pu_particles

        generated = 0
        attempts = 0
        max_attempts = n_pu * 20

        while generated < n_pu and attempts < max_attempts:
            attempts += 1

            if not self._pythia_pu.next():
                continue

            pu_particles.extend(
                self._extract_stable(self._pythia_pu.event, is_lv=False)
            )
            generated += 1

        if generated < n_pu:
            warnings.warn(
                f"PileupOverlay: requested {n_pu} PU events "
                f"but only generated {generated}."
            )

        return pu_particles


def tagged_from_snapshot(snapshot: list[dict[str, Any]]) -> list[TaggedParticle]:
    """
    Convert a stored leading-vertex snapshot into TaggedParticle objects.

    The snapshot comes from the Pythia stage and represents the hard-scatter
    event before explicit pileup overlay.
    """
    return [
        TaggedParticle(
            px=float(p["px"]),
            py=float(p["py"]),
            pz=float(p["pz"]),
            e=float(p["e"]),
            is_lv=True,
            charge=float(p["charge"]),
            pdg_id=int(p["pdg_id"]),
        )
        for p in snapshot
    ]