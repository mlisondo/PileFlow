"""
Pythia8 event handling for the PileFlow generator.

This module handles:
    1. Pythia initialization from an LHE file.
    2. Extraction of visible final-state particles.
    3. Extraction of partons for flavour matching.
    4. Building stored event records for later FastJet reconstruction.
"""

from __future__ import annotations

from typing import Any

import fastjet as fj
import numpy as np
import pythia8

from pileflow_generator.physics.pdg import (
    NEUTRINO_IDS_ABS,
    QUARK_GLUON_IDS_ABS,
    RELEVANT_STATUS_ABS,
)


class PythiaRunner:
    """
    Wrapper around Pythia8 for the generator workflow.

    Parameters
    ----------
    lhe_file:
        Path to the uncompressed LHE file that Pythia should read.
    n_events:
        Maximum number of Pythia event attempts.
    pythia_seed:
        Random seed passed to Pythia.
    min_hard_parton_pt:
        Optional event-level hardness cut, using the maximum selected parton
        transverse momentum as a proxy.
    """

    def __init__(
        self,
        lhe_file: str,
        n_events: int,
        pythia_seed: int,
        min_hard_parton_pt: float = 0.0,
    ) -> None:
        self.lhe_file = lhe_file
        self.n_events = int(n_events)
        self.pythia_seed = int(pythia_seed)
        self.min_hard_parton_pt = float(min_hard_parton_pt)

    def initialize(self):
        """
        Initialize Pythia8 from the configured LHE file.

        Returns
        -------
        pythia8.Pythia
            Initialized Pythia object.
        """
        pythia = pythia8.Pythia()

        # Read events from a Les Houches Event file instead of generating
        # the hard process internally.
        pythia.readString("Beams:frameType = 4")
        pythia.readString(f"Beams:LHEF = {self.lhe_file}")

        # MPI here is the underlying event associated with the hard collision.
        # It is conceptually separate from the explicit pileup overlay created
        # later by the image/pileup stage.
        pythia.readString("PartonLevel:MPI = on")
        pythia.readString("PartonLevel:ISR = on")
        pythia.readString("PartonLevel:FSR = on")
        pythia.readString("HadronLevel:Hadronize = on")

        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {self.pythia_seed}")
        pythia.readString("Print:quiet = on")

        if not pythia.init():
            raise RuntimeError(
                "Pythia8 could not initialize using the provided LHE file."
            )

        return pythia

    @staticmethod
    def extract_partons_for_matching(pythia) -> list[tuple[int, float, float, float]]:
        """
        Extract hard-process partons used later for jet-flavour matching.

        The returned flavour is stored as an absolute PDG ID. Antiparticles are
        therefore not distinguished.

        Parameters
        ----------
        pythia:
            Active Pythia object whose current event has already been loaded.

        Returns
        -------
        list[tuple[int, float, float, float]]
            Each entry is ``(abs_pid, eta, phi, pt)``.
        """
        partons: list[tuple[int, float, float, float]] = []

        for i in range(pythia.event.size()):
            particle = pythia.event[i]
            pid_abs = abs(particle.id())

            # Keep only quarks and gluons.
            if pid_abs not in QUARK_GLUON_IDS_ABS:
                continue

            # Keep only relevant hard-process / parton-level statuses.
            if abs(particle.status()) not in RELEVANT_STATUS_ABS:
                continue

            d1 = particle.daughter1()
            d2 = particle.daughter2()

            has_same_flavour_daughter = False

            if d1 > 0 and d2 >= d1:
                for daughter_index in range(d1, d2 + 1):
                    if 0 <= daughter_index < pythia.event.size():
                        daughter = pythia.event[daughter_index]

                        same_flavour = abs(daughter.id()) == pid_abs
                        relevant_status = (
                            abs(daughter.status()) in RELEVANT_STATUS_ABS
                        )

                        if same_flavour and relevant_status:
                            has_same_flavour_daughter = True
                            break

            # Keep the last copy in the relevant parton chain.
            if not has_same_flavour_daughter:
                partons.append(
                    (
                        int(pid_abs),
                        float(particle.eta()),
                        float(particle.phi()),
                        float(particle.pT()),
                    )
                )

        return partons

    @staticmethod
    def event_hardness_proxy_pt(
        partons: list[tuple[int, float, float, float]]
    ) -> float:
        """
        Compute a simple event hardness proxy.

        The proxy is the maximum transverse momentum among the selected
        matching partons.

        Parameters
        ----------
        partons:
            List of ``(abs_pid, eta, phi, pt)`` tuples.

        Returns
        -------
        float
            Maximum selected parton pT. Returns 0 if no partons were found.
        """
        if not partons:
            return 0.0

        return float(max(parton[3] for parton in partons))

    def read_events(self) -> list[dict[str, Any]]:
        """
        Read events from Pythia and store them in a workflow-friendly format.

        Returns
        -------
        list[dict[str, Any]]
            Stored event records. Each record contains:

            - ``source_event_idx``
            - ``accepted_event_idx``
            - ``particles``
            - ``particle_map``
            - ``partons``
            - ``hard_proxy_pt``
            - ``lv_snapshot``
        """
        pythia = self.initialize()

        stored_events: list[dict[str, Any]] = []
        accepted_counter = 0

        for i_ev in range(self.n_events):
            if not pythia.next():
                print(f"Pythia stopped at attempt {i_ev}.")
                break

            particles = []
            particle_map: dict[int, tuple[int, bool]] = {}

            for i in range(pythia.event.size()):
                particle = pythia.event[i]

                # Keep only stable final-state particles.
                if not particle.isFinal():
                    continue

                # Remove invisible neutrinos before jet clustering.
                if abs(particle.id()) in NEUTRINO_IDS_ABS:
                    continue

                px = particle.px()
                py = particle.py()
                pz = particle.pz()
                energy = particle.e()

                if (
                    not np.isfinite(px)
                    or not np.isfinite(py)
                    or not np.isfinite(pz)
                    or not np.isfinite(energy)
                ):
                    continue

                if energy <= 0.0:
                    continue

                pseudojet = fj.PseudoJet(px, py, pz, energy)
                pseudojet.set_user_index(i)

                pid = int(abs(particle.id()))
                is_charged = bool(particle.isCharged())

                # This is the object consumed by the FastJet clustering stage.
                particles.append((pseudojet, pid, is_charged))

                # This map lets later reconstruction recover PID/charge info
                # from FastJet constituents using user_index().
                particle_map[i] = (pid, is_charged)

            partons = self.extract_partons_for_matching(pythia)
            hard_proxy_pt = self.event_hardness_proxy_pt(partons)

            if (
                self.min_hard_parton_pt > 0.0
                and hard_proxy_pt < self.min_hard_parton_pt
            ):
                continue

            # Snapshot of the leading-vertex hard-scatter particles before any
            # explicit pileup overlay. This is needed by the PUMML image stage.
            lv_snapshot = []

            for pseudojet, pid, is_charged in particles:
                lv_snapshot.append(
                    {
                        "px": pseudojet.px(),
                        "py": pseudojet.py(),
                        "pz": pseudojet.pz(),
                        "e": pseudojet.e(),
                        "charge": 1.0 if is_charged else 0.0,
                        "pdg_id": pid,
                        "is_lv": True,
                    }
                )

            stored_events.append(
                {
                    "source_event_idx": int(i_ev),
                    "accepted_event_idx": int(accepted_counter),
                    "particles": particles,
                    "particle_map": particle_map,
                    "partons": partons,
                    "hard_proxy_pt": hard_proxy_pt,
                    "lv_snapshot": lv_snapshot,
                }
            )

            accepted_counter += 1

            if (i_ev + 1) % 200 == 0:
                print(
                    f"Read attempts: {i_ev + 1}/{self.n_events} "
                    f"| accepted: {accepted_counter}"
                )

        try:
            pythia.stat()
        except Exception:
            pass

        return stored_events