"""
FastJet clustering stage.

This module owns:
    - FastJet anti-kT clustering,
    - event-level jet loops,
    - jet pT/eta cuts,
    - constituent lookup,
    - feature-row construction,
    - PUMML/PUPPI image accumulation hook.

Feature definitions live in ``stages.features``.
Parton matching lives in ``physics.matching``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import fastjet as fj

from pileflow_generator.physics.matching import match_flavour
from pileflow_generator.schemas.jet_features import N_FEATURES
from pileflow_generator.stages.features import (
    build_feature_row,
    compute_fractions,
)
from pileflow_generator.stages.images import produce_images


IMAGE_ACCUMULATION_KEYS = [
    # PUMML inputs.
    "ch_charged_lv",
    "ch_charged_pu",
    "ch_neutral_all",
    "ch_neutral_all_raw",
    "ch_neutral_lv",

    # Clean reference.
    "clean_neutral_lv",
    "clean_neutral_all",

    # Metadata.
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "n_pu",

    # Full LV + PU particles before PUPPI.
    # These are needed to rerun PUPPI standalone after generation.
    "full_px",
    "full_py",
    "full_pz",
    "full_e",
    "full_charge",
    "full_is_lv",
    "full_n",

    # True LV-only constituents.
    "true_px",
    "true_py",
    "true_pz",
    "true_e",
    "true_n",

    # LV + pileup constituents.
    "pileup_px",
    "pileup_py",
    "pileup_pz",
    "pileup_e",
    "pileup_n",

    # PUPPI-mitigated constituents.
    "puppi_px",
    "puppi_py",
    "puppi_pz",
    "puppi_e",
    "puppi_n",
]


class FastJetRunner:
    """
    Run FastJet clustering and construct generator outputs.

    Parameters
    ----------
    jet_pt_min:
        Minimum jet pT for the 25-feature ``.npy`` dataset.

    jet_R:
        Anti-kT jet radius.

    rng_seed:
        NumPy seed used for detector smearing, tag proxies, and image-stage
        pileup-count sampling.

    pileup_overlay:
        Optional PileupOverlay object. If provided together with
        ``image_builder``, image production is enabled.

    image_builder:
        Optional JetImageBuilder object.

    n_pu:
        Mean pileup count used for image production.

    image_pt_min:
        Independent jet pT cut for the PUMML image dataset.

    image_output_path:
        Path where the accumulated ``.npz`` image file is saved.
    """

    def __init__(
        self,
        jet_pt_min: float,
        jet_R: float,
        rng_seed: int,
        pileup_overlay: Any | None = None,
        image_builder: Any | None = None,
        n_pu: int = 50,
        image_pt_min: float = 15.0,
        image_output_path: str | None = None,
    ):
        self.jet_pt_min = float(jet_pt_min)
        self.jet_R = float(jet_R)
        self.rng = np.random.default_rng(rng_seed)

        self.pileup_overlay = pileup_overlay
        self.image_builder = image_builder
        self.n_pu = int(n_pu)
        self.image_pt_min = float(image_pt_min)
        self.image_output_path = image_output_path

        self._image_accum: dict[str, list[np.ndarray]] = {
            key: [] for key in IMAGE_ACCUMULATION_KEYS
        }

    def _make_jet_definition(self):
        """
        Create the anti-kT FastJet definition.
        """
        return fj.JetDefinition(fj.antikt_algorithm, self.jet_R)

    def _make_area_definition(self):
        """
        Try to create FastJet active-area support.

        Returns
        -------
        tuple[bool, object | None]
            ``(area_supported, area_def)``.
        """
        try:
            ghost_spec = fj.GhostedAreaSpec(5.0)
            area_def = fj.AreaDefinition(fj.active_area, ghost_spec)
            return True, area_def
        except Exception:
            print("Warning: FastJet area support is not available. jetArea will be set to 0.")
            return False, None

    @staticmethod
    def _sorted_inclusive_jets(cluster_sequence) -> list:
        """
        Return inclusive jets sorted by descending pT.
        """
        try:
            return list(fj.sorted_by_pt(cluster_sequence.inclusive_jets()))
        except Exception:
            return sorted(cluster_sequence.inclusive_jets(), key=lambda j: -j.pt())

    def _cluster_event(
        self,
        pseudojets: list,
        jet_def,
        area_supported: bool,
        area_def,
    ):
        """
        Cluster one event with or without active-area support.
        """
        if area_supported:
            return fj.ClusterSequenceArea(pseudojets, jet_def, area_def)

        return fj.ClusterSequence(pseudojets, jet_def)

    def _select_jets(self, jets_all: list) -> list:
        """
        Apply the old jet selection.

        The eta cut preserves the old tracker-acceptance-inspired behavior:
            pT >= jet_pt_min and |eta| < 2.5
        """
        return [
            jet for jet in jets_all
            if jet.pt() >= self.jet_pt_min and abs(jet.eta()) < 2.5
        ]

    @staticmethod
    def _constituent_info(jet, particle_map: dict) -> list[tuple]:
        """
        Recover ``(constituent, pid, is_charged)`` for real jet constituents.

        FastJet ghost particles may have missing or invalid user indices.
        Those are skipped.
        """
        const_info = []

        for constituent in jet.constituents():
            idx = constituent.user_index()
            info = particle_map.get(idx)

            if info is None:
                continue

            pid, is_charged = info
            const_info.append((constituent, pid, is_charged))

        return const_info

    @staticmethod
    def _jet_area(jet, area_supported: bool) -> float:
        """
        Read active jet area, returning 0 if unavailable.
        """
        if not area_supported:
            return 0.0

        try:
            return float(jet.area())
        except Exception:
            return 0.0

    def _image_production_enabled(self) -> bool:
        """
        Check whether image production is enabled.
        """
        return self.pileup_overlay is not None and self.image_builder is not None

    def _accumulate_event_images(self, event_data: dict, jets: list) -> None:
        """
        Run image production for one event and append arrays to the accumulator.
        """
        if not self._image_production_enabled():
            return

        n_pu = int(self.rng.poisson(lam=self.n_pu))

        event_imgs = produce_images(
            hard_event=event_data["lv_snapshot"],
            jets=jets,
            overlay=self.pileup_overlay,
            builder=self.image_builder,
            n_pu=n_pu,
            jet_pt_min=self.image_pt_min,
            output_path=None,
        )

        for key in self._image_accum:
            arr = event_imgs[key]

            if len(arr):
                self._image_accum[key].append(arr)

    def _save_accumulated_images(self) -> None:
        """
        Save accumulated PUMML/PUPPI image arrays to ``image_output_path``.
        """
        if not self._image_production_enabled():
            return

        if not any(self._image_accum.values()):
            return

        if self.image_output_path is None:
            raise ValueError(
                "Image production was enabled, but image_output_path is None."
            )

        final = {
            key: np.concatenate(value, axis=0)
            for key, value in self._image_accum.items()
            if value
        }

        np.savez_compressed(self.image_output_path, **final)

        n_jets = len(final.get("jet_pt", []))

        print(
            f"[PUMML] Saved {n_jets} jets -> {self.image_output_path}\n"
            f"  images       : ch_neutral_all {final['ch_neutral_all'].shape}\n"
            f"  true consts  : {final['true_px'].shape}\n"
            f"  pileup consts: {final['pileup_px'].shape}\n"
            f"  puppi consts : {final['puppi_px'].shape}"
        )

    def cluster_events(self, stored_events: list[dict]) -> tuple[np.ndarray, list]:
        """
        Cluster stored Pythia events and build the final jet dataset.

        Parameters
        ----------
        stored_events:
            Event dictionaries produced by ``PythiaRunner.read_events``.

        Returns
        -------
        tuple[np.ndarray, list]
            ``(dataset, event_figures)`` where ``dataset`` has shape
            ``(N_jets, 25)`` and ``event_figures`` stores event-level jet
            coordinates for diagnostics.
        """
        dataset_rows: list[np.ndarray] = []
        event_figures: list[dict] = []

        jet_def = self._make_jet_definition()
        area_supported, area_def = self._make_area_definition()

        for event_data in stored_events:
            particles = event_data["particles"]
            particle_map = event_data["particle_map"]
            partons = event_data["partons"]

            pseudojets = [pj for pj, _pid, _is_charged in particles]

            if not pseudojets:
                continue

            cluster_sequence = self._cluster_event(
                pseudojets=pseudojets,
                jet_def=jet_def,
                area_supported=area_supported,
                area_def=area_def,
            )

            jets_all = self._sorted_inclusive_jets(cluster_sequence)
            jets = self._select_jets(jets_all)

            if not jets:
                continue

            jets_eta_phi_pt = [
                (float(j.eta()), float(j.phi()), float(j.pt()))
                for j in jets
            ]

            event_figures.append(
                {
                    "source_event_idx": int(event_data["source_event_idx"]),
                    "accepted_event_idx": int(event_data["accepted_event_idx"]),
                    "jets_eta_phi_pt": jets_eta_phi_pt,
                }
            )

            for jet in jets:
                const_info = self._constituent_info(jet, particle_map)

                if not const_info:
                    continue

                fracs = compute_fractions(const_info)

                flavour = match_flavour(
                    jet_eta=jet.eta(),
                    jet_phi=jet.phi(),
                    partons=partons,
                    R=self.jet_R,
                )

                jet_area = self._jet_area(jet, area_supported)

                feature_row = build_feature_row(
                    rng=self.rng,
                    jet=jet,
                    fracs=fracs,
                    flavour=flavour,
                    jet_R=self.jet_R,
                    jet_area=jet_area,
                )

                dataset_rows.append(feature_row)

            self._accumulate_event_images(event_data, jets)

        if dataset_rows:
            dataset = np.asarray(dataset_rows, dtype=np.float32)
        else:
            dataset = np.empty((0, N_FEATURES), dtype=np.float32)

        self._save_accumulated_images()

        return dataset, event_figures