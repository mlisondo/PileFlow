"""
FastJet clustering stage.

This module owns:
    - FastJet anti-kT clustering,
    - event-level jet loops,
    - jet pT/eta cuts,
    - constituent lookup,
    - feature-row construction,
    - PUMML/PUPPI image accumulation,
    - event-ID and within-event jet-rank propagation.

Feature definitions live in ``stages.features``.
Parton matching lives in ``physics.matching``.
The image-array contract lives in ``schemas.image_arrays``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import fastjet as fj

from pileflow_generator.physics.matching import match_flavour
from pileflow_generator.schemas.jet_features import N_FEATURES
from pileflow_generator.schemas.image_arrays import (
    REQUIRED_NPZ_KEYS,
    empty_image_arrays,
)
from pileflow_generator.stages.features import (
    build_feature_row,
    compute_fractions,
)
from pileflow_generator.stages.images import produce_images


# Use the formal NPZ schema as the single source of truth.
#
# This prevents the accumulation list from drifting out of sync with
# schemas/image_arrays.py when new arrays are added.
IMAGE_ACCUMULATION_KEYS = list(REQUIRED_NPZ_KEYS)


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
        Independent jet pT cut for the PUMML/PUPPI image dataset.

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
            key: []
            for key in IMAGE_ACCUMULATION_KEYS
        }

    def _make_jet_definition(self):
        """
        Create the anti-kT FastJet definition.
        """
        return fj.JetDefinition(
            fj.antikt_algorithm,
            self.jet_R,
        )

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
            area_def = fj.AreaDefinition(
                fj.active_area,
                ghost_spec,
            )
            return True, area_def

        except Exception:
            print(
                "Warning: FastJet area support is not available. "
                "jetArea will be set to 0."
            )
            return False, None

    @staticmethod
    def _sorted_inclusive_jets(
        cluster_sequence,
    ) -> list:
        """
        Return inclusive jets sorted by descending pT.
        """
        try:
            return list(
                fj.sorted_by_pt(
                    cluster_sequence.inclusive_jets()
                )
            )

        except Exception:
            return sorted(
                cluster_sequence.inclusive_jets(),
                key=lambda jet: -jet.pt(),
            )

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
            return fj.ClusterSequenceArea(
                pseudojets,
                jet_def,
                area_def,
            )

        return fj.ClusterSequence(
            pseudojets,
            jet_def,
        )

    def _select_jets(
        self,
        jets_all: list,
    ) -> list:
        """
        Select jets for the legacy 25-feature NPY dataset.

        The image dataset has its own independent pT cut, applied later
        inside ``produce_images``.
        """
        return [
            jet
            for jet in jets_all
            if jet.pt() >= self.jet_pt_min
            and abs(jet.eta()) < 2.5
        ]

    @staticmethod
    def _constituent_info(
        jet,
        particle_map: dict,
    ) -> list[tuple]:
        """
        Recover ``(constituent, pid, is_charged)`` for real constituents.

        FastJet ghost particles may have missing or invalid user indices.
        Those are skipped.
        """
        const_info = []

        for constituent in jet.constituents():
            index = constituent.user_index()
            info = particle_map.get(index)

            if info is None:
                continue

            pid, is_charged = info

            const_info.append(
                (
                    constituent,
                    pid,
                    is_charged,
                )
            )

        return const_info

    @staticmethod
    def _jet_area(
        jet,
        area_supported: bool,
    ) -> float:
        """
        Read active jet area, returning zero if unavailable.
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
        return (
            self.pileup_overlay is not None
            and self.image_builder is not None
        )

    def _accumulate_event_images(
        self,
        event_data: dict,
        jets_all: list,
    ) -> None:
        """
        Build image-level rows for one event and append them to the run-level
        accumulator.

        ``jets_all`` is passed rather than the feature-selected jets so that
        ``image_pt_min`` remains independent from ``jet_pt_min``.
        """
        if not self._image_production_enabled():
            return

        n_pu = int(
            self.rng.poisson(
                lam=self.n_pu,
            )
        )

        event_imgs = produce_images(
            hard_event=event_data["lv_snapshot"],
            jets=jets_all,
            overlay=self.pileup_overlay,
            builder=self.image_builder,
            event_id=int(
                event_data["source_event_idx"]
            ),
            n_pu=n_pu,
            jet_pt_min=self.image_pt_min,
            output_path=None,
        )

        missing_keys = [
            key
            for key in IMAGE_ACCUMULATION_KEYS
            if key not in event_imgs
        ]

        unexpected_keys = [
            key
            for key in event_imgs
            if key not in IMAGE_ACCUMULATION_KEYS
        ]

        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Per-event image payload does not match the NPZ schema. "
                f"Missing keys: {missing_keys}; "
                f"unexpected keys: {unexpected_keys}"
            )

        row_counts = {
            key: len(event_imgs[key])
            for key in IMAGE_ACCUMULATION_KEYS
        }

        unique_row_counts = set(
            row_counts.values()
        )

        if len(unique_row_counts) != 1:
            raise RuntimeError(
                "Per-event image payload is not row-aligned: "
                f"{row_counts}"
            )

        n_event_jets = next(
            iter(unique_row_counts),
            0,
        )

        if n_event_jets == 0:
            return

        for key in IMAGE_ACCUMULATION_KEYS:
            self._image_accum[key].append(
                event_imgs[key]
            )

    def _save_accumulated_images(self) -> None:
        """
        Save the complete detector-image NPZ payload.

        Every required key must exist and every array must have the same
        leading dimension.
        """
        if not self._image_production_enabled():
            return

        if self.image_output_path is None:
            raise ValueError(
                "Image production was enabled, but "
                "image_output_path is None."
            )

        populated_keys = [
            key
            for key, chunks in self._image_accum.items()
            if chunks
        ]

        if not populated_keys:
            final = empty_image_arrays(
                n_charged=(
                    self.image_builder.n_pixels_charged
                ),
                n_neutral=(
                    self.image_builder.n_pixels_neutral
                ),
            )

        else:
            missing_accumulators = [
                key
                for key in IMAGE_ACCUMULATION_KEYS
                if not self._image_accum[key]
            ]

            if missing_accumulators:
                raise RuntimeError(
                    "Cannot save an incomplete NPZ payload. "
                    "These required accumulators are empty: "
                    f"{missing_accumulators}"
                )

            final = {
                key: np.concatenate(
                    self._image_accum[key],
                    axis=0,
                )
                for key in IMAGE_ACCUMULATION_KEYS
            }

        missing_final_keys = [
            key
            for key in IMAGE_ACCUMULATION_KEYS
            if key not in final
        ]

        unexpected_final_keys = [
            key
            for key in final
            if key not in IMAGE_ACCUMULATION_KEYS
        ]

        if missing_final_keys or unexpected_final_keys:
            raise RuntimeError(
                "Final image payload does not match the NPZ schema. "
                f"Missing keys: {missing_final_keys}; "
                f"unexpected keys: {unexpected_final_keys}"
            )

        final["event_id"] = np.asarray(
            final["event_id"],
            dtype=np.int64,
        )

        final["jet_rank"] = np.asarray(
            final["jet_rank"],
            dtype=np.int32,
        )

        row_counts = {
            key: len(value)
            for key, value in final.items()
        }

        unique_row_counts = set(
            row_counts.values()
        )

        if len(unique_row_counts) != 1:
            raise RuntimeError(
                "Accumulated NPZ arrays are not row-aligned: "
                f"{row_counts}"
            )

        n_jets = next(
            iter(unique_row_counts),
            0,
        )

        # Verify that each event has at most one row for each jet rank.
        if n_jets:
            event_rank_pairs = np.stack(
                [
                    final["event_id"],
                    final["jet_rank"].astype(
                        np.int64,
                        copy=False,
                    ),
                ],
                axis=1,
            )

            unique_pairs = np.unique(
                event_rank_pairs,
                axis=0,
            )

            if len(unique_pairs) != n_jets:
                raise RuntimeError(
                    "Duplicate (event_id, jet_rank) pairs were found "
                    "in the accumulated image dataset."
                )

        np.savez_compressed(
            self.image_output_path,
            **final,
        )

        n_events = (
            len(
                np.unique(
                    final["event_id"]
                )
            )
            if n_jets
            else 0
        )

        print(
            f"[PUMML] Saved {n_jets} jets from {n_events} events "
            f"-> {self.image_output_path}\n"
            f"  neutral input      : "
            f"{final['ch_neutral_all'].shape}\n"
            f"  neutral target     : "
            f"{final['ch_neutral_lv'].shape}\n"
            f"  PUPPI neutral      : "
            f"{final['puppi_neutral_9x9'].shape}\n"
            f"  PUPPI charged      : "
            f"{final['puppi_charged_36x36'].shape}\n"
            f"  event IDs          : "
            f"{final['event_id'].shape}\n"
            f"  jet ranks          : "
            f"{final['jet_rank'].shape}\n"
            f"  true constituents  : "
            f"{final['true_px'].shape}\n"
            f"  pileup constituents: "
            f"{final['pileup_px'].shape}\n"
            f"  PUPPI constituents : "
            f"{final['puppi_px'].shape}"
        )

    def cluster_events(
        self,
        stored_events: list[dict],
    ) -> tuple[np.ndarray, list]:
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
        area_supported, area_def = (
            self._make_area_definition()
        )

        for event_data in stored_events:
            particles = event_data["particles"]
            particle_map = event_data["particle_map"]
            partons = event_data["partons"]

            pseudojets = [
                pseudojet
                for pseudojet, _pid, _is_charged
                in particles
            ]

            if not pseudojets:
                continue

            cluster_sequence = self._cluster_event(
                pseudojets=pseudojets,
                jet_def=jet_def,
                area_supported=area_supported,
                area_def=area_def,
            )

            jets_all = self._sorted_inclusive_jets(
                cluster_sequence
            )

            # Image production has an independent pT selection.
            #
            # This must occur before the legacy feature-dataset ``continue``
            # so events with no feature-level jets can still contribute image
            # jets if they pass ``image_pt_min``.
            self._accumulate_event_images(
                event_data,
                jets_all,
            )

            jets = self._select_jets(
                jets_all
            )

            if not jets:
                continue

            jets_eta_phi_pt = [
                (
                    float(jet.eta()),
                    float(jet.phi()),
                    float(jet.pt()),
                )
                for jet in jets
            ]

            event_figures.append(
                {
                    "source_event_idx": int(
                        event_data["source_event_idx"]
                    ),
                    "accepted_event_idx": int(
                        event_data["accepted_event_idx"]
                    ),
                    "jets_eta_phi_pt": jets_eta_phi_pt,
                }
            )

            for jet in jets:
                const_info = self._constituent_info(
                    jet,
                    particle_map,
                )

                if not const_info:
                    continue

                fractions = compute_fractions(
                    const_info
                )

                flavour = match_flavour(
                    jet_eta=jet.eta(),
                    jet_phi=jet.phi(),
                    partons=partons,
                    R=self.jet_R,
                )

                jet_area = self._jet_area(
                    jet,
                    area_supported,
                )

                feature_row = build_feature_row(
                    rng=self.rng,
                    jet=jet,
                    fracs=fractions,
                    flavour=flavour,
                    jet_R=self.jet_R,
                    jet_area=jet_area,
                )

                dataset_rows.append(
                    feature_row
                )

        if dataset_rows:
            dataset = np.asarray(
                dataset_rows,
                dtype=np.float32,
            )

        else:
            dataset = np.empty(
                (0, N_FEATURES),
                dtype=np.float32,
            )

        self._save_accumulated_images()

        return dataset, event_figures