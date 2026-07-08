"""
PUMML image-building stage.

This module builds the PUMML/PUPPI `.npz` output for one hard event.

It handles:
    - PUMML image pixelization,
    - true / pileup / PUPPI constituent packing,
    - jet matching between clean, pileup, and PUPPI collections,
    - optional saving to compressed `.npz`.

It assumes PUPPI is temporarily available inside the generator package as
``stages.baseline_puppi``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import fastjet as fj

from pileflow_generator.schemas.image_arrays import (
    MAX_CONST,
    N_PIXELS_CHARGED,
    N_PIXELS_NEUTRAL,
    ETA_RANGE,
    PHI_RANGE,
    PT_CHARGED_CUT,
    empty_image_arrays,
)
from pileflow_generator.stages.pileup import (
    TaggedParticle,
    PileupOverlay,
    _is_stable_visible,
    tagged_from_snapshot,
)
# from pileflow_generator.stages.temporary_baseline_puppi import (
#     Particle as PuppiParticle,
#     run_puppi,
# )
from puppi import Particle as PuppiParticle, run_puppi


class JetImageBuilder:
    """
    Build 2D pT histograms in Delta eta / Delta phi around a jet axis.

    Default grid:
        charged images: 36 x 36
        neutral images: 9 x 9
        image window: +/- 0.45 in eta and phi
    """

    def __init__(
        self,
        eta_range: float = ETA_RANGE,
        phi_range: float = PHI_RANGE,
        n_pixels_charged: int = N_PIXELS_CHARGED,
        n_pixels_neutral: int = N_PIXELS_NEUTRAL,
        pt_charged_cut: float = PT_CHARGED_CUT,
    ):
        self.eta_range = eta_range
        self.phi_range = phi_range
        self.n_pixels_charged = n_pixels_charged
        self.n_pixels_neutral = n_pixels_neutral
        self.pt_charged_cut = pt_charged_cut

        self._edges_c = (
            np.linspace(-eta_range, eta_range, n_pixels_charged + 1),
            np.linspace(-phi_range, phi_range, n_pixels_charged + 1),
        )
        self._edges_n = (
            np.linspace(-eta_range, eta_range, n_pixels_neutral + 1),
            np.linspace(-phi_range, phi_range, n_pixels_neutral + 1),
        )

    @staticmethod
    def _dphi(phi1: float, phi2: float) -> float:
        """
        Signed wrapped delta-phi in (-pi, pi].
        """
        dphi = phi1 - phi2

        while dphi > np.pi:
            dphi -= 2.0 * np.pi

        while dphi < -np.pi:
            dphi += 2.0 * np.pi

        return float(dphi)

    def _fill_grid(
        self,
        particles: list[TaggedParticle],
        jet_eta: float,
        jet_phi: float,
        edges: tuple[np.ndarray, np.ndarray],
        n_pixels: int,
    ) -> np.ndarray:
        """
        Fill one pT image around a jet axis.
        """
        grid = np.zeros((n_pixels, n_pixels), dtype=np.float32)

        for p in particles:
            deta = p.eta - jet_eta
            dphi = self._dphi(p.phi, jet_phi)

            if abs(deta) >= self.eta_range or abs(dphi) >= self.phi_range:
                continue

            ieta = int(np.searchsorted(edges[0], deta, side="right")) - 1
            iphi = int(np.searchsorted(edges[1], dphi, side="right")) - 1

            ieta = int(np.clip(ieta, 0, n_pixels - 1))
            iphi = int(np.clip(iphi, 0, n_pixels - 1))

            grid[ieta, iphi] += p.pt

        return grid

    @staticmethod
    def _upsample(img9: np.ndarray) -> np.ndarray:
        """
        Upsample 9x9 neutral image to 36x36.

        Division by 16 preserves total pT because each 9x9 cell becomes
        4 x 4 = 16 cells.
        """
        return np.repeat(np.repeat(img9, 4, axis=0), 4, axis=1) / 16.0

    def build(
        self,
        jet_eta: float,
        jet_phi: float,
        particles: list[TaggedParticle],
    ) -> dict[str, np.ndarray]:
        """
        Build the PUMML image arrays for one jet.

        Returns
        -------
        dict
            Keys:
                ch_charged_lv
                ch_charged_pu
                ch_neutral_all
                ch_neutral_all_raw
                ch_neutral_lv
        """
        nc = self.n_pixels_charged
        nn = self.n_pixels_neutral

        charged_lv: list[TaggedParticle] = []
        charged_pu: list[TaggedParticle] = []
        neutral_all: list[TaggedParticle] = []
        neutral_lv: list[TaggedParticle] = []

        for p in particles:
            is_reco_charged = (abs(p.charge) > 0.0) and (p.pt > self.pt_charged_cut)

            if is_reco_charged:
                if p.is_lv:
                    charged_lv.append(p)
                else:
                    charged_pu.append(p)
            else:
                neutral_all.append(p)

                if p.is_lv:
                    neutral_lv.append(p)

        img_ch_lv = self._fill_grid(charged_lv, jet_eta, jet_phi, self._edges_c, nc)
        img_ch_pu = self._fill_grid(charged_pu, jet_eta, jet_phi, self._edges_c, nc)
        img_n_all9 = self._fill_grid(neutral_all, jet_eta, jet_phi, self._edges_n, nn)
        img_n_lv9 = self._fill_grid(neutral_lv, jet_eta, jet_phi, self._edges_n, nn)
        img_n_all36 = self._upsample(img_n_all9)

        return {
            "ch_charged_lv": img_ch_lv,
            "ch_charged_pu": img_ch_pu,
            "ch_neutral_all": img_n_all36,
            "ch_neutral_all_raw": img_n_all9,
            "ch_neutral_lv": img_n_lv9,
        }


def _pack_constituents(pseudojets: list) -> dict[str, np.ndarray | np.int32]:
    """
    Pack FastJet constituents into fixed-length zero-padded arrays.
    """
    n = min(len(pseudojets), MAX_CONST)

    px = np.zeros(MAX_CONST, dtype=np.float32)
    py = np.zeros(MAX_CONST, dtype=np.float32)
    pz = np.zeros(MAX_CONST, dtype=np.float32)
    e = np.zeros(MAX_CONST, dtype=np.float32)

    for i, pj in enumerate(pseudojets[:n]):
        px[i] = float(pj.px())
        py[i] = float(pj.py())
        pz[i] = float(pj.pz())
        e[i] = float(pj.e())

    return {
        "px": px,
        "py": py,
        "pz": pz,
        "e": e,
        "n": np.int32(n),
    }

def _pack_tagged_particles(
    particles: list[TaggedParticle],
) -> dict[str, np.ndarray | np.int32]:
    """
    Pack full LV + PU TaggedParticles into fixed-length zero-padded arrays.

    This is the pre-PUPPI particle collection. It includes charge and LV/PU
    truth labels so that PUPPI can be rerun standalone after generation.
    """
    n = min(len(particles), MAX_CONST)

    px = np.zeros(MAX_CONST, dtype=np.float32)
    py = np.zeros(MAX_CONST, dtype=np.float32)
    pz = np.zeros(MAX_CONST, dtype=np.float32)
    e = np.zeros(MAX_CONST, dtype=np.float32)
    charge = np.zeros(MAX_CONST, dtype=np.float32)
    is_lv = np.zeros(MAX_CONST, dtype=np.float32)

    for i, p in enumerate(particles[:n]):
        px[i] = float(p.px)
        py[i] = float(p.py)
        pz[i] = float(p.pz)
        e[i] = float(p.e)
        charge[i] = float(p.charge)
        is_lv[i] = 1.0 if p.is_lv else 0.0

    return {
        "px": px,
        "py": py,
        "pz": pz,
        "e": e,
        "charge": charge,
        "is_lv": is_lv,
        "n": np.int32(n),
    }


def _empty_constituents() -> dict[str, np.ndarray | np.int32]:
    """
    Return an empty fixed-length constituent payload.
    """
    return {
        "px": np.zeros(MAX_CONST, dtype=np.float32),
        "py": np.zeros(MAX_CONST, dtype=np.float32),
        "pz": np.zeros(MAX_CONST, dtype=np.float32),
        "e": np.zeros(MAX_CONST, dtype=np.float32),
        "n": np.int32(0),
    }


def _tagged_to_pseudojets(particles: list[TaggedParticle]) -> list:
    """
    Convert TaggedParticle objects to FastJet PseudoJets.
    """
    pjs = []

    for p in particles:
        if p.e <= 0.0:
            continue

        pjs.append(
            fj.PseudoJet(
                float(p.px),
                float(p.py),
                float(p.pz),
                float(p.e),
            )
        )

    return pjs


def _puppi_to_pseudojets(particles: list) -> list:
    """
    Convert PUPPI Particle objects to FastJet PseudoJets.
    """
    pjs = []

    for p in particles:
        if p.e <= 0.0:
            continue

        pjs.append(
            fj.PseudoJet(
                float(p.px),
                float(p.py),
                float(p.pz),
                float(p.e),
            )
        )

    return pjs


def _cluster(pseudojets: list, R: float = 0.4):
    """
    Cluster pseudojets using anti-kT.

    Returns
    -------
    tuple
        ``(jets, cluster_sequence)``.

    Important:
        Keep the cluster sequence alive while using ``jet.constituents()``.
    """
    if not pseudojets:
        return [], None

    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    cs = fj.ClusterSequence(pseudojets, jet_def)

    try:
        jets = list(fj.sorted_by_pt(cs.inclusive_jets()))
    except Exception:
        jets = sorted(cs.inclusive_jets(), key=lambda j: -j.pt())

    return jets, cs


def _match_jet(
    ref_eta: float,
    ref_phi: float,
    jets: list,
    dr_max: float = 0.4,
) -> Optional[object]:
    """
    Match a reference jet axis to the closest jet within ``dr_max``.
    """
    best_jet = None
    best_dr = dr_max

    for j in jets:
        dphi = ref_phi - j.phi()

        while dphi > np.pi:
            dphi -= 2.0 * np.pi

        while dphi < -np.pi:
            dphi += 2.0 * np.pi

        dr = float(np.sqrt((ref_eta - j.eta()) ** 2 + dphi**2))

        if dr < best_dr:
            best_dr = dr
            best_jet = j

    return best_jet


def _clean_particles_from_hard_event(hard_event) -> list[TaggedParticle]:
    """
    Convert a Pythia hard event into clean leading-vertex TaggedParticles.
    """
    clean_particles: list[TaggedParticle] = []

    for i in range(hard_event.size()):
        p = hard_event[i]

        if not _is_stable_visible(p):
            continue

        clean_particles.append(
            TaggedParticle(
                px=float(p.px()),
                py=float(p.py()),
                pz=float(p.pz()),
                e=float(p.e()),
                is_lv=True,
                charge=float(p.charge()),
                pdg_id=int(p.id()),
            )
        )

    return clean_particles


def _copy_tagged_particles(particles: list[TaggedParticle]) -> list[TaggedParticle]:
    """
    Deep-copy TaggedParticle objects.

    This prevents pileup overlay from mutating or aliasing the clean LV list.
    """
    return [
        TaggedParticle(
            px=p.px,
            py=p.py,
            pz=p.pz,
            e=p.e,
            is_lv=p.is_lv,
            charge=p.charge,
            pdg_id=p.pdg_id,
        )
        for p in particles
    ]


def _initial_results() -> dict[str, list]:
    """
    Create the accumulator dictionary used by produce_images.
    """
    return {
        "ch_charged_lv": [],
        "ch_charged_pu": [],
        "ch_neutral_all": [],
        "ch_neutral_all_raw": [],
        "ch_neutral_lv": [],
        "clean_neutral_lv": [],
        "clean_neutral_all": [],
        "jet_pt": [],
        "jet_eta": [],
        "jet_phi": [],
        "n_pu": [],
        "full_px": [],
        "full_py": [],
        "full_pz": [],
        "full_e": [],
        "full_charge": [],
        "full_is_lv": [],
        "full_n": [],
        "true_px": [],
        "true_py": [],
        "true_pz": [],
        "true_e": [],
        "true_n": [],
        "pileup_px": [],
        "pileup_py": [],
        "pileup_pz": [],
        "pileup_e": [],
        "pileup_n": [],
        "puppi_px": [],
        "puppi_py": [],
        "puppi_pz": [],
        "puppi_e": [],
        "puppi_n": [],
    }


def _append_packed(results: dict[str, list], prefix: str, packed: dict) -> None:
    """
    Append one packed constituent payload into ``results``.
    """
    results[f"{prefix}_px"].append(packed["px"])
    results[f"{prefix}_py"].append(packed["py"])
    results[f"{prefix}_pz"].append(packed["pz"])
    results[f"{prefix}_e"].append(packed["e"])
    results[f"{prefix}_n"].append(packed["n"])

def _append_full_packed(results: dict[str, list], packed: dict) -> None:
    """
    Append one packed full-event LV+PU payload into ``results``.
    """
    results["full_px"].append(packed["px"])
    results["full_py"].append(packed["py"])
    results["full_pz"].append(packed["pz"])
    results["full_e"].append(packed["e"])
    results["full_charge"].append(packed["charge"])
    results["full_is_lv"].append(packed["is_lv"])
    results["full_n"].append(packed["n"])


def produce_images(
    hard_event,
    jets: list,
    overlay: PileupOverlay,
    builder: JetImageBuilder,
    n_pu: int = 140,
    jet_pt_min: float = 100.0,
    output_path: Optional[str] = None,
    save_clean: bool = True,
    puppi_R0: float = 0.3,
    puppi_Rmin: float = 0.02,
    puppi_wcut: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Build PUMML/PUPPI image arrays for one hard event.

    Parameters
    ----------
    hard_event:
        Either a Pythia event or a stored ``lv_snapshot`` list.
    jets:
        Reference jets from the main reconstruction event loop.
    overlay:
        PileupOverlay instance.
    builder:
        JetImageBuilder instance.
    n_pu:
        Actual number of pileup events to overlay.
    jet_pt_min:
        Minimum pT for saving image-level jets.
    output_path:
        Optional `.npz` path. If provided, this function saves the result.
    save_clean:
        Whether to include clean LV-only reference images.
    puppi_R0, puppi_Rmin, puppi_wcut:
        PUPPI baseline parameters.

    Returns
    -------
    dict[str, np.ndarray]
        Final `.npz` payload for this event.
    """
    if isinstance(hard_event, list):
        clean_particles = tagged_from_snapshot(hard_event)
        pu_particles = _copy_tagged_particles(clean_particles)
        pu_particles.extend(overlay._get_pu_particles(n_pu))
    else:
        clean_particles = _clean_particles_from_hard_event(hard_event)
        pu_particles = overlay.overlay(hard_event, n_pu=n_pu)

    # Save the full LV+PU particle collection before PUPPI.
    # This is duplicated for each saved jet from this event.
    full_packed = _pack_tagged_particles(pu_particles)

    # Run PUPPI on the full LV+PU event.
    puppi_input = [
        PuppiParticle(
            px=p.px,
            py=p.py,
            pz=p.pz,
            e=p.e,
            charge=p.charge,
            is_lv=p.is_lv,
        )
        for p in pu_particles
    ]

    puppi_output = run_puppi(
        particles=puppi_input,
        n_pu=n_pu,
        R0=puppi_R0,
        Rmin=puppi_Rmin,
        w_cut=puppi_wcut,
    )

    # Cluster clean, full-pileup, and PUPPI collections once per event.
    lv_jets, cs_lv = _cluster(_tagged_to_pseudojets(clean_particles))
    full_jets, cs_full = _cluster(_tagged_to_pseudojets(pu_particles))
    puppi_jets, cs_puppi = _cluster(_puppi_to_pseudojets(puppi_output))

    results = _initial_results()

    for jet in jets:
        if jet.pt() < jet_pt_min:
            continue

        if abs(jet.eta()) >= 2.5:
            continue

        jet_eta = float(jet.eta())
        jet_phi = float(jet.phi())

        # PUMML input/target images from the pileup-overlaid event.
        imgs_pu = builder.build(jet_eta, jet_phi, pu_particles)

        results["ch_charged_lv"].append(imgs_pu["ch_charged_lv"])
        results["ch_charged_pu"].append(imgs_pu["ch_charged_pu"])
        results["ch_neutral_all"].append(imgs_pu["ch_neutral_all"])
        results["ch_neutral_all_raw"].append(imgs_pu["ch_neutral_all_raw"])
        results["ch_neutral_lv"].append(imgs_pu["ch_neutral_lv"])

        # Clean LV-only reference images.
        if save_clean:
            imgs_clean = builder.build(jet_eta, jet_phi, clean_particles)
            results["clean_neutral_lv"].append(imgs_clean["ch_neutral_lv"])
            results["clean_neutral_all"].append(imgs_clean["ch_neutral_all_raw"])

        results["jet_pt"].append(float(jet.pt()))
        results["jet_eta"].append(jet_eta)
        results["jet_phi"].append(jet_phi)
        results["n_pu"].append(int(n_pu))

        _append_full_packed(results, full_packed)

        # True constituents.
        matched = _match_jet(jet_eta, jet_phi, lv_jets)
        packed = (
            _pack_constituents(list(matched.constituents()))
            if matched is not None
            else _empty_constituents()
        )
        _append_packed(results, "true", packed)

        # Pileup constituents.
        matched = _match_jet(jet_eta, jet_phi, full_jets)
        packed = (
            _pack_constituents(list(matched.constituents()))
            if matched is not None
            else _empty_constituents()
        )
        _append_packed(results, "pileup", packed)

        # PUPPI constituents.
        matched = _match_jet(jet_eta, jet_phi, puppi_jets)
        packed = (
            _pack_constituents(list(matched.constituents()))
            if matched is not None
            else _empty_constituents()
        )
        _append_packed(results, "puppi", packed)

    if not results["jet_pt"]:
        final = empty_image_arrays(
            n_charged=builder.n_pixels_charged,
            n_neutral=builder.n_pixels_neutral,
        )
    else:
        final = {key: np.array(value) for key, value in results.items()}

    if output_path is not None:
        np.savez_compressed(output_path, **final)

        n_jets = len(final["jet_pt"])

        print(
            f"[pumml_jet_images] Saved {n_jets} jets -> {output_path}\n"
            f"  ch_neutral_all   : {final['ch_neutral_all'].shape} (contaminated)\n"
            f"  ch_neutral_lv    : {final['ch_neutral_lv'].shape} (PUMML target)\n"
            f"  clean_neutral_all: {final['clean_neutral_all'].shape} (True curve)\n"
            f"  full particles   : {final['full_px'].shape}\n"
            f"  true consts      : {final['true_px'].shape}\n"
            f"  pileup consts    : {final['pileup_px'].shape}\n"
            f"  puppi consts     : {final['puppi_px'].shape}"
        )

    return final