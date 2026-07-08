from __future__ import annotations

import numpy as np
import fastjet as fj

from .algorithm import run_puppi
from .particles import Particle, unpack_particles


def _to_pseudojets(particles: list[Particle]) -> list:
    """
    Convert PUPPI Particle objects into FastJet PseudoJets.
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
):
    """
    Match a reference jet axis to the closest jet within dr_max.
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


def pack_pseudojets(
    pseudojets: list,
    max_const: int = 500,
) -> dict[str, np.ndarray | np.int32]:
    """
    Pack FastJet constituents into fixed-length arrays.
    """
    n = min(len(pseudojets), max_const)

    px = np.zeros(max_const, dtype=np.float32)
    py = np.zeros(max_const, dtype=np.float32)
    pz = np.zeros(max_const, dtype=np.float32)
    e = np.zeros(max_const, dtype=np.float32)

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


def _empty_packed(max_const: int = 500) -> dict[str, np.ndarray | np.int32]:
    return {
        "px": np.zeros(max_const, dtype=np.float32),
        "py": np.zeros(max_const, dtype=np.float32),
        "pz": np.zeros(max_const, dtype=np.float32),
        "e": np.zeros(max_const, dtype=np.float32),
        "n": np.int32(0),
    }


def _require_keys(data: np.lib.npyio.NpzFile, keys: list[str]) -> None:
    missing = [key for key in keys if key not in data.files]

    if missing:
        raise ValueError(
            "Input .npz is missing required keys for standalone PUPPI: "
            f"{missing}"
        )


def run_puppi_on_npz(
    npz_path: str,
    output_path: str | None = None,
    n_pu_override: int | None = None,
    R0: float = 0.3,
    Rmin: float = 0.02,
    w_cut: float = 0.1,
    eta_tracker: float = 2.5,
    max_const: int = 500,
    jet_R: float = 0.4,
) -> dict[str, np.ndarray]:
    """
    Run standalone PUPPI on generator `.npz` output.

    This mimics the generator-side saved PUPPI output:
        full_* arrays
            -> run PUPPI
            -> cluster PUPPI particles
            -> match to stored jet_eta / jet_phi
            -> save matched PUPPI jet constituents
    """
    data = np.load(npz_path, allow_pickle=False)

    required = [
        "full_px",
        "full_py",
        "full_pz",
        "full_e",
        "full_charge",
        "full_is_lv",
        "full_n",
        "n_pu",
        "jet_eta",
        "jet_phi",
    ]

    _require_keys(data, required)

    n_items = len(data["full_n"])

    puppi_px = np.zeros((n_items, max_const), dtype=np.float32)
    puppi_py = np.zeros((n_items, max_const), dtype=np.float32)
    puppi_pz = np.zeros((n_items, max_const), dtype=np.float32)
    puppi_e = np.zeros((n_items, max_const), dtype=np.float32)
    puppi_n = np.zeros(n_items, dtype=np.int32)

    for i in range(n_items):
        n_pu = int(n_pu_override if n_pu_override is not None else data["n_pu"][i])

        particles = unpack_particles(
            px=data["full_px"][i],
            py=data["full_py"][i],
            pz=data["full_pz"][i],
            e=data["full_e"][i],
            charge=data["full_charge"][i],
            is_lv=data["full_is_lv"][i],
            n=int(data["full_n"][i]),
        )

        mitigated = run_puppi(
            particles=particles,
            n_pu=n_pu,
            R0=R0,
            Rmin=Rmin,
            w_cut=w_cut,
            eta_tracker=eta_tracker,
        )

        pjs = _to_pseudojets(mitigated)
        puppi_jets, cs_puppi = _cluster(pjs, R=jet_R)

        matched = _match_jet(
            ref_eta=float(data["jet_eta"][i]),
            ref_phi=float(data["jet_phi"][i]),
            jets=puppi_jets,
            dr_max=jet_R,
        )

        packed = (
            pack_pseudojets(list(matched.constituents()), max_const=max_const)
            if matched is not None
            else _empty_packed(max_const=max_const)
        )

        puppi_px[i] = packed["px"]
        puppi_py[i] = packed["py"]
        puppi_pz[i] = packed["pz"]
        puppi_e[i] = packed["e"]
        puppi_n[i] = packed["n"]

        if (i + 1) % 1000 == 0 or (i + 1) == n_items:
            print(f"[PUPPI] {i + 1}/{n_items} rows done")

    output = {
        "puppi_px": puppi_px,
        "puppi_py": puppi_py,
        "puppi_pz": puppi_pz,
        "puppi_e": puppi_e,
        "puppi_n": puppi_n,
    }

    if output_path is not None:
        np.savez_compressed(output_path, **output)
        print(f"[PUPPI] Saved {output_path}")

    return output


def compare_puppi_outputs(
    generator_npz_path: str,
    standalone_npz_path: str,
) -> dict[str, float]:
    """
    Compare generator-side PUPPI arrays against standalone PUPPI arrays.
    """
    gen = np.load(generator_npz_path, allow_pickle=False)
    standalone = np.load(standalone_npz_path, allow_pickle=False)

    keys = ["puppi_px", "puppi_py", "puppi_pz", "puppi_e", "puppi_n"]

    out: dict[str, float] = {}

    for key in keys:
        if key not in gen.files:
            raise ValueError(f"Generator file missing {key}")
        if key not in standalone.files:
            raise ValueError(f"Standalone file missing {key}")

        diff = np.asarray(gen[key]) - np.asarray(standalone[key])
        out[f"{key}_max_abs_diff"] = float(np.max(np.abs(diff)))

    return out