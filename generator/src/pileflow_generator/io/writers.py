"""
Output writers for the PileFlow generator.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from pileflow_generator.config import WorkflowConfig
from pileflow_generator.io.paths import ensure_dir, timestamp_string
from pileflow_generator.schemas.jet_features import (
    FEATURE_NAMES,
    N_FEATURES,
    ALGO_CODE_TO_NAME,
)
from pileflow_generator.diagnostics.plots import (
    plot_global_dataset_figures,
    plot_event_jets_eta_phi,
)


def save_json(path: str, payload: dict[str, Any]) -> None:
    """
    Save a dictionary as pretty-printed JSON.
    """
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_text(path: str, text: str) -> None:
    """
    Save plain text to a file.
    """
    with open(path, "w") as f:
        f.write(text)


def save_outputs(
    config: WorkflowConfig,
    dataset: np.ndarray,
    event_figures: list[dict],
    ts: str | None = None,
    cfg_dir: str | None = None,
) -> tuple[str, str]:
    """
    Save the jet-feature dataset, metadata, preview, README, and optional plots.

    Parameters
    ----------
    config:
        Full workflow configuration.

    dataset:
        Jet-feature table with shape ``(N_jets, 25)``.

    event_figures:
        Event-level plotting records returned by ``FastJetRunner``.

    ts:
        Optional precomputed timestamp. Used so the `.npy` file and `.npz`
        image file share the same run directory.

    cfg_dir:
        Optional precomputed configuration directory, usually
        ``data/run_<process>_<timestamp>/antikt_R0.4``.

    Returns
    -------
    tuple[str, str]
        ``(run_dir, npy_path)``.
    """
    if ts is None:
        ts = timestamp_string()

    cfg_key = f"antikt_R{config.jet_config.R:g}"

    if cfg_dir is None:
        run_dir = ensure_dir(
            os.path.join(config.output_dir, f"run_{config.process_name}_{ts}")
        )
        cfg_dir = ensure_dir(os.path.join(run_dir, cfg_key))
    else:
        run_dir = os.path.dirname(cfg_dir)

    base = os.path.join(cfg_dir, f"jets_{config.process_name}_{cfg_key}")
    npy_path = f"{base}.npy"

    np.save(npy_path, dataset)

    metadata = {
        "timestamp": ts,
        "process": config.process_name,
        "lhe_file": config.lhe_file,
        "n_events_requested": int(config.n_events),
        "n_jets": int(dataset.shape[0]),
        "algorithm": "antikt",
        "algorithm_code": int(config.jet_config.algo_code),
        "algorithm_code_map": {str(k): v for k, v in ALGO_CODE_TO_NAME.items()},
        "R": float(config.jet_config.R),
        "jet_pt_min": float(config.jet_pt_min),
        "image_pt_min": float(config.image_pt_min),
        "min_hard_parton_pt_proxy": float(config.min_hard_parton_pt),
        "n_pu": int(config.n_pu),
        "n_features": int(N_FEATURES),
        "features": {str(i): name for i, name in enumerate(FEATURE_NAMES)},
        "feature_notes": {
            "flavour": "Absolute PDG ID only. Antiparticles are stored with positive IDs.",
            "reco": "Reco-like variables come from a simple parametric smearing.",
            "btag_ctag": "Proxy values built from flavour matching and constituent content.",
            "fractions": "Computed from final visible Pythia constituents.",
            "jetR": "Jet radius used in clustering.",
            "algoCode": "1 = anti-kT.",
            "jetArea": "Active area if available in FastJet Python bindings; else 0.",
        },
        "notes": {
            "visible_jets": "Neutrinos are excluded from clustering.",
            "algorithm_restriction": "This version only uses anti-kT with R=0.4.",
        },
    }

    save_json(f"{base}_metadata.json", metadata)

    preview_lines = [
        f"# {cfg_key} | {dataset.shape[0]} jets",
        f"# process = {config.process_name}",
        f"# jet_pt_min = {config.jet_pt_min} GeV",
        f"# image_pt_min = {config.image_pt_min} GeV",
        f"# min_hard_parton_pt_proxy = {config.min_hard_parton_pt} GeV",
        "# " + "  ".join(f"{i}:{name}" for i, name in enumerate(FEATURE_NAMES)),
    ]

    if dataset.shape[0] > 0:
        sample = "\n".join(
            " ".join(f"{x:10.4f}" for x in row) for row in dataset[:10]
        )
        preview_lines.append(sample)
    else:
        preview_lines.append("# Empty dataset")

    write_text(f"{base}_preview.txt", "\n".join(preview_lines) + "\n")

    if config.save_figures:
        figures_dir = ensure_dir(os.path.join(cfg_dir, "figures"))
        rng = np.random.default_rng(config.rng_seed)

        plot_global_dataset_figures(
            dataset=dataset,
            cfg_key=cfg_key,
            out_dir=figures_dir,
            max_scatter_points=config.max_scatter_points_global,
            rng=rng,
        )

        event_fig_dir = ensure_dir(os.path.join(cfg_dir, "event_figures"))

        for item in event_figures[: config.max_event_figures]:
            plot_event_jets_eta_phi(
                jets_eta_phi_pt=item["jets_eta_phi_pt"],
                cfg_key=cfg_key,
                source_event_idx=item["source_event_idx"],
                accepted_event_idx=item["accepted_event_idx"],
                out_dir=event_fig_dir,
            )

    readme_lines = [
        f"Run: {config.process_name}",
        f"LHE file: {config.lhe_file}",
        f"Events requested: {config.n_events}",
        f"Jets saved: {dataset.shape[0]}",
        "Algorithm: anti-kT",
        f"R: {config.jet_config.R}",
        f"Minimum jet pT: {config.jet_pt_min} GeV",
        f"Minimum image jet pT: {config.image_pt_min} GeV",
        f"Mean pileup vertices: {config.n_pu}",
        "",
        "Columns:",
    ]

    for i, name in enumerate(FEATURE_NAMES):
        readme_lines.append(f"  [{i:2d}] {name}")

    readme_lines.extend(
        [
            "",
            "Important note:",
            "  flavour uses absolute PDG ID values only.",
            "  anti-particles are stored with positive IDs.",
        ]
    )

    write_text(os.path.join(run_dir, "README.txt"), "\n".join(readme_lines) + "\n")

    return run_dir, npy_path