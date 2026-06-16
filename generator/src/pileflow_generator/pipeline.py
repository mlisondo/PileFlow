"""
Main generator pipeline.

This module replaces the old ``src/workflow.py::execute_workflow`` function.
It coordinates the migrated stages without owning their internal logic.
"""

from __future__ import annotations

import os

from pileflow_generator.config import WorkflowConfig
from pileflow_generator.diagnostics.sanity import print_header, print_sanity
from pileflow_generator.io.paths import ensure_dir, timestamp_string
from pileflow_generator.io.readers import decompress_lhe_if_needed
from pileflow_generator.io.writers import save_outputs
from pileflow_generator.stages.clustering import FastJetRunner
from pileflow_generator.stages.images import JetImageBuilder
from pileflow_generator.stages.pileup import PileupOverlay
from pileflow_generator.stages.pythia import PythiaRunner


def _maybe_run_madgraph(config: WorkflowConfig) -> None:
    """
    Run MadGraph automatic mode if requested.

    This is isolated so fixed-LHE workflows still work even if the MadGraph
    stage has not been fully migrated yet.
    """
    if not config.use_mg5_auto:
        return

    try:
        from pileflow_generator.stages.madgraph import MadGraphRunner
    except ImportError as exc:
        raise ImportError(
            "Automatic MG5 mode was requested, but "
            "pileflow_generator.stages.madgraph is not available yet. "
            "Migrate old src/generator.py into stages/madgraph.py, or run from "
            "an existing --lhe-file."
        ) from exc

    if not config.process_name:
        raise ValueError("process_name is required when automatic MG5 mode is enabled.")

    mg_runner = MadGraphRunner(config.mg5_path, config.work_dir)

    mg_runner.run_automatic(
        process_name=config.process_name,
        process_command=config.mg5_process_command,
        mg5_nevents=config.mg5_nevents,
        mg5_ptj=config.mg5_ptj,
        extra_run_card_edits=config.mg5_run_card_edits,
    )

    found_lhe = mg_runner.find_lhe_file(config.process_name)

    if not found_lhe:
        raise FileNotFoundError("MG5 finished, but no LHE file could be found automatically.")

    config.lhe_file = found_lhe


def _maybe_collect_feynman_diagrams(config: WorkflowConfig, run_dir: str) -> None:
    """
    Copy MG5 Feynman diagrams if requested and if the MadGraph stage exists.
    """
    if not config.save_feynman_diagrams:
        return

    try:
        from pileflow_generator.stages.madgraph import MadGraphRunner
    except ImportError as exc:
        raise ImportError(
            "save_feynman_diagrams=True, but stages.madgraph is not available yet."
        ) from exc

    mg_runner = MadGraphRunner(config.mg5_path, config.work_dir)
    mg_runner.collect_feynman_diagrams(config.process_name, run_dir)


def execute_workflow(config: WorkflowConfig) -> tuple[str, str]:
    """
    Execute the full generator workflow.

    Returns
    -------
    tuple[str, str]
        ``(run_dir, npy_path)``.
    """
    print_header()
    ensure_dir(config.output_dir)

    _maybe_run_madgraph(config)

    if not config.process_name:
        raise ValueError("process_name is required.")

    if not config.lhe_file:
        raise ValueError("lhe_file is required.")

    config.lhe_file = decompress_lhe_if_needed(config.lhe_file)

    print("\n[1/3] Reading events with Pythia8...")

    pythia_runner = PythiaRunner(
        lhe_file=config.lhe_file,
        n_events=config.n_events,
        pythia_seed=config.pythia_seed,
        min_hard_parton_pt=config.min_hard_parton_pt,
    )

    stored_events = pythia_runner.read_events()
    print(f"Accepted stored events: {len(stored_events)}")

    # Pre-compute the run folder so the .npy and .npz outputs share it.
    ts = timestamp_string()
    cfg_key = f"antikt_R{config.jet_config.R:g}"

    run_dir = ensure_dir(
        os.path.join(config.output_dir, f"run_{config.process_name}_{ts}")
    )
    cfg_dir = ensure_dir(os.path.join(run_dir, cfg_key))

    npz_name = f"jets_{config.process_name}_{cfg_key}_pileup_images.npz"
    npz_path = os.path.join(cfg_dir, npz_name)

    pileup_overlay = PileupOverlay(pythia_seed=config.pythia_seed + 1)

    image_builder = JetImageBuilder(
        eta_range=0.45,
        phi_range=0.45,
        n_pixels_charged=36,
        n_pixels_neutral=9,
        pt_charged_cut=0.5,
    )

    print("\n[2/3] Clustering jets with FastJet...")

    fastjet_runner = FastJetRunner(
        jet_pt_min=config.jet_pt_min,
        jet_R=config.jet_config.R,
        rng_seed=config.rng_seed,
        pileup_overlay=pileup_overlay,
        image_builder=image_builder,
        n_pu=config.n_pu,
        image_pt_min=config.image_pt_min,
        image_output_path=npz_path,
    )

    dataset, event_figures = fastjet_runner.cluster_events(stored_events)

    print(f"Jets stored in dataset: {dataset.shape[0]}")
    print_sanity(dataset, cfg_key)

    print("\n[3/3] Saving outputs...")

    run_dir, npy_path = save_outputs(
        config=config,
        dataset=dataset,
        event_figures=event_figures,
        ts=ts,
        cfg_dir=cfg_dir,
    )

    _maybe_collect_feynman_diagrams(config, run_dir)

    print("\n" + "=" * 80)
    print("WORKFLOW FINISHED")
    print("=" * 80)
    print(f"Process name : {config.process_name}")
    print(f"LHE file     : {config.lhe_file}")
    print(f"Output dir   : {run_dir}")
    print(f"Dataset file : {npy_path}")
    print(f"N jets       : {dataset.shape[0]}")
    print("=" * 80)

    return run_dir, npy_path