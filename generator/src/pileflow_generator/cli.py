"""
Command-line interface for the PileFlow generator.
"""

from __future__ import annotations

import argparse
import traceback

from pileflow_generator.config import WorkflowConfig
from pileflow_generator.pipeline import execute_workflow


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="PileFlow generator: MadGraph -> Pythia8 -> FastJet workflow"
    )

    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")

    parser.add_argument("--mg5-path", type=str, default=None, help="Path to mg5_aMC executable.")
    parser.add_argument("--lhe-file", type=str, default=None, help="Path to .lhe or .lhe.gz file.")
    parser.add_argument("--process-name", type=str, default=None, help="Logical process name used for output saving.")
    parser.add_argument("--output-dir", type=str, default="data", help="Base output directory.")

    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to process with Pythia.")
    parser.add_argument("--jet-pt-min", type=float, default=15.0, help="Minimum reconstructed jet pT in GeV.")
    parser.add_argument(
        "--image-pt-min",
        type=float,
        default=15.0,
        help="Minimum jet pT for PUMML image building in GeV.",
    )
    parser.add_argument("--min-hard-parton-pt", type=float, default=0.0, help="Minimum hardness proxy in GeV.")

    parser.add_argument("--rng-seed", type=int, default=42, help="NumPy random seed.")
    parser.add_argument("--pythia-seed", type=int, default=42, help="Pythia random seed.")

    parser.add_argument("--save-figures", action="store_true", help="Save plots.")
    parser.add_argument("--save-feynman-diagrams", action="store_true", help="Copy MG5 diagrams if available.")
    parser.add_argument("--max-event-figures", type=int, default=30, help="Maximum number of event figures.")
    parser.add_argument("--max-scatter-points-global", type=int, default=20000, help="Maximum scatter points in global plots.")

    parser.add_argument("--auto-mg5", action="store_true", help="Run MG5 automatically using a generated script.")
    parser.add_argument("--mg5-process-command", type=str, default="generate p p > j j", help="MG5 process command.")
    parser.add_argument("--mg5-nevents", type=int, default=1000, help="MG5 run_card nevents.")
    parser.add_argument("--mg5-ptj", type=float, default=None, help="MG5 run_card ptj cut in GeV.")

    parser.add_argument(
        "--n-pu",
        type=int,
        default=50,
        help="Mean number of pileup vertices overlaid per event. Set to 0 for no pileup overlay.",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    """
    Convert parsed command-line arguments into a WorkflowConfig.
    """
    config = WorkflowConfig()

    if args.mg5_path is not None:
        config.mg5_path = args.mg5_path

    config.lhe_file = args.lhe_file
    config.process_name = args.process_name
    config.output_dir = args.output_dir

    config.n_events = args.n_events
    config.jet_pt_min = args.jet_pt_min
    config.image_pt_min = args.image_pt_min
    config.min_hard_parton_pt = args.min_hard_parton_pt

    config.rng_seed = args.rng_seed
    config.pythia_seed = args.pythia_seed

    config.save_figures = args.save_figures
    config.save_feynman_diagrams = args.save_feynman_diagrams
    config.max_event_figures = args.max_event_figures
    config.max_scatter_points_global = args.max_scatter_points_global

    config.use_mg5_auto = args.auto_mg5
    config.mg5_process_command = args.mg5_process_command
    config.mg5_nevents = args.mg5_nevents
    config.mg5_ptj = args.mg5_ptj

    config.n_pu = args.n_pu

    # Preserve old behavior: anti-kT with R = 0.4 only.
    config.jet_config.R = 0.4
    config.jet_config.algo_name = "antikt"
    config.jet_config.algo_code = 1

    return config


def run_from_args(args: argparse.Namespace) -> None:
    """
    Run the workflow from parsed command-line arguments.
    """
    try:
        config = config_from_args(args)
        execute_workflow(config)

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")

    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()


def run_interactive() -> None:
    """
    Run the workflow in interactive mode.

    Keep this only if you actually use it. For reproducible research runs,
    command-line or YAML config is better.
    """
    try:
        from pileflow_generator.stages.madgraph import MadGraphRunner
    except ImportError:
        MadGraphRunner = None

    try:
        config = WorkflowConfig()
        config.interactive = True

        mg5_default = config.mg5_path
        mg5_in = input(f"MG5 path [{mg5_default}]: ").strip()
        if mg5_in:
            config.mg5_path = mg5_in

        auto_in = input("Use automatic MG5 mode with scripted run_card edits? [y/N]: ").strip().lower()
        config.use_mg5_auto = auto_in in {"y", "yes"}

        if config.use_mg5_auto:
            process_name = input("MadGraph process/output name [my_process]: ").strip()
            config.process_name = process_name if process_name else "my_process"

            proc_cmd = input("MG5 process command [generate p p > j j]: ").strip()
            if proc_cmd:
                config.mg5_process_command = proc_cmd

            mg5_nev = input(f"MG5 run_card nevents [{config.mg5_nevents}]: ").strip()
            if mg5_nev:
                config.mg5_nevents = int(mg5_nev)

            mg5_ptj = input("MG5 run_card ptj [leave blank for none]: ").strip()
            if mg5_ptj:
                config.mg5_ptj = float(mg5_ptj)

        else:
            use_mg = input("Launch MadGraph interactively first? [y/N]: ").strip().lower()

            if use_mg in {"y", "yes"}:
                if MadGraphRunner is None:
                    raise ImportError(
                        "MadGraphRunner is not available. Migrate old src/generator.py first."
                    )

                mg_runner = MadGraphRunner(config.mg5_path, config.work_dir)
                mg_runner.run_interactive()

                process_name = input("MadGraph output directory name: ").strip()
                if not process_name:
                    raise ValueError("You must provide a process name.")

                config.process_name = process_name

                found_lhe = mg_runner.find_lhe_file(process_name)
                if found_lhe:
                    print(f"Found LHE file: {found_lhe}")
                    config.lhe_file = found_lhe
                else:
                    config.lhe_file = input("Full path to .lhe or .lhe.gz: ").strip()

            else:
                config.process_name = input("Process name: ").strip()
                config.lhe_file = input("Full path to .lhe or .lhe.gz: ").strip()

        n_events_in = input(f"Number of Pythia events to process [{config.n_events}]: ").strip()
        if n_events_in:
            config.n_events = int(n_events_in)

        jet_pt_in = input(f"Minimum reconstructed jet pT [{config.jet_pt_min} GeV]: ").strip()
        if jet_pt_in:
            config.jet_pt_min = float(jet_pt_in)

        image_pt_in = input(f"Minimum jet pT for PUMML images [{config.image_pt_min} GeV]: ").strip()
        if image_pt_in:
            config.image_pt_min = float(image_pt_in)

        hard_in = input(f"Minimum hardness proxy [{config.min_hard_parton_pt} GeV]: ").strip()
        if hard_in:
            config.min_hard_parton_pt = float(hard_in)

        out_in = input(f"Output directory [{config.output_dir}]: ").strip()
        if out_in:
            config.output_dir = out_in

        n_pu_in = input(f"Mean number of pileup vertices [{config.n_pu}]: ").strip()
        if n_pu_in:
            config.n_pu = int(n_pu_in)

        figs_in = input("Save figures? [y/N]: ").strip().lower()
        config.save_figures = figs_in in {"y", "yes"}

        diag_in = input("Copy Feynman diagrams if available? [y/N]: ").strip().lower()
        config.save_feynman_diagrams = diag_in in {"y", "yes"}

        config.jet_config.R = 0.4
        config.jet_config.algo_name = "antikt"
        config.jet_config.algo_code = 1

        execute_workflow(config)

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")

    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()


def main() -> None:
    """
    CLI entry point.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_from_args(args)