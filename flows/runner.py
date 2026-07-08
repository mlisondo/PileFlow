#!/usr/bin/env python
"""
flows/runner.py
===============

Single entry point for running the PileFlow model package.

This runner does not generate collision data. It consumes existing generator
outputs:

    jets_*.npy
    jets_*_pileup_images.npz

The generator should be run first from `generator/`. Then this file trains,
generates, and optionally evaluates the PileFlow model.

Run from the repository root:

    python -m flows.runner \
        --skip-gen \
        --data-npy path/to/jets_..._antikt_R0.4.npy \
        --data-npz path/to/jets_..._antikt_R0.4_pileup_images.npz \
        --outdir data/flows/exp1 \
        --device cpu

Pipeline stages
---------------
Stage 1 — Load generator data
Stage 2 — Train PileFlow
Stage 3 — Generate mitigated jets and optionally run comparison plots

PileFlow input context
----------------------
253-dimensional context vector:

    7      generator-level scalar features
    3      jet-flavour one-hot labels
    81     neutral-all 9x9 image
    81     charged-pileup 9x9 image
    81     charged-LV 9x9 image

PileFlow target
---------------
97-dimensional target vector:

    81     neutral-LV 9x9 image
    16     reconstructed scalar jet observables
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .training.train_flow import generate_and_save, train_pileflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PileFlow — pileup mitigation via target conditional flow matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage toggles
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help=(
            "Required for flows/runner.py. Data generation is handled by "
            "generator/, not by flows/."
        ),
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip PileFlow training and use an existing --flow-ckpt.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Stage 3 generation and comparison plots.",
    )

    # Input/output paths
    parser.add_argument(
        "--data-npy",
        default=None,
        help="Path to generator jet-feature table, shape (N, 25).",
    )
    parser.add_argument(
        "--data-npz",
        default=None,
        help="Path to generator image/constituent file.",
    )
    parser.add_argument(
        "--flow-ckpt",
        default=None,
        help="Path to an existing or output PileFlow checkpoint.",
    )
    parser.add_argument(
        "--pumml-ckpt",
        default=None,
        help=(
            "Optional external PUMML checkpoint. Used only for comparison plots. "
            "If omitted, the PUMML column is skipped."
        ),
    )

    # Metadata / debugging
    parser.add_argument(
        "--process-name",
        default="ppjj",
        help="Process label used for metadata/logging only.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Optional cap on the number of jets for smoke tests/debugging.",
    )

    # PileFlow training hyperparameters
    group = parser.add_argument_group("PileFlow hyperparameters")
    group.add_argument("--flow-epochs", type=int, default=800)
    group.add_argument("--flow-batch", type=int, default=512)
    group.add_argument("--flow-lr", type=float, default=1e-4)
    group.add_argument("--flow-hidden", type=int, default=512)
    group.add_argument("--flow-blocks", type=int, default=8)
    group.add_argument("--flow-sigma-min", type=float, default=1e-4)
    group.add_argument("--flow-dropout", type=float, default=0.1)
    group.add_argument(
        "--flow-patience",
        type=int,
        default=60,
        help="Early stop after N epochs with no validation improvement. Use 0 to disable.",
    )

    # Generation / evaluation
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=512,
        help="Batch size for PileFlow generation/evaluation.",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="Number of Euler integration steps for PileFlow generation.",
    )

    # Generic
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="output")

    return parser


def _abs_path(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _ensure_output_dirs(outdir: str) -> None:
    for subdir in ("checkpoints", "plots", "data"):
        Path(outdir, subdir).mkdir(parents=True, exist_ok=True)


def _validate_input_data(npy_path: str | None, npz_path: str | None) -> tuple[str, str]:
    if not npy_path or not npz_path:
        raise ValueError(
            "--skip-gen requires both --data-npy and --data-npz. "
            "Run generator first, then pass its output files here."
        )

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Data .npy not found: {npy_path}")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Data .npz not found: {npz_path}")

    return npy_path, npz_path


def _print_header(cfg: Config) -> None:
    print(f"\n{'=' * 60}")
    print(f"  PileFlow | device={cfg.device} | outdir={cfg.outdir}")
    print(
        f"  context_dim = {cfg.context_dim} "
        f"(7 gen + 3 flavour + 3×81 image channels)"
    )
    print(
        f"  target_dim  = {cfg.n_target} "
        f"(81 neutral LV + 16 scalar observables)"
    )

    if cfg.max_jets is not None:
        print(f"  max_jets    = {cfg.max_jets:,}")

    if cfg.pumml_ckpt:
        print(f"  PUMML ckpt  = {cfg.pumml_ckpt} [comparison only]")
    else:
        print("  PUMML ckpt  = not provided [PUMML column skipped]")

    print(f"  early stop  = patience {cfg.flow_patience} epochs")
    print(f"{'=' * 60}\n")


def _mean_npu(npz_path: str) -> float:
    data = np.load(npz_path, allow_pickle=False)
    try:
        return float(data["n_pu"].mean()) if "n_pu" in data.files else 50.0
    finally:
        data.close()


def main() -> None:
    args = build_parser().parse_args()

    data_npy = _abs_path(args.data_npy)
    data_npz = _abs_path(args.data_npz)
    flow_ckpt_arg = _abs_path(args.flow_ckpt)
    pumml_ckpt = _abs_path(args.pumml_ckpt)

    cfg = Config(
        outdir=args.outdir,
        process_name=args.process_name,
        skip_gen=args.skip_gen,
        skip_flow=args.skip_flow,
        skip_eval=args.skip_eval,
        data_npy=data_npy,
        data_npz=data_npz,
        flow_ckpt=flow_ckpt_arg,
        pumml_ckpt=pumml_ckpt,
        max_jets=args.max_jets,
        flow_epochs=args.flow_epochs,
        flow_batch=args.flow_batch,
        flow_lr=args.flow_lr,
        flow_hidden=args.flow_hidden,
        flow_blocks=args.flow_blocks,
        flow_sigma_min=args.flow_sigma_min,
        flow_dropout=args.flow_dropout,
        flow_patience=args.flow_patience,
        eval_batch=args.eval_batch,
        device=args.device,
        seed=args.seed,
    )

    _ensure_output_dirs(cfg.outdir)
    _print_header(cfg)

    # Stage 1: Load existing generator data.
    if not cfg.skip_gen:
        raise ValueError(
            "flows/runner.py does not generate data. "
            "Run generator first, then rerun this command with "
            "--skip-gen --data-npy <path> --data-npz <path>."
        )

    npy_path, npz_path = _validate_input_data(cfg.data_npy, cfg.data_npz)

    print(
        "[Stage 1/3] Using existing generator data\n"
        f"  .npy: {npy_path}\n"
        f"  .npz: {npz_path}\n"
    )

    # Stage 2: Train or load PileFlow checkpoint.
    flow_ckpt = cfg.flow_ckpt or os.path.join(
        cfg.outdir,
        "checkpoints",
        "pileflow_best.pt",
    )

    if not cfg.skip_flow:
        print("[Stage 2/3] Training PileFlow ...")
        print(
            f"  Context: {cfg.context_dim}-dim "
            f"(gen scalars + flavour + 3 image channels)"
        )
        print(
            f"  Target : {cfg.n_target}-dim "
            f"(neutral LV image + scalar observables)"
        )

        train_pileflow(
            npy_path=npy_path,
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
        )

        print(f"  Checkpoint: {flow_ckpt}\n")

    else:
        if not os.path.isfile(flow_ckpt):
            raise FileNotFoundError(
                "--skip-flow requires an existing checkpoint. "
                f"Not found: {flow_ckpt}"
            )

        print(f"[Stage 2/3] Skipped training — using checkpoint:\n  {flow_ckpt}\n")

    # Stage 3: Generate PileFlow predictions and run comparison plots.
    if cfg.skip_eval:
        print("[Stage 3/3] Skipped generation/evaluation.\n")
    else:
        print("[Stage 3/3] Generating pileup-mitigated jets ...")

        results = generate_and_save(
            npy_path=npy_path,
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
            n_steps=args.ode_steps,
            out_dir=os.path.join(cfg.outdir, "data"),
        )

        n_generated = results["neutral_lv_pred"].shape[0]
        print(f"  Generated {n_generated:,} jets\n")

        print("  Running comparison plots ...")
        try:
            from comparison.observable_comparison import run_comparison

            run_comparison(
                npz_path=npz_path,
                npy_path=npy_path,
                pumml_ckpt=cfg.pumml_ckpt,
                results=results,
                cfg=cfg,
                mean_npu=_mean_npu(npz_path),
            )

        except Exception as exc:
            print(f"  [skipped] comparison plots: {exc}")
            import traceback

            traceback.print_exc()

        print(f"  Plots -> {os.path.join(cfg.outdir, 'plots')}/")

    print(f"{'=' * 60}")
    print(f"  Done. Outputs in: {cfg.outdir}/")
    print("  checkpoints/  pileflow_best.pt, *_history.npz")
    print("  plots/        pileflow_loss.png, figure4_distributions.{png,pdf},")
    print("                figure5_percent_errors.{png,pdf}, tables_1_2.{txt,csv}")
    print("  data/         generated_jets.npz")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()