#!/usr/bin/env python
"""
flows/runner.py
===============

Entry point for mixed-resolution image-only PileFlow.

PileFlow consumes an existing generator image file:

    jets_*_pileup_images.npz

Model inputs:
    ch_neutral_all_raw : 9x9   -> 81
    ch_charged_pu      : 36x36 -> 1296
    ch_charged_lv      : 36x36 -> 1296

The charged images retain their native 36x36 resolution. They are flattened
directly without pooling.

Total context dimension:
    81 + 1296 + 1296 = 2673

Flow target:
    ch_neutral_lv : 9x9 -> 81

The optional generator .npy table is never supplied to PileFlow. It may still
be provided for compatibility with the current comparison code.

Example
-------
python -m flows.runner \
    --skip-gen \
    --data-npz path/to/jets_..._pileup_images.npz \
    --data-npy path/to/jets_...npy \
    --outdir runs/pileflow_native36 \
    --device mps

Pipeline stages
---------------
Stage 1 - Load existing image data
Stage 2 - Train or load PileFlow
Stage 3 - Generate neutral-LV predictions and optionally compare results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .models.pileflow import (
    CHARGED_DIM,
    CHARGED_SIDE,
    NEUTRAL_DIM,
    NEUTRAL_SIDE,
    N_CONTEXT,
    N_TARGET,
)
from .training.train_flow import generate_and_save, train_pileflow


def _default_device() -> str:
    """Choose CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mixed-resolution image-only PileFlow using target conditional flow matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage toggles
    parser.add_argument("--skip-gen", action="store_true", help="Use existing generator output.")
    parser.add_argument("--skip-flow", action="store_true", help="Skip training and use --flow-ckpt.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip prediction generation and comparison.")

    # Input/output paths
    parser.add_argument("--data-npz", default=None, help="Generator image/constituent .npz file.")
    parser.add_argument("--data-npy", default=None, help="Optional .npy table used only by comparison code.")
    parser.add_argument("--flow-ckpt", default=None, help="Existing or output PileFlow checkpoint.")
    parser.add_argument("--pumml-ckpt", default=None, help="Optional PUMML checkpoint for comparisons.")

    # Metadata and debugging
    parser.add_argument("--process-name", default="ppjj", help="Process label.")
    parser.add_argument("--max-jets", type=int, default=None, help="Optional jet limit.")

    # PileFlow hyperparameters
    group = parser.add_argument_group("PileFlow hyperparameters")
    group.add_argument("--flow-epochs", type=int, default=800)
    group.add_argument("--flow-batch", type=int, default=512)
    group.add_argument("--flow-lr", type=float, default=1e-4)
    group.add_argument("--flow-hidden", type=int, default=512)
    group.add_argument("--flow-blocks", type=int, default=8)
    group.add_argument("--flow-time-emb", type=int, default=64)
    group.add_argument("--flow-sigma-min", type=float, default=1e-4)
    group.add_argument("--flow-dropout", type=float, default=0.1)
    group.add_argument("--flow-patience", type=int, default=60, help="Use 0 to disable early stopping.")

    # Generation and evaluation
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--ode-steps", type=int, default=100)

    # Generic
    parser.add_argument("--device", default=_default_device(), help="Torch device.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="output")

    return parser

def _abs_path(path: str | None) -> str | None:
    """Convert an optional path to an absolute path."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _ensure_output_dirs(outdir: str) -> None:
    """Create standard PileFlow output directories."""
    for subdir in ("checkpoints", "plots", "data"):
        Path(outdir, subdir).mkdir(parents=True, exist_ok=True)


def _validate_image_data(npz_path: str | None) -> str:
    """Validate the generator image file required by PileFlow."""
    if not npz_path:
        raise ValueError(
            "--data-npz is required. Run the generator first and pass its "
            "jets_*_pileup_images.npz output."
        )
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Generator image .npz file not found: {npz_path}")
    return npz_path


def _validate_optional_npy(npy_path: str | None) -> str | None:
    """Validate the optional comparison-only generator table."""
    if npy_path is None:
        return None
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Comparison-only generator .npy file not found: {npy_path}")
    return npy_path


def _validate_config_contract(cfg: Config) -> None:
    """Ensure Config and the model module describe the same architecture."""
    expected = {
        "neutral_side": NEUTRAL_SIDE,
        "charged_side": CHARGED_SIDE,
        "neutral_dim": NEUTRAL_DIM,
        "charged_dim": CHARGED_DIM,
        "context_dim": N_CONTEXT,
        "target_dim": N_TARGET,
    }
    actual = {
        "neutral_side": cfg.image_size,
        "charged_side": cfg.charged_image_size,
        "neutral_dim": cfg.neutral_dim,
        "charged_dim": cfg.charged_dim,
        "context_dim": cfg.context_dim,
        "target_dim": cfg.n_target,
    }

    if actual != expected:
        raise ValueError(
            "Config/model contract mismatch.\n"
            f"Expected: {expected}\n"
            f"Found:    {actual}"
        )


def _print_header(cfg: Config) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Mixed-resolution image-only PileFlow | device={cfg.device}")
    print(f"  outdir      = {cfg.outdir}")
    print(
        f"  context_dim = {cfg.context_dim} "
        f"({cfg.neutral_dim} neutral + {cfg.charged_dim} charged-PU + "
        f"{cfg.charged_dim} charged-LV)"
    )
    print(f"  target_dim  = {cfg.n_target} ({cfg.image_size}x{cfg.image_size} neutral-LV)")
    print(
        "  channels    = ch_neutral_all_raw (9x9), "
        "ch_charged_pu (36x36), ch_charged_lv (36x36)"
    )

    if cfg.max_jets is not None:
        print(f"  max_jets    = {cfg.max_jets:,}")

    if cfg.pumml_ckpt:
        print(f"  PUMML ckpt  = {cfg.pumml_ckpt} [comparison only]")
    else:
        print("  PUMML ckpt  = not provided [comparison skipped]")

    if cfg.flow_patience > 0:
        print(f"  early stop  = patience {cfg.flow_patience} epochs")
    else:
        print("  early stop  = disabled")

    print(f"{'=' * 72}\n")


def _mean_npu(npz_path: str) -> float:
    """Read the average generated pileup count for plot labels."""
    with np.load(npz_path, allow_pickle=False) as data:
        if "n_pu" in data.files:
            return float(data["n_pu"].mean())
    return 50.0


def main() -> None:
    args = build_parser().parse_args()

    data_npz = _abs_path(args.data_npz)
    comparison_npy = _abs_path(args.data_npy)
    flow_ckpt_arg = _abs_path(args.flow_ckpt)
    pumml_ckpt = _abs_path(args.pumml_ckpt)

    cfg = Config(
        outdir=args.outdir,
        process_name=args.process_name,
        skip_gen=args.skip_gen,
        skip_flow=args.skip_flow,
        skip_eval=args.skip_eval,
        data_npz=data_npz,
        flow_ckpt=flow_ckpt_arg,
        pumml_ckpt=pumml_ckpt,
        max_jets=args.max_jets,
        flow_epochs=args.flow_epochs,
        flow_batch=args.flow_batch,
        flow_lr=args.flow_lr,
        flow_hidden=args.flow_hidden,
        flow_blocks=args.flow_blocks,
        flow_time_emb=args.flow_time_emb,
        flow_sigma_min=args.flow_sigma_min,
        flow_dropout=args.flow_dropout,
        flow_patience=args.flow_patience,
        eval_batch=args.eval_batch,
        device=args.device,
        seed=args.seed,
    )

    _validate_config_contract(cfg)
    _ensure_output_dirs(cfg.outdir)
    _print_header(cfg)

    # Stage 1: load existing generator data
    if not cfg.skip_gen:
        raise ValueError(
            "flows/runner.py does not generate collision data. "
            "Run generator/ first, then use --skip-gen --data-npz <path>."
        )

    npz_path = _validate_image_data(cfg.data_npz)
    comparison_npy = _validate_optional_npy(comparison_npy)

    print("[Stage 1/3] Using existing generator image data")
    print(f"  model input .npz: {npz_path}")

    if comparison_npy is not None:
        print(f"  comparison .npy : {comparison_npy} [not used by PileFlow]")

    print()

    # Stage 2: train or load PileFlow
    flow_ckpt = cfg.flow_ckpt or os.path.join(
        cfg.outdir,
        "checkpoints",
        "pileflow_best.pt",
    )

    if not cfg.skip_flow:
        print("[Stage 2/3] Training mixed-resolution PileFlow ...")
        print(
            f"  Context: {cfg.context_dim}-dim "
            f"({cfg.image_size}x{cfg.image_size} neutral + two "
            f"{cfg.charged_image_size}x{cfg.charged_image_size} charged images)"
        )
        print(f"  Target : {cfg.n_target}-dim neutral-LV 9x9 image")

        train_pileflow(
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
        )

        print(f"  Checkpoint: {flow_ckpt}\n")
    else:
        if not os.path.isfile(flow_ckpt):
            raise FileNotFoundError(
                "--skip-flow requires an existing mixed-resolution checkpoint. "
                f"Not found: {flow_ckpt}"
            )

        print("[Stage 2/3] Skipped training - using checkpoint:")
        print(f"  {flow_ckpt}\n")

    # Stage 3: generate predictions and optionally compare
    if cfg.skip_eval:
        print("[Stage 3/3] Skipped generation/evaluation.\n")
    else:
        print("[Stage 3/3] Generating neutral-LV images ...")

        results = generate_and_save(
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
            n_steps=args.ode_steps,
            out_dir=os.path.join(cfg.outdir, "data"),
        )

        n_generated = results["neutral_lv_pred"].shape[0]
        print(f"  Generated {n_generated:,} jets\n")

        if comparison_npy is None:
            print(
                "  Comparison plots skipped because --data-npy was not provided. "
                "Prediction generation completed successfully."
            )
        else:
            print("  Running comparison plots ...")

            try:
                from comparison.observable_comparison import run_comparison

                run_comparison(
                    npz_path=npz_path,
                    npy_path=comparison_npy,
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

    print(f"{'=' * 72}")
    print(f"  Done. Outputs in: {cfg.outdir}/")
    print("  checkpoints/  pileflow_best.pt, *_history.npz")
    print("  plots/        pileflow_loss.png and optional comparison plots")
    print("  data/         generated_jets.npz")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()