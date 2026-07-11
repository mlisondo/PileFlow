#!/usr/bin/env python
"""
flows/runner.py
===============

Entry point for image-only PileFlow.

PileFlow consumes an existing generator image file:

    jets_*_pileup_images.npz

The model uses only these three input channels:

    ch_neutral_all_raw
    ch_charged_pu
    ch_charged_lv

The charged images are sum-pooled from 36x36 to 9x9. All three images are
then flattened and concatenated into a 243-dimensional context vector.

The flow generates only:

    ch_neutral_lv, flattened from 9x9 to 81 dimensions

The optional generator `.npy` table is not used by PileFlow. It may still be
provided temporarily for the existing comparison code.

Example
-------
Run from the repository root:

    python -m flows.runner \
        --skip-gen \
        --data-npz path/to/jets_..._pileup_images.npz \
        --data-npy path/to/jets_...npy \
        --outdir runs/pileflow_image_only \
        --device mps

Pipeline stages
---------------
Stage 1 — Load existing image data
Stage 2 — Train or load image-only PileFlow
Stage 3 — Generate neutral-LV predictions and optionally compare results
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .training.train_flow import generate_and_save, train_pileflow


def _default_device() -> str:
    """
    Choose CUDA, then Apple MPS, then CPU.
    """
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Image-only PileFlow pileup mitigation using "
            "target conditional flow matching"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage toggles
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help=(
            "Use an existing generator image file. Data generation is "
            "handled by generator/, not flows/."
        ),
    )

    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip training and use an existing --flow-ckpt.",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip prediction generation and comparison plots.",
    )

    # Input/output paths
    parser.add_argument(
        "--data-npz",
        default=None,
        help=(
            "Generator image/constituent .npz file. This is the only "
            "dataset file used by the PileFlow model."
        ),
    )

    parser.add_argument(
        "--data-npy",
        default=None,
        help=(
            "Optional generator jet-feature table used only by the existing "
            "comparison code. It is never supplied to PileFlow."
        ),
    )

    parser.add_argument(
        "--flow-ckpt",
        default=None,
        help="Existing or output image-only PileFlow checkpoint.",
    )

    parser.add_argument(
        "--pumml-ckpt",
        default=None,
        help=(
            "Optional external PUMML checkpoint used only for comparison "
            "plots. If omitted, the PUMML column is skipped."
        ),
    )

    # Metadata and debugging
    parser.add_argument(
        "--process-name",
        default="ppjj",
        help="Process label used for metadata and logging only.",
    )

    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Optional jet limit for smoke tests and debugging.",
    )

    # PileFlow hyperparameters
    group = parser.add_argument_group("PileFlow hyperparameters")

    group.add_argument(
        "--flow-epochs",
        type=int,
        default=800,
    )

    group.add_argument(
        "--flow-batch",
        type=int,
        default=512,
    )

    group.add_argument(
        "--flow-lr",
        type=float,
        default=1e-4,
    )

    group.add_argument(
        "--flow-hidden",
        type=int,
        default=512,
    )

    group.add_argument(
        "--flow-blocks",
        type=int,
        default=8,
    )

    group.add_argument(
        "--flow-time-emb",
        type=int,
        default=64,
        help="Sinusoidal flow-time embedding dimension.",
    )

    group.add_argument(
        "--flow-sigma-min",
        type=float,
        default=1e-4,
    )

    group.add_argument(
        "--flow-dropout",
        type=float,
        default=0.1,
    )

    group.add_argument(
        "--flow-patience",
        type=int,
        default=60,
        help=(
            "Stop after this many epochs without validation improvement. "
            "Use 0 to disable early stopping."
        ),
    )

    # Generation and evaluation
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=512,
        help="Batch size for PileFlow prediction generation.",
    )

    parser.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="Euler integration steps used during flow generation.",
    )

    # Generic
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Torch device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--outdir",
        default="output",
    )

    return parser


def _abs_path(path: str | None) -> str | None:
    """
    Convert an optional path to an absolute path.
    """
    if path is None:
        return None

    return str(
        Path(path).expanduser().resolve()
    )


def _ensure_output_dirs(outdir: str) -> None:
    """
    Create the standard PileFlow output directories.
    """
    for subdir in (
        "checkpoints",
        "plots",
        "data",
    ):
        Path(
            outdir,
            subdir,
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def _validate_image_data(
    npz_path: str | None,
) -> str:
    """
    Validate the image file required by image-only PileFlow.
    """
    if not npz_path:
        raise ValueError(
            "--data-npz is required. Run the generator first and pass its "
            "jets_*_pileup_images.npz output."
        )

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Generator image .npz file not found: {npz_path}"
        )

    return npz_path


def _validate_optional_npy(
    npy_path: str | None,
) -> str | None:
    """
    Validate the optional comparison-only generator table.
    """
    if npy_path is None:
        return None

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(
            f"Comparison-only generator .npy file not found: {npy_path}"
        )

    return npy_path


def _print_header(cfg: Config) -> None:
    print(f"\n{'=' * 68}")
    print(
        f"  Image-only PileFlow | "
        f"device={cfg.device} | "
        f"outdir={cfg.outdir}"
    )

    print(
        f"  context_dim = {cfg.context_dim} "
        f"(3 image channels × 81 pixels)"
    )

    print(
        f"  target_dim  = {cfg.n_target} "
        f"(81 neutral-LV pixels)"
    )

    print(
        "  channel order:"
        "\n    1. ch_neutral_all_raw"
        "\n    2. ch_charged_pu"
        "\n    3. ch_charged_lv"
    )

    if cfg.max_jets is not None:
        print(
            f"  max_jets    = {cfg.max_jets:,}"
        )

    if cfg.pumml_ckpt:
        print(
            f"  PUMML ckpt  = {cfg.pumml_ckpt} "
            "[comparison only]"
        )
    else:
        print(
            "  PUMML ckpt  = not provided "
            "[PUMML comparison skipped]"
        )

    print(
        f"  early stop  = patience "
        f"{cfg.flow_patience} epochs"
    )

    print(f"{'=' * 68}\n")


def _mean_npu(npz_path: str) -> float:
    """
    Read the average generated pileup count for plot labels.
    """
    data = np.load(
        npz_path,
        allow_pickle=False,
    )

    try:
        if "n_pu" in data.files:
            return float(
                data["n_pu"].mean()
            )

        return 50.0

    finally:
        data.close()


def main() -> None:
    args = build_parser().parse_args()

    data_npz = _abs_path(
        args.data_npz,
    )

    # This file is retained only for the current comparison implementation.
    # It is never supplied to the model, context encoder, or target processor.
    comparison_npy = _abs_path(
        args.data_npy,
    )

    flow_ckpt_arg = _abs_path(
        args.flow_ckpt,
    )

    pumml_ckpt = _abs_path(
        args.pumml_ckpt,
    )

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

    _ensure_output_dirs(
        cfg.outdir,
    )

    _print_header(
        cfg,
    )

    # ------------------------------------------------------------------
    # Stage 1: load existing generator image data
    # ------------------------------------------------------------------

    if not cfg.skip_gen:
        raise ValueError(
            "flows/runner.py does not generate collision data. "
            "Run generator/ first, then use --skip-gen --data-npz <path>."
        )

    npz_path = _validate_image_data(
        cfg.data_npz,
    )

    comparison_npy = _validate_optional_npy(
        comparison_npy,
    )

    print(
        "[Stage 1/3] Using existing generator image data\n"
        f"  model input .npz: {npz_path}"
    )

    if comparison_npy is not None:
        print(
            f"  comparison .npy : {comparison_npy} "
            "[not used by PileFlow]"
        )

    print()

    # ------------------------------------------------------------------
    # Stage 2: train or load PileFlow
    # ------------------------------------------------------------------

    flow_ckpt = cfg.flow_ckpt or os.path.join(
        cfg.outdir,
        "checkpoints",
        "pileflow_best.pt",
    )

    if not cfg.skip_flow:
        print(
            "[Stage 2/3] Training image-only PileFlow ..."
        )

        print(
            f"  Context: {cfg.context_dim}-dim "
            "(three flattened 9x9 image channels)"
        )

        print(
            f"  Target : {cfg.n_target}-dim "
            "(neutral-LV 9x9 image only)"
        )

        train_pileflow(
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
        )

        print(
            f"  Checkpoint: {flow_ckpt}\n"
        )

    else:
        if not os.path.isfile(flow_ckpt):
            raise FileNotFoundError(
                "--skip-flow requires an existing image-only checkpoint. "
                f"Not found: {flow_ckpt}"
            )

        print(
            "[Stage 2/3] Skipped training — using checkpoint:\n"
            f"  {flow_ckpt}\n"
        )

    # ------------------------------------------------------------------
    # Stage 3: generate predictions and optionally compare
    # ------------------------------------------------------------------

    if cfg.skip_eval:
        print(
            "[Stage 3/3] Skipped generation/evaluation.\n"
        )

    else:
        print(
            "[Stage 3/3] Generating neutral-LV images ..."
        )

        results = generate_and_save(
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
            n_steps=args.ode_steps,
            out_dir=os.path.join(
                cfg.outdir,
                "data",
            ),
        )

        n_generated = results[
            "neutral_lv_pred"
        ].shape[0]

        print(
            f"  Generated {n_generated:,} jets\n"
        )

        # The current comparison script may still use the generator .npy
        # table for evaluation metadata. That table is not a model input.
        if comparison_npy is None:
            print(
                "  Comparison plots skipped because --data-npy was not "
                "provided. Prediction generation completed successfully."
            )

        else:
            print(
                "  Running comparison plots ..."
            )

            try:
                from comparison.observable_comparison import run_comparison

                run_comparison(
                    npz_path=npz_path,
                    npy_path=comparison_npy,
                    pumml_ckpt=cfg.pumml_ckpt,
                    results=results,
                    cfg=cfg,
                    mean_npu=_mean_npu(
                        npz_path,
                    ),
                )

            except Exception as exc:
                print(
                    f"  [skipped] comparison plots: {exc}"
                )

                import traceback

                traceback.print_exc()

            print(
                f"  Plots -> "
                f"{os.path.join(cfg.outdir, 'plots')}/"
            )

    print(f"{'=' * 68}")
    print(
        f"  Done. Outputs in: {cfg.outdir}/"
    )
    print(
        "  checkpoints/  pileflow_best.pt, *_history.npz"
    )
    print(
        "  plots/        pileflow_loss.png and optional comparison plots"
    )
    print(
        "  data/         generated_jets.npz"
    )
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()