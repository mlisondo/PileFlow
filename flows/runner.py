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

Evaluation sampling:
    PileFlow can generate multiple independent neutral-LV samples for each
    conditioning input. These samples are averaged pixel-by-pixel before
    computing detector images, jet observables, plots, and metrics.

    --eval-samples 1 preserves the original single-sample behavior.

Example
-------
python -m flows.runner \
    --skip-gen \
    --data-npz path/to/jets_..._pileup_images.npz \
    --data-npy path/to/jets_...npy \
    --outdir runs/pileflow_native36 \
    --eval-samples 10 \
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

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed-resolution image-only PileFlow using "
            "target conditional flow matching"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage toggles
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Use existing generator output.",
    )
    parser.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip training and use --flow-ckpt.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip prediction generation and comparison.",
    )

    # Input/output paths
    parser.add_argument(
        "--data-npz",
        default=None,
        help="Generator image/constituent .npz file.",
    )
    parser.add_argument(
        "--data-npy",
        default=None,
        help="Optional .npy table used only by comparison code (training set).",
    )
    parser.add_argument(
        "--eval-npz",
        default=None,
        help=(
            "Held-out evaluation .npz file.  When provided, physics metrics are "
            "computed on this dataset instead of the training .npz.  "
            "Recommended: use the *_eval.npz split generated alongside *_train.npz."
        ),
    )
    parser.add_argument(
        "--eval-npy",
        default=None,
        help="Optional .npy table for the held-out evaluation set (comparison plots).",
    )
    parser.add_argument(
        "--flow-ckpt",
        default=None,
        help="Existing or output PileFlow checkpoint.",
    )
    parser.add_argument(
        "--pumml-ckpt",
        default=None,
        help="Optional PUMML checkpoint for comparisons.",
    )

    # Metadata and debugging
    parser.add_argument(
        "--process-name",
        default="ppjj",
        help="Process label.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Optional jet limit.",
    )

    # PileFlow hyperparameters
    group = parser.add_argument_group(
        "PileFlow hyperparameters"
    )
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
        default=120,
        help="Use 0 to disable early stopping.",
    )

    # Minor improvement flags
    improve = parser.add_argument_group(
        "Minor improvements (all enabled by default)"
    )
    improve.add_argument(
        "--use-log-pt",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply log(1+pT) normalization to the target image.",
    )
    improve.add_argument(
        "--use-minibatch-ot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use minibatch OT noise coupling (requires scipy).",
    )
    improve.add_argument(
        "--use-ema",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Maintain an EMA of model weights during training.",
    )
    improve.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate (used only when --use-ema).",
    )
    improve.add_argument(
        "--pt-threshold",
        type=float,
        default=0.0,
        help=(
            "Pixel pT threshold in GeV applied at decode time. "
            "Pixels below this value are zeroed. 0 = disabled."
        ),
    )
    improve.add_argument(
        "--use-cnn-context",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Use standalone CNN context encoder (superseded by --use-transformer). "
            "Only active when --no-use-transformer is also passed."
        ),
    )
    improve.add_argument(
        "--cnn-neutral-feat",
        type=int,
        default=128,
        help="CNN output features for the 9x9 neutral channel (standalone CNN encoder).",
    )
    improve.add_argument(
        "--cnn-charged-feat",
        type=int,
        default=256,
        help="CNN output features for each 36x36 charged channel (standalone CNN encoder).",
    )

    # Patch-token set transformer
    pst = parser.add_argument_group(
        "Patch-token set transformer (patch-per-pixel, global attention)"
    )
    pst.add_argument(
        "--use-set-transformer",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Replace the velocity field with a patch-token set transformer. "
            "81 tokens, one per output pixel; each token receives a 4×4 patch "
            "from each charged channel + one neutral pixel + explicit (η,φ). "
            "Full global SAB attention O(81²) by default; use --st-n-inducing "
            "for ISAB (O(N·m))."
        ),
    )
    pst.add_argument(
        "--st-d-model",
        type=int,
        default=256,
        help="Set-transformer hidden dimension (must be divisible by 4 and n_heads).",
    )
    pst.add_argument(
        "--st-n-heads",
        type=int,
        default=4,
        help="Number of attention heads per SAB/ISAB block.",
    )
    pst.add_argument(
        "--st-n-layers",
        type=int,
        default=6,
        help="Number of SAB or ISAB blocks.",
    )
    pst.add_argument(
        "--st-n-inducing",
        type=int,
        default=0,
        help=(
            "Number of inducing points per ISAB block. "
            "0 = use full SAB (global O(N²) attention). "
            ">0 = use ISAB (O(N·m) attention)."
        ),
    )

    # U-Net velocity field
    unet = parser.add_argument_group(
        "U-Net velocity field (drastic change)"
    )
    unet.add_argument(
        "--use-unet",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Replace the flat residual MLP with a U-Net velocity field. "
            "Processes charged images at native 36×36, encodes 36→18→9, "
            "merges with neutral+z_t at 9×9, bottleneck at 4×4 with skip. "
            "Pass --no-use-unet to revert to the flat ResNet."
        ),
    )
    unet.add_argument(
        "--unet-base-channels",
        type=int,
        default=64,
        help=(
            "Base channel count C for the U-Net. "
            "Encoder uses C→2C→4C; bottleneck uses 8C. "
            "Use 32 for speed (~1M params), 64 for quality (~4M params)."
        ),
    )

    # Transformer velocity field
    xfm = parser.add_argument_group(
        "Transformer velocity field (drastic change)"
    )
    xfm.add_argument(
        "--use-transformer",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Replace the flat residual MLP with a Transformer velocity field. "
            "Each output pixel cross-attends to all 243 context tokens. "
            "Pass --no-use-transformer to revert to the flat ResNet."
        ),
    )
    xfm.add_argument(
        "--transformer-d-model",
        type=int,
        default=256,
        help="Transformer hidden dimension.",
    )
    xfm.add_argument(
        "--transformer-n-heads",
        type=int,
        default=4,
        help="Number of attention heads (must evenly divide --transformer-d-model).",
    )
    xfm.add_argument(
        "--transformer-n-self-attn",
        type=int,
        default=2,
        help="Number of self-attention layers over the 243 context tokens.",
    )
    xfm.add_argument(
        "--transformer-n-cross-attn",
        type=int,
        default=4,
        help="Number of cross-attention layers (81 target tokens × 243 context tokens).",
    )
    xfm.add_argument(
        "--transformer-cnn-channels",
        type=int,
        default=32,
        help="CNN feature channels used to map each image channel to 9×9 tokens.",
    )

    # Generation and evaluation
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=512,
        help="Jet batch size used during generation.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=8,
        help=(
            "Number of independent PileFlow samples generated per jet. "
            "The decoded images are averaged pixel-by-pixel before "
            "observable evaluation. Use 1 for the original behavior."
        ),
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="Number of Euler steps used for each generated sample.",
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
    """Convert an optional path to an absolute path."""
    if path is None:
        return None

    return str(
        Path(path)
        .expanduser()
        .resolve()
    )


def _ensure_output_dirs(outdir: str) -> None:
    """Create standard PileFlow output directories."""
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
    """Validate the generator image file required by PileFlow."""
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
    """Validate the optional comparison-only generator table."""
    if npy_path is None:
        return None

    if not os.path.isfile(npy_path):
        raise FileNotFoundError(
            "Comparison-only generator .npy file not found: "
            f"{npy_path}"
        )

    return npy_path


def _validate_config_contract(
    cfg: Config,
) -> None:
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


def _print_header(
    cfg: Config,
) -> None:
    print(f"\n{'=' * 72}")
    print(
        "  Mixed-resolution image-only PileFlow "
        f"| device={cfg.device}"
    )
    print(
        f"  outdir       = {cfg.outdir}"
    )
    print(
        f"  context_dim  = {cfg.context_dim} "
        f"({cfg.neutral_dim} neutral + "
        f"{cfg.charged_dim} charged-PU + "
        f"{cfg.charged_dim} charged-LV)"
    )
    print(
        f"  target_dim   = {cfg.n_target} "
        f"({cfg.image_size}x{cfg.image_size} neutral-LV)"
    )
    print(
        "  channels     = ch_neutral_all_raw (9x9), "
        "ch_charged_pu (36x36), "
        "ch_charged_lv (36x36)"
    )
    print(
        f"  eval samples = {cfg.eval_samples} per jet"
    )

    if cfg.eval_samples == 1:
        print(
            "  aggregation  = single generated image "
            "[original behavior]"
        )
    else:
        print(
            "  aggregation  = pixelwise mean of generated images "
            "before observables"
        )

    if cfg.max_jets is not None:
        print(
            f"  max_jets     = {cfg.max_jets:,}"
        )

    if cfg.pumml_ckpt:
        print(
            f"  PUMML ckpt   = {cfg.pumml_ckpt} "
            "[comparison only]"
        )
    else:
        print(
            "  PUMML ckpt   = not provided "
            "[comparison skipped]"
        )

    if cfg.flow_patience > 0:
        print(
            "  early stop   = patience "
            f"{cfg.flow_patience} epochs"
        )
    else:
        print(
            "  early stop   = disabled"
        )

    print(f"{'=' * 72}\n")


def _mean_npu(
    npz_path: str,
) -> float:
    """Read the average generated pileup count for plot labels."""
    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:
        if "n_pu" in data.files:
            return float(
                data["n_pu"].mean()
            )

    return 50.0


def main() -> None:
    args = build_parser().parse_args()

    data_npz = _abs_path(args.data_npz)
    comparison_npy = _abs_path(args.data_npy)
    eval_npz = _abs_path(args.eval_npz)
    eval_npy = _abs_path(args.eval_npy)
    flow_ckpt_arg = _abs_path(
        args.flow_ckpt
    )
    pumml_ckpt = _abs_path(
        args.pumml_ckpt
    )

    cfg = Config(
        outdir=args.outdir,
        process_name=args.process_name,
        skip_gen=args.skip_gen,
        skip_flow=args.skip_flow,
        skip_eval=args.skip_eval,
        data_npz=data_npz,
        eval_npz=eval_npz,
        eval_npy=eval_npy,
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
        use_log_pt=args.use_log_pt,
        use_minibatch_ot=args.use_minibatch_ot,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        pt_threshold=args.pt_threshold,
        use_cnn_context=args.use_cnn_context,
        cnn_neutral_feat=args.cnn_neutral_feat,
        cnn_charged_feat=args.cnn_charged_feat,
        use_set_transformer=args.use_set_transformer,
        st_d_model=args.st_d_model,
        st_n_heads=args.st_n_heads,
        st_n_layers=args.st_n_layers,
        st_n_inducing=args.st_n_inducing,
        use_unet=args.use_unet,
        unet_base_channels=args.unet_base_channels,
        use_transformer=args.use_transformer,
        transformer_d_model=args.transformer_d_model,
        transformer_n_heads=args.transformer_n_heads,
        transformer_n_self_attn=args.transformer_n_self_attn,
        transformer_n_cross_attn=args.transformer_n_cross_attn,
        transformer_cnn_channels=args.transformer_cnn_channels,
        eval_batch=args.eval_batch,
        eval_samples=args.eval_samples,
        ode_steps=args.ode_steps,
        device=args.device,
        seed=args.seed,
    )

    _validate_config_contract(
        cfg
    )
    _ensure_output_dirs(
        cfg.outdir
    )
    _print_header(
        cfg
    )

    # Stage 1: load existing generator data
    if not cfg.skip_gen:
        raise ValueError(
            "flows/runner.py does not generate collision data. "
            "Run generator/ first, then use "
            "--skip-gen --data-npz <path>."
        )

    npz_path = _validate_image_data(
        cfg.data_npz
    )
    comparison_npy = _validate_optional_npy(
        comparison_npy
    )

    print(
        "[Stage 1/3] Using existing generator image data"
    )
    print(
        f"  model input .npz: {npz_path}"
    )

    if comparison_npy is not None:
        print(
            f"  comparison .npy : {comparison_npy} "
            "[not used by PileFlow]"
        )

    print()

    # Stage 2: train or load PileFlow
    flow_ckpt = (
        cfg.flow_ckpt
        or os.path.join(
            cfg.outdir,
            "checkpoints",
            "pileflow_best.pt",
        )
    )

    if not cfg.skip_flow:
        print(
            "[Stage 2/3] Training mixed-resolution PileFlow ..."
        )
        print(
            f"  Context: {cfg.context_dim}-dim "
            f"({cfg.image_size}x{cfg.image_size} neutral + two "
            f"{cfg.charged_image_size}x"
            f"{cfg.charged_image_size} charged images)"
        )
        print(
            f"  Target : {cfg.n_target}-dim "
            "neutral-LV 9x9 image"
        )
        print(
            "  Training samples per target: 1"
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
                "--skip-flow requires an existing "
                "mixed-resolution checkpoint. "
                f"Not found: {flow_ckpt}"
            )

        print(
            "[Stage 2/3] Skipped training - using checkpoint:"
        )
        print(
            f"  {flow_ckpt}\n"
        )

    # Stage 3: generate predictions and optionally compare
    if cfg.skip_eval:
        print(
            "[Stage 3/3] Skipped generation/evaluation.\n"
        )

    else:
        if cfg.eval_samples == 1:
            print(
                "[Stage 3/3] Generating one neutral-LV "
                "image per jet ..."
            )
        else:
            print(
                "[Stage 3/3] Generating and averaging "
                f"{cfg.eval_samples} neutral-LV samples per jet ..."
            )

        # Use held-out eval set when available; warn if falling back to train set.
        gen_npz = cfg.eval_npz or npz_path
        gen_npy = cfg.eval_npy or comparison_npy

        results = generate_and_save(
            npz_path=npz_path,
            flow_ckpt=flow_ckpt,
            cfg=cfg,
            n_steps=cfg.ode_steps,
            out_dir=os.path.join(
                cfg.outdir,
                "data",
            ),
            eval_npz=cfg.eval_npz,
        )

        n_generated = results[
            "neutral_lv_pred"
        ].shape[0]

        print(
            f"  Generated predictions for "
            f"{n_generated:,} jets"
        )
        print(
            f"  Samples per jet: {cfg.eval_samples}\n"
        )

        if gen_npy is None:
            print(
                "  Comparison plots skipped because no .npy table was provided "
                "(use --eval-npy or --data-npy). Prediction generation "
                "completed successfully."
            )

        else:
            print(
                "  Running comparison plots ..."
            )

            try:
                from comparison.observable_comparison import (
                    run_comparison,
                )

                run_comparison(
                    npz_path=gen_npz,
                    npy_path=gen_npy,
                    pumml_ckpt=cfg.pumml_ckpt,
                    results=results,
                    cfg=cfg,
                    mean_npu=_mean_npu(
                        npz_path
                    ),
                )

            except Exception as exc:
                print(
                    f"  [skipped] comparison plots: {exc}"
                )

                import traceback

                traceback.print_exc()

            print(
                "  Plots -> "
                f"{os.path.join(cfg.outdir, 'plots')}/"
            )

    print(
        f"{'=' * 72}"
    )
    print(
        f"  Done. Outputs in: {cfg.outdir}/"
    )
    print(
        "  checkpoints/  pileflow_best.pt, *_history.npz"
    )
    print(
        "  plots/        pileflow_loss.png and optional "
        "comparison plots"
    )
    print(
        "  data/         generated_jets.npz"
    )
    print(
        f"{'=' * 72}\n"
    )


if __name__ == "__main__":
    main()