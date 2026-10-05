"""Train PileFlow and evaluate it on the held-out split.

    python -m pileflow.run --data DATA.npz --outdir runs/pileflow_v2 --arch spatial_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import PileFlowImages, SplitIndices, check_splits, describe_split_mode, make_splits
from .evaluate import evaluate_model
from .model import PRESETS, build_model, count_parameters, model_config_from_preset
from .preprocess import ImagePreprocessor
from .train import train_model


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_sampling_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("sampling")
    g.add_argument("--eval-samples", type=int, default=1,
                   help="Samples per jet. >1 enables posterior mean/median and coverage metrics.")
    g.add_argument("--sample-seed", type=int, default=2024)
    g.add_argument("--ode-solver", choices=["euler", "midpoint", "rk4", "dopri5"], default="dopri5")
    g.add_argument("--ode-steps", type=int, default=50, help="Steps for fixed-step solvers.")
    g.add_argument("--ode-atol", type=float, default=1e-5)
    g.add_argument("--ode-rtol", type=float, default=1e-5)
    g.add_argument("--no-plots", action="store_true")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="PileFlow/PUMML NPZ dataset")
    p.add_argument("--outdir", default="runs/pileflow")
    p.add_argument("--arch", choices=sorted(PRESETS), default="spatial_v2")
    p.add_argument("--device", default=default_device())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-jets", type=int, default=None,
                   help="Debug only: use the first N jets (predefined split is intersected, not dropped).")
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=400)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--patience", type=int, default=60)
    g.add_argument("--sigma-min", type=float, default=1e-4)
    g.add_argument("--alpha", type=float, default=1.0)
    g.add_argument("--ema-decay", type=float, default=0.999, help="0 disables the weight EMA.")
    g.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
    g.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau")
    g.add_argument("--init-from", default=None,
                   help="Warm-start the weights from this checkpoint (same architecture).")
    g.add_argument("--preprocess", choices=["log1p", "linear"], default="log1p")
    g.add_argument("--target-mode", choices=["log1p", "fraction"], default="fraction",
                   help="fraction: neutral-LV fraction of the total neutral pT per cell with dequantised "
                        "zeros (default); log1p: draft parameterisation.")
    g.add_argument("--zero-width", type=float, default=0.1)
    g.add_argument("--frac-scale", type=float, default=2.0)
    g.add_argument("--base-channels", type=int, default=None, help="Override the preset width.")
    g.add_argument("--n-blocks", type=int, default=None, help="Override the preset depth.")
    g.add_argument("--pt-scale", type=float, default=1.0)
    g.add_argument("--eval-fraction", type=float, default=0.10,
                   help="Only used when the file has no event_train_idx/event_test_idx.")
    g.add_argument("--val-fraction-of-train", type=float, default=0.10)
    g.add_argument("--splits-file", default=None,
                   help="NPZ with train/val/eval index arrays to use instead of building the split "
                        "(e.g. to reuse an earlier checkpoint's split).")
    g.add_argument("--morph-lambda", type=float, default=0.0,
                   help="Weight of the low-activity morphology auxiliary loss (0 = pure flow matching).")
    g.add_argument("--morph-threshold", type=float, default=0.1, help="GeV; defines low-activity target cells.")
    g.add_argument("--morph-tmin", type=float, default=0.5, help="Apply the morphology loss only for t >= tmin.")
    g.add_argument("--eval-splits", nargs="*", default=["val", "eval"], choices=["val", "eval"],
                   help="Splits to sample after training (val is needed for threshold selection).")
    add_sampling_args(p)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = PileFlowImages(args.data, max_jets=args.max_jets)
    if args.splits_file:
        with np.load(args.splits_file) as f:
            splits = SplitIndices.from_dict({k: f[k] for k in ("train", "val", "eval")})
        check_splits(splits, dataset.event_id)
        if dataset.has_predefined_event_split and set(splits.eval.tolist()) != set(dataset.eval_idx.tolist()):
            raise RuntimeError("--splits-file eval split differs from the dataset's event_test_idx")
    else:
        splits = make_splits(dataset, seed=args.seed, eval_fraction=args.eval_fraction,
                             val_fraction_of_train=args.val_fraction_of_train)
    print(f"Split ({describe_split_mode(dataset)}): train={len(splits.train)} "
          f"val={len(splits.val)} eval={len(splits.eval)}")

    overrides = {"base_channels": args.base_channels, "n_blocks": args.n_blocks} \
        if PRESETS[args.arch]["kind"] == "spatial" else {}
    model_config = model_config_from_preset(args.arch, **overrides)
    model = build_model(model_config)
    print(f"Architecture: {args.arch} {model_config}\nTrainable parameters: {count_parameters(model):,}")

    pre = ImagePreprocessor(mode=args.preprocess, pt_scale=args.pt_scale, target_mode=args.target_mode,
                            zero_width=args.zero_width, frac_scale=args.frac_scale)
    model, pre, history, ckpt = train_model(
        model, model_config, dataset, splits, pre, outdir, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience, sigma_min=args.sigma_min,
        alpha=args.alpha, seed=args.seed, morph_lambda=args.morph_lambda,
        morph_threshold=args.morph_threshold, morph_tmin=args.morph_tmin,
        ema_decay=args.ema_decay, amp=not args.no_amp, lr_schedule=args.lr_schedule,
        init_state=(torch.load(args.init_from, map_location="cpu", weights_only=False)["model_state"]
                    if args.init_from else None),
        extra_metadata={"data_path": str(Path(args.data).resolve()), "max_jets": args.max_jets,
                        "split_mode": describe_split_mode(dataset), "n_jets": len(dataset)})

    _loss_plot(history, outdir / "plots")
    for split in args.eval_splits:
        evaluate_model(model, dataset, getattr(splits, split), pre, outdir, split_name=split,
                       device=args.device, batch_size=args.batch_size, solver=args.ode_solver,
                       ode_steps=args.ode_steps, atol=args.ode_atol, rtol=args.ode_rtol,
                       eval_samples=args.eval_samples, sigma_min=args.sigma_min, alpha=args.alpha,
                       sample_seed=args.sample_seed, checkpoint_path=ckpt,
                       make_plots=not args.no_plots)


def _loss_plot(history, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"], label="train")
    ax.plot(ep, history["val"], label="validation (fixed noise)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Flow-matching loss"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "loss.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
