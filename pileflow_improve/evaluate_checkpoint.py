#!/usr/bin/env python3
"""Sample a trained PileFlow checkpoint on the val or eval split.

Architecture, preprocessing, flow settings and the train/val/eval split are
read from the checkpoint, so evaluation cannot silently use a different
configuration (or leak training jets into the evaluation set).

    python evaluate_checkpoint.py --data DATA.npz \
        --checkpoint runs/pileflow_v2/checkpoints/pileflow_best.pt \
        --split val  --eval-samples 8 --outdir runs/pileflow_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pileflow.data import PileFlowImages, SplitIndices, check_splits, make_splits
from pileflow.evaluate import evaluate_model
from pileflow.model import PRESETS, count_parameters, model_config_from_preset
from pileflow.run import add_sampling_args, default_device
from pileflow.train import load_checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--split", choices=["val", "eval"], default="eval")
    p.add_argument("--device", default=default_device())
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--legacy-arch", choices=sorted(PRESETS), default="spatial",
                   help="Architecture for draft checkpoints that did not store model_config.")
    p.add_argument("--seed", type=int, default=42,
                   help="Only used to rebuild splits for draft checkpoints without stored splits.")
    add_sampling_args(p)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    model, pre, flow_cfg, ckpt = load_checkpoint(
        args.checkpoint, device=args.device, fallback_model_config=model_config_from_preset(args.legacy_arch))
    print(f"Loaded {args.checkpoint}\n  model: {ckpt['model_config']} ({count_parameters(model):,} params)"
          f"\n  preprocess: {pre.get_config()}\n  flow: {flow_cfg}")

    max_jets = (ckpt.get("metadata") or {}).get("max_jets")
    dataset = PileFlowImages(args.data, max_jets=max_jets)
    if "splits" in ckpt:
        splits = SplitIndices.from_dict(ckpt["splits"])
        if max(int(np.max(v)) for v in splits.as_dict().values()) >= len(dataset):
            raise ValueError("Checkpoint splits index beyond this dataset: wrong --data file?")
        check_splits(splits, dataset.event_id)
    else:
        print("WARNING: checkpoint stores no splits; rebuilding them (cannot guarantee they match training)")
        splits = make_splits(dataset, seed=args.seed)
    if dataset.has_predefined_event_split and args.split == "eval" \
            and not np.array_equal(np.sort(splits.eval), np.sort(dataset.eval_idx)):
        raise RuntimeError("Checkpoint eval split differs from the dataset's event_test_idx")

    evaluate_model(model, dataset, getattr(splits, args.split), pre, Path(args.outdir),
                   split_name=args.split, device=args.device, batch_size=args.batch_size,
                   solver=args.ode_solver, ode_steps=args.ode_steps, atol=args.ode_atol,
                   rtol=args.ode_rtol, eval_samples=args.eval_samples, sample_seed=args.sample_seed,
                   checkpoint_path=args.checkpoint, make_plots=not args.no_plots, **flow_cfg)


if __name__ == "__main__":
    main()
