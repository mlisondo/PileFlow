#!/usr/bin/env python3
"""PUMML reproduction check: w. Pileup / PUPPI / PUMML vs the paper tables.

Uses exactly the same observables and statistics as compare_methods.py
(pileflow.benchmark), so reproduction numbers and the PileFlow comparison are
consistent. Run from the repository root:

    python -m pumml.evaluate --data DATA.npz --model runs/pumml/pumml_event.pt \
        --split event --output-dir results/pumml_repro
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pileflow import benchmark as bm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=["event", "image", "all"], default="event")
    p.add_argument("--output-dir", default="results/pumml_repro")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--cut-mode", choices=bm.CUT_MODES, default="both",
                   help="both (default): evaluate without and with the neutral-cell cut; off / on: one only")
    p.add_argument("--cell-threshold", type=float, default=bm.DEFAULT_CELL_THRESHOLD,
                   help="GeV, applied to every method's neutral image in the cut variant")
    p.add_argument("--threshold-truth", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)
    if args.split == "all":
        print("WARNING: --split all includes training jets (debug only)")

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = bm.load_eval_data(args.data, args.split)
    reps = bm.base_representations(d)
    reps["pumml"] = (d["charged_lv36"], bm.pumml_predict(args.model, d["X"], device)[0])
    methods = [m for m in ("with_pileup", "puppi", "pumml") if m in reps]
    variants = []
    if args.cut_mode in ("both", "off"):
        variants.append(("nocut", 0.0, reps))
    if args.cut_mode in ("both", "on"):
        variants.append((bm.cut_tag(args.cell_threshold), args.cell_threshold,
                         bm.apply_cell_cut(reps, args.cell_threshold, args.threshold_truth)))
    values = {}
    for tag, c, vreps in variants:
        title = "NO neutral-cell cut" if c == 0 else f"Neutral-cell cut {1000 * c:.0f} MeV on all methods"
        values[tag], _ = bm.run_variant(d, vreps, methods, out / tag, args.bootstrap_samples, 12345, "pumml",
                                        paper=True, dpi=args.dpi, extra={"split": args.split,
                                                                         "cell_threshold_GeV": c}, title=title)
    if len(variants) == 2:
        bm.compare_cut_variants(values["nocut"], values[variants[1][0]], methods, out, args.cell_threshold,
                                args.bootstrap_samples, threshold_truth=args.threshold_truth, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
