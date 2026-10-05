#!/usr/bin/env python3
"""PUMML-paper benchmark: w. Pileup / PUPPI / PUMML / PileFlow.

All methods are evaluated on the event-grouped held-out jets
(event_test_idx) with the observable definitions of pileflow.observables and
paired-bootstrap uncertainties, including the difference of every method
with respect to PUMML.

    python compare_methods.py --data DATA.npz \
        --pumml-model runs/pumml/pumml_event.pt \
        --pileflow runs/pileflow_v2/data/generated_eval.npz \
        --threshold-json runs/pileflow_v2/threshold.json \
        --outdir results/comparison

By default the benchmark is run twice (--cut-mode both):

  <outdir>/nocut/         no neutral-cell threshold for any method
  <outdir>/cut_100MeV/    the same --cell-threshold (default 0.1 GeV) on every method
  <outdir>/tables/cut_vs_nocut.{csv,json,tex}, <outdir>/plots/cut_vs_nocut_*.{png,pdf}

Each variant directory holds tables/{summary.json, correlation_table.{csv,tex},
iqr_table.{csv,tex}} and plots/fig4_*, fig5_*, fig6_*. Use --cut-mode off or
--cut-mode on to run only one variant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pileflow import benchmark as bm


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["event", "image"], default="event")
    p.add_argument("--pumml-model", required=True, help="PUMML checkpoint (paper recipe)")
    p.add_argument("--pumml-tuned-model", default=None, help="Optional val-selected PUMML checkpoint")
    p.add_argument("--pileflow", required=True, help="generated_eval.npz from PileFlow")
    p.add_argument("--pileflow-small", default=None, help="Optional parameter-matched PileFlow generated_eval.npz")
    p.add_argument("--pileflow-estimator", choices=["sample", "mean", "median"], default="sample",
                   help="'sample' = one stochastic draw per jet (default, honest generative output).")
    p.add_argument("--add-posterior-mean", action="store_true",
                   help="Also report the observables of the per-pixel mean image over the stored samples.")
    p.add_argument("--add-posterior-obs", action="store_true",
                   help="Also report PileFlow's posterior-mean observables: each observable averaged over "
                        "the K stored samples of the jet (needs --eval-samples > 1 at sampling time).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--threshold-json", default=None,
                   help="Optional PileFlow-only cut from select_threshold.py (chosen on val); used in the cut "
                        "variant only, and only if larger than --cell-threshold.")
    g.add_argument("--pileflow-threshold", type=float, default=0.0,
                   help="Same as --threshold-json but given directly in GeV (must come from validation data).")
    p.add_argument("--cut-mode", choices=bm.CUT_MODES, default="both",
                   help="Neutral-cell cut: 'both' (default) runs the benchmark without and with the cut and "
                        "writes a side-by-side comparison; 'off' = no cut only; 'on' = cut only.")
    p.add_argument("--cell-threshold", type=float, default=bm.DEFAULT_CELL_THRESHOLD,
                   help="Neutral-cell zero-suppression in GeV applied UNIFORMLY to every reconstructed "
                        "neutral image (w. Pileup, PUPPI, PUMML, PileFlow) in the 'cut' variant. "
                        "Default 0.1 GeV, fixed a priori (see threshold_scan.py).")
    p.add_argument("--threshold-truth", action="store_true",
                   help="Also apply --cell-threshold to the true neutral image.")
    p.add_argument("--spread-samples", type=int, default=8,
                   help="Max number of stored samples used to quantify sampling spread (0 = off).")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--auto-ranges", action="store_true", help="Histogram ranges from data, not the paper")
    p.add_argument("--outdir", default="results/comparison")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def resolve_threshold(args) -> tuple[float, dict]:
    if args.threshold_json:
        info = json.loads(Path(args.threshold_json).read_text())
        if info.get("selected_on_split") != "val":
            raise ValueError(f"{args.threshold_json} was not selected on the validation split")
        return float(info["threshold"]), info
    return float(args.pileflow_threshold), {"threshold": args.pileflow_threshold, "source": "command line"}


def main(argv=None):
    args = parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    d = bm.load_eval_data(args.data, args.split)
    print(f"Held-out jets: {len(d['indices'])} ({(d['jet_rank'] == 0).sum()} leading)")
    reps = bm.base_representations(d)
    speed = {}

    pumml9, dt = bm.pumml_predict(args.pumml_model, d["X"], device)
    reps["pumml"] = (d["charged_lv36"], pumml9)
    speed["pumml"] = {"jets_per_second": len(pumml9) / max(dt, 1e-12), "n_parameters": 4711, "device": device}
    if args.pumml_tuned_model:
        reps["pumml_tuned"] = (d["charged_lv36"], bm.pumml_predict(args.pumml_tuned_model, d["X"], device)[0])

    thr, thr_info = resolve_threshold(args)
    pf9, samples, pf_info = bm.load_pileflow(args.pileflow, d["indices"], args.pileflow_estimator)
    reps["pileflow"] = (d["charged_lv36"], pf9)
    with np.load(args.pileflow, allow_pickle=False) as pf:
        speed["pileflow"] = {k: pf[f][()].item() if f in pf.files else None for k, f in
                             (("jets_per_second", "timing_jets_per_second"), ("n_parameters", "n_parameters"),
                              ("nfe", "timing_nfe"))}
        speed["pileflow"]["device"] = str(pf["device"]) if "device" in pf.files else None
    if args.add_posterior_obs:
        if samples is None:
            raise ValueError("--add-posterior-obs needs neutral_lv_samples in the PileFlow file")
        reps["pileflow_obsmean"] = (d["charged_lv36"], samples)
    if args.add_posterior_mean:
        reps["pileflow_mean"] = (d["charged_lv36"], bm.load_pileflow(args.pileflow, d["indices"], "mean")[0])
    if args.pileflow_small:
        reps["pileflow_small"] = (d["charged_lv36"],
                                  bm.load_pileflow(args.pileflow_small, d["indices"], args.pileflow_estimator)[0])
    methods = [m for m in ("with_pileup", "puppi", "pumml", "pumml_tuned", "pileflow", "pileflow_obsmean",
                           "pileflow_mean", "pileflow_small") if m in reps]

    # ---------------------------------------------------------------- variants
    variants = []
    if args.cut_mode in ("both", "off"):
        variants.append(("nocut", 0.0, reps))
    if args.cut_mode in ("both", "on"):
        c = args.cell_threshold
        cut_reps = bm.apply_cell_cut(reps, c, args.threshold_truth)
        if thr > c:   # optional extra PileFlow-only cut chosen on validation (select_threshold.py)
            print(f"NOTE: PileFlow models get their validation-selected {thr} GeV cut instead of the common "
                  f"{c} GeV; the other methods keep {c} GeV.")
            for m in ("pileflow", "pileflow_obsmean", "pileflow_mean", "pileflow_small"):
                if m in cut_reps:
                    cut_reps[m] = (cut_reps[m][0], bm.apply_threshold(reps[m][1], thr))
        variants.append((bm.cut_tag(c), c, cut_reps))
    if not variants:
        raise ValueError("nothing to do")

    common = {"speed": speed, "pileflow": pf_info, "pileflow_val_threshold": thr_info,
              "threshold_truth": args.threshold_truth}
    values = {}
    for tag, c, vreps in variants:
        title = "NO neutral-cell cut" if c == 0 else (
            f"Neutral-cell cut {1000 * c:.0f} MeV on all methods" + (" and truth" if args.threshold_truth else ""))
        values[tag], _ = bm.run_variant(
            d, vreps, methods, outdir / tag, args.bootstrap_samples, args.bootstrap_seed, "pumml",
            paper=True, paper_ranges=not args.auto_ranges, dpi=args.dpi,
            extra={**common, "cell_threshold_GeV": c}, title=title)

    if len(variants) == 2:
        bm.compare_cut_variants(values["nocut"], values[variants[1][0]], methods, outdir,
                                args.cell_threshold, min(args.bootstrap_samples, 1000),
                                threshold_truth=args.threshold_truth, dpi=args.dpi)

    if samples is not None and args.spread_samples > 0:
        print("\nPer-sample Pearson spread of PileFlow (sampling uncertainty, no cut)")
        spread = bm.stochastic_spread(d, samples, args.spread_samples)
        for obs, s_ in spread.items():
            print(f"  {obs:12s} r = {100 * s_['mean']:.2f} +- {100 * s_['std']:.2f} %")
        (outdir / "pileflow_sample_spread.json").write_text(json.dumps(spread, indent=2))
    print("\nThroughput:", json.dumps(speed, indent=1))
    print(f"\nWrote {', '.join(str(outdir / t) for t, _, _ in variants)}"
          + (f" and the cut-vs-no-cut summary in {outdir}/tables, {outdir}/plots" if len(variants) == 2 else ""))


if __name__ == "__main__":
    main()
