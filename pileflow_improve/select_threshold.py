#!/usr/bin/env python3
"""Choose the PileFlow neutral-cell pT threshold on the VALIDATION split.

Generated images carry small positive pT in cells the truth leaves empty,
which mainly inflates N95 and the ECFs. A per-cell threshold removes it; to
keep the test set blind it is chosen here, on validation jets only, by
maximising the mean Pearson correlation over the six PUMML observables.

    python evaluate_checkpoint.py --data DATA.npz --checkpoint CKPT --split val --outdir RUN
    python select_threshold.py --data DATA.npz --generated RUN/data/generated_val.npz \
        --output RUN/threshold.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pileflow import benchmark as bm
from pileflow.observables import OBS_ORDER, compute_observables
from pileflow.stats import pearson


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--generated", required=True, help="generated_val.npz")
    p.add_argument("--estimator", choices=["sample", "mean", "median"], default="sample")
    p.add_argument("--grid", type=float, nargs="*",
                   default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    with np.load(args.generated, allow_pickle=False) as g:
        if str(g["split_name"]) != "val":
            raise ValueError("select_threshold.py must be run on generated_val.npz (split 'val')")
        idx = np.sort(np.asarray(g["source_indices"], np.int64))
    d = bm.load_eval_data(args.data, indices=idx)
    truth = compute_observables(d["charged_lv36"], d["truth9"], d["eta"], d["phi"], d["event_id"], d["jet_rank"])

    scan = []
    for thr in args.grid:
        pred, _, _ = bm.load_pileflow(args.generated, idx, args.estimator, thr, expected_split="val")
        v = compute_observables(d["charged_lv36"], pred, d["eta"], d["phi"], d["event_id"], d["jet_rank"])
        corr = {obs: pearson(truth[obs], v[obs]) for obs in OBS_ORDER}
        score = float(np.nanmean(list(corr.values())))
        scan.append({"threshold": thr, "objective_mean_pearson": score, "pearson": corr})
        print(f"threshold {thr:5.2f} GeV  mean r = {100 * score:.3f}%  "
              + " ".join(f"{k}={100 * c:.1f}" for k, c in corr.items()))
    best = max(scan, key=lambda r: r["objective_mean_pearson"])
    out = {"threshold": best["threshold"], "selected_on_split": "val", "estimator": args.estimator,
           "objective": "mean Pearson over " + ", ".join(OBS_ORDER), "n_val_jets": int(len(idx)),
           "generated": str(Path(args.generated).resolve()), "scan": scan}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSelected threshold {best['threshold']} GeV -> {args.output}")


if __name__ == "__main__":
    main()
