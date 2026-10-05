#!/usr/bin/env python
"""Mean +- std over seeds of the compare_methods.py tables.

    python aggregate_seeds.py results/test_frac_seed*      # -> results/seeds_summary/

Each input folder is one compare_methods.py output (one PileFlow seed paired with one
PUMML seed). For every cut variant (nocut, cut_100MeV) and table (correlation, IQR) it
writes <variant>_<table>_mean_std.csv and prints the table.
"""
import argparse
import os
import sys

import pandas as pd

LABELS = {"with_pileup": "w. Pileup", "puppi": "PUPPI", "pumml": "PUMML",
          "pileflow": "PileFlow (1 sample)", "pileflow_obsmean": "PileFlow (post. mean)"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="compare_methods.py output folders, one per seed")
    p.add_argument("--outdir", default="results/seeds_summary")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for variant in ("nocut", "cut_100MeV"):
        for table in ("correlation", "iqr"):
            frames = []
            for d in a.dirs:
                f = os.path.join(d, variant, "tables", f"{table}_table.csv")
                if os.path.exists(f):
                    df = pd.read_csv(f).set_index("observable")
                    frames.append(df[[c for c in LABELS if c in df.columns]])
            if not frames:
                continue
            allv = pd.concat(frames, keys=range(len(frames)))
            mean = allv.groupby(level=1, sort=False).mean()
            std = allv.groupby(level=1, sort=False).std(ddof=1).fillna(0.0)
            out = mean.copy().astype(object)
            for c in mean.columns:
                out[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean[c], std[c])]
            out = out.rename(columns=LABELS)
            path = os.path.join(a.outdir, f"{variant}_{table}_mean_std.csv")
            out.to_csv(path)
            title = "PEARSON r (%)" if table == "correlation" else "IQR (%)"
            print(f"\n=== {variant}: {title}, mean ± std over {len(frames)} seeds")
            print(out.to_string())
    print(f"\nWrote {a.outdir}")


if __name__ == "__main__":
    sys.exit(main())
