#!/usr/bin/env python
"""Per-seed paired-bootstrap differences vs PUMML, plus Wasserstein mean +- std (numpy only).

    python seed_significance.py results/test_frac_seed{1..5}
    python seed_significance.py results/test_frac_seed{1..5} --method pileflow   # single sample

Reads <dir>/<variant>/tables/summary.json written by compare_methods.py. For every seed pair it
prints Delta = method - PUMML with its 95% CI; '*' marks a CI that excludes zero in PileFlow's
favour (higher correlation, lower IQR), '-' a CI that excludes zero in PUMML's favour.
"""
import argparse
import json
import os

import numpy as np

OBS = ["jet_mass", "dijet_mass", "jet_pt", "neutral_n95", "ln_ecf2", "ln_ecf3"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--method", default="pileflow_obsmean", help="pileflow_obsmean (default) or pileflow")
    p.add_argument("--variants", nargs="+", default=["cut_100MeV", "nocut"])
    a = p.parse_args()
    for variant in a.variants:
        S = []
        for d in a.dirs:
            f = os.path.join(d, variant, "tables", "summary.json")
            if os.path.exists(f):
                S.append((os.path.basename(d.rstrip("/")), json.load(open(f))))
        if not S:
            continue
        print(f"\n{'=' * 100}\n{variant}: {a.method} - PUMML, per seed pair "
              f"(value [95% CI], percent points)\n{'=' * 100}")
        for metric, better in (("correlation", +1), ("iqr", -1)):
            print(f"\n{metric.upper()}  ({'higher' if better > 0 else 'lower'} is better for PileFlow)")
            print("observable     " + "".join(f"{n[-5:]:>24s}" for n, _ in S) + "   PileFlow wins")
            for o in OBS:
                cells, wins, losses = [], 0, 0
                for _, s in S:
                    dd = s.get(metric, {}).get(o, {}).get("diff_vs_pumml", {}).get(a.method)
                    if not dd:
                        cells.append(" " * 24)
                        continue
                    scale = 100.0 if metric == "correlation" else 1.0     # IQRs are stored in percent already
                    v, lo, hi = (scale * dd[k] for k in ("value", "ci_low", "ci_high"))
                    tag = "*" if (lo > 0 if better > 0 else hi < 0) else ("-" if (hi < 0 if better > 0 else lo > 0) else " ")
                    wins += tag == "*"
                    losses += tag == "-"
                    cells.append(f"{v:+6.2f} [{lo:+5.2f},{hi:+5.2f}]{tag}".rjust(24))
                if any(c.strip() for c in cells):
                    print(f"{o:14s} " + "".join(cells) + f"   {wins}/{len(S)}  (PUMML {losses})")
        print("\nWASSERSTEIN-1 to truth, mean ± std over seeds (lower is better)")
        methods = ["puppi", "pumml", "pileflow", "pileflow_obsmean"]
        print("observable     " + "".join(f"{m:>22s}" for m in methods))
        for o in OBS:
            row = []
            for m in methods:
                v = np.array([s["wasserstein"].get(o, {}).get(m, np.nan) for _, s in S], float)
                row.append(f"{np.nanmean(v):.3f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.3f}".rjust(22))
            print(f"{o:14s} " + "".join(row))


if __name__ == "__main__":
    main()
