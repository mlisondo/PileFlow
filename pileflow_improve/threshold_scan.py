#!/usr/bin/env python3
"""Neutral-cell pT threshold scan applied UNIFORMLY to every method (appendix study).

Motivation: CNN (ReLU) and flow outputs leave small positive pT in cells that
the truth leaves empty. N95 counts cells, so it is very sensitive to this soft
residue, while the other observables barely move. The PUMML paper states no
cell threshold, yet its Fig. 4 shows the PUMML N95 distribution on top of the
truth, which suggests an O(100 MeV) zero-suppression was applied.

For each threshold c the neutral 9x9 image of every reconstruction is set to
zero in cells with pT < c (charged cells are untouched, and the truth too
unless --threshold-truth). For every method and observable we report the
Pearson correlation with a *paired* bootstrap band (the same resampled jets for
every threshold and method), the mean N95 bias and the fraction of neutral pT
removed. Two independent "paper-matching" cuts are extracted for PUMML, each
with a bootstrap 68% interval:

  * r-match    : c where Pearson(N95) equals the paper value (0.753),
  * bias-zero  : c where <N95_pred - N95_true> = 0 (Fig. 4 shows no bias).

    python threshold_scan.py --data DATA.npz --pumml-model runs/pumml/pumml_event.pt \
        --pileflow runs/pileflow_v2/data/generated_eval.npz --outdir results/threshold_scan

The threshold used in the main result should be fixed a priori or chosen on
validation jets (select_threshold.py), never taken from this test-set scan.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pileflow import benchmark as bm
from pileflow.observables import OBS_LABELS, OBS_ORDER, PAPER_CORRELATION, compute_observables

DEFAULT_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
                0.13, 0.14, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0]


def corr_matrix(a, b):
    """Row-wise Pearson for bootstrap matrices a, b of shape (B, n)."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / den


def crossing(x, y, target):
    """First x where the piecewise-linear y(x) crosses `target` (NaN if never)."""
    d = np.asarray(y) - target
    for i in range(len(d) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] <= 0 and d[i] != d[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * (-d[i]) / (d[i + 1] - d[i]))
        if d[i] == 0:
            return float(x[i])
    return float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["event", "image"], default="event")
    p.add_argument("--pumml-model", required=True)
    p.add_argument("--pileflow", default=None, help="generated_eval.npz (optional)")
    p.add_argument("--pileflow-estimator", choices=["sample", "mean", "median"], default="sample")
    p.add_argument("--no-puppi", action="store_true")
    p.add_argument("--threshold-truth", action="store_true",
                   help="Apply the cut to the truth as well (a detector zero-suppression); default: truth untouched")
    p.add_argument("--grid", type=float, nargs="*", default=DEFAULT_GRID)
    p.add_argument("--paper-n95-pearson", type=float, default=PAPER_CORRELATION["pumml"]["neutral_n95"] / 100)
    p.add_argument("--bootstrap-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--outdir", default="results/threshold_scan")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    grid = np.array(sorted(set(args.grid)), dtype=float)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = bm.load_eval_data(args.data, args.split)
    neutral = {"pumml": bm.pumml_predict(args.pumml_model, d["X"], device)[0]}
    charged = {"pumml": d["charged_lv36"]}
    if "puppi_charged36" in d and not args.no_puppi:
        neutral["puppi"], charged["puppi"] = d["puppi_neutral9"], d["puppi_charged36"]
    if args.pileflow:
        neutral["pileflow"] = bm.load_pileflow(args.pileflow, d["indices"], args.pileflow_estimator)[0]
        charged["pileflow"] = d["charged_lv36"]
    methods = list(neutral)

    def obs_for(c36, n9):
        return compute_observables(c36, n9, d["eta"], d["phi"], d["event_id"], d["jet_rank"])

    lead = np.where(d["jet_rank"] == 0)[0]
    truth_raw = obs_for(d["charged_lv36"], d["truth9"])
    n_lead, n_dijet = len(lead), len(truth_raw["dijet_mass"])
    rng = np.random.default_rng(args.seed)
    boot = {"single": rng.integers(0, n_lead, (args.bootstrap_samples, n_lead)),
            "dijet": rng.integers(0, n_dijet, (args.bootstrap_samples, n_dijet))}

    # results[method][obs] -> arrays over the grid; reps[method][obs] -> (n_grid, B)
    res = {m: {o: {"r": [], "lo": [], "hi": []} for o in OBS_ORDER} for m in methods}
    reps = {m: {o: [] for o in OBS_ORDER} for m in methods}
    extra = {m: {"n95_bias": [], "n95_mean_pred": [], "n95_mean_true": [], "removed_pt_frac": []}
             for m in methods}
    n95_bias_reps = {m: [] for m in methods}

    for c in grid:
        truth = obs_for(d["charged_lv36"], bm.apply_threshold(d["truth9"], c)) if args.threshold_truth else truth_raw
        for m in methods:
            n9 = bm.apply_threshold(neutral[m], c)
            v = obs_for(charged[m], n9)
            for o in OBS_ORDER:
                t, y = np.asarray(truth[o], float), np.asarray(v[o], float)
                b = boot["dijet" if o == "dijet_mass" else "single"]
                tb, yb = t[b], y[b]
                ok = np.isfinite(tb) & np.isfinite(yb)
                if not ok.all():            # rare NaN ECFs: fall back to masked per-replica values
                    r_b = np.array([np.corrcoef(tb[i][ok[i]], yb[i][ok[i]])[0, 1] for i in range(len(b))])
                else:
                    r_b = corr_matrix(tb, yb)
                m_ok = np.isfinite(t) & np.isfinite(y)
                res[m][o]["r"].append(float(np.corrcoef(t[m_ok], y[m_ok])[0, 1]))
                lo, hi = np.nanpercentile(r_b, [2.5, 97.5])
                res[m][o]["lo"].append(float(lo)); res[m][o]["hi"].append(float(hi))
                reps[m][o].append(r_b)
            diff = v["neutral_n95"] - truth["neutral_n95"]
            extra[m]["n95_bias"].append(float(diff.mean()))
            extra[m]["n95_mean_pred"].append(float(v["neutral_n95"].mean()))
            extra[m]["n95_mean_true"].append(float(truth["neutral_n95"].mean()))
            extra[m]["removed_pt_frac"].append(float(1 - n9[lead].sum() / max(neutral[m][lead].sum(), 1e-12)))
            n95_bias_reps[m].append(diff[boot["single"]].mean(1))
        print(f"c = {c:6.3f} GeV  " + "  ".join(
            f"{bm.label(m)}: r(N95)={res[m]['neutral_n95']['r'][-1]:.3f} bias={extra[m]['n95_bias'][-1]:+.2f}"
            for m in methods))

    # paper-matching cuts
    matches = {}
    for m in methods:
        rN = np.array(res[m]["neutral_n95"]["r"])
        R = np.stack(reps[m]["neutral_n95"])            # (n_grid, B)
        Bz = np.stack(n95_bias_reps[m])
        c_r = crossing(grid, rN, args.paper_n95_pearson)
        c_b = crossing(grid, extra[m]["n95_bias"], 0.0)
        c_r_b = np.array([crossing(grid, R[:, i], args.paper_n95_pearson) for i in range(R.shape[1])])
        c_b_b = np.array([crossing(grid, Bz[:, i], 0.0) for i in range(Bz.shape[1])])
        best = int(np.nanargmax(rN))
        matches[m] = {
            "r_match_threshold": c_r,
            "r_match_68": [float(x) for x in np.nanpercentile(c_r_b, [16, 84])] if np.isfinite(c_r_b).any() else None,
            "bias_zero_threshold": c_b,
            "bias_zero_68": [float(x) for x in np.nanpercentile(c_b_b, [16, 84])] if np.isfinite(c_b_b).any() else None,
            "max_r_n95": float(rN[best]), "max_r_threshold": float(grid[best]),
        }

    # outputs
    with open(out / "threshold_scan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold_GeV", "method", "observable", "pearson", "ci95_low", "ci95_high",
                    "n95_bias", "n95_mean_pred", "n95_mean_true", "removed_neutral_pt_frac"])
        for i, c in enumerate(grid):
            for m in methods:
                for o in OBS_ORDER:
                    w.writerow([c, m, o, res[m][o]["r"][i], res[m][o]["lo"][i], res[m][o]["hi"][i],
                                extra[m]["n95_bias"][i], extra[m]["n95_mean_pred"][i],
                                extra[m]["n95_mean_true"][i], extra[m]["removed_pt_frac"][i]])
    summary = {"grid": grid.tolist(), "threshold_truth": args.threshold_truth, "n_leading": n_lead,
               "n_dijet": n_dijet, "paper_n95_pearson": args.paper_n95_pearson,
               "bootstrap_samples": args.bootstrap_samples, "matches": matches}
    (out / "threshold_scan_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPaper-matching neutral-cell cuts (68% bootstrap intervals)")
    for m, s in matches.items():
        fmt = lambda v, iv: "none" if not np.isfinite(v) else (
            f"{1000 * v:.0f} MeV" + (f" [{1000 * iv[0]:.0f}, {1000 * iv[1]:.0f}]" if iv else ""))
        print(f"  {bm.label(m):10s} r(N95)={args.paper_n95_pearson:.3f} at {fmt(s['r_match_threshold'], s['r_match_68'])}"
              f" | zero N95 bias at {fmt(s['bias_zero_threshold'], s['bias_zero_68'])}"
              f" | max r={s['max_r_n95']:.3f} at {1000 * s['max_r_threshold']:.0f} MeV")

    _plot(grid, res, extra, methods, args, out)
    print(f"\nWrote {out}")


def _plot(grid, res, extra, methods, args, out):
    plt = bm._style()
    x = 1000 * grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for ax, o in zip(axes.flat, OBS_ORDER):
        for m in methods:
            ax.plot(x, res[m][o]["r"], "-", color=bm.color(m), label=bm.label(m), lw=1.8)
            ax.fill_between(x, res[m][o]["lo"], res[m][o]["hi"], color=bm.color(m), alpha=0.18, lw=0)
            paper = PAPER_CORRELATION.get(m, {}).get(o)
            if paper is not None:
                ax.axhline(paper / 100, color=bm.color(m), ls=":", lw=1)
        ax.set_title(OBS_LABELS[o].replace(" [GeV]", ""))
        ax.set_ylabel("Pearson r"); ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Neutral cell threshold [MeV]"); ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(0, x.max())
    axes[0, 0].legend(title="dotted: paper value")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"threshold_scan_correlations.{ext}", dpi=args.dpi)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        a1.plot(x, res[m]["neutral_n95"]["r"], color=bm.color(m), label=bm.label(m))
        a1.fill_between(x, res[m]["neutral_n95"]["lo"], res[m]["neutral_n95"]["hi"], color=bm.color(m), alpha=0.18, lw=0)
        a2.plot(x, extra[m]["n95_bias"], color=bm.color(m), label=bm.label(m))
    a1.axhline(args.paper_n95_pearson, color="k", ls=":", lw=1, label="PUMML paper")
    a2.axhline(0, color="k", ls=":", lw=1)
    a1.set_ylabel(r"Pearson r, neutral $N_{95}$"); a2.set_ylabel(r"$\langle N_{95}^{reco} - N_{95}^{true}\rangle$")
    for a in (a1, a2):
        a.set_xlabel("Neutral cell threshold [MeV]"); a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, x.max())
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"threshold_scan_n95.{ext}", dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()