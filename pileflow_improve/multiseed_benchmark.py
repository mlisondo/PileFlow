#!/usr/bin/env python3
"""Paper-style benchmark over several training seeds of PUMML and PileFlow.

For every variant (no neutral-cell cut / a common cut on every method):

  * paper figures (PUMML Figs. 4, 5, 6) and paired-bootstrap tables for the
    *reference* seeds (the first --pumml-models / --pileflow entry, fixed a
    priori, no cherry-picking), in <outdir>/<variant>/reference_seed/;
  * per-seed metrics for every seed (CSV) and a seed-aggregated table
    (mean +- std over seeds) in the format of PUMML Tables 1 and 2, with the
    PileFlow - PUMML difference and its standard error;
  * a reproduction table: our PUMML/PUPPI next to the published values;
  * seed-spread figures (Pearson per seed, Fig. 6 with the seed band).

With --cut-mode both, a cut-vs-no-cut comparison and, with --n95-scan, the
N95 correlation/bias versus threshold for every seed are added (appendix).

    python multiseed_benchmark.py --data DATA.npz \
        --pumml-models runs/pumml/pumml_seed{1..5}.pt \
        --pileflow runs/pf_seed{42..46}/data/generated_eval.npz \
        --outdir results/paper
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pileflow import benchmark as bm
from pileflow.observables import (OBS_ORDER, OBS_PLAIN, PAPER_CORRELATION, PAPER_IQR,
                                  compute_observables, n95)
from pileflow.stats import iqr_metric, pearson, wasserstein_1d

SEED_METHODS = ("pumml", "pileflow")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["event", "image"], default="event")
    p.add_argument("--pumml-models", nargs="+", required=True)
    p.add_argument("--pileflow", nargs="+", required=True, help="generated_eval.npz of each PileFlow seed")
    p.add_argument("--pumml-labels", nargs="*", default=None)
    p.add_argument("--pileflow-labels", nargs="*", default=None)
    p.add_argument("--pileflow-estimator", choices=["sample", "mean", "median"], default="sample")
    p.add_argument("--cut-mode", choices=bm.CUT_MODES, default="both")
    p.add_argument("--cell-threshold", type=float, default=bm.DEFAULT_CELL_THRESHOLD)
    p.add_argument("--threshold-truth", action="store_true")
    p.add_argument("--n95-scan", action="store_true", help="N95 r/bias vs threshold for all seeds (appendix)")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--outdir", default="results/paper")
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args(argv)


def seed_metrics(values_truth, values_m) -> dict:
    out = {}
    for obs in OBS_ORDER:
        t, y = values_truth[obs], values_m[obs]
        out[obs] = {"pearson": pearson(t, y),
                    "iqr": iqr_metric(obs)(t, y) if obs != "neutral_n95" else float("nan"),
                    "wasserstein": wasserstein_1d(t, y),
                    "mean_bias": float(np.nanmean(np.asarray(y, float) - np.asarray(t, float)))}
    return out


def aggregate(per_seed: list[dict]) -> dict:
    agg = {}
    for obs in OBS_ORDER:
        agg[obs] = {}
        for k in ("pearson", "iqr", "wasserstein", "mean_bias"):
            v = np.array([s[obs][k] for s in per_seed], float)
            agg[obs][k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                           "min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "n": int(len(v)),
                           "values": v.tolist()}
    return agg


def run_variant(args, d, base_reps, preds, tag, c, outdir):
    """Everything for one variant (c = cell threshold, 0 = no cut)."""
    vdir = outdir / tag
    vdir.mkdir(parents=True, exist_ok=True)
    cut = (lambda n9: bm.apply_threshold(n9, c)) if c > 0 else (lambda n9: n9)
    reps = bm.apply_cell_cut(base_reps, c, args.threshold_truth) if c > 0 else dict(base_reps)

    # ---- reference seeds: full paper-style benchmark with paired bootstrap
    ref = dict(reps)
    for m in SEED_METHODS:
        ref[m] = (d["charged_lv36"], cut(preds[m][0][1]))
    methods = [m for m in ("with_pileup", "puppi", "pumml", "pileflow") if m in ref]
    title = ("NO neutral-cell cut" if c == 0 else f"{1000 * c:.0f} MeV neutral-cell cut on all methods") + \
        f" | reference seeds: PUMML {preds['pumml'][0][0]}, PileFlow {preds['pileflow'][0][0]}"
    ref_values, ref_res = bm.run_variant(d, ref, methods, vdir / "reference_seed", args.bootstrap_samples,
                                         args.seed, "pumml", paper=True, dpi=args.dpi,
                                         extra={"cell_threshold_GeV": c, "reference_seeds": {
                                             m: preds[m][0][0] for m in SEED_METHODS}}, title=title)

    # ---- every seed
    print(f"\n[{tag}] per-seed observables")
    single = {m: seed_metrics(ref_values["truth"], ref_values[m]) for m in ("with_pileup", "puppi") if m in ref}
    per_seed = {m: [] for m in SEED_METHODS}
    seed_values = {m: [] for m in SEED_METHODS}
    for m in SEED_METHODS:
        for i, (lab, n9) in enumerate(preds[m]):
            v = ref_values[m] if i == 0 else compute_observables(
                d["charged_lv36"], cut(n9), d["eta"], d["phi"], d["event_id"], d["jet_rank"])
            seed_values[m].append(v)
            per_seed[m].append(seed_metrics(ref_values["truth"], v))
            print(f"  {bm.label(m):9s} {lab:28s} " + " ".join(
                f"{OBS_PLAIN[o].split(' (')[0]}={100 * per_seed[m][-1][o]['pearson']:.1f}" for o in OBS_ORDER))
    agg = {m: aggregate(per_seed[m]) for m in SEED_METHODS}

    # ---- NPU robustness with seed band
    lead_npu = ref_values["_meta"]["npu_leading"]
    rob = bm.npu_robustness(ref_values, methods)
    bands = {}
    if lead_npu is not None:
        for m in SEED_METHODS:
            curves = []
            for v in seed_values[m]:
                tmp = {"truth": ref_values["truth"], m: v, "_meta": ref_values["_meta"]}
                curves.append(bm.npu_robustness(tmp, [m])[m])
            curves = np.array(curves, float)
            bands[m] = (np.nanmin(curves, 0), np.nanmax(curves, 0))
            rob[f"{m}_seed_mean"] = np.nanmean(curves, 0).tolist()
            rob[f"{m}_seed_min"], rob[f"{m}_seed_max"] = bands[m][0].tolist(), bands[m][1].tolist()
        bm.plot_npu(rob, methods, vdir, dpi=args.dpi, bands=bands, name="fig6_jet_mass_correlation_vs_npu_seedband")

    summary = {"variant": tag, "cell_threshold_GeV": c, "threshold_truth": args.threshold_truth,
               "n_leading": ref_values["_meta"]["n_leading"], "n_dijet": ref_values["_meta"]["n_dijet"],
               "seeds": {m: [lab for lab, _ in preds[m]] for m in SEED_METHODS},
               "single": single, "aggregate": agg, "npu_robustness": rob}
    _write_tables(vdir, summary, c)
    _plot_seed_spread(vdir, summary, args.dpi, c)
    (vdir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    _print_summary(summary)
    return ref_values, summary


# ------------------------------------------------------------------ output
def _json(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _pm(a, key, scale=100.0, digits=1):
    s = a[key]
    return f"{scale * s['mean']:.{digits}f} ± {scale * s['std']:.{digits}f}"


def _diff(summary, obs, key, scale):
    a, b = summary["aggregate"]["pileflow"][obs][key], summary["aggregate"]["pumml"][obs][key]
    se = np.sqrt(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"])
    return scale * (a["mean"] - b["mean"]), scale * se


def _print_summary(summary):
    agg, single = summary["aggregate"], summary["single"]
    print("\n" + "=" * 118)
    print(f"SEED-AVERAGED RESULTS ({summary['variant']}): mean ± std over "
          f"{agg['pumml']['jet_mass']['pearson']['n']} PUMML / {agg['pileflow']['jet_mass']['pearson']['n']} "
          "PileFlow seeds")
    print("-" * 118)
    print(f"{'Correlation (%)':18s}{'w. Pileup':>12s}{'PUPPI':>10s}{'PUMML':>16s}{'PileFlow':>16s}"
          f"{'PF - PUMML (±SE)':>22s}{'paper PUMML':>14s}")
    for obs in OBS_ORDER:
        dm, se = _diff(summary, obs, "pearson", 100)
        wp = 100 * single["with_pileup"][obs]["pearson"]
        pp = 100 * single["puppi"][obs]["pearson"] if "puppi" in single else float("nan")
        print(f"{OBS_PLAIN[obs]:18s}{wp:12.1f}{pp:10.1f}{_pm(agg['pumml'][obs], 'pearson'):>16s}"
              f"{_pm(agg['pileflow'][obs], 'pearson'):>16s}{f'{dm:+.2f} ± {se:.2f}':>22s}"
              f"{PAPER_CORRELATION['pumml'][obs]:14.1f}")
    print(f"\n{'IQR (%)':18s}{'PUPPI':>22s}{'PUMML':>16s}{'PileFlow':>16s}{'PF - PUMML (±SE)':>22s}{'paper PUMML':>14s}")
    for obs in OBS_ORDER:
        if obs == "neutral_n95":
            continue
        dm, se = _diff(summary, obs, "iqr", 1)
        pp = single["puppi"][obs]["iqr"] if "puppi" in single else float("nan")
        print(f"{OBS_PLAIN[obs]:18s}{pp:22.2f}{_pm(agg['pumml'][obs], 'iqr', 1, 2):>16s}"
              f"{_pm(agg['pileflow'][obs], 'iqr', 1, 2):>16s}{f'{dm:+.2f} ± {se:.2f}':>22s}"
              f"{PAPER_IQR['pumml'][obs]:14.2f}")
    print("=" * 118)


def _write_tables(vdir, summary, c):
    agg, single = summary["aggregate"], summary["single"]
    tdir = vdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    # per-seed CSV
    with open(tdir / "per_seed_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "observable", "pearson", "iqr", "wasserstein", "mean_bias"])
        for m in ("with_pileup", "puppi"):
            if m in single:
                for obs in OBS_ORDER:
                    s = single[m][obs]
                    w.writerow([m, "-", obs, s["pearson"], s["iqr"], s["wasserstein"], s["mean_bias"]])
        for m in SEED_METHODS:
            for i, lab in enumerate(summary["seeds"][m]):
                for obs in OBS_ORDER:
                    w.writerow([m, lab, obs] + [agg[m][obs][k]["values"][i]
                                                for k in ("pearson", "iqr", "wasserstein", "mean_bias")])
    # aggregated CSV
    with open(tdir / "seed_aggregated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["observable", "metric", "method", "mean", "std", "min", "max", "n_seeds"])
        for obs in OBS_ORDER:
            for k in ("pearson", "iqr", "wasserstein", "mean_bias"):
                for m in SEED_METHODS:
                    a = agg[m][obs][k]
                    w.writerow([obs, k, m, a["mean"], a["std"], a["min"], a["max"], a["n"]])
                for m in ("with_pileup", "puppi"):
                    if m in single:
                        w.writerow([obs, k, m, single[m][obs][k], 0, single[m][obs][k], single[m][obs][k], 1])

    cut_txt = "without a neutral-cell threshold" if c == 0 else \
        f"with a {1000 * c:.0f}~MeV neutral-cell threshold applied to every method"
    nP, nF = agg["pumml"]["jet_mass"]["pearson"]["n"], agg["pileflow"]["jet_mass"]["pearson"]["n"]
    tex_names = {"jet_mass": "Jet mass", "dijet_mass": "Dijet mass", "jet_pt": "Jet $p_T$",
                 "neutral_n95": "Neutral $N_{95}$", "ln_ecf2": "$\\ln\\mathrm{ECF}^{(\\beta=4)}_{N=2}$",
                 "ln_ecf3": "$\\ln\\mathrm{ECF}^{(\\beta=4)}_{N=3}$"}

    def pm(a, scale, dig):
        return f"{scale * a['mean']:.{dig}f} $\\pm$ {scale * a['std']:.{dig}f}"

    with open(tdir / "table1_correlation.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Correlation coefficients (\\%) between the true and corrected values of the jet "
                f"observables on an event-by-event level, {cut_txt}. PUMML and PileFlow: mean $\\pm$ standard "
                f"deviation over {nP} and {nF} training seeds. Larger is better.}}\n\\label{{tab:corr_{summary['variant']}}}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n"
                "Correlation (\\%) & w. Pileup & PUMML & PUPPI & PileFlow \\\\\n\\hline\n")
        for obs in OBS_ORDER:
            wp = 100 * single["with_pileup"][obs]["pearson"]
            pp = f"{100 * single['puppi'][obs]['pearson']:.1f}" if "puppi" in single else "--"
            f.write(f"{tex_names[obs]} & {wp:.1f} & {pm(agg['pumml'][obs]['pearson'], 100, 1)} & {pp} & "
                    f"{pm(agg['pileflow'][obs]['pearson'], 100, 1)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    with open(tdir / "table2_iqr.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Interquartile ranges (\\%) of the centered percent-error distributions, "
                f"{cut_txt}. Mean $\\pm$ standard deviation over training seeds. Lower is better.}}\n"
                f"\\label{{tab:iqr_{summary['variant']}}}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\nIQR (\\%) & PUMML & PUPPI & PileFlow \\\\\n\\hline\n")
        for obs in OBS_ORDER:
            if obs == "neutral_n95":
                continue
            pp = f"{single['puppi'][obs]['iqr']:.2f}" if "puppi" in single else "--"
            f.write(f"{tex_names[obs]} & {pm(agg['pumml'][obs]['iqr'], 1, 2)} & {pp} & "
                    f"{pm(agg['pileflow'][obs]['iqr'], 1, 2)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    with open(tdir / "reproduction_vs_paper.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Reproduction check: published PUMML/PUPPI values "
                f"(arXiv:1707.08600, Tables 1--2) and this work, {cut_txt}.}}\n"
                f"\\label{{tab:repro_{summary['variant']}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n & \\multicolumn{2}{c}{w. Pileup} & "
                "\\multicolumn{2}{c}{PUMML} & \\multicolumn{2}{c}{PUPPI} \\\\\n"
                "Correlation (\\%) & paper & ours & paper & ours & paper & ours \\\\\n\\hline\n")
        for obs in OBS_ORDER:
            pc = PAPER_CORRELATION
            pp = f"{100 * single['puppi'][obs]['pearson']:.1f}" if "puppi" in single else "--"
            f.write(f"{tex_names[obs]} & {pc['with_pileup'][obs]:.1f} & "
                    f"{100 * single['with_pileup'][obs]['pearson']:.1f} & {pc['pumml'][obs]:.1f} & "
                    f"{pm(agg['pumml'][obs]['pearson'], 100, 1)} & {pc['puppi'][obs]:.1f} & {pp} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _plot_seed_spread(vdir, summary, dpi, c):
    plt = bm._style()
    agg, single = summary["aggregate"], summary["single"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, obs in zip(axes.flat, OBS_ORDER):
        for j, m in enumerate(SEED_METHODS):
            vals = np.array(agg[m][obs]["pearson"]["values"])
            x = j + np.linspace(-0.15, 0.15, len(vals))
            ax.plot(x, vals, "o", color=bm.color(m), ms=5, alpha=0.8)
            mu, sd = agg[m][obs]["pearson"]["mean"], agg[m][obs]["pearson"]["std"]
            ax.errorbar(j + 0.3, mu, yerr=sd, fmt="s", color=bm.color(m), ms=6, capsize=3)
        if "puppi" in single:
            ax.axhline(single["puppi"][obs]["pearson"], color=bm.color("puppi"), lw=1.3, label="PUPPI")
        ax.axhline(PAPER_CORRELATION["pumml"][obs] / 100, color="k", ls=":", lw=1, label="PUMML (paper)")
        ax.set_xticks([0, 1]); ax.set_xticklabels([bm.label(m) for m in SEED_METHODS])
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(bm.PAPER_XLABELS[obs].replace(" (GeV)", "")); ax.set_ylabel("Correlation coefficient")
        ax.grid(axis="y", alpha=0.3)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-seed correlations (dots), mean ± std (squares) — "
                 + ("no cut" if c == 0 else f"{1000 * c:.0f} MeV cut"), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    bm._save(fig, vdir, "seed_spread_correlations", dpi)
    plt.close(fig)


def n95_scan(d, base_reps, preds, outdir, dpi, threshold_truth=False,
             grid=(0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
                   0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)):
    """N95 Pearson and mean bias versus a common cell threshold for EVERY seed."""
    lead = np.where(d["jet_rank"] == 0)[0]
    grid = np.array(grid, float)
    curves = {"pumml": [], "pileflow": [], "puppi": []}
    curves_b = {"pumml": [], "pileflow": [], "puppi": []}
    truth0 = n95(d["truth9"][lead])
    for m in SEED_METHODS:
        for lab, n9 in preds[m]:
            r, b = [], []
            for c in grid:
                t = n95(bm.apply_threshold(d["truth9"][lead], c)) if threshold_truth else truth0
                y = n95(bm.apply_threshold(n9[lead], c))
                r.append(pearson(t, y)); b.append(float((y - t).mean()))
            curves[m].append(r); curves_b[m].append(b)
    if "puppi" in base_reps:
        r, b = [], []
        for c in grid:
            t = n95(bm.apply_threshold(d["truth9"][lead], c)) if threshold_truth else truth0
            y = n95(bm.apply_threshold(base_reps["puppi"][1][lead], c))
            r.append(pearson(t, y)); b.append(float((y - t).mean()))
        curves["puppi"].append(r); curves_b["puppi"].append(b)

    plt = bm._style()
    x = 1000 * grid
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in ("puppi", "pumml", "pileflow"):
        for i, (r, b) in enumerate(zip(curves[m], curves_b[m])):
            lw, al = (0.8, 0.35) if m in SEED_METHODS else (1.8, 1)
            a1.plot(x, r, color=bm.color(m), lw=lw, alpha=al)
            a2.plot(x, b, color=bm.color(m), lw=lw, alpha=al)
        if m in SEED_METHODS and curves[m]:
            a1.plot(x, np.mean(curves[m], 0), color=bm.color(m), lw=2, label=f"{bm.label(m)} (seed mean)")
            a2.plot(x, np.mean(curves_b[m], 0), color=bm.color(m), lw=2, label=f"{bm.label(m)} (seed mean)")
        elif curves[m]:
            a1.plot([], [], color=bm.color(m), lw=1.8, label=bm.label(m))
            a2.plot([], [], color=bm.color(m), lw=1.8, label=bm.label(m))
    a1.axhline(PAPER_CORRELATION["pumml"]["neutral_n95"] / 100, color="k", ls=":", lw=1, label="PUMML (paper)")
    a2.axhline(0, color="k", ls=":", lw=1)
    a1.set_ylabel(r"Neutral $N_{95}$ correlation coefficient")
    a2.set_ylabel(r"$\langle N_{95}^{\rm reco} - N_{95}^{\rm true} \rangle$")
    for a in (a1, a2):
        a.set_xlabel("Neutral cell threshold (MeV)"); a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, x.max()); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    bm._save(fig, outdir, "appendix_n95_threshold_all_seeds", dpi)
    plt.close(fig)
    out = {"grid_GeV": grid.tolist(), "pearson": curves, "bias": curves_b,
           "labels": {m: [lab for lab, _ in preds[m]] for m in SEED_METHODS}}
    (outdir / "appendix_n95_threshold_all_seeds.json").write_text(json.dumps(out, indent=2, default=_json))
    with open(outdir / "appendix_n95_threshold_all_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold_GeV", "method", "seed", "pearson_n95", "mean_bias_n95"])
        for m in curves:
            labs = out["labels"].get(m, ["-"])
            for lab, r, b in zip(labs, curves[m], curves_b[m]):
                for c, ri, bi in zip(grid, r, b):
                    w.writerow([c, m, lab, ri, bi])


def main(argv=None):
    args = parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = bm.load_eval_data(args.data, args.split)
    print(f"Held-out jets: {len(d['indices'])} ({(d['jet_rank'] == 0).sum()} leading)")
    base = bm.base_representations(d)

    labels_p = args.pumml_labels or [Path(p).stem for p in args.pumml_models]
    labels_f = args.pileflow_labels or [Path(p).parent.parent.name for p in args.pileflow]
    preds = {"pumml": [(lab, bm.pumml_predict(p, d["X"], device)[0]) for lab, p in zip(labels_p, args.pumml_models)],
             "pileflow": [(lab, bm.load_pileflow(p, d["indices"], args.pileflow_estimator)[0])
                          for lab, p in zip(labels_f, args.pileflow)]}

    variants = []
    if args.cut_mode in ("both", "off"):
        variants.append(("nocut", 0.0))
    if args.cut_mode in ("both", "on"):
        variants.append((bm.cut_tag(args.cell_threshold), args.cell_threshold))
    results = {}
    for tag, c in variants:
        results[tag] = run_variant(args, d, base, preds, tag, c, outdir)

    if len(variants) == 2:
        (v0, s0), (v1, s1) = results[variants[0][0]], results[variants[1][0]]
        methods = [m for m in ("with_pileup", "puppi", "pumml", "pileflow") if m in v0]
        bm.compare_cut_variants(v0, v1, methods, outdir / "cut_vs_nocut_reference_seed", args.cell_threshold,
                                min(args.bootstrap_samples, 1000), threshold_truth=args.threshold_truth, dpi=args.dpi)
        rows = []
        for obs in OBS_ORDER:
            for m in SEED_METHODS:
                a0, a1 = s0["aggregate"][m][obs]["pearson"], s1["aggregate"][m][obs]["pearson"]
                dv = np.array(a1["values"]) - np.array(a0["values"])
                rows.append([obs, m, a0["mean"], a0["std"], a1["mean"], a1["std"], dv.mean(), dv.std(ddof=1)])
        with open(outdir / "cut_vs_nocut_seed_aggregated.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["observable", "method", "pearson_nocut_mean", "pearson_nocut_std", "pearson_cut_mean",
                        "pearson_cut_std", "delta_mean", "delta_std"])
            w.writerows(rows)
    if args.n95_scan:
        n95_scan(d, base, preds, outdir, args.dpi, args.threshold_truth)
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
