"""PUMML-paper benchmark shared by compare_methods.py and pumml/evaluate.py.

One code path computes every method's observables from the same detector
representation (pileflow.observables), on the same jets, with the same
statistics (pileflow.stats), so numbers are directly comparable.

A representation is (charged36, neutral9):
    truth        charged LV (36x36)             + true neutral LV (9x9)
    with_pileup  charged LV + charged PU        + total neutral
    puppi        PUPPI charged (36x36)          + PUPPI neutral (9x9)
    pumml        charged LV                     + PUMML neutral prediction
    pileflow     charged LV                     + PileFlow neutral prediction
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .observables import OBS_LABELS, OBS_ORDER, OBS_PLAIN, PAPER_CORRELATION, PAPER_IQR, compute_observables
from .stats import iqr_metric, paired_bootstrap, pearson, reconstruction_error, wasserstein_1d

METHOD_LABELS = {"truth": "True", "with_pileup": "w. Pileup", "puppi": "PUPPI", "pumml": "PUMML",
                 "pumml_tuned": "PUMML (val-selected)", "pileflow": "PileFlow",
                 "pileflow_mean": "PileFlow (mean image)", "pileflow_small": "PileFlow-small",
                 "pileflow_obsmean": "PileFlow (posterior mean)"}
COLORS = {"truth": "green", "with_pileup": "red", "puppi": "orange", "pumml": "black",
          "pumml_tuned": "0.45", "pileflow": "blue", "pileflow_mean": "deepskyblue",
          "pileflow_small": "purple", "pileflow_obsmean": "royalblue"}
PAPER_XLIMS = {"jet_mass": (0, 100), "dijet_mass": (0, 1100), "jet_pt": (0, 600),
               "neutral_n95": (0, 55), "ln_ecf2": (-15, 0), "ln_ecf3": (-35, -5)}
ERROR_XLIMS = {"jet_mass": (-100, 100), "dijet_mass": (-15, 15), "jet_pt": (-20, 20),
               "neutral_n95": (-10, 10), "ln_ecf2": (-40, 40), "ln_ecf3": (-40, 40)}
NPU_EDGES = (0, 120, 130, 140, 150, 160, 1000)
IQR_OBS = ("jet_mass", "dijet_mass", "jet_pt", "ln_ecf2", "ln_ecf3")


def label(m):
    return METHOD_LABELS.get(m, m)


def color(m):
    return COLORS.get(m, None)


# loading
def load_eval_data(path, split="event", indices=None) -> dict:
    """Arrays of the held-out jets. split: 'event' (event_test_idx), 'image'
    (test_idx) or 'all' (debug); explicit `indices` override `split`."""
    with np.load(path, allow_pickle=False) as d:
        n = len(d["y_neutral_lv_9"])
        key = {"event": "event_test_idx", "image": "test_idx"}.get(split)
        if indices is not None:
            idx = np.asarray(indices, np.int64)
        else:
            idx = np.arange(n) if split == "all" else np.asarray(d[key], np.int64)
        req = ["X", "y_neutral_lv_9", "event_id", "jet_rank", "image_center_eta", "image_center_phi"]
        missing = [k for k in req if k not in d.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        X = np.asarray(d["X"][idx], np.float32)
        # Slim files (test jets only) keep the original row numbers in `orig_index`;
        # `indices` always refers to rows of the ORIGINAL dataset, as stored by PileFlow.
        ids = np.asarray(d["orig_index"], np.int64)[idx] if "orig_index" in d.files else idx
        out = {
            "indices": ids, "X": X,
            "truth9": np.asarray(d["y_neutral_lv_9"][idx], np.float64),
            "event_id": np.asarray(d["event_id"][idx], np.int64),
            "jet_rank": np.asarray(d["jet_rank"][idx], np.int64),
            "eta": np.asarray(d["image_center_eta"][idx], np.float64),
            "phi": np.asarray(d["image_center_phi"][idx], np.float64),
            "npu": np.asarray(d["npu"][idx], np.float64) if "npu" in d.files else None,
        }
        if "puppi_charged_36" in d.files and "puppi_neutral_9" in d.files:
            out["puppi_charged36"] = np.asarray(d["puppi_charged_36"][idx], np.float64)
            out["puppi_neutral9"] = np.asarray(d["puppi_neutral_9"][idx], np.float64)
        else:
            print("WARNING: no puppi_charged_36/puppi_neutral_9 in the file; PUPPI is omitted")
    out["total9"] = X[:, 0].reshape(-1, 9, 4, 9, 4).sum((2, 4)).astype(np.float64)
    out["charged_pu36"] = X[:, 1].astype(np.float64)
    out["charged_lv36"] = X[:, 2].astype(np.float64)
    return out


def base_representations(d) -> dict:
    reps = {"truth": (d["charged_lv36"], d["truth9"]),
            "with_pileup": (d["charged_lv36"] + d["charged_pu36"], d["total9"])}
    if "puppi_charged36" in d:
        reps["puppi"] = (d["puppi_charged36"], d["puppi_neutral9"])
    return reps


def pumml_predict(model_path, X, device="cpu", batch_size=512):
    """Run a PUMML checkpoint. Returns (neutral9 (N,9,9) GeV, seconds)."""
    import torch
    from pumml.pumml_arch import PUMML

    model = PUMML().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[s:s + batch_size]).to(device))[:, 0].cpu().numpy())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return np.clip(np.concatenate(out).astype(np.float64), 0.0, None), dt


def load_pileflow(path, eval_indices, estimator="sample", threshold=0.0, expected_split="eval"):
    """Load generated_eval.npz aligned to `eval_indices`.

    estimator: 'sample' (one stochastic draw, the honest generative output),
               'mean' or 'median' (per-pixel over the K stored samples).
    Returns (neutral9 (N,9,9), samples (N,K,9,9) or None, info dict).
    """
    with np.load(path, allow_pickle=False) as pf:
        split = str(pf["split_name"]) if "split_name" in pf.files else "unknown"
        src = np.asarray(pf["source_indices"], np.int64)
        key = {"sample": "neutral_lv_pred", "mean": "neutral_lv_pred_mean",
               "median": "neutral_lv_pred_median"}[estimator]
        if key not in pf.files:
            raise KeyError(f"{path} has no {key!r} (regenerate with the current evaluate code)")
        pred = np.asarray(pf[key], np.float64)
        samples = np.asarray(pf["neutral_lv_samples"], np.float64) if "neutral_lv_samples" in pf.files else None
    if split not in {expected_split, "unknown"}:
        raise ValueError(f"{path} was generated on split '{split}', expected '{expected_split}'")
    eval_indices = np.asarray(eval_indices, np.int64)
    if len(src) != len(eval_indices) or set(src.tolist()) != set(eval_indices.tolist()):
        raise RuntimeError("PileFlow source_indices do not match the evaluation indices")
    pos = {int(v): i for i, v in enumerate(src)}
    order = np.array([pos[int(i)] for i in eval_indices])
    pred = apply_threshold(pred[order], threshold)
    if samples is not None:
        samples = apply_threshold(samples[order], threshold)
    return pred, samples, {"split": split, "estimator": estimator, "threshold": threshold}


def apply_threshold(neutral9, threshold):
    x = np.clip(neutral9, 0.0, None)
    return np.where(x < threshold, 0.0, x) if threshold > 0 else x


# observables
def posterior_mean_observables(d, c36, samples) -> dict:
    """Posterior-mean estimate of every observable: the observable is computed for each of
    the K generated images of a jet and averaged (for N95 and the log ECFs as well).

    This is the minimum-MSE estimate of each observable given the inputs, which a single
    regressed image (PUMML) cannot provide because the observables are non-linear in the
    image."""
    per = [compute_observables(c36, samples[:, k], d["eta"], d["phi"], d["event_id"], d["jet_rank"])
           for k in range(samples.shape[1])]
    out = dict(per[0])
    for o in OBS_ORDER:
        out[o] = np.nanmean(np.stack([v[o] for v in per]), axis=0)
    return out


def compute_values(d, reps: dict, verbose=True) -> dict:
    """reps[name] = (charged36, neutral9) with neutral9 of shape (N,9,9), or (N,K,9,9) for K
    posterior samples, in which case the posterior mean of each observable is used."""
    values = {}
    for name, (c36, n9) in reps.items():
        t0 = time.time()
        if np.ndim(n9) == 4:
            values[name] = posterior_mean_observables(d, c36, n9)
        else:
            values[name] = compute_observables(c36, n9, d["eta"], d["phi"], d["event_id"], d["jet_rank"])
        if verbose:
            print(f"  observables for {label(name):28s} ({time.time() - t0:.1f} s)")
    lead = values["truth"]["leading_idx"]
    for v in values.values():
        if not np.array_equal(v["dijet_events"], values["truth"]["dijet_events"]):
            raise RuntimeError("Dijet event lists differ between methods")
    values["_meta"] = {"leading_idx": lead, "npu_leading": None if d["npu"] is None else d["npu"][lead],
                       "n_leading": len(lead), "n_dijet": len(values["truth"]["dijet_mass"])}
    return values


# tables
def benchmark_tables(values, methods, n_boot=2000, seed=12345, reference="pumml") -> dict:
    """Pearson correlation, IQR and 1-D Wasserstein distance per observable,
    with paired-bootstrap 95% CIs and differences w.r.t. `reference`."""
    ref = reference if reference in methods else None
    res = {"correlation": {}, "iqr": {}, "wasserstein": {}, "n": {}}
    for i, obs in enumerate(OBS_ORDER):
        truth = values["truth"][obs]
        reco = {m: values[m][obs] for m in methods}
        res["correlation"][obs] = paired_bootstrap(truth, reco, pearson, n_boot, seed + i, ref, True)
        if obs in IQR_OBS:
            res["iqr"][obs] = paired_bootstrap(truth, {m: reco[m] for m in methods if m != "with_pileup"},
                                               iqr_metric(obs), n_boot, seed + 100 + i, ref, False)
        res["wasserstein"][obs] = {m: wasserstein_1d(truth, reco[m]) for m in methods}
        res["n"][obs] = res["correlation"][obs]["methods"][methods[0]]["n"]
    return res


def npu_robustness(values, methods, obs="jet_mass", edges=NPU_EDGES) -> dict:
    """Pearson(truth, reco) of `obs` in bins of NPU (PUMML Fig. 6 analogue)."""
    npu = values["_meta"]["npu_leading"]
    if npu is None or obs == "dijet_mass":
        return {}
    out = {"edges": list(edges), "n": [], "npu_mean": [], **{m: [] for m in methods}}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (npu >= lo) & (npu < hi)
        out["n"].append(int(sel.sum()))
        out["npu_mean"].append(float(npu[sel].mean()) if sel.any() else float("nan"))
        for m in methods:
            out[m].append(pearson(values["truth"][obs][sel], values[m][obs][sel]) if sel.sum() > 10 else float("nan"))
    return out


def stochastic_spread(d, samples, max_samples=8, verbose=True) -> dict:
    """Pearson of each observable for each individual PileFlow sample: the
    spread quantifies the sampling (not jet-population) uncertainty."""
    k = min(samples.shape[1], max_samples)
    truth = compute_observables(d["charged_lv36"], d["truth9"], d["eta"], d["phi"], d["event_id"], d["jet_rank"])
    per = {obs: [] for obs in OBS_ORDER}
    for s in range(k):
        v = compute_observables(d["charged_lv36"], samples[:, s], d["eta"], d["phi"], d["event_id"], d["jet_rank"])
        for obs in OBS_ORDER:
            per[obs].append(pearson(truth[obs], v[obs]))
        if verbose:
            print(f"  sample {s + 1}/{k} done")
    return {obs: {"mean": float(np.mean(r)), "std": float(np.std(r, ddof=1)) if k > 1 else float("nan"),
                  "values": [float(x) for x in r]} for obs, r in per.items()}


# printing
def _fmt_ci(s, scale=100.0, digits=1):
    v, lo, hi = s["value"] * scale, s["ci_low"] * scale, s["ci_high"] * scale
    if not np.isfinite(lo):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_tables(res, methods, reference="pumml", paper=True):
    w = max(24, max(len(label(m)) for m in methods) + 2)
    iqr_methods = [m for m in methods if m != "with_pileup"]
    line = "=" * (18 + w * len(methods))
    print("\n" + line + "\nPEARSON CORRELATION (%)   value [95% paired-bootstrap CI], higher is better\n" + line)
    print(f"{'':18s}" + "".join(f"{label(m):>{w}s}" for m in methods))
    for obs, r in res["correlation"].items():
        print(f"{OBS_PLAIN[obs]:18s}" + "".join(f"{_fmt_ci(r['methods'][m]):>{w}s}" for m in methods))
    print("\nIQR OF CENTRED PERCENT ERROR (%), lower is better")
    print(f"{'':18s}" + "".join(f"{label(m):>{w}s}" for m in iqr_methods))
    for obs, r in res["iqr"].items():
        print(f"{OBS_PLAIN[obs]:18s}" + "".join(f"{_fmt_ci(r['methods'][m], 1.0, 2):>{w}s}" for m in iqr_methods))
    key = "diff_vs_" + reference
    others = [m for m in methods if m not in {reference, "with_pileup"}]
    if others and any(key in r for r in res["correlation"].values()):
        print(f"\nDIFFERENCE vs {label(reference)} (paired bootstrap, 95% CI; p = P(not better))")
        for obs in OBS_ORDER:
            print(f"  {OBS_PLAIN[obs]}")
            for m in others:
                c = res["correlation"][obs][key][m]
                txt = (f"    {label(m):28s} d(corr) = {100 * c['value']:+6.2f} "
                       f"[{100 * c['ci_low']:+6.2f}, {100 * c['ci_high']:+6.2f}]  p={c['p_not_better']:.3f}")
                if obs in res["iqr"]:
                    q = res["iqr"][obs][key][m]
                    txt += (f"   d(IQR) = {q['value']:+6.2f} [{q['ci_low']:+6.2f}, {q['ci_high']:+6.2f}]"
                            f"  p={q['p_not_better']:.3f}")
                print(txt)
    if paper:
        print("\nPUMML paper reference, <NPU>=140 (Table 1 corr. % | Table 2 IQR %):")
        for obs in OBS_ORDER:
            c = " ".join(f"{label(m)}={PAPER_CORRELATION[m][obs]}" for m in ("with_pileup", "puppi", "pumml"))
            q = " ".join(f"{label(m)}={PAPER_IQR[m][obs]}" for m in ("puppi", "pumml") if obs in PAPER_IQR[m])
            print(f"  {OBS_PLAIN[obs]:18s} {c}" + (f" | {q}" if q else ""))
    print(line)


# writing
def write_outputs(outdir, res, methods, extra: dict, reference="pumml"):
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    with open(outdir / "tables" / "summary.json", "w") as f:
        json.dump(clean({"methods": methods, "reference": reference, **res, **extra}), f, indent=2)

    iqr_methods = [m for m in methods if m != "with_pileup"]
    rows_c = [{"observable": obs, **{f"{m}": 100 * r["methods"][m]["value"] for m in methods},
               **{f"{m}_ci_low": 100 * r["methods"][m]["ci_low"] for m in methods},
               **{f"{m}_ci_high": 100 * r["methods"][m]["ci_high"] for m in methods}}
              for obs, r in res["correlation"].items()]
    rows_i = [{"observable": obs, **{f"{m}": r["methods"][m]["value"] for m in iqr_methods},
               **{f"{m}_ci_low": r["methods"][m]["ci_low"] for m in iqr_methods},
               **{f"{m}_ci_high": r["methods"][m]["ci_high"] for m in iqr_methods}}
              for obs, r in res["iqr"].items()]
    for name, rows in (("correlation_table.csv", rows_c), ("iqr_table.csv", rows_i)):
        with open(outdir / "tables" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def latex(fname, caption, lab, rows_obs, meths, getter, digits):
        with open(outdir / "tables" / fname, "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + lab + "}\n")
            f.write("\\begin{tabular}{l" + "c" * len(meths) + "}\n\\hline\n")
            f.write("Observable & " + " & ".join(label(m) for m in meths) + " \\\\\n\\hline\n")
            for obs in rows_obs:
                cells = []
                for m in meths:
                    s = getter(obs, m)
                    txt = f"{s['value']:.{digits}f}"
                    if np.isfinite(s["std"]):
                        txt += f" $\\pm$ {s['std']:.{digits}f}"
                    cells.append(txt)
                f.write(OBS_LABELS[obs].replace(" [GeV]", "") + " & " + " & ".join(cells) + " \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    def scaled(s, k):
        return {"value": k * s["value"], "std": k * s["std"]}

    latex("correlation_table.tex",
          "Pearson correlation (\\%) between true and reconstructed jet observables on the held-out "
          "event-grouped test set; uncertainties are paired-bootstrap standard deviations. Single-jet "
          "observables use leading jets, the dijet mass uses complete leading/subleading pairs.",
          "tab:correlations", list(res["correlation"]), methods,
          lambda o, m: scaled(res["correlation"][o]["methods"][m], 100.0), 1)
    latex("iqr_table.tex",
          "Interquartile range (\\%) of the median-centred percent reconstruction error; lower is better.",
          "tab:iqr", list(res["iqr"]), iqr_methods, lambda o, m: res["iqr"][o]["methods"][m], 2)


# plots
def _hist(ax, x, bins, **kw):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= bins[0]) & (x <= bins[-1])]
    if len(x):
        ax.hist(x, bins, density=True, **kw)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
                         "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "0.3",
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


# Axis labels, ranges and binning as in PUMML (arXiv:1707.08600) Figs. 4 and 5.
PAPER_XLABELS = {
    "jet_mass": "Jet Mass (GeV)", "dijet_mass": "Dijet Mass (GeV)", "jet_pt": "Jet pT (GeV)",
    "neutral_n95": r"Neutral $N_{95}$",
    "ln_ecf2": r"$\ln\,\mathrm{ECF}^{(\beta=4)}_{N=2}$", "ln_ecf3": r"$\ln\,\mathrm{ECF}^{(\beta=4)}_{N=3}$",
}
PAPER_ERROR_XLABELS = {
    "jet_mass": "Jet Mass Percent Error, Centered", "dijet_mass": "Dijet Mass Percent Error, Centered",
    "jet_pt": "Jet pT Percent Error, Centered", "neutral_n95": r"Neutral $N_{95}$ Difference, Centered",
    "ln_ecf2": r"$\ln\,\mathrm{ECF}^{(\beta=4)}_{N=2}$ Percent Error, Centered",
    "ln_ecf3": r"$\ln\,\mathrm{ECF}^{(\beta=4)}_{N=3}$ Percent Error, Centered",
}
PAPER_NBINS = {"jet_mass": 50, "dijet_mass": 55, "jet_pt": 60, "ln_ecf2": 60, "ln_ecf3": 60}
PAPER_NBINS_ERR = {"jet_mass": 50, "dijet_mass": 50, "jet_pt": 50, "ln_ecf2": 50, "ln_ecf3": 50}
PAPER_ORDER = ("with_pileup", "puppi", "pumml", "pumml_tuned", "pileflow", "pileflow_obsmean", "pileflow_mean",
               "pileflow_small")


def _paper_bins(obs, lo, hi, nbins):
    if obs == "neutral_n95":
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    return np.linspace(lo, hi, nbins + 1)


def _save(fig, outdir, name, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(Path(outdir) / f"{name}.{ext}", dpi=dpi)


def plot_distributions(values, methods, outdir, dpi=200, paper_ranges=True, name="fig4_observable_distributions"):
    """PUMML Fig. 4: normalised distributions, truth as a filled green histogram."""
    plt = _style()
    ms = [m for m in PAPER_ORDER if m in methods]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.5))
    for ax, obs in zip(axes.flat, OBS_ORDER):
        truth = values["truth"][obs]
        if paper_ranges:
            lo, hi = PAPER_XLIMS[obs]
        else:
            pool = np.concatenate([np.asarray(values[m][obs], float) for m in ["truth", *ms]])
            lo, hi = np.nanpercentile(pool, [0.5, 99.5])
        bins = _paper_bins(obs, lo, hi, PAPER_NBINS.get(obs, 50))
        _hist(ax, truth, bins, histtype="stepfilled", color=color("truth"), alpha=0.25, lw=0)
        _hist(ax, truth, bins, histtype="step", color=color("truth"), lw=1.6)
        for m in ms:
            _hist(ax, values[m][obs], bins, histtype="step", lw=1.4, color=color(m))
        handles = [plt.Rectangle((0, 0), 1, 1, fc=(0.0, 0.5, 0.0, 0.25), ec=color("truth"), lw=1.4, label="True")]
        handles += [plt.Rectangle((0, 0), 1, 1, fc="none", ec=color(m), lw=1.4, label=label(m)) for m in ms]
        if len(ms) > 4:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(handles=handles, loc="upper right", fontsize=7.5 if len(ms) > 4 else 8.5)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(PAPER_XLABELS[obs]); ax.set_ylabel("Cross-section (normalized)")
        ax.set_yticks([])
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    _save(fig, outdir, name, dpi)
    plt.close(fig)


def plot_errors(values, methods, outdir, dpi=200, name="fig5_centred_errors"):
    """PUMML Fig. 5: median-centred percent errors (difference for N95)."""
    plt = _style()
    ms = [m for m in PAPER_ORDER if m in methods and m != "with_pileup"]
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 10.5))
    for ax, obs in zip(axes.flat, OBS_ORDER):
        lo, hi = ERROR_XLIMS[obs]
        bins = _paper_bins(obs, lo, hi, PAPER_NBINS_ERR.get(obs, 50))
        for m in ms:
            err = reconstruction_error(values["truth"][obs], values[m][obs], obs)
            _hist(ax, err, bins, histtype="step", lw=1.4, color=color(m))
        handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec=color(m), lw=1.4, label=label(m)) for m in ms]
        ax.set_ylim(0, ax.get_ylim()[1] * (1.0 + 0.12 * len(ms)))   # headroom so the legend never hides the peak
        ax.legend(handles=handles, loc="upper right", fontsize=7.5)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(PAPER_ERROR_XLABELS[obs]); ax.set_ylabel("Cross-section (normalized)")
        ax.set_yticks([])
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    _save(fig, outdir, name, dpi)
    plt.close(fig)


def plot_npu(rob, methods, outdir, obs="jet_mass", dpi=200, bands=None, name=None):
    """PUMML Fig. 6 analogue: jet-mass correlation vs NPU (bins of the test sample).

    `bands` optionally maps method -> (lo, hi) arrays (e.g. min/max over seeds)."""
    if not rob:
        return
    plt = _style()
    if "npu_mean" in rob:
        centres = np.asarray(rob["npu_mean"], float)          # mean NPU of the jets in each bin
    else:
        e = np.asarray(rob["edges"], float)
        centres = np.array([(a + min(b, 185)) / 2 for a, b in zip(e[:-1], e[1:])])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for m in [m for m in PAPER_ORDER if m in methods and m != "with_pileup"]:
        ax.plot(centres, rob[m], "-", color=color(m), label=label(m), lw=1.6, marker="o", ms=3.5)
        if bands and m in bands:
            ax.fill_between(centres, bands[m][0], bands[m][1], color=color(m), alpha=0.18, lw=0)
    ax.set_xlabel("NPU (mean of bin)")
    ax.set_ylabel("Jet Mass Correlation Coefficient" if obs == "jet_mass" else f"{OBS_PLAIN[obs]} correlation")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, outdir, name or f"fig6_{obs}_correlation_vs_npu", dpi)
    plt.close(fig)


# cell-cut variants
DEFAULT_CELL_THRESHOLD = 0.1   # GeV. Round O(100 MeV) value fixed a priori (see threshold_scan.py)
CUT_MODES = ("both", "on", "off")


def cut_tag(threshold: float) -> str:
    return f"cut_{int(round(1000 * threshold))}MeV"


def apply_cell_cut(reps: dict, threshold: float, threshold_truth: bool = False) -> dict:
    """Zero neutral cells below `threshold` GeV in every reconstruction (and the
    truth if requested). Charged images are never touched."""
    if threshold <= 0:
        return dict(reps)
    return {m: (c36, apply_threshold(n9, threshold) if (m != "truth" or threshold_truth) else n9)
            for m, (c36, n9) in reps.items()}


def run_variant(d, reps, methods, outdir, n_boot, seed, reference="pumml", paper=True,
                paper_ranges=True, dpi=200, extra=None, title=""):
    """Observables, tables and figures of one variant, written to `outdir`."""
    outdir = Path(outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 100 + f"\n# {title}\n" + "#" * 100)
    values = compute_values(d, reps)
    res = benchmark_tables(values, methods, n_boot, seed, reference)
    print_tables(res, methods, reference, paper)
    rob = npu_robustness(values, methods)
    info = {"npu_robustness_jet_mass": rob, "n_leading": values["_meta"]["n_leading"],
            "n_dijet": values["_meta"]["n_dijet"], "variant": title, **(extra or {})}
    write_outputs(outdir, res, methods, info, reference)
    plot_distributions(values, methods, outdir / "plots", dpi, paper_ranges)
    plot_errors(values, methods, outdir / "plots", dpi)
    plot_npu(rob, methods, outdir / "plots")
    return values, res


def compare_cut_variants(values_off, values_on, methods, outdir, threshold, n_boot=1000, seed=4242,
                         threshold_truth=False, dpi=200):
    """Side-by-side 'no cut' vs 'cut' summary: paired-bootstrap change of each
    method's Pearson correlation and IQR (same jets, same resamples), a
    combined CSV/LaTeX table and comparison figures."""
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    tcut = f"{1000 * threshold:.0f} MeV"
    rows = []
    for i, obs in enumerate(OBS_ORDER):
        for m in methods:
            # truth of each variant (identical unless threshold_truth)
            t_off, t_on = values_off["truth"][obs], values_on["truth"][obs]
            if threshold_truth:
                # different truths -> bootstrap each variant separately on shared indices
                r_off = paired_bootstrap(t_off, {"v": values_off[m][obs]}, pearson, n_boot, seed + i)["methods"]["v"]
                r_on = paired_bootstrap(t_on, {"v": values_on[m][obs]}, pearson, n_boot, seed + i)["methods"]["v"]
                d_corr = {"value": r_on["value"] - r_off["value"], "ci_low": float("nan"), "ci_high": float("nan")}
            else:
                b = paired_bootstrap(t_off, {"off": values_off[m][obs], "on": values_on[m][obs]},
                                     pearson, n_boot, seed + i, reference="off")
                r_off, r_on, d_corr = b["methods"]["off"], b["methods"]["on"], b["diff_vs_off"]["on"]
            row = {"observable": obs, "method": m,
                   "pearson_nocut": r_off["value"], "pearson_cut": r_on["value"],
                   "delta_pearson": d_corr["value"], "delta_ci_low": d_corr["ci_low"],
                   "delta_ci_high": d_corr["ci_high"],
                   "pearson_nocut_ci": [r_off["ci_low"], r_off["ci_high"]],
                   "pearson_cut_ci": [r_on["ci_low"], r_on["ci_high"]]}
            if obs in IQR_OBS and m != "with_pileup":
                row["iqr_nocut"] = iqr_metric(obs)(t_off, values_off[m][obs])
                row["iqr_cut"] = iqr_metric(obs)(t_on, values_on[m][obs])
            if obs == "neutral_n95":
                row["n95_bias_nocut"] = float(np.nanmean(values_off[m][obs] - t_off))
                row["n95_bias_cut"] = float(np.nanmean(values_on[m][obs] - t_on))
            rows.append(row)

    print("\n" + "=" * 100)
    print(f"EFFECT OF THE {tcut} NEUTRAL-CELL CUT (applied to every method"
          + (", truth included)" if threshold_truth else ", truth untouched)"))
    print("-" * 100)
    print(f"{'observable':14s} {'method':26s} {'r no cut':>9s} {'r cut':>9s} {'delta r [95% CI]':>28s}")
    for r in rows:
        ci = "" if not np.isfinite(r["delta_ci_low"]) else \
            f" [{100 * r['delta_ci_low']:+.2f}, {100 * r['delta_ci_high']:+.2f}]"
        print(f"{OBS_PLAIN[r['observable']]:14s} {label(r['method']):26s} {100 * r['pearson_nocut']:9.2f} "
              f"{100 * r['pearson_cut']:9.2f} {100 * r['delta_pearson']:+9.2f}{ci}")
    print("=" * 100)

    flat_keys = ["observable", "method", "pearson_nocut", "pearson_cut", "delta_pearson", "delta_ci_low",
                 "delta_ci_high", "iqr_nocut", "iqr_cut", "n95_bias_nocut", "n95_bias_cut"]
    with open(outdir / "tables" / "cut_vs_nocut.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(outdir / "tables" / "cut_vs_nocut.json", "w") as f:
        json.dump({"threshold_GeV": threshold, "threshold_truth": threshold_truth, "rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else float(o))
    with open(outdir / "tables" / "cut_vs_nocut.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Pearson correlation (\\%) without / with a "
                f"{tcut} neutral-cell threshold applied to every method"
                + (" and to the truth" if threshold_truth else " (truth untouched)") + ".}\n")
        f.write("\\label{tab:cellcut}\n\\begin{tabular}{l" + "c" * len(methods) + "}\n\\hline\n")
        f.write("Observable & " + " & ".join(label(m) for m in methods) + " \\\\\n\\hline\n")
        for obs in OBS_ORDER:
            cells = [f"{100 * r['pearson_nocut']:.1f} / {100 * r['pearson_cut']:.1f}"
                     for m in methods for r in rows if r["observable"] == obs and r["method"] == m]
            f.write(OBS_LABELS[obs].replace(" [GeV]", "") + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    _plot_cut_effect(rows, methods, outdir / "plots", tcut, dpi)
    _plot_n95_cut(values_off, values_on, methods, outdir / "plots", tcut, dpi)
    return rows


def _plot_cut_effect(rows, methods, plot_dir, tcut, dpi):
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(methods))
    for ax, obs in zip(axes.flat, OBS_ORDER):
        sel = {r["method"]: r for r in rows if r["observable"] == obs}
        for j, m in enumerate(methods):
            r = sel[m]
            for dx, key, mk, fill in ((-0.12, "nocut", "o", "none"), (0.12, "cut", "s", color(m))):
                v = r[f"pearson_{key}"]
                lo, hi = r[f"pearson_{key}_ci"]
                err = None if not np.isfinite(lo) else [[v - lo], [hi - v]]
                ax.errorbar(j + dx, v, yerr=err, fmt=mk, color=color(m), mfc=fill, ms=7, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([label(m) for m in methods], rotation=0, fontsize=9)
        ax.set_title(OBS_LABELS[obs].replace(" [GeV]", ""))
        ax.set_ylabel("Pearson r"); ax.grid(axis="y", alpha=0.3)
    h = [plt.Line2D([], [], marker="o", mfc="none", color="0.3", ls="", label="no cut"),
         plt.Line2D([], [], marker="s", color="0.3", ls="", label=f"cut {tcut}")]
    axes.flat[0].legend(handles=h, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(plot_dir) / f"cut_vs_nocut_correlations.{ext}", dpi=dpi)
    plt.close(fig)


def _plot_n95_cut(values_off, values_on, methods, plot_dir, tcut, dpi):
    plt = _style()
    ms = [m for m in methods if m != "with_pileup"]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 4), squeeze=False)
    hi = np.nanpercentile(np.concatenate([values_off[m]["neutral_n95"] for m in ["truth", *ms]]), 99.5)
    bins = np.arange(-0.5, max(hi, 5) + 1.5, 1.0)
    for ax, m in zip(axes[0], ms):
        _hist(ax, values_off["truth"]["neutral_n95"], bins, histtype="stepfilled", alpha=0.25,
              color=color("truth"), label="True")
        _hist(ax, values_off[m]["neutral_n95"], bins, histtype="step", ls="--", lw=1.6, color=color(m),
              label=f"{label(m)}, no cut")
        _hist(ax, values_on[m]["neutral_n95"], bins, histtype="step", lw=1.9, color=color(m),
              label=f"{label(m)}, cut {tcut}")
        ax.set_xlabel(OBS_LABELS["neutral_n95"]); ax.set_yticks([]); ax.legend(fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(plot_dir) / f"cut_vs_nocut_n95.{ext}", dpi=dpi)
    plt.close(fig)
