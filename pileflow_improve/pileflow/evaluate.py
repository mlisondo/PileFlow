"""Sample PileFlow on a split and compute neutral-image level diagnostics.

Jet-level PUMML observables (mass, dijet mass, pT, N95, ECFs) are computed
by compare_methods.py from the generated_<split>.npz written here, using the
shared pileflow.observables module.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import BatchLoader
from .flow import TargetConditionalFlowMatching
from .observables import n95
from .stats import pearson, wasserstein_1d


@torch.no_grad()
def generate(model, dataset, indices, pre, device="cuda", batch_size=256, solver="dopri5",
             ode_steps=50, atol=1e-5, rtol=1e-5, n_samples=1, sigma_min=1e-4, alpha=1.0,
             seed=2024):
    """Return physical-GeV samples of shape (N, K, 9, 9) plus timing info.

    Sampling is reproducible: the initial noise of sample k of every jet is
    drawn once (CPU generator seeded with seed + k) for all jets, so it does
    not depend on the batch size or device. (With the adaptive dopri5 solver
    the step sizes are shared within a batch, so results can still differ at
    the solver-tolerance level between batch sizes.)
    """
    device = torch.device(device)
    model = model.to(device).eval()
    pre = pre.to(device)
    flow = TargetConditionalFlowMatching(sigma_min=sigma_min, alpha=alpha)
    loader = BatchLoader(dataset, indices, batch_size, shuffle=False)
    noise = [torch.randn((len(indices), 1, 9, 9), generator=torch.Generator().manual_seed(seed + 1000003 * k))
             for k in range(n_samples)]

    samples, nfe_total = [], 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    start = 0
    for raw in loader:
        stop = start + len(raw["index"])
        total_gev = raw["neutral_all_9"].to(device).float()
        cond = {k: pre.transform(k, raw[src].to(device).float()) for k, src in
                (("neutral_all", "neutral_all_9"), ("charged_pu", "charged_pu_36"),
                 ("charged_lv", "charged_lv_36"))}
        batch = []
        for k in range(n_samples):
            z1, nfe = flow.sample(model, cond["neutral_all"], cond["charged_pu"], cond["charged_lv"],
                                  solver=solver, steps=ode_steps, atol=atol, rtol=rtol,
                                  z0=noise[k][start:stop], return_nfe=True)
            nfe_total += nfe
            batch.append(pre.decode_target(z1, total_gev).float().cpu())
        samples.append(torch.stack(batch, dim=1)[:, :, 0])
        start = stop
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out = torch.cat(samples).numpy() if samples else np.zeros((0, n_samples, 9, 9), np.float32)
    timing = {"seconds": elapsed, "jets_per_second": len(indices) * n_samples / max(elapsed, 1e-12),
              "nfe_per_batch_and_sample": nfe_total / max(len(loader) * n_samples, 1)}
    return np.clip(out, 0.0, None), timing


def image_metrics(truth, pred) -> dict:
    """Per-image and distribution-level diagnostics of the neutral-LV image."""
    tt, pp = truth.sum((1, 2)), pred.sum((1, 2))
    resid = pp - tt
    ok = tt > 1e-6
    resp = pp[ok] / tt[ok]
    nt, npred = n95(truth), n95(pred)
    return {
        "pixel_mse": float(np.mean((pred - truth) ** 2)),
        "pixel_mae": float(np.mean(np.abs(pred - truth))),
        "neutral_pt_bias": float(resid.mean()),
        "neutral_pt_median_bias": float(np.median(resid)),
        "neutral_pt_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "neutral_pt_pearson": pearson(tt, pp),
        "neutral_pt_response_median": float(np.median(resp)) if len(resp) else float("nan"),
        "neutral_pt_response_iqr": float(np.subtract(*np.percentile(resp, [75, 25]))) if len(resp) else float("nan"),
        "neutral_pt_wasserstein": wasserstein_1d(tt, pp),
        "n95_mae": float(np.mean(np.abs(npred - nt))),
        "n95_pearson": pearson(nt, npred),
        "n95_wasserstein": wasserstein_1d(nt, npred),
    }


def calibration_metrics(truth, samples) -> dict:
    """Coverage of the per-jet neutral-pT posterior (needs K > 1 samples).

    A calibrated generative model has the truth inside its central 68% (90%)
    sample interval for ~68% (~90%) of the jets. PUMML, being deterministic,
    cannot provide this.
    """
    k = samples.shape[1]
    if k < 2:
        return {}
    tt = truth.sum((1, 2))
    s = samples.sum((2, 3))
    out = {"posterior_samples_per_jet": k,
           "neutral_pt_posterior_std_median": float(np.median(s.std(1)))}
    for level in (68, 90):
        lo, hi = np.percentile(s, [50 - level / 2, 50 + level / 2], axis=1)
        out[f"neutral_pt_coverage_{level}"] = float(np.mean((tt >= lo) & (tt <= hi)))
    return out


def evaluate_model(model, dataset, indices, pre, outdir, split_name="eval", device="cuda",
                   batch_size=256, solver="dopri5", ode_steps=50, atol=1e-5, rtol=1e-5,
                   eval_samples=1, sigma_min=1e-4, alpha=1.0, sample_seed=2024,
                   checkpoint_path=None, make_plots=True):
    indices = np.asarray(indices, dtype=np.int64)
    outdir = Path(outdir)
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    print(f"\nPileFlow sampling on '{split_name}' ({len(indices)} jets, {eval_samples} sample(s)/jet, "
          f"solver={solver}{'' if solver == 'dopri5' else f', steps={ode_steps}'}, seed={sample_seed})")

    samples, timing = generate(model, dataset, indices, pre, device, batch_size, solver, ode_steps,
                               atol, rtol, eval_samples, sigma_min, alpha, sample_seed)
    truth = dataset.target_neutral_lv_9[indices, 0].numpy().astype(np.float64)
    nominal = samples[:, 0]
    mean, median = samples.mean(1), np.median(samples, axis=1)

    metrics = {f"sample0_{k}": v for k, v in image_metrics(truth, nominal).items()}
    if eval_samples > 1:
        metrics.update({f"mean_{k}": v for k, v in image_metrics(truth, mean).items()})
        metrics.update(calibration_metrics(truth, samples))
    metrics.update({f"timing_{k}": v for k, v in timing.items()})

    save = {
        "neutral_lv_true": truth.astype(np.float32),
        "neutral_lv_pred": nominal.astype(np.float32),
        "neutral_lv_pred_mean": mean.astype(np.float32),
        "neutral_lv_pred_median": median.astype(np.float32),
        "source_indices": indices,
        "split_name": np.asarray(split_name),
        "sample_seed": np.asarray(sample_seed),
        "solver": np.asarray(solver),
        "checkpoint": np.asarray(str(checkpoint_path or "")),
        "data_path": np.asarray(str(dataset.path)),
        "n_parameters": np.asarray(sum(p.numel() for p in model.parameters())),
        "timing_seconds": np.asarray(timing["seconds"]),
        "timing_jets_per_second": np.asarray(timing["jets_per_second"]),
        "timing_nfe": np.asarray(timing["nfe_per_batch_and_sample"]),
        "device": np.asarray(str(device)),
    }
    if eval_samples > 1:
        save["neutral_lv_samples"] = samples.astype(np.float32)
    if dataset.event_id is not None:
        save["event_id"] = dataset.event_id[indices]
    if dataset.jet_rank is not None:
        save["jet_rank"] = dataset.jet_rank[indices]
    path = outdir / "data" / f"generated_{split_name}.npz"
    np.savez_compressed(path, **save)

    with open(outdir / f"metrics_{split_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / f"metrics_{split_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(metrics.items())
    if make_plots:
        _plots(truth, nominal, outdir / "plots", split_name)
    print(f"Wrote {path}")
    for k, v in metrics.items():
        print(f"  {k:44s}: {v:.6g}")
    return metrics


def _plots(truth, pred, plot_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    tt, pp = truth.sum((1, 2)), pred.sum((1, 2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(0, np.percentile(np.r_[tt, pp], 99.5), 50)
    axes[0].hist(tt, bins, histtype="stepfilled", alpha=0.3, color="green", label="Truth", density=True)
    axes[0].hist(pp, bins, histtype="step", color="blue", lw=1.8, label="PileFlow", density=True)
    axes[0].set_xlabel(r"Neutral LV $p_T$ [GeV]"); axes[0].legend()
    hi = float(max(tt.max(), pp.max(), 1e-6))
    axes[1].scatter(tt, pp, s=4, alpha=0.4)
    axes[1].plot([0, hi], [0, hi], "k--", lw=1)
    axes[1].set_xlabel(r"Truth neutral LV $p_T$ [GeV]"); axes[1].set_ylabel("PileFlow [GeV]")
    axes[1].set_title(f"Pearson r = {pearson(tt, pp):.3f}")
    ok = tt > 1e-6
    resp = pp[ok] / tt[ok]
    if len(resp):
        lo, hi2 = np.percentile(resp, [1, 99])
        axes[2].hist(resp, np.linspace(lo, hi2, 50), histtype="step", lw=1.8, density=True)
    axes[2].axvline(1.0, ls="--", c="k"); axes[2].set_xlabel("PileFlow / truth neutral $p_T$")
    fig.tight_layout(); fig.savefig(plot_dir / f"neutral_pt_{tag}.png", dpi=180); plt.close(fig)

    n_show = min(6, len(truth))
    if n_show:
        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show), squeeze=False)
        for i in range(n_show):
            vmax = max(truth[i].max(), pred[i].max(), 1e-8)
            for ax, img, title in ((axes[i, 0], truth[i], "Truth"), (axes[i, 1], pred[i], "PileFlow"),
                                   (axes[i, 2], pred[i] - truth[i], "PileFlow - truth")):
                kw = dict(vmin=0, vmax=vmax, cmap="viridis") if title != "PileFlow - truth" else \
                    dict(vmin=-vmax, vmax=vmax, cmap="coolwarm")
                im = ax.imshow(img.T, origin="lower", **kw)  # x = eta, y = phi
                ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(plot_dir / f"examples_{tag}.png", dpi=150); plt.close(fig)
