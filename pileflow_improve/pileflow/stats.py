"""Metrics and *paired* bootstrap used to compare pileup-mitigation methods."""
from __future__ import annotations

import numpy as np


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def iqr(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def reconstruction_error(truth, reco, obs: str) -> np.ndarray:
    """PUMML Fig. 5: percent error (difference for N95), centred at median 0."""
    truth, reco = np.asarray(truth, float), np.asarray(reco, float)
    if obs == "neutral_n95":
        err = reco - truth
    else:
        err = np.full_like(truth, np.nan)
        m = np.isfinite(truth) & np.isfinite(reco) & (np.abs(truth) > 1e-12)
        err[m] = 100.0 * (reco[m] - truth[m]) / truth[m]
    if np.isfinite(err).any():
        err = err - np.nanmedian(err)
    return err


def iqr_metric(obs: str):
    return lambda truth, reco: iqr(reconstruction_error(truth, reco, obs))


def wasserstein_1d(a, b) -> float:
    """Earth mover's distance between two 1-D samples (distribution fidelity)."""
    from scipy.stats import wasserstein_distance
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")


def common_mask(*arrays) -> np.ndarray:
    m = np.ones(len(arrays[0]), bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, float))
    return m


def paired_bootstrap(truth, methods: dict, metric, n_boot: int = 2000, seed: int = 12345,
                     reference: str | None = None, higher_is_better: bool = True) -> dict:
    """Bootstrap a metric for several methods on the SAME resampled jets.

    Only jets finite for truth and every method are used, so all methods are
    compared on an identical population. Returns per method
    {value, std, ci_low, ci_high, n} and, if `reference` is given, for each
    other method the difference (method - reference) with its 95% CI and the
    one-sided bootstrap probability that the method is NOT better.
    """
    truth = np.asarray(truth, float)
    reco = {k: np.asarray(v, float) for k, v in methods.items()}
    m = common_mask(truth, *reco.values())
    truth = truth[m]
    reco = {k: v[m] for k, v in reco.items()}
    n = len(truth)
    central = {k: metric(truth, v) for k, v in reco.items()}
    reps = {k: np.full(n_boot, np.nan) for k in reco}
    if n >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            for k, v in reco.items():
                reps[k][b] = metric(truth[idx], v[idx])

    def summary(value, r):
        r = r[np.isfinite(r)]
        if len(r) < 2:
            return {"value": value, "std": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "n": n}
        lo, hi = np.percentile(r, [2.5, 97.5])
        return {"value": value, "std": float(r.std(ddof=1)), "ci_low": float(lo),
                "ci_high": float(hi), "n": n}

    out = {"methods": {k: summary(central[k], reps[k]) for k in reco}}
    if reference is not None and reference in reco:
        diffs = {}
        for k in reco:
            if k == reference:
                continue
            d = reps[k] - reps[reference]
            s = summary(central[k] - central[reference], d)
            d = d[np.isfinite(d)]
            worse = (d <= 0) if higher_is_better else (d >= 0)
            s["p_not_better"] = float(worse.mean()) if len(d) else float("nan")
            diffs[k] = s
        out["diff_vs_" + reference] = diffs
    return out
