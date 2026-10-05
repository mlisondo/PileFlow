"""Dev-time helpers: validation subset, sampling K posterior samples, estimator-aware scoring."""
import numpy as np, torch
from pileflow import benchmark as bm
from pileflow.observables import OBS_ORDER, OBS_PLAIN, compute_observables
from pileflow.stats import pearson, iqr_metric, wasserstein_1d

import os
DATA = os.environ.get('PILEFLOW_DATA', 'data/full.npz')          # full-format NPZ
SPLITS = os.environ.get('PILEFLOW_SPLITS', 'runs/splits_train_val_eval.npz')
RUNS = os.environ.get('PILEFLOW_RUNS', 'runs')


def val_subset(n_events=1000, seed=0, split='val'):
    """Indices of all jets from `n_events` random hard events of the split (dijet-complete)."""
    d = np.load(DATA)
    idx = np.load(SPLITS)[split]
    ev = d['event_id'][idx]
    rng = np.random.default_rng(seed)
    u = rng.permutation(np.unique(ev))[:n_events]
    return np.sort(idx[np.isin(ev, u)])


def load(indices):
    return bm.load_eval_data(DATA, indices=indices)


def observables(d, neutral9, cut=0.0):
    n9 = bm.apply_threshold(neutral9, cut) if cut > 0 else neutral9
    return compute_observables(d['charged_lv36'], n9, d['eta'], d['phi'], d['event_id'], d['jet_rank'])


def combine(obs_list, how='mean'):
    """Posterior summary of each observable over K samples (per jet / per dijet event)."""
    out = {}
    for o in OBS_ORDER:
        a = np.stack([v[o] for v in obs_list])
        out[o] = np.nanmean(a, 0) if how == 'mean' else np.nanmedian(a, 0)
    return out


def score(truth, reco):
    r = {}
    for o in OBS_ORDER:
        r[o] = {'r': pearson(truth[o], reco[o]), 'iqr': iqr_metric(o)(truth[o], reco[o]) if o != 'neutral_n95' else np.nan,
                'w1': wasserstein_1d(truth[o], reco[o]), 'bias': float(np.nanmean(np.asarray(reco[o]) - truth[o]))}
    return r


def table(rows: dict, title=''):
    print('\n' + title)
    print(f"{'':34s}" + ''.join(f"{OBS_PLAIN[o][:11]:>12s}" for o in OBS_ORDER) + '   | IQR: ' +
          ' '.join(f"{OBS_PLAIN[o][:8]:>8s}" for o in OBS_ORDER if o != 'neutral_n95'))
    for name, s in rows.items():
        print(f"{name:34s}" + ''.join(f"{100 * s[o]['r']:12.2f}" for o in OBS_ORDER) + '   |      ' +
              ' '.join(f"{s[o]['iqr']:8.2f}" for o in OBS_ORDER if o != 'neutral_n95'))


@torch.no_grad()
def sample_k(model, pre, flow_cfg, dataset, indices, k=8, solver='midpoint', steps=8, seed=7, batch=512):
    from pileflow.evaluate import generate
    s, tm = generate(model, dataset, indices, pre, 'cpu', batch, solver, steps, 1e-5, 1e-5, k, **flow_cfg, seed=seed)
    return s, tm
