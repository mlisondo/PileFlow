"""Score one or more PileFlow checkpoints on the validation subset against PUMML (train-split)."""
import sys, time, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
from common import *
from pileflow.train import load_checkpoint
from pileflow.model import model_config_from_preset
from pileflow.data import PileFlowImages
torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument('ckpts', nargs='+'); ap.add_argument('--k', type=int, default=8)
ap.add_argument('--solver', default='midpoint'); ap.add_argument('--steps', type=int, default=8)
ap.add_argument('--events', type=int, default=1000); ap.add_argument('--tag', default='exp')
a = ap.parse_args()
idx = val_subset(a.events); d = load(idx); ds = PileFlowImages(DATA, verbose=False)
T = observables(d, d['truth9'])
p9, _ = bm.pumml_predict(f'{RUNS}/pumml_trainsplit_seed1.pt', d['X'])
rows = {}
for cut in (0.0, 0.1):
    rows[f'PUMML (cut {cut})'] = score(T, observables(d, p9, cut))
for c in a.ckpts:
    m, pre, fc, ck = load_checkpoint(c, fallback_model_config=model_config_from_preset('spatial'))
    name = c.split('/')[-3]
    t = time.time(); S, tm = sample_k(m, pre, fc, ds, idx, k=a.k, solver=a.solver, steps=a.steps)
    print(name, 'sampling %.0fs' % (time.time() - t), flush=True)
    np.save(f'{RUNS}/{a.tag}_{name}_samples.npy', S)
    for cut in (0.0, 0.1):
        obs = [observables(d, S[:, j], cut) for j in range(a.k)]
        rows[f'{name[:18]} 1 sample (cut {cut})'] = score(T, obs[0])
        rows[f'{name[:18]} mean-obs K={a.k} (cut {cut})'] = score(T, combine(obs, 'mean'))
    tt, pp = d['truth9'].sum((1, 2)), S[:, 0].sum((1, 2))
    print(name, 'neutral pT bias %.2f GeV, rmse %.2f' % ((pp - tt).mean(), np.sqrt(((pp - tt) ** 2).mean())))
table(rows, f'VALIDATION subset ({len(idx)} jets): Pearson % | IQR %')
json.dump(rows, open(f'{RUNS}/{a.tag}.json', 'w'), indent=1, default=float)
