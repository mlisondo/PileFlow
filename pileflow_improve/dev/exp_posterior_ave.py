import os
"""Experiment 1: does averaging OBSERVABLES over K posterior samples close the gap to PUMML?"""
import sys, time, json, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
from common import *
from pileflow.train import load_checkpoint
from pileflow.model import model_config_from_preset
from pileflow.data import PileFlowImages
torch.set_num_threads(2)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
idx = val_subset(1000)
d = load(idx)
ds = PileFlowImages(DATA, verbose=False)
m, pre, fc, _ = load_checkpoint(os.environ.get('PILEFLOW_OLD_CKPT', '../pileflow_draft/runs/pileflow_56k_morph001_seed42/checkpoints/pileflow_best.pt'),
                                fallback_model_config=model_config_from_preset('spatial'))

t = time.time()
S, tm = sample_k(m, pre, fc, ds, idx, k=K)
print('sampling', time.time() - t, tm)

np.save(f'{RUNS}/exp1_samples_old_seed42.npy', S)

p9, _ = bm.pumml_predict(f'{RUNS}/pumml_trainsplit_seed1.pt', d['X'])
rows = {}

for cut in (0.0, 0.1):
    T = observables(d, d['truth9'], 0.0)
    rows[f'PUMML (cut {cut})'] = score(T, observables(d, p9, cut))
    if 'puppi_neutral9' in d:
        rows[f'PUPPI (cut {cut})'] = score(T, compute_observables(d['puppi_charged36'], bm.apply_threshold(d['puppi_neutral9'], cut), d['eta'], d['phi'], d['event_id'], d['jet_rank']))
    obs = [observables(d, S[:, j], cut) for j in range(K)]
    rows[f'PF old: 1 sample (cut {cut})'] = score(T, obs[0])
    rows[f'PF old: mean obs K={K} (cut {cut})'] = score(T, combine(obs, 'mean'))
    rows[f'PF old: median obs K={K} (cut {cut})'] = score(T, combine(obs, 'median'))
    rows[f'PF old: obs of mean img (cut {cut})'] = score(T, observables(d, S.mean(1), cut))
table(rows, f'VALIDATION subset ({len(idx)} jets), Pearson % | IQR %')
json.dump(rows, open(f'{RUNS}/exp1.json', 'w'), indent=1, default=float)
