"""K-dependence and solver check for a checkpoint (validation subset)."""
import sys, time, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'dev')
from common import *
from pileflow.train import load_checkpoint
from pileflow.data import PileFlowImages
torch.set_num_threads(2)

ck = sys.argv[1]
K = int(sys.argv[2])
solver = sys.argv[3]
steps = int(sys.argv[4])
tag = sys.argv[5]
idx = val_subset(1000)
d = load(idx)
ds = PileFlowImages(DATA, verbose=False)

m, pre, fc, _ = load_checkpoint(ck)

t = time.time()
S, _ = sample_k(m, pre, fc, ds, idx, k=K, solver=solver, steps=steps, seed=11)
print('sampling %.0fs' % (time.time() - t))

np.save(f'{RUNS}/{tag}_samples.npy', S)
T = observables(d, d['truth9'])
obs = [observables(d, S[:, j]) for j in range(K)]
rows = {}
kk = 1

while kk <= K:
    rows[f'{tag} mean-obs K={kk}'] = score(T, combine(obs[:kk], 'mean')); kk *= 2
table(rows, f'{tag}: K dependence ({solver} {steps})')
