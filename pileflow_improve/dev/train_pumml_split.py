"""Dev baseline: PUMML (paper recipe) trained on the PileFlow TRAIN split only, so that
the PileFlow validation split is also unseen by PUMML and can be used for model selection."""
import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from pumml.pumml_arch import PUMML, PUMMLLoss
import os
RUNS = os.environ.get('PILEFLOW_RUNS', 'runs')
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = sys.argv[2] if len(sys.argv) > 2 else f'{RUNS}/pumml_trainsplit_seed{seed}.pt'
torch.manual_seed(seed); rng = np.random.default_rng(seed)
d = np.load(os.environ.get('PILEFLOW_DATA', 'data/full.npz')); sp = np.load(os.environ.get('PILEFLOW_SPLITS', 'runs/splits_train_val_eval.npz'))
X = torch.from_numpy(np.ascontiguousarray(d['X'])); y = torch.from_numpy(d['y_neutral_lv_9']).unsqueeze(1)
tr = sp['train']; m = PUMML(); crit = PUMMLLoss(10.0); opt = torch.optim.Adam(m.parameters(), 1e-3)
t0 = time.time()
for ep in range(25):
    perm = rng.permutation(tr); tot = 0
    for s in range(0, len(perm), 50):
        i = torch.from_numpy(perm[s:s+50]); opt.zero_grad(); l = crit(m(X[i]), y[i]); l.backward(); opt.step(); tot += l.item()*len(i)
    print(ep+1, tot/len(tr), f'{time.time()-t0:.0f}s', flush=True)
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({'model_state_dict': m.state_dict(), 'train_idx': tr, 'split_name': 'pileflow_train_split'}, out)
