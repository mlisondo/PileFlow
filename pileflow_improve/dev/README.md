# Development experiments (validation split only)

Environment variables (defaults relative to this folder): `PILEFLOW_DATA` (full-format NPZ),
`PILEFLOW_SPLITS` (train/val/eval indices, `runs/splits_train_val_eval.npz`), `PILEFLOW_RUNS`.

| script | purpose |
|---|---|
| `train_pumml_split.py SEED` | PUMML (paper recipe) trained on the PileFlow train split only (dev baseline) |
| `exp_posterior_avg.py K` | Exp. 1: draft checkpoint, 1 sample vs posterior-mean observables |
| `exp_eval.py CKPT --k K` | score checkpoints on 1000 validation events vs the dev PUMML |
| `exp_k.py CKPT K SOLVER STEPS TAG` | dependence on the number of posterior samples K |

Logs and JSON outputs of the runs reported in `IMPROVEMENTS.md` are in `runs/`.
