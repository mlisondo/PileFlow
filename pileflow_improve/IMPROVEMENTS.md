# PileFlow improvements (pileflow_improve)

This folder starts from `pileflow_draft`, which is left unchanged, together with its 5-seed
results. The aim was to make PileFlow competitive with PUMML on the paper's metrics.

## Diagnosis

Two facts about the data and the metrics explained why the draft PileFlow lost to PUMML.

1. **The target is almost empty and bounded.**
   - 89% of the neutral-LV target cells are exactly zero.
   - Where a cell has any neutral pT, 84% of those cells hold no LV neutral pT.
   - The LV neutral pT never exceeds the total neutral pT (true in 99.94% of cells).

   The draft modelled standardised log1p(pT). A continuous flow cannot put mass exactly on
   zero, so it smeared small fake pT into empty cells. This caused:
   - the N95 cut dependence;
   - a neutral-pT bias varying from −1.4 to +4.2 GeV between seeds;
   - extra jet-mass and ECF noise.

2. **Per-jet metrics favour point estimates.** A single draw from a calibrated posterior has
   about twice the error variance of the posterior mean. Pearson correlations and IQRs of
   per-jet errors therefore penalise any generative model that reports a single sample.

## Changes

- **`--target-mode fraction`** (`pileflow/preprocess.py`; now the default).
  - The flow models the per-cell LV fraction of the measured neutral pT, f = y / total ∈ [0, 1].
  - Exact zeros are dequantised uniformly into [−0.1, 0), as FlowSim does for integer targets.
  - On output: f = clip(x, 0, 1), y = f · total, and cells with no neutral pT are exactly zero.
  - Consequences, by construction:
    - PileFlow can never create neutral pT that was not measured.
    - Empty cells come out exactly empty.
    - No cell threshold is needed.
- **Posterior-mean observables** (`compare_methods.py --add-posterior-obs`,
  `pileflow.benchmark.posterior_mean_observables`).
  - Each observable is computed on each of the K samples of a jet and then averaged. This is the
    minimum-MSE estimate of the observable.
  - A regression model like PUMML cannot provide this: its single image is only an average, and
    the observables are non-linear in the image.
  - The single-sample output is still reported. It remains the right estimator for
    distribution-level results (Fig. 4, Wasserstein).
- **Architecture:** `spatial_v2` with width 48 and 4 blocks (220,801 parameters).
  - Charged encoder aligned with the neutral cells (PixelUnshuffle).
  - FiLM time conditioning.
  - Scaled time embedding.
- **Training:**
  - EMA of the weights.
  - Fixed-noise validation.
  - 80 epochs with Adam at 1e-3.
  - The train/val split is the one used by the earlier checkpoints (`--splits-file`).
- **Sampling:** midpoint ODE solver, 8 steps (16 network evaluations). Measured with the draft
  model on 256 validation jets, its per-jet neutral pT differs from dopri5 by 0.18 GeV on average
  (0.25%).
- **Other additions:**
  - `spatial_v3` preset (adds coordinate channels)
  - `--lr-schedule cosine`
  - `--init-from`
  - `dev/` experiment scripts

## Development on the validation split

- **Validation set:** 1,000 events (1,849 jets) of the PileFlow validation split.
- **Comparison model:** PUMML retrained with the paper recipe on the PileFlow training split only
  (`dev/train_pumml_split.py`), so both models are unseen on these jets.

| Model | Jet mass r | Dijet r | Jet pT r | N95 r | ECF2 r | ECF3 r | Jet mass IQR |
|---|---|---|---|---|---|---|---|
| PUMML (100 MeV cut) | 97.34 | 99.55 | 99.66 | 77.7 | 90.1 | 76.8 | 12.9 |
| Draft PileFlow, 1 sample | 94.6 | 98.76 | 98.84 | 67.5 | 83.5 | 65.4 | 19.7 |
| Draft PileFlow, posterior mean (K = 8) | 97.09 | 99.38 | 99.44 | 75.6 | 89.9 | 77.4 | 15.2 |
| New PileFlow, 1 sample | 95.7 | 99.31 | 99.49 | 71.8 | 82.6 | 65.4 | 17.3 |
| New PileFlow, posterior mean (K = 32) | **97.39** | **99.57** | **99.69** | **83.8** | **90.7** | **80.9** | 13.6 |

What else was tried:

- **Number of samples K:** every metric improves up to about K = 32 and then saturates.
- **40 more epochs with a cosine schedule:** no gain.
- **A bigger model** (`spatial_v3`, 64 × 6, 530k parameters, with coordinate channels): same
  results within the noise.
- **Final choice:** the 220k-parameter model with K = 32 and the posterior mean.

## Final result on the held-out test set (one run, `event_test_idx`)

- 5,594 jets: 2,998 leading jets and 2,596 dijet events.
- PUMML is the paper-recipe seed1 trained on `event_train_idx`.
- Paired bootstrap with 2,000 replicas.
- Full outputs are in `results/test_A_K32/`.

Pearson correlation (%) with the 100 MeV cut applied to every method; PUMML is at its best here:

| | w. Pileup | PUPPI | PUMML | PileFlow, 1 sample | **PileFlow, posterior mean** | Δ vs PUMML [95% CI] |
|---|---|---|---|---|---|---|
| Jet mass | 64.0 | 93.2 | 96.9 | 95.0 | **97.0** | +0.16 [+0.04, +0.27] |
| Dijet mass | 87.2 | 97.2 | 99.5 | 99.2 | **99.5** | +0.04 [+0.01, +0.07] |
| Jet pT | 93.7 | 98.0 | 99.5 | 99.3 | **99.6** | +0.04 [+0.02, +0.06] |
| Neutral N95 | 31.6 | 73.3 | 77.4 | 71.5 | **84.2** | +6.75 [+5.50, +7.95] |
| ln ECF2 | 59.7 | 80.7 | 89.6 | 83.8 | **91.1** | +1.48 [+0.89, +2.12] |
| ln ECF3 | 41.8 | 62.9 | 76.7 | 67.5 | **81.4** | +4.67 [+3.11, +6.31] |

IQR of centred percent errors (%), 100 MeV cut, lower is better:

| | PUPPI | PUMML | PileFlow, posterior mean | Δ vs PUMML [95% CI] |
|---|---|---|---|---|
| Jet mass | 30.6 | 13.73 | 13.79 | +0.07 [−0.44, +0.82] (tie) |
| Dijet mass | 3.23 | 2.04 | **1.96** | −0.08 [−0.15, +0.02] |
| Jet pT | 3.47 | 2.37 | **2.25** | −0.12 [−0.22, −0.02] |
| ln ECF2 | 9.58 | 5.96 | **5.68** | −0.28 [−0.57, −0.01] |
| ln ECF3 | 12.3 | 8.52 | **8.16** | −0.36 [−0.76, −0.03] |

Without any cut, PUMML's N95 correlation drops to 55.7, while PileFlow's is unchanged, because its
zeros are exact. The posterior-mean PileFlow then beats PUMML significantly on
every correlation and on the dijet, jet pT and ECF3 IQRs. The jet-mass and ECF2 IQRs also favour
PileFlow, but their 95% intervals touch zero.

Distribution fidelity: 1-D Wasserstein distance to the truth, 100 MeV cut (lower is better):

| | Jet mass | Dijet | Jet pT | N95 | ECF2 | ECF3 |
|---|---|---|---|---|---|---|
| PUMML | 0.94 | 5.64 | 2.44 | 0.22 | 0.18 | 0.48 |
| PileFlow, 1 sample | **0.13** | **1.46** | **0.60** | **0.10** | **0.03** | **0.08** |

**Calibration.** The truth falls inside the central 68% (90%) of the 32-sample posterior for 62%
(83%) of the jets' neutral pT. The posterior is slightly too narrow.

**Cost.** Each jet needs K × 16 = 512 network evaluations. Throughput on 2 CPU cores:

- PileFlow: ~6 jets/s
- PUMML: ~45,000 jets/s

A GPU (and fewer samples: K = 16 already beats PUMML on most metrics) closes much of this gap.
PUMML will always be faster.

## Summary

With the fraction parameterisation and posterior-mean observables, PileFlow now:

- **Per jet:** beats PUMML significantly on all six correlations, and on the IQRs of jet pT and
  both ECFs. The dijet-mass IQR also favours PileFlow but its interval touches zero (p = 0.075),
  and the jet-mass IQR is a statistical tie.
- **Distributions:** beats PUMML on every observable, using single samples.
- **Uncertainty:** provides a per-jet uncertainty.
- **Cuts:** needs no neutral-cell cut.

The only area where PileFlow is behind is speed.

## Reproduce

```bash
python -m pileflow.run --data DATA.npz --splits-file splits.npz --arch spatial_v2 --base-channels 48 \
       --n-blocks 4 --target-mode fraction --epochs 80 --patience 20 --eval-splits --outdir runs/A
python evaluate_checkpoint.py --data DATA.npz --checkpoint runs/A/checkpoints/pileflow_best.pt \
       --split eval --eval-samples 32 --ode-solver midpoint --ode-steps 8 --outdir runs/A
python compare_methods.py --data DATA.npz --pumml-model pumml_56k_seed1.pt \
       --pileflow runs/A/data/generated_eval.npz --add-posterior-obs --outdir results/test_A_K32
```

On a GPU, drop `--splits-file` to use the default event-grouped split, and train 5 seeds before
the final paper numbers.
