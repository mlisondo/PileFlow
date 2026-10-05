# PileFlow (improved)

> This folder (`pileflow_improve`) continues `pileflow_draft`. The draft folder and its results are
> unchanged. See `IMPROVEMENTS.md` for what changed and the validation/test results.

Pileup mitigation for jet images using conditional flow matching. PileFlow is benchmarked against
PUMML (Komiske, Metodiev, Nachman, Schwartz, [arXiv:1707.08600](https://arxiv.org/abs/1707.08600))
and PUPPI.

Like PUMML, PileFlow takes three images of a jet as input:

- total neutral pT on the 9×9 calorimeter grid
- charged pileup pT on the 36×36 tracker grid
- charged leading-vertex pT on the 36×36 tracker grid

It predicts the **neutral leading-vertex** 9×9 image. The difference from PUMML is that PileFlow is
*generative*. PUMML regresses a single image. PileFlow learns the conditional distribution
p(neutral LV | inputs), using the target-conditional flow matching of FlowSim
([arXiv:2402.13684](https://arxiv.org/abs/2402.13684)). It can therefore give several plausible
images per jet, with a calibrated per-jet uncertainty.

## Repository layout

```
pileflow/            PileFlow package
  data.py            dataset, event-grouped splits, fast in-memory batch loader
  preprocess.py      log1p + standardisation, fitted on the training split only
  model.py           velocity networks + presets (spatial, spatial_v2, spatial_small, flat)
  flow.py            target-conditional flow matching; euler/midpoint/rk4/dopri5 samplers
  train.py           training (EMA, fixed-noise validation, early stopping), checkpoint I/O
  evaluate.py        sampling on a split -> generated_<split>.npz + neutral-image metrics
  observables.py     PUMML observables (single implementation used by every script)
  stats.py           Pearson, IQR, Wasserstein, paired bootstrap
  benchmark.py       shared benchmark: representations, tables, plots
  run.py             train + evaluate CLI (python -m pileflow.run)
pumml/               PyTorch PUMML reproduction (arch, training, paper-table check)
generator/gen4pf.py  Pythia8/FastJet dataset generator (PUMML setup, <NPU> = 140)
evaluate_checkpoint.py   sample a checkpoint on val or eval (config + splits read from the checkpoint)
select_threshold.py      choose the PileFlow cell threshold on validation jets
compare_methods.py       final benchmark: w. Pileup / PUPPI / PUMML / PileFlow
plot_average_channels.py, plot_qualitative_jets.py   figures
tests/               unit tests + end-to-end test on a synthetic dataset
tools/inspect_npz.py
```

## Installation

```bash
pip install -r requirements.txt
python -m pytest -q tests        # ~1 min on CPU
```

Run every command from the repository root.

## Full workflow

```bash
DATA=generator/ppjj_scalar500_npu140_56k_v1.npz

# 1. PUMML baseline: the paper recipe, plus an optional val-selected variant
python -m pumml.run_pumml --data $DATA --split event --output runs/pumml/pumml_event.pt
python -m pumml.run_pumml --data $DATA --split event --select-on-val --output runs/pumml/pumml_event_val.pt
python -m pumml.evaluate  --data $DATA --model runs/pumml/pumml_event.pt --output-dir results/pumml_repro

# 2. PileFlow (samples val + eval once trained)
python -m pileflow.run --data $DATA --arch spatial_v2 --outdir runs/pileflow_v2
#    optional parameter-matched model (~18k parameters) for a fair-size comparison
python -m pileflow.run --data $DATA --arch spatial_small --outdir runs/pileflow_small

# 3. Several samples per jet, for posterior summaries and coverage (use >= 20 for coverage)
python evaluate_checkpoint.py --data $DATA --checkpoint runs/pileflow_v2/checkpoints/pileflow_best.pt \
       --split val  --eval-samples 20 --outdir runs/pileflow_v2
python evaluate_checkpoint.py --data $DATA --checkpoint runs/pileflow_v2/checkpoints/pileflow_best.pt \
       --split eval --eval-samples 20 --outdir runs/pileflow_v2

# 4. Freeze the neutral-cell threshold on VALIDATION jets
python select_threshold.py --data $DATA --generated runs/pileflow_v2/data/generated_val.npz \
       --output runs/pileflow_v2/threshold.json

# 5. Benchmark on the held-out event-grouped test set
python compare_methods.py --data $DATA \
       --pumml-model runs/pumml/pumml_event.pt --pumml-tuned-model runs/pumml/pumml_event_val.pt \
       --pileflow runs/pileflow_v2/data/generated_eval.npz \
       --pileflow-small runs/pileflow_small/data/generated_eval.npz \
       --threshold-json runs/pileflow_v2/threshold.json --add-posterior-mean \
       --outdir results/comparison
```

Appendix study: a neutral-cell threshold applied uniformly to all methods (Pearson vs. threshold
with paired-bootstrap bands, N95 bias, and the cuts that reproduce the PUMML paper):

```bash
python threshold_scan.py --data $DATA --pumml-model runs/pumml/pumml_event.pt \
       --pileflow runs/pileflow_v2/data/generated_eval.npz --outdir results/threshold_scan
```

### Several training seeds (paper figures and tables)

```bash
python multiseed_benchmark.py --data $DATA \
       --pumml-models runs/pumml/pumml_seed{1..5}.pt \
       --pileflow runs/pf_seed{42..46}/data/generated_eval.npz \
       --n95-scan --outdir results/paper
```

For each variant (`nocut/`, `cut_100MeV/`): PUMML-style Figs. 4, 5, 6 and paired-bootstrap tables for the
reference seeds (the first model of each list, fixed a priori) in `reference_seed/`; Tables 1-2 in the paper
format with mean ± std over seeds (`tables/table1_correlation.tex`, `table2_iqr.tex`), a reproduction table
next to the published values, per-seed CSVs, the seed-spread figure and Fig. 6 with the seed band. With
`--n95-scan`, the appendix figure of N95 correlation/bias versus threshold for every seed. The data file may be
a slim test-only copy that stores `orig_index` (original row numbers).

### Neutral-cell cut (on / off / both)

`compare_methods.py` and `python -m pumml.evaluate` run the whole benchmark **twice by default**:
once with no cut and once with the same neutral-cell zero-suppression applied to every method
(`--cell-threshold`, default 0.1 GeV; the truth stays untouched unless `--threshold-truth`).

```
results/comparison/
  nocut/        tables + Fig. 4/5/6 without any cut
  cut_100MeV/   same, with the 100 MeV cut on every method
  tables/cut_vs_nocut.{csv,json,tex}   r without/with the cut, paired-bootstrap delta r, IQR, N95 bias
  plots/cut_vs_nocut_correlations.*    Pearson r per method, no cut (open) vs cut (filled)
  plots/cut_vs_nocut_n95.*             N95 distributions, no cut (dashed) vs cut (solid)
```

Switch it with `--cut-mode off` (no cut only), `--cut-mode on` (cut only) or `--cut-mode both`.
`plot_qualitative_jets.py --cell-threshold 0.1` shows the jets with the same cut.

`compare_methods.py` produces the following in `results/comparison/`:

- **Correlation and IQR tables** (PUMML Tables 1–2) with paired-bootstrap 95% CIs, as CSV, LaTeX
  and JSON
- **The difference of every method relative to PUMML**, with a CI and a one-sided
  p = P(not better)
- **1-D Wasserstein distances**, measuring distribution fidelity (as in FlowSim)
- **Jet-mass correlation in bins of NPU** (analogue of PUMML Fig. 6)
- **Sampling spread** of PileFlow's correlations across independent samples
- **Throughput and parameter counts**
- **Figures** in the style of PUMML Figs. 4, 5 and 6

## Evaluation protocol (what makes the comparison fair)

- **Same jets.** Every method is evaluated on `event_test_idx`, an event-grouped split in which
  no hard event contributes jets to both training and test. PileFlow's train/validation split is
  carved out of `event_train_idx` in the same way. The checkpoint stores the split, and the
  evaluation scripts refuse a mismatched one.
- **Same observables.** `pileflow/observables.py` is the only implementation.
  - Charged pT sits at 36×36 cell centres and neutral pT at 9×9 calorimeter-cell centres,
    following PUMML's "detector discretization".
  - Positions are in global (η, φ) coordinates.
  - N95 counts neutral cells.
  - ECFs use β = 4 with z_i = pT_i/ΣpT; log(0) is recorded as NaN.
  - Single-jet observables use leading jets only; the dijet mass uses complete rank-0/rank-1 pairs.
- **Same statistics.** Correlations and IQRs are computed on the jets that are finite for truth
  *and* for every method. Uncertainties come from a *paired* bootstrap (the same resampled jets
  for every method), so the uncertainty on a difference between methods is honest.
- **No tuning on the test set.** The cell threshold is chosen by `select_threshold.py` on
  validation jets. `compare_methods.py` rejects a threshold file that was not produced on the
  validation split, and a PileFlow file that was not generated on the evaluation split.
- **Honest generative output.** By default a single stochastic sample per jet is scored
  (`--pileflow-estimator sample`). Posterior means or medians are reported separately and are
  labelled as such.
- **Size and speed.** The `spatial_small` preset (~18k parameters) allows a comparison at a size
  close to PUMML's (4,711 parameters). Throughput is written to `summary.json`.

## Model presets

| preset | params | notes |
|---|---|---|
| `spatial` | 505k | Original draft architecture. Kept exactly so that old checkpoints load. |
| `spatial_v2` | 528k | **Recommended.** Charged encoder aligned with the neutral cells (PixelUnshuffle 4×4 → 9×9 cells, exact cell match); FiLM time conditioning; time embedding scaled so it covers its full frequency range. |
| `spatial_small` | 18k | `spatial_v2` design at a size close to PUMML's. |
| `flat` | 3.4M | Flattened MLP velocity field in the style of FlowSim's CRT, kept as an ablation. |

Training defaults:

- FlowSim settings: σ_min = 1e-4, α = 1, Adam at 1e-3, ReduceLROnPlateau.
- EMA of the weights (decay 0.999, with warm-up).
- The validation loss uses fixed noise and fixed t, so early stopping tracks the model rather
  than Monte-Carlo noise.
- AMP is enabled on CUDA.
- The optional "morphology" loss (`--morph-lambda`) only acts at t ≥ `--morph-tmin`.

## Checkpoints

Checkpoints in format 2 store the following, and `evaluate_checkpoint.py` rebuilds everything from
them:

- `model_config`
- the preprocessing config and its fitted statistics
- the flow config
- the splits
- the history
- the training config
- metadata

Checkpoints from the original draft still load: architecture `spatial` is assumed, which can be
changed with `--legacy-arch`.
