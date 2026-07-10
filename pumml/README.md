# PUMML — Pileup Mitigation with Machine Learning

Implementation of the PUMML CNN for pileup mitigation in hadron collider jets,
reproducing the results from:

> **Pileup Mitigation with Machine Learning (PUMML)**  
> Komiske, Metodiev, Thaler (2017)  
> arXiv: [1707.08600](https://arxiv.org/abs/1707.08600)

---

## Overview

In high-luminosity hadron collider experiments, each hard-scatter event is
accompanied by tens to hundreds of additional soft inelastic collisions called
**pileup**. PUMML uses a small CNN (4,711 parameters) to predict the neutral
local-vertex pT image of a jet from its pileup-contaminated counterpart,
significantly outperforming the standard PUPPI algorithm on jet mass, dijet
mass, jet pT, and energy correlation functions.

### What the CNN does

The input is a **3 × 36 × 36** jet image with three channels:

| Channel | Content |
|---------|---------|
| 0 | All neutral pT (9×9 downsampled, then upsampled to 36×36) |
| 1 | Charged pileup pT (36×36) |
| 2 | Charged local-vertex (LV) pT (36×36) |

The output is a **1 × 9 × 9** image predicting neutral LV pT — the pileup-free
neutral component of the jet.

---

## Directory Structure

```
pumml/
├── run.py                     # Single entry point: train + evaluate
│
├── scripts/
│   ├── train.py               # Train the PUMML CNN
│   ├── check_prediction.py    # Quick sanity check on one jet
│   └── plot_average_images.py # Visualise average jet images
│
├── src/
│   ├── models/
│   │   ├── pumml_model.py     # PUMMLNet CNN architecture
│   │   └── loss.py            # Modified log-squared loss (Eq. 2.1)
│   ├── data/
│   │   └── dataset.py         # PUMMLDataset and train/val split
│   └── utils/
│       └── inference.py       # Batch inference helper
│
└── plotting/
    ├── compare.py             # Physics engine: inference + observable computation
    ├── common.py              # Shared constants, labels, re-exports
    ├── plotting.py            # Figures 4 & 5, Tables 1 & 2 (consolidated)
    ├── run_all.py             # Evaluation-only orchestrator
    ├── obs_distributions.py   # Figure 4 standalone
    ├── percent_errors.py      # Figure 5 standalone
    ├── r_IQR_scores.py        # Tables 1 & 2 standalone
    ├── corr_vs_npu.py         # Figure 6: Pearson r vs mean NPU
    ├── corr_vs_mass.py        # Figure 7: Pearson r vs resonance mass
    ├── jet_display.py         # Figure 3: 3D jet image display
    └── filter_weights.py      # Figure 8: Conv1 filter visualisation
```

---

## Run itttt!!!

### 1. Install dependencies

```bash
pip install torch numpy scipy matplotlib fastjet
# or with the provided environment file:
conda env create -f requirements_pflow.yml
```

### 2. Train + evaluate in one command

```bash
python run.py \
    --npz  data/jets_pileup.npz \
    --name my_run
```

All outputs go to `runs/my_run/`:

```
runs/my_run/
    checkpoints/
        pumml_model.pt              # best model weights
        pumml_model_history.npz     # train/val loss per epoch
        pumml_model_loss_curve.png  # loss curve plot
    plots/
        figure4_distributions.png/pdf
        figure5_percent_errors.png/pdf
        tables_1_2.txt
        tables_1_2.csv
```

### 3. Evaluate only (skip training)

```bash
python run.py \
    --npz        data/jets_pileup.npz \
    --name       my_run \
    --skip-train \
    --model      runs/my_run/checkpoints/pumml_model.pt
```

### 4. Train only (skip evaluation)

```bash
python run.py \
    --npz       data/jets_pileup.npz \
    --name      my_run \
    --skip-eval
```

---

## Input Data Format

The `.npz` file produced by **gen4e2e** must contain the following arrays. All
momentum arrays are in GeV; constituent arrays are zero-padded to a fixed length.

| Key | Shape | Description |
|-----|-------|-------------|
| `jet_pt` | (N,) | Reconstructed jet pT [GeV] |
| `jet_eta` | (N,) | Jet pseudorapidity (image centre η) |
| `jet_phi` | (N,) | Jet azimuthal angle (image centre φ) |
| `true_px/py/pz/e` | (N, M) | True neutral constituent 4-momenta |
| `true_n` | (N,) | Number of true neutral constituents per jet |
| `pileup_px/py/pz/e` | (N, M) | Pileup-contaminated constituent 4-momenta |
| `pileup_n` | (N,) | Number of pileup constituents per jet |
| `puppi_px/py/pz/e` | (N, M) | PUPPI-corrected constituent 4-momenta |
| `puppi_n` | (N,) | Number of PUPPI constituents per jet |
| `ch_charged_lv` | (N, 36, 36) | Charged LV pT image |
| `ch_charged_pu` | (N, 36, 36) | Charged pileup pT image |
| `ch_neutral_all` | (N, 36, 36) | All neutral pT (upsampled from 9×9) |
| `ch_neutral_all_raw` | (N, 9, 9) | All neutral pT (raw 9×9) |
| `ch_neutral_lv` | (N, 9, 9) | True neutral LV pT (target) |
| `n_pu` | (N,) | Number of pileup vertices per jet (optional) |

Jets are stored as **consecutive pairs** from pp → jj events: jet 0 and jet 1
form one dijet event, jets 2 and 3 form the next, and so on.

---

## Observables Computed

The following six observables are computed for each method and used in
Figures 4–5 and Tables 1–2 of the paper:

| Observable | Key | Description |
|-----------|-----|-------------|
| Jet mass | `jet_mass` | Invariant mass from anti-kT (R=0.4) clustering |
| Dijet mass | `dijet_mass` | Invariant mass of the two leading jets per event |
| Jet pT | `jet_pt` | Reconstructed transverse momentum |
| Neutral N95 | `neutral_n95` | Minimum pixels covering 95% of neutral pT |
| ln ECF2(β=4) | `ecf2_log` | Log of 2-point energy correlation function |
| ln ECF3(β=4) | `ecf3_log` | Log of 3-point energy correlation function |

**ECF definition:**

```
ECF(2, β) = Σ_{i<j}   z_i z_j  ΔR_{ij}^β
ECF(3, β) = Σ_{i<j<k} z_i z_j z_k  ΔR_{ij}^β ΔR_{ik}^β ΔR_{jk}^β
```

where `z_i = pT_i / pT_jet` (dimensionless momentum fraction) and β = 4.

---

## Training Details

All hyperparameters default to the paper values (Section 2):

| Hyperparameter | Default | Paper |
|---------------|---------|-------|
| Dataset size | all jets in .npz | ~56k |
| Train / val split | 90% / 10% | 90% / 10% |
| Loss | modified log-squared | Eq. 2.1 |
| p_bar | 10 GeV | 10 GeV |
| Optimiser | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Batch size | 50 | 50 |
| Epochs | 25 | 25 |
| Initialisation | He-uniform | He-uniform |
| Parameters | 4,711 | 4,711 |

The **modified log-squared loss** (Eq. 2.1) is:

```
L = [ log(p_pred + p_bar) - log(p_true + p_bar) ]^2
```

averaged over all 9×9 pixels. The softening term `p_bar = 10 GeV` prevents
the network from ignoring low-pT pixels.

---

## run.py — Command Reference

```
python run.py --npz PATH --name NAME [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--npz` | required | Path to jets_pileup.npz |
| `--name` | `run` | Run name; outputs go to `runs/<name>/` |
| `--skip-train` | — | Skip training; requires `--model` |
| `--skip-eval` | — | Skip evaluation; only train |
| `--model` | auto | Checkpoint path (auto-set within run dir) |
| `--epochs` | 25 | Training epochs |
| `--batch` | 50 | Batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--pbar` | 10.0 | Loss softening [GeV] |
| `--train-frac` | 0.9 | Train fraction |
| `--max-jets` | all | Cap jets for evaluation |
| `--device` | auto | `cpu` or `cuda` |
| `--seed` | 42 | Random seed |

---

## plotting/run_all.py — Re-generate plots only

Use this only when you already have a trained model and want to regenerate
figures without retraining (e.g. after tweaking a plot style).
If you haven't trained yet, use `run.py` above — it does everything in one command.

```bash
cd pumml_in_server/
python plotting/run_all.py \
    --npz   data/jets_pileup.npz \
    --model runs/my_run/checkpoints/pumml_model.pt \
    --name  my_run
```

Optional Figure 6 (Pearson r vs NPU — requires multiple datasets):

```bash
python plotting/run_all.py \
    --npz       data/jets_pileup.npz \
    --model     runs/my_run/checkpoints/pumml_model.pt \
    --name      my_run \
    --npz6      data/npu50.npz data/npu100.npz data/npu140.npz \
    --npu-list  50 100 140
```

Optional Figure 7 (Pearson r vs resonance mass):

```bash
python plotting/run_all.py \
    --npz       data/jets_pileup.npz \
    --model     runs/my_run/checkpoints/pumml_model.pt \
    --name      my_run \
    --npz7      data/m500.npz data/m750.npz data/m1000.npz \
    --mass-list 500 750 1000
```

---

## Standalone Plotting Scripts

Each figure can also be regenerated individually. All scripts accept `--npz`,
`--model`, `--out`, `--max-jets`, and `--device`.

| Script | Figure | Output file |
|--------|--------|-------------|
| `plotting/obs_distributions.py` | Figure 4 | `distributions.{png,pdf}` |
| `plotting/percent_errors.py` | Figure 5 | `percent_errors.{png,pdf}` |
| `plotting/r_IQR_scores.py` | Tables 1 & 2 | `tables_1_2.{txt,csv}` |
| `plotting/corr_vs_npu.py` | Figure 6 | `corr_vs_npu.{png,pdf}` |
| `plotting/corr_vs_mass.py` | Figure 7 | `corr_vs_mass.{png,pdf}` |
| `plotting/jet_display.py` | Figure 3 | `jet_display.{png,pdf}` |
| `plotting/filter_weights.py` | Figure 8 | `filter_weights.{png,pdf}` |

Example:

```bash
python plotting/r_IQR_scores.py \
    --npz   data/jets_pileup.npz \
    --model runs/my_run/checkpoints/pumml_model.pt \
    --out   runs/my_run/plots
```

---

## Note (Important for Statistics)

All observables are stored as **fixed-size NaN arrays of length N**. Index `i`
always refers to the same physical jet for every method (True, Pileup, PUPPI,
PUMML). NaN marks jets where clustering failed or an observable could not be
computed.

Statistical comparisons (Pearson r, IQR, percent-error) always use a boolean
mask:

```python
both = np.isfinite(store["true"]["jet_mass"]) & np.isfinite(store["pumml"]["jet_mass"])
r, _ = pearsonr(store["true"]["jet_mass"][both], store["pumml"]["jet_mass"][both])
```

This ensures we always compare the **same jet** across methods. The
approach of truncating to `min(len(true), len(pred))` doesn't work when
different methods fail clustering for different jets.