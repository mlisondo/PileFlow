# PileFlow Model

`flows/` contains the **PileFlow model package** for pileup mitigation studies.

This package does **not** generate collision data. It consumes data already produced by the repository’s `generator/` package, trains a Target Conditional Flow Matching model, generates pileup-mitigated outputs, and optionally runs comparison plots against truth, pileup, PUPPI, and PUMML.

---

## Role inside the repository

The intended monorepo separation is:

```text
PileFlow/
├── generator/     # produces simulated jet datasets
├── puppi/         # standalone PUPPI baseline
├── pumml/         # PUMML model / external comparison code
├── flows/         # PileFlow model training and generation
└── comparison/    # observable-level comparison plots and tables
```

`flows/` should only do the following:

```text
1. Load generator outputs.
2. Train the PileFlow flow model.
3. Generate PileFlow-mitigated neutral-LV images and scalar observables.
4. Call comparison code if requested.
```

It should **not** run MadGraph, Pythia, FastJet event generation, PUPPI reconstruction, or image construction. Those belong in `generator/` and `puppi/`.

---

## Required input data

Before running `flows/`, first run the generator and produce:

```text
jets_..._antikt_R0.4.npy
jets_..._antikt_R0.4_pileup_images.npz
```

The `.npy` file contains the 25-column tabular jet features.

The `.npz` file contains the image arrays and constituent arrays needed for PileFlow training and evaluation.

Example generator output location:

```text
generator/data/local_test_runs/run_ppjj_20260708_123456/antikt_R0.4/
├── jets_ppjj_antikt_R0.4.npy
├── jets_ppjj_antikt_R0.4_pileup_images.npz
├── jets_ppjj_antikt_R0.4_metadata.json
└── jets_ppjj_antikt_R0.4_preview.txt
```

---

## What PileFlow learns

PileFlow uses Target Conditional Flow Matching to learn a conditional map from pileup-contaminated jet information to a pileup-mitigated target.

The model conditions on a **253-dimensional context vector** and generates a **97-dimensional target vector**.

---

## Context vector

The context vector has dimension:

```text
253 = 7 + 3 + 81 + 81 + 81
```

| Slice | Dim | Content |
|---:|---:|---|
| `[0:7]` | 7 | Generator-level scalar features |
| `[7:10]` | 3 | Jet flavour one-hot encoding |
| `[10:91]` | 81 | Neutral-all image at 9×9 |
| `[91:172]` | 81 | Charged-pileup image at 9×9 |
| `[172:253]` | 81 | Charged-LV image at 9×9 |

The 7 generator-level scalar features are:

```text
pt_gen, eta_gen, phi_gen, m_gen, muon_pT, jetR, jetArea
```

They are read from the `.npy` columns:

```python
[0, 1, 2, 3, 9, 22, 24]
```

The flavour column is:

```python
4
```

The flavour labels are converted into three classes:

```text
light/gluon, c, b
```

---

## Target vector

The target vector has dimension:

```text
97 = 81 + 16
```

| Slice | Dim | Content |
|---:|---:|---|
| `[0:81]` | 81 | Neutral-LV pT image at 9×9 |
| `[81:97]` | 16 | Reconstructed scalar jet observables |

The neutral-LV image target is read from:

```text
ch_neutral_lv
```

The 16 scalar targets are read from the `.npy` columns:

```python
[5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
```

These correspond to reconstructed/tagging/constituent observables such as:

```text
btag, recoPt, recoPhi, recoEta, recoNConst, fractions, qgl, jetId,
ncharged, nneutral, ctag, nSV, recoMass
```

---

## Image handling

The generator `.npz` provides both 36×36 and 9×9 images.

PileFlow uses the following adapter logic:

| Generator key | Shape | PileFlow use |
|---|---:|---|
| `ch_neutral_all_raw` | `(N, 9, 9)` | Used directly as neutral-all context |
| `ch_charged_pu` | `(N, 36, 36)` | Sum-pooled to 9×9 charged-PU context |
| `ch_charged_lv` | `(N, 36, 36)` | Sum-pooled to 9×9 charged-LV context |
| `ch_neutral_lv` | `(N, 9, 9)` | Neutral-LV target |

The charged images are converted from 36×36 to 9×9 by summing each 4×4 block:

```text
36×36 → 9×9 by 4×4 sum-pooling
```

This preserves total pT.

Do **not** use average pooling here.

---

## Package structure

```text
flows/
├── __init__.py
├── config.py
├── environment.yml
├── README.md
├── runner.py
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── pileflow.py
└── training/
    ├── __init__.py
    └── train_flow.py
```

### Main files

| File | Purpose |
|---|---|
| `config.py` | Central configuration dataclass |
| `runner.py` | CLI entry point for training/generation/evaluation |
| `data/dataset.py` | Adapter from generator `.npy`/`.npz` to PileFlow tensors |
| `models/pileflow.py` | CRT velocity field, context encoder, target preprocessor |
| `training/train_flow.py` | Training loop and generation function |
| `environment.yml` | Conda environment for the PileFlow model |

---

## Setup

From the repository root:

```bash
cd ~/PileFlow
conda env create -f flows/environment.yml
conda activate pflow
```

If the environment already exists:

```bash
conda activate pflow
```

---

## How to run

Run all `flows/` commands from the repository root.

Use:

```bash
python -m flows.runner
```

Do **not** run:

```bash
python flows/runner.py
```

because `flows/runner.py` uses package-relative imports.

---

## Smoke test

Use this first after generating a small dataset:

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --data-npy path/to/jets_..._antikt_R0.4.npy \
  --data-npz path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --outdir data/flows/smoke \
  --device cpu \
  --flow-epochs 2 \
  --flow-batch 4 \
  --eval-batch 4 \
  --max-jets 8 \
  --skip-eval
```

This checks that:

```text
1. imports work
2. the dataset adapter works
3. the model trains for two epochs
4. a checkpoint is written
5. the loss curve is written
```

The `--skip-eval` flag skips the comparison plots, which makes the first smoke test simpler.

---

## Smoke test with generation and comparison

After the training smoke test works, run:

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --data-npy path/to/jets_..._antikt_R0.4.npy \
  --data-npz path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --outdir data/flows/smoke_eval \
  --device cpu \
  --flow-epochs 2 \
  --flow-batch 4 \
  --eval-batch 4 \
  --max-jets 8
```

This additionally writes:

```text
data/generated_jets.npz
plots/figure4_distributions.png
plots/figure5_percent_errors.png
plots/tables_1_2.txt
```

If no PUMML checkpoint is provided, the PUMML column is skipped.

---

## Standard training run

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --data-npy generator/data/path/to/jets_..._antikt_R0.4.npy \
  --data-npz generator/data/path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --outdir data/flows/exp1 \
  --device cuda \
  --flow-epochs 800 \
  --flow-batch 512
```

---

## Training with PUMML comparison

PUMML is not trained inside `flows/`.

To include PUMML in the comparison plots, pass an existing trained PUMML checkpoint:

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --data-npy generator/data/path/to/jets_..._antikt_R0.4.npy \
  --data-npz generator/data/path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --pumml-ckpt pumml/checkpoints/pumml_model.pt \
  --outdir data/flows/exp1 \
  --device cuda \
  --flow-epochs 800 \
  --flow-batch 512
```

If `--pumml-ckpt` is omitted, comparison plots include:

```text
True | w/ Pileup | PUPPI | PileFlow
```

If `--pumml-ckpt` is provided, comparison plots include:

```text
True | w/ Pileup | PUPPI | PUMML | PileFlow
```

---

## Reuse an existing PileFlow checkpoint

To skip training and generate/evaluate from an existing checkpoint:

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --skip-flow \
  --data-npy generator/data/path/to/jets_..._antikt_R0.4.npy \
  --data-npz generator/data/path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --flow-ckpt data/flows/exp1/checkpoints/pileflow_best.pt \
  --outdir data/flows/exp1 \
  --device cuda
```

---

## Skip evaluation

To train only:

```bash
cd ~/PileFlow

python -m flows.runner \
  --skip-gen \
  --skip-eval \
  --data-npy generator/data/path/to/jets_..._antikt_R0.4.npy \
  --data-npz generator/data/path/to/jets_..._antikt_R0.4_pileup_images.npz \
  --outdir data/flows/exp1 \
  --device cuda
```

This writes the checkpoint and training loss curve, but does not generate `generated_jets.npz` or comparison plots.

---

## CLI reference

| Flag | Default | Description |
|---|---:|---|
| `--skip-gen` | off | Required. `flows/` does not generate data. |
| `--skip-flow` | off | Skip training and use an existing `--flow-ckpt`. |
| `--skip-eval` | off | Skip generation and comparison plots. |
| `--data-npy` | `None` | Path to generator `.npy` jet table. |
| `--data-npz` | `None` | Path to generator `.npz` image/constituent file. |
| `--flow-ckpt` | `None` | Existing checkpoint path, or output checkpoint path. |
| `--pumml-ckpt` | `None` | Optional external PUMML checkpoint for comparison. |
| `--process-name` | `ppjj` | Metadata/logging label only. |
| `--max-jets` | `None` | Optional cap on jets for smoke tests/debugging. |
| `--flow-epochs` | `800` | Number of training epochs. |
| `--flow-batch` | `512` | Training batch size. |
| `--flow-lr` | `1e-4` | Adam learning rate. |
| `--flow-hidden` | `512` | CRT residual network hidden dimension. |
| `--flow-blocks` | `8` | Number of residual blocks. |
| `--flow-sigma-min` | `1e-4` | TCFM sigma_min. |
| `--flow-dropout` | `0.1` | Dropout during training. |
| `--flow-patience` | `60` | Early stopping patience. Use `0` to disable. |
| `--eval-batch` | `512` | Batch size during generation/evaluation. |
| `--ode-steps` | `100` | Euler integration steps for generation. |
| `--device` | auto | `cuda` if available, otherwise `cpu`. |
| `--seed` | `42` | Random seed. |
| `--outdir` | `output` | Output directory. |

---

## Input data format

### `jets_*.npy`

Expected shape:

```text
(N, 25)
```

Required columns:

| Column | Name | PileFlow role |
|---:|---|---|
| 0 | `pt_gen` | Context scalar |
| 1 | `eta_gen` | Context scalar |
| 2 | `phi_gen` | Context scalar |
| 3 | `m_gen` | Context scalar |
| 4 | `flavour` | Context flavour label |
| 5 | `btag` | Target scalar |
| 6 | `recoPt` | Target scalar |
| 7 | `recoPhi` | Target scalar |
| 8 | `recoEta` | Target scalar |
| 9 | `muon_pT` | Context scalar |
| 10 | `recoNConst` | Target scalar |
| 11 | `nef` | Target scalar |
| 12 | `nhf` | Target scalar |
| 13 | `cef` | Target scalar |
| 14 | `chf` | Target scalar |
| 15 | `qgl` | Target scalar |
| 16 | `jetId` | Target scalar |
| 17 | `ncharged` | Target scalar |
| 18 | `nneutral` | Target scalar |
| 19 | `ctag` | Target scalar |
| 20 | `nSV` | Target scalar |
| 21 | `recoMass` | Target scalar |
| 22 | `jetR` | Context scalar |
| 23 | `algoCode` | Not used by PileFlow |
| 24 | `jetArea` | Context scalar |

---

### `jets_*_pileup_images.npz`

Required for training:

| Key | Shape | PileFlow role |
|---|---:|---|
| `ch_neutral_lv` | `(N, 9, 9)` | Target neutral-LV image |
| `ch_neutral_all_raw` | `(N, 9, 9)` | Neutral-all context image |
| `ch_charged_pu` | `(N, 36, 36)` | Charged-PU context image, sum-pooled to 9×9 |
| `ch_charged_lv` | `(N, 36, 36)` | Charged-LV context image, sum-pooled to 9×9 |

Required for comparison plots:

| Key family | Shape | Purpose |
|---|---:|---|
| `jet_eta`, `jet_phi`, `n_pu` | `(N,)` | Jet metadata |
| `true_px`, `true_py`, `true_pz`, `true_e`, `true_n` | `(N, M)` / `(N,)` | Truth constituents |
| `pileup_px`, `pileup_py`, `pileup_pz`, `pileup_e`, `pileup_n` | `(N, M)` / `(N,)` | Pileup-contaminated constituents |
| `puppi_px`, `puppi_py`, `puppi_pz`, `puppi_e`, `puppi_n` | `(N, M)` / `(N,)` | PUPPI-mitigated constituents |

---

## Output files

A typical run writes:

```text
data/flows/exp1/
├── checkpoints/
│   ├── pileflow_best.pt
│   └── pileflow_best_history.npz
├── plots/
│   ├── pileflow_loss.png
│   ├── figure4_distributions.png
│   ├── figure4_distributions.pdf
│   ├── figure5_percent_errors.png
│   ├── figure5_percent_errors.pdf
│   ├── tables_1_2.txt
│   └── tables_1_2.csv
└── data/
    └── generated_jets.npz
```

---

## `generated_jets.npz`

The generated output contains:

| Key | Shape | Meaning |
|---|---:|---|
| `neutral_lv_pred` | `(N, 81)` | PileFlow-predicted neutral-LV image |
| `neutral_lv_true` | `(N, 81)` | Generator truth neutral-LV image |
| `scalar_obs` | `(N, 16)` | PileFlow-generated scalar observables |
| `neutral_all_9x9` | `(N, 81)` | Input neutral-all context |
| `charged_pu_9x9` | `(N, 81)` | Input charged-PU context |
| `charged_lv_9x9` | `(N, 81)` | Input charged-LV context |

---

## Comparison observables

The comparison code computes the following observables:

```text
jet_mass
dijet_mass
jet_pt
neutral_n95
ecf2_log
ecf3_log
```

The comparison plots are based on the same general observable set used in the PUMML study.

---

## Common errors

### `ImportError: attempted relative import with no known parent package`

You probably ran:

```bash
python flows/runner.py
```

Run this instead from the repository root:

```bash
python -m flows.runner
```

---

### `flows/runner.py does not generate data`

This is expected if you forgot `--skip-gen`.

`flows/` does not generate data. First run `generator/`, then pass the generated `.npy` and `.npz` files:

```bash
python -m flows.runner \
  --skip-gen \
  --data-npy path/to/jets.npy \
  --data-npz path/to/jets_pileup_images.npz
```

---

### `.npy/.npz row-count mismatch`

PileFlow requires the `.npy` table and `.npz` image file to describe the same jets in the same order.

For generator runs intended for PileFlow, use matching cuts:

```text
jet_pt_min == image_pt_min
```

For example:

```bash
--jet-pt-min 15 \
--image-pt-min 15
```

when running the generator.

---

## Development notes

The important adapter is:

```text
flows/data/dataset.py
```

The important model file is:

```text
flows/models/pileflow.py
```

The training/generation logic is:

```text
flows/training/train_flow.py
```

The CLI entry point is:

```text
flows/runner.py
```

The comparison plotting code lives outside `flows/`:

```text
comparison/observable_comparison.py
```

---

## References

- PUMML: Komiske, Metodiev, Thaler — arXiv:1707.08600
- FlowSim: Vaselli et al. — arXiv:2402.13684
- PUPPI: Bertolini et al. — JHEP 2014
- Flow Matching: Lipman et al. — arXiv:2210.02747