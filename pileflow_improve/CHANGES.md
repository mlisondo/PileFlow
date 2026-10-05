# Review fixes (v0.2.0)

This file lists the changes made after reviewing the code against the PUMML paper
(arXiv:1707.08600) and the FlowSim paper (arXiv:2402.13684). The original draft is the first
commit in the git history.

## Bugs that broke runs or could give wrong results

- **`run.py` import error.** `run.py` imported `.preprocess`, but the file was named
  `preproces.py`. The file is now `pileflow/preprocess.py`.
- **`evaluate_checkpoint.py` imported a package that does not exist** (`pileflow5k`). It now
  reads the architecture, preprocessing, flow settings and **splits** from the checkpoint, so it
  can no longer evaluate on jets that were used for training.
- **`--max-jets` silently dropped the predefined event split.** A later evaluation on the full
  file could then include training jets in the evaluation set. The predefined split is now
  intersected with the truncated data instead of being dropped.
- **Checkpoints did not record the model.** They now store `model_config`, `preprocess_config`,
  `flow_config`, metadata and a `fitted` flag for the preprocessor. Draft checkpoints still load
  (tested: the output is identical).
- **Inconsistent PUMML imports.** `pumml/evaluate.py` used a relative import while
  `run_pumml.py` used an absolute one, so each worked only from a different directory. `pumml`
  is now a proper package, and every script runs from the repository root.
- **Small preprocessing inconsistency.** `transform` divided by `std + eps` while the inverse
  multiplied by `std`, so the round trip was not exact. Both now use `std`.

## Making the comparison fair

- **One definition of the observables** (`pileflow/observables.py`).
  - Previously `compare_methods.py` spread each neutral cell over 16 fine pixels before computing
    mass and ECFs, while the PUMML reproduction used one particle per cell centre. The two
    therefore disagreed.
  - Both now follow PUMML's detector discretisation: charged pT at 36×36 cell centres, neutral
    pT at 9×9 cell centres.
  - PUPPI now uses `puppi_charged_36` + `puppi_neutral_9`.
  - ECFs are computed exactly in O(n³) with matrix products, and a unit test checks them against
    a brute-force triple loop.
- **Paired bootstrap.** Every method is resampled on the same jets, and the difference with
  respect to PUMML is reported with a 95% CI and p = P(not better). Before, each method was
  resampled independently.
- **Common population.** Each statistic is computed only on jets that are finite for truth and
  for every method.
- **Threshold chosen on validation data.** The new `select_threshold.py` picks it on the val
  split, and `compare_methods.py` rejects a threshold (or a PileFlow file) that was produced on
  the wrong split.
- **New outputs:**
  - an optional val-selected PUMML
  - an optional parameter-matched PileFlow (`spatial_small`, ~18k parameters)
  - throughput and parameter counts
  - jet-mass correlation versus NPU (PUMML Fig. 6)
  - 1-D Wasserstein distances (FlowSim)
  - PileFlow's sampling spread across independent samples
  - posterior coverage from `evaluate.py`
- **Reproducible sampling.** The initial noise is drawn once per jet from a seeded generator, so
  results do not depend on the batch size or device.
- **Consistent PileFlow output in the qualitative plots.** `plot_qualitative_jets.py` now shows
  the same PileFlow output that is scored (a single sample by default), instead of preferring
  the posterior mean.

## Model and training

- **New `spatial_v2` preset.**
  - The charged encoder is exactly aligned with the neutral cells (PixelUnshuffle). The draft
    encoder was offset by about 1.5 fine pixels.
  - Time conditioning uses FiLM.
  - The time embedding is scaled so its features span the full frequency range. Before, all
    features were close to linear in t.
- **EMA of the weights**, with warm-up.
- **Fixed-noise validation loss**, so early stopping and the learning-rate schedule are no
  longer driven by Monte-Carlo noise.
- **Morphology loss restricted to t ≥ `morph_tmin`.** At small t the endpoint estimate is a
  posterior mean, and pulling it towards zero distorts the vector field.
- **Faster data loading.** Batches are indexed directly in memory instead of going through
  per-item `Subset`.
- **New fixed-step samplers:** midpoint and RK4. The number of network evaluations (NFE) is
  counted.
- **Fewer pickled files.** NPZ files are loaded with `allow_pickle=False`.

## Tests

`tests/` contains:

- unit tests of the splits, preprocessing, flow identities, samplers, models, observables
  (N95, 4-vectors, ECFs against brute force, dijet pairing) and the paired bootstrap
- an end-to-end run of the full pipeline on a synthetic dataset: training, sampling on val and
  eval, threshold selection, PUMML (paper recipe and val-selected), the reproduction check,
  the comparison, and both plotting scripts
