# PileFlow unified environment snapshot

This folder contains the files needed to recreate and validate the PileFlow
software environment on another Apple Silicon macOS machine.

## Tracked files

- `environment-full-portable.yml`

  Main Conda environment file. This should be the first attempt on another
  Apple Silicon Mac. The environment name is `pileflow-unified`, and the local
  `prefix` has been removed.

- `conda-explicit-osx-arm64.txt`

  Exact Conda package export for macOS Apple Silicon. Use this as a fallback if
  the portable YAML fails to solve. This file is not portable to Linux or HPC.

- `pip-freeze.txt`

  Record of pip-installed Python packages from the working environment.

- `check_pileflow_env.py`

  Runtime validation script. This is the main test after recreating the
  environment.

- `env-vars.sh`

  Shell variables needed from the PileFlow repo root. It sets `PYTHONPATH` and
  `MG5_PATH`.

## Important local-package note

Do not include `pileflow-generator` in the pip section of the Conda YAML.

`pileflow-generator` is a local editable package from this repository, not a
public PyPI package. After creating the Conda environment, install it manually
from the cloned repo:

    cd generator
    python -m pip install -e .

## Usage on a new Mac

From the PileFlow repo root:

    conda env create -f repro/environment/environment-full-portable.yml
    conda activate pileflow-unified

    cd generator
    python -m pip install -e .

    cd ..
    source repro/environment/env-vars.sh
    python repro/environment/check_pileflow_env.py

The setup is successful when the check ends with:

    PileFlow environment check passed.

## Fallback exact recreation

If the portable YAML fails on Apple Silicon macOS:

    conda create -n pileflow-unified --file repro/environment/conda-explicit-osx-arm64.txt
    conda activate pileflow-unified

    cd generator
    python -m pip install -e .

    cd ..
    source repro/environment/env-vars.sh
    python repro/environment/check_pileflow_env.py

## Ignored local files

Machine-specific diagnostics, local git snapshots, and redundant pip/conda
reports are intentionally ignored by `repro/environment/.gitignore`.
