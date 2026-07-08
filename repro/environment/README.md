# PileFlow unified environment snapshot

This folder records the local environment used for the successful PileFlow smoke test.

Primary files:

- environment-full-portable.yml
  Conda YAML with environment name changed to pileflow-unified and prefix removed.
  Best first attempt for another Apple Silicon macOS machine.

- conda-explicit-osx-arm64.txt
  Exact Conda package export for macOS Apple Silicon.
  Not portable to Linux/HPC.

- pip-freeze.txt
  Exact pip-installed Python packages.

- check_pileflow_env.py
  Runtime validation script. This is the main test after recreating the environment.

- system-report.txt
  Local Python, Conda, MG5, FastJet, LHAPDF, compiler, and OS report.

- env-vars.sh
  Shell variables needed from the PileFlow repo root.

- git-commit.txt, git-status.txt, git-diff.patch
  Code-state information at the time of the snapshot.

Usage from a recreated environment:

    cd /path/to/PileFlow
    source repro/environment/env-vars.sh
    python repro/environment/check_pileflow_env.py
