# PileFlow PUPPI

Standalone simplified PUPPI baseline for PileFlow.

This package duplicates the generator-side temporary PUPPI implementation so it can be tested independently.

## Inputs

The standalone runner expects a generator `.npz` file containing:

```text
full_px
full_py
full_pz
full_e
full_charge
full_is_lv
full_n
n_pu