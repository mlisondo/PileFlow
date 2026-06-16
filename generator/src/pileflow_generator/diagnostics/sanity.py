"""
Console diagnostics for generator runs.
"""

from __future__ import annotations

import numpy as np


def print_header() -> None:
    """
    Print the workflow header.
    """
    print("\n" + "=" * 80)
    print(" PILEFLOW GENERATOR: MadGraph -> Pythia8 -> FastJet")
    print("=" * 80)


def print_sanity(dataset: np.ndarray, label: str) -> None:
    """
    Print simple sanity checks for the generated jet-feature dataset.

    Assumes the old 25-feature schema:
        column 0  = pt_gen
        column 6  = recoPt
        columns 11-14 = energy fractions
    """
    if dataset.shape[0] == 0:
        print(f"[sanity] {label}: empty dataset")
        return

    frac_sum = dataset[:, 11] + dataset[:, 12] + dataset[:, 13] + dataset[:, 14]
    bad_frac = np.mean((frac_sum < 0.5) | (frac_sum > 1.5))
    ratio = np.median(dataset[:, 6] / np.clip(dataset[:, 0], 1e-6, None))

    if dataset.shape[0] < 2 or np.std(dataset[:, 0]) == 0 or np.std(dataset[:, 6]) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(dataset[:, 0], dataset[:, 6])[0, 1]

    print(f"[sanity] {label}")
    print(f"  gen/reco pT correlation : {corr:.4f}")
    print(f"  median reco/gen pT      : {ratio:.3f}")
    print(f"  unusual fraction sum    : {100.0 * bad_frac:.2f}%")
    print(f"  median fraction sum     : {np.median(frac_sum):.3f}")