"""
General plotting utilities for the PileFlow generator.

This module contains diagnostics for the 25-feature jet table and event-level
eta-phi jet maps.

These functions are called by the output writer when save_figures=True.
They can also be imported from notebooks or standalone diagnostic scripts.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_global_dataset_figures(
    dataset: np.ndarray,
    cfg_key: str,
    out_dir: str,
    max_scatter_points: int = 20_000,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Generate global overview plots for the full jet dataset.

    Produces four files:

        hist_pt.png
        hist_eta.png
        hist_phi.png
        scatter_eta_phi_ptcolor.png

    Parameters
    ----------
    dataset:
        Jet table with shape ``(N_jets, N_features)``. The first three columns
        are expected to be ``pt_gen``, ``eta_gen``, and ``phi_gen``.
    cfg_key:
        Short run label, for example ``antikt_R0.4``.
    out_dir:
        Output directory for PNG files.
    max_scatter_points:
        Maximum number of jets to show in the eta-phi scatter plot.
    rng:
        Optional NumPy random generator for scatter subsampling.
    """
    if dataset.size == 0 or dataset.shape[0] == 0:
        return

    if dataset.ndim != 2 or dataset.shape[1] < 3:
        raise ValueError(
            "Expected dataset with shape (N_jets, >=3), "
            f"got shape {dataset.shape}."
        )

    os.makedirs(out_dir, exist_ok=True)

    pt = dataset[:, 0]
    eta = dataset[:, 1]
    phi = dataset[:, 2]

    plt.figure(figsize=(6, 4))
    plt.hist(pt, bins=60, alpha=0.85)
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | pT")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_pt.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(eta, bins=60, alpha=0.85)
    plt.xlabel(r"$\eta$")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | eta")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_eta.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(phi, bins=60, alpha=0.85)
    plt.xlabel(r"$\phi$")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | phi")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_phi.png"), dpi=140)
    plt.close()

    if dataset.shape[0] > max_scatter_points and rng is not None:
        idx = rng.choice(dataset.shape[0], size=max_scatter_points, replace=False)
        eta_scatter = eta[idx]
        phi_scatter = phi[idx]
        pt_scatter = pt[idx]
    else:
        eta_scatter = eta
        phi_scatter = phi
        pt_scatter = pt

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(eta_scatter, phi_scatter, c=pt_scatter, s=8, alpha=0.65)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")
    plt.title(f"{cfg_key} | eta vs phi (color = pT)")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$p_T$ [GeV]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_eta_phi_ptcolor.png"), dpi=140)
    plt.close()


def plot_event_jets_eta_phi(
    jets_eta_phi_pt,
    cfg_key: str,
    source_event_idx: int,
    accepted_event_idx: int,
    out_dir: str,
) -> None:
    """
    Plot the eta-phi positions of all accepted jets in one event.

    Parameters
    ----------
    jets_eta_phi_pt:
        List of ``(eta, phi, pt)`` tuples.
    cfg_key:
        Short run label, for example ``antikt_R0.4``.
    source_event_idx:
        Event index in the raw Pythia stream.
    accepted_event_idx:
        Event index after event-level acceptance.
    out_dir:
        Output directory for event-level PNGs.
    """
    if not jets_eta_phi_pt:
        return

    os.makedirs(out_dir, exist_ok=True)

    etas = np.array([x[0] for x in jets_eta_phi_pt], dtype=float)
    phis = np.array([x[1] for x in jets_eta_phi_pt], dtype=float)
    pts = np.array([x[2] for x in jets_eta_phi_pt], dtype=float)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        etas,
        phis,
        c=pts,
        s=40 + 2.5 * np.clip(pts, 0, 200),
        alpha=0.85,
    )
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")
    plt.title(f"{cfg_key} | srcEv={source_event_idx} | accEv={accepted_event_idx}")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$p_T$ [GeV]")
    plt.tight_layout()

    out_path = os.path.join(
        out_dir,
        f"srcEv_{source_event_idx:06d}_accEv_{accepted_event_idx:06d}_eta_phi.png",
    )
    plt.savefig(out_path, dpi=140)
    plt.close()