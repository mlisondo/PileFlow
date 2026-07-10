"""
pumml_in_server/run.py

PUMML single entry point — train and/or evaluate in one command.

All outputs go to a self-contained run directory:

    runs/<name>/
        checkpoints/
            pumml_model.pt              best model weights (lowest val loss)
            pumml_model_history.npz     per-epoch train/val loss history
            pumml_model_loss_curve.png  loss curve plot
        plots/
            distributions.png/pdf
            percent_errors.png/pdf
            tables_1_2.txt / .csv

Example usage
-------------
Train + evaluate on 12k jets:

    python run.py \\
        --npz  data/jets_pileup.npz \\
        --name run_12k

Evaluate only (skip training, supply an existing checkpoint):

    python run.py \\
        --npz        data/jets_pileup.npz \\
        --name       run_12k \\
        --skip-train \\
        --model      runs/run_12k/checkpoints/pumml_model.pt

Train only (skip evaluation):

    python run.py \\
        --npz       data/jets_pileup.npz \\
        --name      run_12k \\
        --skip-eval
"""

import os
import sys
import argparse

_HERE    = os.path.dirname(os.path.abspath(__file__))
_SRC     = os.path.join(_HERE, "src")
_SCRIPTS = os.path.join(_HERE, "scripts")
_PLOT    = os.path.join(_HERE, "plotting")

# Make all sub-packages importable without requiring pip install
for _p in [_SRC, _SCRIPTS, _PLOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PUMML — train and/or evaluate the pileup mitigation CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--npz", required=True,
        help="Path to jets_pileup.npz produced by gen4e2e",
    )
    p.add_argument(
        "--name", default="run",
        help="Run name — all outputs go to runs/<name>/ (default: 'run')",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load an existing checkpoint with --model",
    )
    p.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation; only train the model",
    )
    p.add_argument(
        "--model", default=None,
        help="Path to an existing checkpoint (.pt). Required when --skip-train is set.",
    )

    train_g = p.add_argument_group("Training hyperparameters (paper defaults)")
    train_g.add_argument("--epochs",     type=int,   default=25,
                         help="Training epochs (paper: 25)")
    train_g.add_argument("--batch",      type=int,   default=50,
                         help="Jets per gradient update (paper: 50)")
    train_g.add_argument("--lr",         type=float, default=1e-3,
                         help="Adam learning rate (paper: 0.001)")
    train_g.add_argument("--pbar",       type=float, default=10.0,
                         help="Loss softening scale p_bar in GeV (paper: 10.0)")
    train_g.add_argument("--train-frac", type=float, default=0.9,
                         help="Fraction of jets for training (paper: 0.9)")

    eval_g = p.add_argument_group("Evaluation options")
    eval_g.add_argument(
        "--max-jets", type=int, default=None,
        help="Cap the number of jets used for evaluation "
             "(default: all jets in the .npz; with 12k jets you can leave this unset)",
    )
    eval_g.add_argument(
        "--device", default=None,
        help="PyTorch device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    eval_g.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for weight initialisation and train/val split (default: 42)",
    )

    return p


def _auto_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _run_training(args, device: str, model_path: str) -> None:
    """
    Train PUMML and save the best checkpoint to model_path.

    Delegates to scripts/train.py's train() function so there is one canonical
    training implementation. We build a compatible SimpleNamespace args object.
    """
    import types
    from train import train

    train_args = types.SimpleNamespace(
        npz        = args.npz,
        out        = model_path,
        epochs     = args.epochs,
        batch      = args.batch,
        lr         = args.lr,
        pbar       = args.pbar,
        train_frac = args.train_frac,
        max_images = None,       # use all jets in the .npz for training
        device     = device,
        seed       = args.seed,
    )
    train(train_args)


def _run_evaluation(args, device: str, model_path: str, plot_dir: str) -> None:
    """
    Evaluate PUMML and generate all paper figures and tables.

    Figures produced
        Figure 3 : 3D jet image display (standalone — no observable loop needed)
        Figure 4 : Normalised observable distributions
        Figure 5 : Percent-error distributions
        Figure 8 : Conv1 filter weight visualisation (standalone)
        Tables 1 & 2 : Pearson r and IQR scores (.txt and .csv)

    The expensive observable loop (compare.collect_observables) is run once and
    its output is shared across Figures 4, 5, and Tables 1 & 2.
    """
    import numpy as np
    from common import load_store
    from plotting import make_all

    data     = np.load(args.npz, allow_pickle=False)
    mean_npu = float(data["n_pu"].mean()) if "n_pu" in data else 50.0
    total    = int(data["jet_pt"].shape[0])
    n_jets   = args.max_jets or total

    print(f"  Dataset: {total:,} jets total, evaluating {n_jets:,}")
    print(f"  Mean NPU: {mean_npu:.0f}\n")

    # Figure 3 — single-jet 3D image display (fast, no observable loop)
    print("3D jet display")
    try:
        from jet_display import make_figure3
        make_figure3(args.npz, model_path, plot_dir, jet_idx=0, device=device)
    except Exception as exc:
        print(f"  [skipped] {exc}")
    print()

    # Figure 8 — Conv1 filter weights (fast, just loads the model)
    print("Conv1 filter weights")
    try:
        from filter_weights import make_figure8
        make_figure8(model_path, plot_dir, device)
    except Exception as exc:
        print(f"  [skipped] {exc}")
    print()

    # Run the observable loop once — shared by Figures 4, 5, and Tables 1 & 2
    print("Computing observables (this is the slow step — a few minutes on CPU) ...")
    store = load_store(
        npz_path   = args.npz,
        model_path = model_path,
        max_jets   = n_jets,
        device     = device,
    )
    print("Observable computation complete.\n")

    # Figures 4 & 5 + Tables 1 & 2
    make_all(store, plot_dir, mean_npu)
    print(f"\n  All plots saved to: {plot_dir}/")


def main() -> None:
    args   = build_parser().parse_args()
    device = args.device or _auto_device()

    run_dir  = os.path.join(_HERE, "runs", args.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Resolve model path: use --model if given, else the run's default checkpoint path
    model_path = args.model or os.path.join(ckpt_dir, "pumml_model.pt")

    print("=" * 60)
    print("  PUMML")
    print(f"  run name  : {args.name}")
    print(f"  data      : {args.npz}")
    print(f"  model     : {model_path}")
    print(f"  device    : {device}")
    print(f"  run dir   : {run_dir}")
    print("=" * 60)

    # Stage 1: Training
    if not args.skip_train:
        print("\n[Stage 1/2] Training PUMML CNN ...")
        _run_training(args, device, model_path)
    else:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"--skip-train requires an existing checkpoint. Not found: {model_path}"
            )
        print(f"\n[Stage 1/2] Skipped — using checkpoint: {model_path}")

    # Stage 2: Evaluation
    if not args.skip_eval:
        print("\n[Stage 2/2] Evaluating ...")
        _run_evaluation(args, device, model_path, plot_dir)
    else:
        print("\n[Stage 2/2] Skipped.")

    print("\n" + "=" * 60)
    print(f"  Done. All outputs in: {run_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
