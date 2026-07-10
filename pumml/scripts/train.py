# pumml/scripts/train.py
#
# Train the PUMML CNN on jet images produced by gen4e2e.
#
# Paper training setup (Section 2):
#   Dataset  : 56k pileup images, 90/10 train/test split
#   Loss     : modified log squared, p_bar = 10 GeV
#   Optimiser: Adam, lr = 0.001
#   Batch    : 50
#   Epochs   : 25
#   Init     : He-uniform
#
# Usage
# -----
#   python scripts/train.py \
#       --npz  ../../gen4e2e/data/run_XXX/antikt_R0.4/jets_XXX_images.npz \
#       --out  checkpoints/pumml_model.pt \
#       --epochs 25 \
#       --batch  50 \
#       --lr     0.001
#       --device cuda 

import os
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# make src/ importable regardless of where the script is called from
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

from models.pumml_model import PUMMLNet
from models.loss        import PUMMLLoss
from data.dataset       import PUMMLDataset, make_train_val_split

# Training loop
def train(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[train] Using device: {device}")

    #dataset
    dataset = PUMMLDataset(args.npz, max_images=args.max_images)
    train_ds, val_ds = make_train_val_split(
        dataset, train_frac=args.train_frac, seed=args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # model
    model = PUMMLNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}  (paper: 4,711)")

    # loss + optimiser 
    loss_fn   = PUMMLLoss(pbar=args.pbar).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    print(f"\n[train] Starting training: {args.epochs} epochs\n")
    print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'Time (s)':>10}")
    print("-" * 50)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train epoch
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            y_pred = model(X)
            loss   = loss_fn(y_pred, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred   = model(X)
                val_loss += loss_fn(y_pred, y).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {elapsed:>10.1f}"
        )

        # save best checkpoint only (paper doesn't specify saving frequency, so we do it at the end of each epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)

    print(f"\n[train] Best val loss: {best_val_loss:.6f}")
    print(f"[train] Model saved -> {args.out}")

    # save loss history as .npz for easy plotting later; also plot loss curve if matplotlib is available
    hist_path = args.out.replace(".pt", "_history.npz")
    np.savez(
        hist_path,
        train_loss=np.array(history["train"]),
        val_loss=np.array(history["val"]),
    )
    print(f"[train] Loss history saved -> {hist_path}")

    # plot loss curves if matplotlib is available
    _plot_loss(history, args.out.replace(".pt", "_loss_curve.png"))

    return model, history


def _plot_loss(history: dict, save_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train"], label="Train")
    plt.plot(epochs, history["val"],   label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("PUMML loss")
    plt.title("PUMML training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Loss curve saved -> {save_path}")



# CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="Train PUMML CNN — reproduces paper training setup"
    )
    p.add_argument(
        "--npz", required=True,
        help="Path to jet_images.npz produced by gen4e2e"
    )
    p.add_argument(
        "--out", default="runs/default/checkpoints/pumml_model.pt",
        help="Path to save the best model checkpoint. "
             "Prefer using run.py which sets this automatically "
             "(default: runs/default/checkpoints/pumml_model.pt)"
    )
    p.add_argument(
        "--epochs", type=int, default=25,
        help="Number of training epochs (paper: 25)"
    )
    p.add_argument(
        "--batch", type=int, default=50,
        help="Batch size (paper: 50)"
    )
    p.add_argument(
        "--lr", type=float, default=0.001,
        help="Adam learning rate (paper: 0.001)"
    )
    p.add_argument(
        "--pbar", type=float, default=10.0,
        help="Loss softening parameter in GeV (paper: 10.0)"
    )
    p.add_argument(
        "--train-frac", type=float, default=0.9,
        help="Fraction of data for training (paper: 0.9)"
    )
    p.add_argument(
        "--max-images", type=int, default=None,
        help="Cap dataset size (None = use all; paper uses 56k)"
    )
    p.add_argument(
        "--device", default="cpu",
        help="PyTorch device: cpu or cuda (default: cpu)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)