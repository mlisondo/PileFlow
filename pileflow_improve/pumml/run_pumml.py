#!/usr/bin/env python3
"""Train the PyTorch PUMML CNN on a PUMML-style NPZ.

Default = the paper recipe (arXiv:1707.08600, Sec. 2): Adam, lr 1e-3,
batch 50, 25 epochs, He-uniform init, log loss with pbar = 10 GeV, raw
(unnormalised) pT images, final-epoch weights.

--select-on-val additionally holds out an event-grouped validation set from
the training pool and keeps the best-validation epoch ("tuned" baseline), so
PileFlow can be compared with a PUMML that gets the same model selection.

Run from the repository root:

    python -m pumml.run_pumml --data DATA.npz --split event --output runs/pumml/pumml_event.pt
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from pumml.pumml_arch import PUMML, PUMMLLoss, count_parameters


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split(d, split: str):
    keys = {"image": ("train_idx", "test_idx"), "event": ("event_train_idx", "event_test_idx")}[split]
    for k in keys:
        if k not in d.files:
            raise KeyError(f"--split {split} requires NPZ key {k!r}")
    return np.asarray(d[keys[0]], np.int64), np.asarray(d[keys[1]], np.int64)


def check_no_event_leakage(event_id, a, b, what):
    if event_id is not None and np.intersect1d(event_id[a], event_id[b]).size:
        raise RuntimeError(f"Event leakage between {what}")


def batches(X, y, idx, batch_size, device, rng=None):
    order = rng.permutation(idx) if rng is not None else idx
    for s in range(0, len(order), batch_size):
        sel = np.sort(order[s:s + batch_size]) if rng is None else order[s:s + batch_size]
        yield (torch.from_numpy(X[sel]).to(device, non_blocking=True),
               torch.from_numpy(y[sel]).unsqueeze(1).to(device, non_blocking=True))


@torch.no_grad()
def mean_loss(model, crit, X, y, idx, batch_size, device):
    model.eval()
    tot = 0.0
    for xb, yb in batches(X, y, idx, batch_size, device):
        tot += crit(model(xb), yb).item() * len(xb)
    return tot / max(len(idx), 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["image", "event"], default="event",
                   help="event = event-grouped 90/10 (recommended, same eval set as PileFlow).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pbar", type=float, default=10.0, help="GeV, loss hyperparameter of Eq. (2.1)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--select-on-val", action="store_true",
                   help="Hold out an event-grouped validation set and keep the best epoch.")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--output", default="pumml_model.pt")
    args = p.parse_args(argv)

    seed_everything(args.seed)
    device = get_device(args.device)
    with np.load(args.data, allow_pickle=False) as d:
        X = np.ascontiguousarray(d["X"], dtype=np.float32)
        y = np.ascontiguousarray(d["y_neutral_lv_9"], dtype=np.float32)
        event_id = np.asarray(d["event_id"], np.int64) if "event_id" in d.files else None
        train_idx, test_idx = resolve_split(d, args.split)
    if X.shape[1:] != (3, 36, 36) or y.shape[1:] != (9, 9):
        raise ValueError(f"Unexpected shapes X{X.shape} y{y.shape}")
    if args.split == "event":
        check_no_event_leakage(event_id, train_idx, test_idx, "train and test")

    val_idx = np.zeros(0, np.int64)
    if args.select_on_val:
        groups = event_id if event_id is not None else np.arange(len(y))
        rng = np.random.default_rng(args.seed + 1)
        ev = rng.permutation(np.unique(groups[train_idx]))
        n_val = max(1, int(round(args.val_fraction * len(ev))))
        is_val = np.isin(groups[train_idx], ev[:n_val])
        train_idx, val_idx = train_idx[~is_val], train_idx[is_val]
        check_no_event_leakage(event_id, train_idx, val_idx, "train and validation")

    model = PUMML().to(device)
    crit = PUMMLLoss(pbar=args.pbar)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"PUMML ({count_parameters(model)} parameters) on {device}: train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)} split={args.split}")

    rng = np.random.default_rng(args.seed)
    history = {"train_loss": [], "val_loss": [], "test_loss": []}
    best, best_epoch, best_state = float("inf"), args.epochs, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in batches(X, y, train_idx, args.batch_size, device, rng):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        history["train_loss"].append(tot / len(train_idx))
        # The test loss is monitored for information only; it is never used for selection.
        history["test_loss"].append(mean_loss(model, crit, X, y, test_idx, 512, device))
        msg = f"Epoch {epoch:3d} train={history['train_loss'][-1]:.6f} test={history['test_loss'][-1]:.6f}"
        if args.select_on_val:
            v = mean_loss(model, crit, X, y, val_idx, 512, device)
            history["val_loss"].append(v)
            msg += f" val={v:.6f}"
            if v < best:
                best, best_epoch, best_state = v, epoch, copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Selected epoch {best_epoch} (best validation loss {best:.6f})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "split_name": args.split,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "selected_epoch": best_epoch,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                     "optimizer": "Adam", "pbar_GeV": args.pbar, "seed": args.seed,
                     "select_on_val": args.select_on_val, "train_seconds": time.time() - t0},
    }, args.output)
    print(f"Saved {args.output} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
