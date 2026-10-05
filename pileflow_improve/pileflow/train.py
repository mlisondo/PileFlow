"""Training loop for PileFlow (target conditional flow matching)."""
from __future__ import annotations

import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import BatchLoader
from .flow import TargetConditionalFlowMatching

CHECKPOINT_FORMAT = 2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict, device, pre, generator=None) -> dict:
    """Move a raw (GeV) batch to `device` and apply the fitted preprocessing."""
    def get(key):
        return batch[key].to(device, non_blocking=True).float()
    return {
        "target_gev": get("target"),
        "target": pre.encode_target(get("target"), get("neutral_all_9"), generator=generator),
        "neutral_all": pre.transform("neutral_all", get("neutral_all_9")),
        "charged_pu": pre.transform("charged_pu", get("charged_pu_36")),
        "charged_lv": pre.transform("charged_lv", get("charged_lv_36")),
    }


def compute_losses(model, flow, b, morph_lambda=0.0, morph_threshold=0.1, morph_tmin=0.5,
                   generator=None):
    """Flow-matching MSE plus an optional low-activity ("morphology") term.

    The auxiliary term compares the endpoint implied by the predicted velocity,
    x1_hat = (1-s) z_t + (1-(1-s) t) v, with the target on cells whose TRUE
    neutral-LV pT is below `morph_threshold` GeV, using a smooth-L1 loss in
    the preprocessed space. Only samples with t >= morph_tmin contribute:
    at small t x1_hat is a posterior mean over many images, and pulling it
    to zero there would distort the learned vector field.
    """
    t, zt, ut = flow.sample_training_tuple(b["target"], generator=generator)
    pred = model(t, zt, b["neutral_all"], b["charged_pu"], b["charged_lv"])
    fm = F.mse_loss(pred.float(), ut)

    if morph_lambda > 0.0:
        x1_hat = flow.endpoint_from_velocity(t, zt, pred.float())
        mask = (b["target_gev"] < morph_threshold).float()
        mask = mask * (t >= morph_tmin).float().view(-1, 1, 1, 1)
        per_pixel = F.smooth_l1_loss(x1_hat, b["target"], reduction="none")
        morph = (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        morph = fm.new_zeros(())
    return {"total": fm + morph_lambda * morph, "fm": fm, "morph": morph}


class EMA:
    """Exponential moving average of the weights (standard for flow/diffusion models).

    Uses the usual warm-up decay min(decay, (1 + n) / (10 + n)) so the
    average is not dominated by the random initialisation early in training.
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def train_model(model, model_config, dataset, splits, preprocessor, outdir, device="cuda",
                epochs=400, batch_size=256, lr=1e-3, patience=60, sigma_min=1e-4, alpha=1.0,
                seed=42, morph_lambda=0.0, morph_threshold=0.1, morph_tmin=0.5,
                ema_decay=0.999, grad_clip=5.0, amp=True, extra_metadata=None,
                lr_schedule="plateau", init_state=None):
    """Train with early stopping on a *fixed-noise* validation loss.

    The validation (t, x0) draws are regenerated from the same seed every
    epoch, so epoch-to-epoch changes of the validation loss reflect the model
    and not the Monte-Carlo noise of the flow-matching objective.
    Returns (best_model, preprocessor, history, checkpoint_path).
    """
    seed_everything(seed)
    device = torch.device(device)
    outdir = Path(outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if init_state is not None:           # warm start (e.g. continue training with a new schedule)
        model.load_state_dict(init_state)
    model = model.to(device)
    preprocessor = preprocessor.to(device)
    preprocessor.fit(dataset, splits.train)

    flow = TargetConditionalFlowMatching(sigma_min=sigma_min, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                               patience=15, min_lr=1e-6)
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    train_loader = BatchLoader(dataset, splits.train, batch_size, shuffle=True, seed=seed)
    val_loader = BatchLoader(dataset, splits.val, batch_size, shuffle=False)

    history = {k: [] for k in ("train", "val", "train_fm", "train_morph", "val_fm",
                               "val_morph", "lr", "epoch_time_s")}
    best, best_epoch, best_state, wait = float("inf"), 0, None, 0

    print("\n" + "=" * 64)
    print(f"Training PileFlow on {device} | train={len(splits.train)} val={len(splits.val)}")
    print("  objective : flow-matching MSE"
          + (f" + {morph_lambda} x morphology (thr {morph_threshold} GeV, t>={morph_tmin})"
             if morph_lambda > 0 else ""))
    print(f"  EMA decay : {ema_decay if ema else 'off'} | AMP: {use_amp}")
    print("=" * 64)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        sums, count = {"total": 0.0, "fm": 0.0, "morph": 0.0}, 0
        for raw in train_loader:
            b = batch_to_device(raw, device, preprocessor)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = compute_losses(model, flow, b, morph_lambda, morph_threshold, morph_tmin)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            n = b["target"].shape[0]
            for k in sums:
                sums[k] += losses[k].item() * n
            count += n
        train = {k: v / count for k, v in sums.items()}

        eval_model = ema.shadow if ema else model
        eval_model.eval()
        gen = torch.Generator(device=device).manual_seed(seed + 12345)
        sums, count = {"total": 0.0, "fm": 0.0, "morph": 0.0}, 0
        with torch.no_grad():
            for raw in val_loader:
                b = batch_to_device(raw, device, preprocessor, generator=gen)
                losses = compute_losses(eval_model, flow, b, morph_lambda, morph_threshold,
                                        morph_tmin, generator=gen)
                n = b["target"].shape[0]
                for k in sums:
                    sums[k] += losses[k].item() * n
                count += n
        val = {k: v / count for k, v in sums.items()}
        scheduler.step() if lr_schedule == "cosine" else scheduler.step(val["total"])

        history["train"].append(train["total"]); history["val"].append(val["total"])
        history["train_fm"].append(train["fm"]); history["train_morph"].append(train["morph"])
        history["val_fm"].append(val["fm"]); history["val_morph"].append(val["morph"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time_s"].append(time.time() - t0)

        if val["total"] < best - 1e-6:
            best, best_epoch, wait = val["total"], epoch, 0
            best_state = copy.deepcopy(eval_model.state_dict())
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train {train['total']:.5f} (fm {train['fm']:.5f}) | "
                  f"val {val['total']:.5f} (fm {val['fm']:.5f}) | "
                  f"lr {history['lr'][-1]:.1e} | {history['epoch_time_s'][-1]:.1f}s")
        if patience > 0 and wait >= patience:
            print(f"Early stopping at epoch {epoch}; best val {best:.6f} at epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No validation improvement was ever recorded")
    model.load_state_dict(best_state)
    model.eval()

    training_config = dict(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                           seed=seed, morph_lambda=morph_lambda, morph_threshold=morph_threshold,
                           morph_tmin=morph_tmin, ema_decay=ema_decay, grad_clip=grad_clip,
                           amp=use_amp, optimizer="Adam",
                           scheduler="CosineAnnealing(eta_min=lr/100)" if lr_schedule == "cosine" else "ReduceLROnPlateau(0.5,15)",
                           warm_start=init_state is not None)
    ckpt = {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "preprocess_config": preprocessor.get_config(),
        "preprocessor_state": preprocessor.state_dict(),
        "flow_config": {"sigma_min": sigma_min, "alpha": alpha},
        "splits": {k: np.asarray(v) for k, v in splits.as_dict().items()},
        "history": history,
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "training_config": training_config,
        "metadata": {**(extra_metadata or {}), "torch": torch.__version__,
                     "python": platform.python_version()},
    }
    path = outdir / "checkpoints" / "pileflow_best.pt"
    torch.save(ckpt, path)
    np.savez(outdir / "checkpoints" / "history.npz", **{k: np.asarray(v) for k, v in history.items()})
    with open(outdir / "config.json", "w") as f:
        json.dump({k: ckpt[k] for k in ("model_config", "preprocess_config", "flow_config",
                                        "training_config", "metadata", "best_val_loss",
                                        "best_epoch")}, f, indent=2, default=str)
    print(f"\nBest validation loss {best:.6f} (epoch {best_epoch}); checkpoint: {path}")
    return model, preprocessor, history, str(path)


def load_checkpoint(path, device="cpu", fallback_model_config=None, fallback_preprocess=None):
    """Load any PileFlow checkpoint (format 2 or the original draft format).

    Returns (model, preprocessor, flow_config, checkpoint_dict).
    Draft checkpoints stored no architecture; `fallback_model_config` (e.g.
    the 'spatial' preset) is then used, and the flow config falls back to the
    training_config entries or FlowSim defaults.
    """
    from .model import build_model
    from .preprocess import ImagePreprocessor

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config") or fallback_model_config
    if model_config is None:
        raise ValueError(f"{path} stores no model_config; pass the architecture explicitly")
    model = build_model(model_config)
    state = next(ckpt[k] for k in ("model_state", "model_state_dict", "state_dict") if k in ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pre_cfg = ckpt.get("preprocess_config")
    if pre_cfg is None:  # draft checkpoint
        pre_cfg = {"mode": ckpt.get("preprocess_mode", (fallback_preprocess or {}).get("mode", "log1p")),
                   "pt_scale": ckpt.get("pt_scale", (fallback_preprocess or {}).get("pt_scale", 1.0))}
    pre = ImagePreprocessor.from_config(pre_cfg, ckpt.get("preprocessor_state")).to(device)

    flow_cfg = ckpt.get("flow_config")
    if flow_cfg is None:
        tc = ckpt.get("training_config", {})
        flow_cfg = {"sigma_min": tc.get("sigma_min", 1e-4), "alpha": tc.get("alpha", 1.0)}
    ckpt["model_config"] = model_config
    return model, pre, flow_cfg, ckpt
