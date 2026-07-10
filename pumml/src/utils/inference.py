# pumml/src/utils/inference.py
#
# PUMML inference utilities.
# Loads a trained model checkpoint and runs prediction on jet images.

import os
import sys
import numpy as np
import torch
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models.pumml_model import PUMMLNet


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> PUMMLNet:
    """
    Load a trained PUMML model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by scripts/train.py
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    PUMMLNet in eval mode on the requested device
    """
    model = PUMMLNet()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[PUMML] Loaded model from {checkpoint_path}  (device={device})")
    return model


def predict_batch(
    model:   PUMMLNet,
    X:       np.ndarray,
    device:  str = "cpu",
    chunk:   int = 512,
) -> np.ndarray:
    """
    Run PUMML inference on a batch of images.

    Parameters
    ----------
    model  : loaded PUMMLNet (from load_model)
    X      : (N, 3, 36, 36) float32 array — stacked input channels
             channel order: [neutral_total, charged_pu, charged_lv]
    device : 'cpu' or 'cuda'
    chunk  : process this many images at a time to avoid OOM

    Returns
    -------
    (N, 9, 9) float32 array — predicted neutral LV pT images, clipped >= 0
    """
    model.eval()
    preds = []

    for start in range(0, len(X), chunk):
        x_np  = X[start : start + chunk].astype(np.float32)
        x_t   = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            out = model(x_t)          # (B, 1, 9, 9)
            out = out.squeeze(1)      # (B, 9, 9)
            out = torch.clamp(out, min=0.0)
        preds.append(out.cpu().numpy())

    result = np.concatenate(preds, axis=0)   # (N, 9, 9)
    print(f"[PUMML] Predicted {len(result):,} jets  shape={result.shape}")
    return result


def predict_from_npz(
    npz_path:        str,
    checkpoint_path: str,
    device:          str = "cpu",
    max_images:      Optional[int] = None,
    chunk:           int = 512,
) -> np.ndarray:
    """
    Convenience wrapper: load npz, stack channels, run inference.

    Parameters
    ----------
    npz_path        : path to jet_images.npz from gen4e2e
    checkpoint_path : path to trained model .pt file
    device          : 'cpu' or 'cuda'
    max_images      : cap number of jets processed (None = all)
    chunk           : batch size for inference

    Returns
    -------
    (N, 9, 9) predicted neutral LV images
    """
    data = np.load(npz_path, allow_pickle=False)

    N = len(data["jet_pt"])
    if max_images is not None:
        N = min(N, max_images)

    # Stack input channels in the same order as dataset.py
    X = np.stack([
        data["ch_neutral_all"][:N],   # channel 0  RED
        data["ch_charged_pu"][:N],    # channel 1  GREEN
        data["ch_charged_lv"][:N],    # channel 2  BLUE
    ], axis=1).astype(np.float32)     # (N, 3, 36, 36)

    model = load_model(checkpoint_path, device=device)
    return predict_batch(model, X, device=device, chunk=chunk)
