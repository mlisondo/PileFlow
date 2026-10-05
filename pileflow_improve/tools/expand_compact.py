#!/usr/bin/env python3
"""Rebuild the full-format NPZ (X with 3x36x36 channels) from the compact file
(neutral9 + sparse charged images, saved with savez_compressed; ~40 MB instead of 2.7 GB).

    python tools/expand_compact.py generator/..._compact.npz data/full.npz
"""
import sys
from pathlib import Path

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
c = np.load(src, allow_pickle=False)
n = len(c["y_neutral_lv_9"])
X = np.empty((n, 3, 36, 36), np.float32)
X[:, 0] = np.repeat(np.repeat(c["neutral9"], 4, 1), 4, 2) / 16.0
X[:, 1], X[:, 2] = c["charged_pu36"], c["charged_lv36"]
out = {k: c[k] for k in c.files if k not in ("neutral9", "charged_pu36", "charged_lv36")}
Path(dst).parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, X=X, **out)
print(f"wrote {dst} ({n} jets)")
