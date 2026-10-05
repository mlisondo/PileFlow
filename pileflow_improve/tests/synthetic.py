"""Small synthetic dataset in the generator's NPZ format (for tests only)."""
from __future__ import annotations

import numpy as np


def make_synthetic_npz(path, n_events=150, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_events
    yy, xx = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
    core = np.exp(-((yy - 4) ** 2 + (xx - 4) ** 2) / 3.0)
    npu = rng.poisson(140, n).clip(0, 180)
    lv9 = rng.gamma(2.0, 3.0, (n, 1, 1)) * core * (rng.random((n, 9, 9)) < 0.7)
    pu9 = rng.exponential(0.004, (n, 1, 1)) * npu[:, None, None] * (rng.random((n, 9, 9)) < 0.8)
    total9 = lv9 + pu9
    up = lambda a: np.repeat(np.repeat(a, 4, -2), 4, -1) / 16.0
    fine_core = np.repeat(np.repeat(core, 4, 0), 4, 1)
    ch_lv = rng.gamma(2.0, 2.0, (n, 1, 1)) * fine_core * (rng.random((n, 36, 36)) < 0.08)
    ch_pu = rng.exponential(0.8, (n, 36, 36)) * (rng.random((n, 36, 36)) < 0.0004 * npu[:, None, None])
    X = np.stack([up(total9), ch_pu, ch_lv], 1).astype(np.float32)
    event_id = np.repeat(np.arange(n_events), 2)
    jet_rank = np.tile([0, 1], n_events)
    perm = rng.permutation(n_events)
    test_ev = perm[: n_events // 10]
    is_test = np.isin(event_id, test_ev)
    img = rng.permutation(n)
    np.savez(
        path, X=X, y_neutral_lv_9=lv9.astype(np.float32), event_id=event_id, jet_rank=jet_rank,
        event_train_idx=np.where(~is_test)[0], event_test_idx=np.where(is_test)[0],
        train_idx=img[: int(0.9 * n)], test_idx=img[int(0.9 * n):],
        channel_names=np.array(["total_neutral", "charged_pileup", "charged_leading_vertex"]),
        image_center_eta=rng.uniform(-2, 2, n).astype(np.float32),
        image_center_phi=rng.uniform(-np.pi, np.pi, n).astype(np.float32),
        npu=npu.astype(np.int16),
        puppi_charged_36=ch_lv.astype(np.float32),
        puppi_neutral_9=(lv9 + 0.3 * pu9).astype(np.float32),
    )
    return path
