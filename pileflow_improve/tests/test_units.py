import itertools

import numpy as np
import pytest
import torch

from pileflow.data import PileFlowImages, make_splits, neutral_36_to_9
from pileflow.flow import TargetConditionalFlowMatching
from pileflow.model import PRESETS, build_model, count_parameters, model_config_from_preset
from pileflow.observables import dijet_mass, four_vectors, n95, normalized_ecfs, wrap_phi
from pileflow.preprocess import ImagePreprocessor
from pileflow.stats import paired_bootstrap, pearson


def test_neutral_roundtrip():
    x9 = torch.rand(5, 1, 9, 9)
    x36 = x9.repeat_interleave(4, -2).repeat_interleave(4, -1) / 16
    assert torch.allclose(neutral_36_to_9(x36), x9, atol=1e-6)


def test_splits_disjoint_and_predefined(synthetic_npz):
    ds = PileFlowImages(synthetic_npz, verbose=False)
    s = make_splits(ds)
    assert np.array_equal(np.sort(s.eval), np.sort(ds.eval_idx))
    ev = ds.event_id
    assert not set(ev[s.train]) & set(ev[s.val])
    assert not set(ev[s.train]) & set(ev[s.eval])


def test_max_jets_keeps_eval_blind(synthetic_npz):
    full = PileFlowImages(synthetic_npz, verbose=False)
    small = PileFlowImages(synthetic_npz, max_jets=120, verbose=False)
    s = make_splits(small)
    assert small.has_predefined_event_split
    assert set(s.eval) <= set(full.eval_idx)
    assert not set(s.train) & set(full.eval_idx)


def test_preprocess_roundtrip_and_state(synthetic_npz):
    ds = PileFlowImages(synthetic_npz, verbose=False)
    pre = ImagePreprocessor().fit(ds, np.arange(50), verbose=False)
    x = ds.target_neutral_lv_9[:10]
    assert torch.allclose(pre.inverse_target(pre.transform("target", x)), x, atol=1e-4)
    pre2 = ImagePreprocessor.from_config(pre.get_config(), pre.state_dict())
    assert pre2.fitted and torch.equal(pre2.target_std, pre.target_std)


def test_flow_endpoint_identity():
    f = TargetConditionalFlowMatching(1e-4, 1.0)
    x1 = torch.randn(16, 1, 9, 9)
    t, zt, ut = f.sample_training_tuple(x1)
    assert torch.allclose(f.endpoint_from_velocity(t, zt, ut), x1, atol=1e-5)


@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4", "dopri5"])
def test_sampler_exact_for_constant_field(solver):
    class Const(torch.nn.Module):
        def forward(self, t, z, *a):
            return torch.ones_like(z)
    f = TargetConditionalFlowMatching()
    z0 = torch.zeros(3, 1, 9, 9)
    c9, c36 = torch.zeros(3, 1, 9, 9), torch.zeros(3, 1, 36, 36)
    z1 = f.sample(Const(), c9, c36, c36, solver=solver, steps=7, z0=z0)
    assert torch.allclose(z1, torch.ones_like(z1), atol=1e-4)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_models_shapes(preset):
    m = build_model(model_config_from_preset(preset))
    out = m(torch.rand(4), torch.randn(4, 1, 9, 9), torch.randn(4, 1, 9, 9),
            torch.randn(4, 1, 36, 36), torch.randn(4, 1, 36, 36))
    assert out.shape == (4, 1, 9, 9)


def test_legacy_param_count_unchanged():
    # The draft SpatialPileFlowVelocity had 505,441 parameters.
    assert count_parameters(build_model(model_config_from_preset("spatial"))) == 505441
    assert count_parameters(build_model(model_config_from_preset("spatial_small"))) < 30000


def test_unshuffle_encoder_is_cell_aligned():
    from pileflow.model import UnshuffleChargedEncoder
    enc = UnshuffleChargedEncoder(4)
    x = torch.zeros(1, 1, 36, 36)
    x[0, 0, 4 * 3 + 2, 4 * 5 + 1] = 1.0            # one hit inside neutral cell (3, 5)
    cells = enc.unshuffle(x)[0].sum(0)
    assert cells[3, 5] == 1 and cells.sum() == 1


def test_n95():
    img = np.zeros((3, 9, 9))
    img[0, 0, 0] = 10
    img[1, 0, :4] = [5, 3, 1.4, 0.6]                  # 3 cells hold 94% -> 4 cells
    assert list(n95(img)) == [1, 4, 0]


def test_four_vectors_single_and_pair():
    c = np.zeros((1, 36, 36)); n = np.zeros((1, 9, 9))
    n[0, 4, 4] = 50.0
    fv = four_vectors(c, n, [0.3], [1.0])
    assert abs(fv["pt"][0] - 50) < 1e-9 and fv["mass"][0] < 1e-6
    c[0, 0, 0] = 20.0                                  # second massless particle
    fv = four_vectors(c, n, [0.3], [1.0])
    from pileflow.observables import COARSE_DETA, COARSE_DPHI, FINE_DETA, FINE_DPHI
    deta = COARSE_DETA[4, 4] - FINE_DETA[0, 0]; dphi = COARSE_DPHI[4, 4] - FINE_DPHI[0, 0]
    m2 = 2 * 50 * 20 * (np.cosh(deta) - np.cos(dphi))
    assert abs(fv["mass"][0] - np.sqrt(m2)) < 1e-6


def test_ecf_matches_bruteforce():
    rng = np.random.default_rng(1)
    pt, eta, phi = rng.exponential(5, 12), rng.normal(0, .2, 12), wrap_phi(rng.normal(3.1, .2, 12))
    z = pt / pt.sum()
    dr = lambda i, j: np.hypot(eta[i] - eta[j], wrap_phi(phi[i] - phi[j])) ** 4
    e2 = sum(z[i] * z[j] * dr(i, j) for i, j in itertools.combinations(range(12), 2))
    e3 = sum(z[i] * z[j] * z[k] * dr(i, j) * dr(i, k) * dr(j, k)
             for i, j, k in itertools.combinations(range(12), 3))
    a, b = normalized_ecfs(pt, eta, phi)
    assert np.isclose(a, e2, rtol=1e-10) and np.isclose(b, e3, rtol=1e-10)


def test_dijet_pairing():
    fv = {k: np.array([1.0, 2.0, 3.0]) for k in ("e", "px", "py", "pz")}
    fv["e"] = np.array([10.0, 10.0, 10.0])
    ev, m = dijet_mass([7, 7, 8], [0, 1, 0], fv)
    assert list(ev) == [7] and len(m) == 1


def test_paired_bootstrap_detects_better_method():
    rng = np.random.default_rng(0)
    t = rng.normal(size=2000)
    res = paired_bootstrap(t, {"a": t + rng.normal(0, .5, 2000), "b": t + rng.normal(0, .2, 2000)},
                           pearson, 300, reference="a")
    d = res["diff_vs_a"]["b"]
    assert d["value"] > 0 and d["ci_low"] > 0 and d["p_not_better"] < 0.01


def test_fraction_target_roundtrip_and_bounds():
    pre = ImagePreprocessor(target_mode="fraction", zero_width=0.1, frac_scale=2.0)
    total = torch.tensor([[[[0.0, 5.0, 10.0, 3.0]]]])
    y = torch.tensor([[[[0.0, 0.0, 4.0, 3.5]]]])            # last cell exceeds total -> clipped
    x = pre.encode_target(y, total)
    assert (x[..., :2] < 0).all() and (x[..., :2] >= -0.2).all()   # zeros dequantised below 0
    back = pre.decode_target(x, total)
    assert torch.allclose(back, torch.tensor([[[[0.0, 0.0, 4.0, 3.0]]]]), atol=1e-6)
    # any generated value: output is non-negative, never above the measured total, 0 where total == 0
    out = pre.decode_target(torch.randn(1, 1, 1, 4) * 5, total)
    assert (out >= 0).all() and (out <= total + 1e-6).all() and out[..., 0] == 0
    pre2 = ImagePreprocessor.from_config(pre.get_config(), pre.state_dict())
    assert pre2.target_mode == "fraction"
