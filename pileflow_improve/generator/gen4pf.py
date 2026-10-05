#!/usr/bin/env python3
"""
generator_fixed_v7.py

From-scratch PUMML-style generator for an apples-to-apples
PUMML / PUPPI / PileFlow benchmark.

This version incorporates the validation work performed against the released
PUMML m_phi=500 GeV sample.

PAPER-SPECIFIED SETTINGS
------------------------
PUMML (arXiv:1707.08600):
  * pp collisions at sqrt(s) = 13 TeV
  * 500 GeV scalar -> light-quark jets
  * Pythia 8.183, default tune
  * soft-QCD pileup overlay
  * NPU in [0,180], nominal Poisson mean 140
  * final-state muons and neutrinos removed
  * all particles treated as massless
  * FastJet anti-kT R=0.4
  * parton-level pT cut 95 GeV
  * up to two leading jets/event with pT>100 GeV and |eta|<2.5
  * jet image: 0.9 x 0.9 in eta-phi
  * image center: charged-LV pT-weighted centroid of the selected jet
  * charged image: 36x36, 0.025 x 0.025
  * neutral image: 9x9, 0.1 x 0.1
  * charged reconstruction threshold: 500 MeV
  * lower-pT charged particles are treated as neutral
  * neutral upsampling: each 9x9 pixel -> 4x4 fine pixels / 16
  * channels: total neutral, charged PU, charged LV
  * target: neutral LV
  * 56k images, 90/10 train/test split

PUPPI (as quoted by PUMML and defined in arXiv:1407.6013):
  * R0=0.3, Rmin=0.02, wcut=0.1
  * pTcut = 0.1 + 0.007*NPU GeV
  * central alpha^C uses charged-LV neighbours
  * charged-PU alpha values define event-by-event pileup median / left RMS
  * PUPPI is run on the FULL particle-level event BEFORE jet/image restriction
  * input images use matched reconstructed-jet constituents; the neutral-LV target uses truth-jet constituents

REPRODUCTION ASSUMPTIONS / VALIDATED PROXIES
--------------------------------------------
The PUMML paper does not publish the exact scalar production card.

Tests against the released PUMML m_phi=500 sample found that:
  HiggsSM:ffbar2H + default ISR + an explicit pT>95 GeV cut on both
  scalar daughter quarks reproduces the released leading-jet pT and dijet-mass
  spectra far more closely than HiggsSM:gg2H.

We therefore use ffbar2H as a documented, released-data-validated reproduction
assumption.  It must NOT be described as an explicitly published PUMML setting.

Modern Pythia is used with explicit Tune 4C settings to approximate the
Pythia-8.183 default pp tune.

IMPORTANT
---------
The PUPPI implementation is a faithful implementation of the algorithm in the
original paper, but exact historical source-code identity is not guaranteed.
It should be validated at the observable level before publication.

Requires:
  numpy, scipy, fastjet, pythia8
"""

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

try:
    import fastjet as fj
except Exception as exc:
    raise RuntimeError("FastJet Python bindings are required.") from exc

try:
    import pythia8
except Exception as exc:
    raise RuntimeError("Pythia8 Python bindings are required.") from exc



# Constants
SQRT_S = 13000.0
M_SCALAR = 500.0

R_JET = 0.4
PT_JET_MIN = 100.0
ETA_JET_MAX = 2.5
PARTON_PT_MIN = 95.0

CHARGED_PT_MIN = 0.5

IMAGE_HALF = 0.45
FINE_N = 36
COARSE_N = 9

PUPPI_R0 = 0.3
PUPPI_RMIN = 0.02
PUPPI_WCUT = 0.1
PUPPI_TRACKER_ETA_MAX = 2.5

# rows = [pt, eta, phi, charge, vertex]
PT, ETA, PHI, CHARGE, VTX = range(5)

JET_DEF = fj.JetDefinition(fj.antikt_algorithm, R_JET)

# Basic utilities
def wrap_phi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.hypot(
        np.asarray(eta1) - np.asarray(eta2),
        wrap_phi(np.asarray(phi1) - np.asarray(phi2)),
    )


def poisson_truncated(rng, mean, lo=0, hi=180):
    """Rejection-sample a Poisson distribution, avoiding clipping at the edges."""
    while True:
        n = int(rng.poisson(mean))
        if lo <= n <= hi:
            return n

def is_neutrino(pid):
    return abs(int(pid)) in (12, 14, 16)

def keep_final_particle(p):
    """PUMML keeps final-state particles except muons and neutrinos."""
    if not p.isFinal():
        return False
    pid = int(p.id())
    if abs(pid) == 13 or is_neutrino(pid):
        return False
    return True

def particle_to_row(p, vertex):
    return [
        float(p.pT()),
        float(p.eta()),
        float(p.phi()),
        float(p.charge()),
        float(vertex),
    ]

def rows_to_pseudojets(rows):
    """Convert rows to massless FastJet PseudoJets."""
    out = []
    for i, r in enumerate(rows):
        pt = float(r[PT])
        eta = float(r[ETA])
        phi = float(r[PHI])

        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz = pt * math.sinh(eta)
        e = pt * math.cosh(eta)

        pj = fj.PseudoJet(px, py, pz, e)
        pj.set_user_index(i)
        out.append(pj)

    return out

def pseudojet_mass(j):
    try:
        return float(j.m())
    except Exception:
        e = float(j.e())
        px = float(j.px())
        py = float(j.py())
        pz = float(j.pz())
        return math.sqrt(max(e*e - px*px - py*py - pz*pz, 0.0))

def quiet_pythia(p):
    for cmd in (
        "Print:quiet = on",
        "Init:showProcesses = off",
        "Init:showMultipartonInteractions = off",
        "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
        "Next:numberShowEvent = 0",
    ):
        try:
            p.readString(cmd)
        except Exception:
            pass

# Pythia configuration
def configure_hard(seed):
    """
    Configure the hard event.

    Published:
      sqrt(s)=13 TeV, m_phi=500 GeV, scalar -> light qq, Pythia 8.183 default.

    Reproduction assumption:
      HiggsSM:ffbar2H is used as the scalar production proxy because it closely
      reproduces the released PUMML hard-jet spectrum.

    Modern-Pythia compatibility:
      Tune 4C is selected explicitly to approximate the Pythia-8.183 default.
    """
    p = pythia8.Pythia()

    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {SQRT_S}")

    # Approximate the Pythia-8.183 default pp tune (4C) in modern Pythia.
    p.readString("Tune:pp = 5")
    p.readString("Tune:ee = 3")
    p.readString("TimeShower:weightGluonToQuark = 1")

    # Best released-data-validated scalar-production proxy.
    p.readString("HiggsSM:ffbar2H = on")

    # Narrow 500 GeV scalar.
    p.readString(f"25:m0 = {M_SCALAR}")
    p.readString("25:mWidth = 0.001")
    p.readString("25:doForceWidth = on")
    p.readString(f"25:mMin = {M_SCALAR - 0.01}")
    p.readString(f"25:mMax = {M_SCALAR + 0.01}")

    # Force decays to light u,d,s,c quark-antiquark pairs.
    p.readString("25:onMode = off")
    for q in (1, 2, 3, 4):
        p.readString(f"25:onIfMatch = {q} {-q}")

    # Paper-quoted setting.  For this 2->1 proxy this alone does not enforce
    # the intended daughter-level cut, so an explicit daughter cut is also
    # applied after event generation.
    p.readString(f"PhaseSpace:pTHatMin = {PARTON_PT_MIN}")

    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {int(seed) % 900000000 + 1}")

    quiet_pythia(p)

    if not p.init():
        raise RuntimeError("Hard-scatter Pythia initialization failed.")
    return p

def configure_pileup(seed):
    """Configure zero-bias/soft-QCD pileup."""
    p = pythia8.Pythia()

    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eCM = {SQRT_S}")

    # Match the historical tune used for the hard event.
    p.readString("Tune:pp = 5")
    p.readString("Tune:ee = 3")
    p.readString("TimeShower:weightGluonToQuark = 1")

    # PUMML only states "soft QCD" / zero-bias pileup.  In PYTHIA,
    # SoftQCD:all includes single/double diffraction, which lowers the average
    # visible activity per overlaid interaction.  Our SoftQCD:all samples were
    # ~40% low in charged-PU pT inside R=0.4 relative to the released PUMML
    # sample.  For the reproduction benchmark we therefore use the
    # non-diffractive minimum-bias component, which is the higher-activity
    # zero-bias-like choice and is recorded explicitly as a reproduction
    # assumption.
    p.readString("SoftQCD:nonDiffractive = on")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {int(seed) % 900000000 + 1}")
    quiet_pythia(p)

    if not p.init():
        raise RuntimeError("Pileup Pythia initialization failed.")

    return p


def find_scalar_daughters(pythia):
    """
    Find the two light-quark daughters of the last scalar/Higgs copy.
    Returns a metadata dictionary or None.
    """
    candidates = []

    for i in range(pythia.event.size()):
        h = pythia.event[i]

        if abs(int(h.id())) != 25:
            continue

        d1 = int(h.daughter1())
        d2 = int(h.daughter2())

        if d1 <= 0 or d2 <= 0:
            continue

        qs = []

        for j in range(d1, d2 + 1):
            if j < 0 or j >= pythia.event.size():
                continue

            q = pythia.event[j]

            if abs(int(q.id())) in (1, 2, 3, 4):
                qs.append(q)

        if len(qs) >= 2:
            candidates.append(qs[:2])

    if not candidates:
        return None

    q1, q2 = candidates[-1]

    return {
        "daughter1_id": int(q1.id()),
        "daughter1_pt": float(q1.pT()),
        "daughter1_eta": float(q1.eta()),
        "daughter1_phi": float(q1.phi()),
        "daughter2_id": int(q2.id()),
        "daughter2_pt": float(q2.pT()),
        "daughter2_eta": float(q2.eta()),
        "daughter2_phi": float(q2.phi()),
    }

def scalar_daughters_pass_95(d):
    if d is None:
        return False
    return (
        float(d["daughter1_pt"]) > PARTON_PT_MIN
        and float(d["daughter2_pt"]) > PARTON_PT_MIN
    )

def extract_final_rows(pythia, vertex):
    rows = []

    for i in range(pythia.event.size()):
        p = pythia.event[i]

        if keep_final_particle(p):
            rows.append(particle_to_row(p, vertex))

    if not rows:
        return np.empty((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)

def generate_one_pileup_event(pythia, vertex):
    while True:
        if not pythia.next():
            continue

        rows = extract_final_rows(pythia, vertex)

        if len(rows):
            return rows

# Jet finding
def cluster_jets(rows):
    if len(rows) == 0:
        return [], None

    pjs = rows_to_pseudojets(rows)
    cs = fj.ClusterSequence(pjs, JET_DEF)
    jets = fj.sorted_by_pt(cs.inclusive_jets())

    return jets, cs

def select_truth_jets(lv_rows):
    """
    Select up to two leading LV jets exactly as stated in PUMML.
    """
    jets, _ = cluster_jets(lv_rows)

    selected = [
        j
        for j in jets
        if j.pt() > PT_JET_MIN and abs(j.eta()) < ETA_JET_MAX
    ][:2]

    out = []

    for rank, j in enumerate(selected):
        idx = np.asarray(
            [c.user_index() for c in j.constituents()],
            dtype=np.int32,
        )

        out.append({
            "rank": rank,
            "pt": float(j.pt()),
            "eta": float(j.eta()),
            "phi": float(j.phi()),
            "mass": pseudojet_mass(j),
            "constituent_idx": idx,
        })

    return out

def match_reco_jet(full_rows, truth_eta, truth_phi):
    """
    Match a selected LV truth jet to the nearest anti-kT R=0.4 jet clustered
    from the COMPLETE hard+pileup event.

    The returned constituent_idx values index full_rows. These constituents
    define the PUMML input-image particle support in v3.
    """
    jets, _ = cluster_jets(full_rows)

    if not jets:
        return None

    drs = [float(delta_r(j.eta(), j.phi(), truth_eta, truth_phi)) for j in jets]
    k = int(np.argmin(drs))

    if drs[k] >= R_JET:
        return None

    j = jets[k]

    idx = np.asarray(
        [c.user_index() for c in j.constituents()],
        dtype=np.int32,
    )

    return {
        "pt": float(j.pt()),
        "eta": float(j.eta()),
        "phi": float(j.phi()),
        "mass": pseudojet_mass(j),
        "dr_match": float(drs[k]),
        "constituent_idx": idx,
    }



# PUPPI
def _charged_reco_mask(rows):
    """
    Detector-image charged reconstruction mask.

    IMPORTANT: this 500 MeV threshold belongs to the PUMML image detector
    model, not to the PUPPI particle-level tracker logic.
    """
    return (
        (np.abs(rows[:, CHARGE]) > 1e-6)
        & (rows[:, PT] > CHARGED_PT_MIN)
    )


def _local_alpha(rows, target_idx, neighbour_idx, chunk=2048):
    """
    Compute

        alpha_i = log sum_j pT_j / DeltaR_ij

    for arbitrary target and neighbour sets with
    Rmin <= DeltaR_ij <= R0.
    """
    target_idx = np.asarray(target_idx, dtype=np.int64)
    neighbour_idx = np.asarray(neighbour_idx, dtype=np.int64)

    out = np.full(len(target_idx), -np.inf, dtype=np.float64)

    if len(target_idx) == 0 or len(neighbour_idx) == 0:
        return out

    ne = rows[neighbour_idx, ETA].astype(np.float64)
    np_ = rows[neighbour_idx, PHI].astype(np.float64)
    npt = rows[neighbour_idx, PT].astype(np.float64)

    for start in range(0, len(target_idx), chunk):
        stop = min(start + chunk, len(target_idx))
        ids = target_idx[start:stop]

        te = rows[ids, ETA].astype(np.float64)[:, None]
        tp = rows[ids, PHI].astype(np.float64)[:, None]

        deta = te - ne[None, :]
        dphi = wrap_phi(tp - np_[None, :])
        dr = np.hypot(deta, dphi)

        mask = (dr >= PUPPI_RMIN) & (dr <= PUPPI_R0)

        contrib = np.zeros_like(dr)
        np.divide(npt[None, :], dr, out=contrib, where=mask)

        sums = np.sum(contrib, axis=1)
        good = sums > 0.0

        vals = np.full(len(ids), -np.inf, dtype=np.float64)
        vals[good] = np.log(sums[good])
        out[start:stop] = vals

    return out


def _median_left_rms(alpha):
    """
    Event-level PUPPI pileup characterization.

    The original PUPPI paper uses the median and the left-side RMS of the
    charged-PU alpha distribution. Values with no valid neighbours correspond
    to alpha -> -inf and are intrinsically pileup-like; they are excluded from
    the numerical median/RMS estimate because no finite alpha exists.
    """
    a = np.asarray(alpha, dtype=np.float64)
    a = a[np.isfinite(a)]

    if len(a) < 4:
        return None

    med = float(np.median(a))
    left = a[a <= med]

    if len(left) == 0:
        return None

    sigma = math.sqrt(max(float(np.mean((left - med) ** 2)), 1e-12))
    return med, sigma


def puppi_weights_full_event(rows, npu):
    """
    PUPPI implementation aligned with Bertolini et al. (arXiv:1407.6013)
    and with the parameters quoted in the PUMML paper.

    Key distinctions from generator v5
    -----------------------------------
    1. PUPPI tracking logic does NOT use the PUMML 500 MeV charged-image
       threshold.  The original PUPPI toy detector has a perfect tracker in
       |eta| < 2.5.  All charged particles in that region therefore carry
       vertex information.

    2. Charged LV particles get weight 1 and charged PU particles get weight 0
       only inside the tracker acceptance.  Outside |eta| < 2.5 particles are
       treated as unknown, as in the original central/forward PUPPI split.

    3. alpha^C uses charged-LV neighbours (Eq. 2.3), while alpha^F uses all
       particles (Eq. 2.4).  Charged-PU particles provide the event-by-event
       median and left-side RMS for both alpha distributions (Eqs. 2.5-2.6).

    4. After weights are assigned, the weight cut and the rescaled-pT cut are
       applied to the COMPLETE particle collection, including weight-1 charged
       LV particles.  This follows the algorithm summary in the PUPPI paper:
       rescale p_i^mu -> w_i p_i^mu, then discard particles with small weight
       or small rescaled pT.

    PUMML quotes:
        R0=0.3, Rmin=0.02, wcut=0.1,
        pTcut = 0.1 + 0.007*NPU GeV.

    For the PUMML benchmark we use that quoted pTcut for the full collection,
    including the small fraction of particles outside tracker acceptance.
    """
    n = len(rows)

    if n == 0:
        return np.empty(0, dtype=np.float32)

    weights = np.ones(n, dtype=np.float64)

    is_charged = np.abs(rows[:, CHARGE]) > 1e-6
    in_tracker = np.abs(rows[:, ETA]) < PUPPI_TRACKER_ETA_MAX

    ch_lv = is_charged & in_tracker & (rows[:, VTX] == 0)
    ch_pu = is_charged & in_tracker & (rows[:, VTX] > 0)

    central_unknown = in_tracker & ~(ch_lv | ch_pu)
    forward_unknown = ~in_tracker

    weights[ch_lv] = 1.0
    weights[ch_pu] = 0.0

    lv_idx = np.where(ch_lv)[0]
    pu_idx = np.where(ch_pu)[0]
    central_idx = np.where(central_unknown)[0]
    forward_idx = np.where(forward_unknown)[0]
    all_idx = np.arange(n, dtype=np.int64)

    # alpha^C: charged-LV neighbours in the tracker region.
    alphaC_pu = _local_alpha(rows, pu_idx, lv_idx)
    statsC = _median_left_rms(alphaC_pu)

    # alpha^F: all event particles as neighbours.  The pileup reference is
    # still measured with central charged-PU targets.
    alphaF_pu = _local_alpha(rows, pu_idx, all_idx)
    statsF = _median_left_rms(alphaF_pu)

    def assign_unknown(target_idx, neighbour_idx, stats):
        if len(target_idx) == 0:
            return

        if stats is None:
            # Conservative failure mode: without a stable PU reference, do not
            # promote unknown particles to LV-like.
            weights[target_idx] = 0.0
            return

        med, sigma = stats
        alpha = _local_alpha(rows, target_idx, neighbour_idx)

        for k, i in enumerate(target_idx):
            a = float(alpha[k])

            # Eq. 2.7: fluctuations below the PU median receive chi2=0.
            if (not np.isfinite(a)) or a <= med:
                w = 0.0
            else:
                chi2_i = ((a - med) / sigma) ** 2
                w = float(chi2.cdf(chi2_i, df=1))

            weights[i] = w

    assign_unknown(central_idx, lv_idx, statsC)
    assign_unknown(forward_idx, all_idx, statsF)

    # Charged-PU weights are fixed to zero; charged-LV weights are fixed to one.
    # Apply the final cuts to the complete weighted particle collection.
    ptcut = 0.1 + 0.007 * float(npu)

    reject = (
        (weights < PUPPI_WCUT)
        | (weights * rows[:, PT] < ptcut)
    )
    weights[reject] = 0.0

    return weights.astype(np.float32)


def make_puppi_corrected_rows(rows, weights):
    """
    Build the particle-level PUPPI-corrected event.

    PUPPI is a particle-level algorithm: surviving particle four-momenta are
    rescaled before jet finding.  Since all particles are treated as massless
    in this benchmark, rescaling the four-vector by w is equivalent to
    pt -> w*pt with eta/phi unchanged.

    Rows with zero final weight are removed.
    """
    rows = np.asarray(rows)
    weights = np.asarray(weights, dtype=np.float64)

    keep = weights > 0.0
    if not np.any(keep):
        return np.empty((0, rows.shape[1]), dtype=np.float32)

    out = rows[keep].copy().astype(np.float32)
    out[:, PT] *= weights[keep].astype(np.float32)
    return out


def match_puppi_jet(puppi_rows, truth_eta, truth_phi):
    """
    Cluster the PUPPI-corrected particle collection independently and match the
    nearest anti-kT R=0.4 PUPPI jet to the selected LV truth jet.

    This is required by the PUPPI paper: weights are applied at particle level,
    then the corrected particle collection is used as input to jet finding.
    """
    jets, _ = cluster_jets(puppi_rows)

    if not jets:
        return None

    drs = [
        float(delta_r(j.eta(), j.phi(), truth_eta, truth_phi))
        for j in jets
    ]
    k = int(np.argmin(drs))

    if drs[k] >= R_JET:
        return None

    j = jets[k]
    idx = np.asarray(
        [c.user_index() for c in j.constituents()],
        dtype=np.int32,
    )

    return {
        "pt": float(j.pt()),
        "eta": float(j.eta()),
        "phi": float(j.phi()),
        "mass": pseudojet_mass(j),
        "dr_match": float(drs[k]),
        "constituent_idx": idx,
    }


def build_puppi_images(puppi_jet_rows, eta0, phi0):
    """
    Detector-discretize the independently reclustered PUPPI jet.

    Separate charged and neutral outputs are stored so the evaluator never has
    to recover a neutral PUPPI image by subtracting the original charged-LV
    image.  This avoids restoring charged particles that PUPPI itself removed.

    The detector discretization is applied after PUPPI, as stated in PUMML.
    """
    if len(puppi_jet_rows) == 0:
        z36 = np.zeros((FINE_N, FINE_N), dtype=np.float32)
        z9 = np.zeros((COARSE_N, COARSE_N), dtype=np.float32)
        return {
            "charged36": z36,
            "neutral9": z9,
            "neutral36": z36.copy(),
            "total36": z36.copy(),
        }

    charged = _charged_reco_mask(puppi_jet_rows)
    neutral_like = ~charged

    charged36 = hist(
        puppi_jet_rows[charged],
        eta0,
        phi0,
        FINE_N,
    )
    neutral9 = hist(
        puppi_jet_rows[neutral_like],
        eta0,
        phi0,
        COARSE_N,
    )
    neutral36 = upsample9(neutral9)
    total36 = charged36 + neutral36

    return {
        "charged36": charged36.astype(np.float32),
        "neutral9": neutral9.astype(np.float32),
        "neutral36": neutral36.astype(np.float32),
        "total36": total36.astype(np.float32),
    }



# Image construction
def charged_lv_centroid_of_truth_jet(truth_jet_rows):
    """
    pT-weighted centroid of CHARGED LV constituents of the selected truth jet.
    """
    m = (
        (np.abs(truth_jet_rows[:, CHARGE]) > 1e-6)
        & (truth_jet_rows[:, PT] > CHARGED_PT_MIN)
        & (truth_jet_rows[:, VTX] == 0)
    )

    x = truth_jet_rows[m]

    if len(x) == 0:
        return None

    w = x[:, PT].astype(np.float64)

    eta0 = float(np.average(x[:, ETA], weights=w))

    s = float(np.sum(w * np.sin(x[:, PHI])))
    c = float(np.sum(w * np.cos(x[:, PHI])))
    phi0 = float(np.arctan2(s, c))

    return eta0, phi0


def crop_mask(rows, eta0, phi0):
    if len(rows) == 0:
        return np.zeros(0, dtype=bool)

    deta = rows[:, ETA] - eta0
    dphi = wrap_phi(rows[:, PHI] - phi0)

    return (
        (deta >= -IMAGE_HALF)
        & (deta < IMAGE_HALF)
        & (dphi >= -IMAGE_HALF)
        & (dphi < IMAGE_HALF)
    )

def hist(rows, eta0, phi0, nbin, weights=None):
    """
    Geometric eta-phi image histogram.

    Cropping is geometric (0.9 x 0.9 around eta0/phi0). Which particles are
    passed in is decided by the caller: build_images() passes the
    constituents of the matched reco jet (inputs) or of the truth jet
    (target), detector_image() may pass any collection.
    """
    if len(rows) == 0:
        return np.zeros((nbin, nbin), dtype=np.float32)

    deta = rows[:, ETA] - eta0
    dphi = wrap_phi(rows[:, PHI] - phi0)

    m = (
        (deta >= -IMAGE_HALF)
        & (deta < IMAGE_HALF)
        & (dphi >= -IMAGE_HALF)
        & (dphi < IMAGE_HALF)
    )

    if not np.any(m):
        return np.zeros((nbin, nbin), dtype=np.float32)

    w = rows[m, PT].astype(np.float64)

    if weights is not None:
        w = w * np.asarray(weights, dtype=np.float64)[m]

    edges = np.linspace(
        -IMAGE_HALF,
        IMAGE_HALF,
        nbin + 1,
        dtype=np.float64,
    )

    h, _, _ = np.histogram2d(
        deta[m],
        dphi[m],
        bins=[edges, edges],
        weights=w,
    )

    return h.astype(np.float32)


def upsample9(x):
    return (
        np.repeat(
            np.repeat(x, 4, axis=0),
            4,
            axis=1,
        ).astype(np.float32)
        / 16.0
    )


def detector_image(rows, eta0, phi0, weights=None):
    """
    Detector-discretized total image:
      charged reconstructed particles -> 36x36
      neutrals + charged below 500 MeV -> 9x9 then upsampled
    """
    if len(rows) == 0:
        return np.zeros((FINE_N, FINE_N), dtype=np.float32)

    charged = _charged_reco_mask(rows)
    neutral_like = ~charged

    wc = None if weights is None else np.asarray(weights)[charged]
    wn = None if weights is None else np.asarray(weights)[neutral_like]

    charged36 = hist(
        rows[charged],
        eta0,
        phi0,
        FINE_N,
        wc,
    )

    neutral9 = hist(
        rows[neutral_like],
        eta0,
        phi0,
        COARSE_N,
        wn,
    )

    return charged36 + upsample9(neutral9)


def build_images(
    reco_jet_rows,
    truth_jet_rows,
    reco_puppi_weights,
):
    """
    Build one PUMML jet image.

    V3 support:
      * input channels: constituents of the matched reconstructed LV+PU jet
      * target: constituents of the selected LV truth jet
      * center: charged-LV pT-weighted centroid of the selected truth jet
      * canvas: 0.9 x 0.9 square around that centroid

    PUPPI weights must already have been computed on the COMPLETE event.
    """
    center = charged_lv_centroid_of_truth_jet(truth_jet_rows)

    if center is None:
        return None

    eta0, phi0 = center

    # Reconstructed full-event jet -> PUMML inputs
    reco_charged = _charged_reco_mask(reco_jet_rows)
    reco_ch_lv = reco_charged & (reco_jet_rows[:, VTX] == 0)
    reco_ch_pu = reco_charged & (reco_jet_rows[:, VTX] > 0)
    reco_neutral_like = ~reco_charged

    neutral_total9 = hist(
        reco_jet_rows[reco_neutral_like],
        eta0,
        phi0,
        COARSE_N,
    )

    charged_pu36 = hist(
        reco_jet_rows[reco_ch_pu],
        eta0,
        phi0,
        FINE_N,
    )

    charged_lv36 = hist(
        reco_jet_rows[reco_ch_lv],
        eta0,
        phi0,
        FINE_N,
    )

    # Selected LV truth jet -> neutral-LV target
    truth_charged = _charged_reco_mask(truth_jet_rows)
    truth_neutral_like = ~truth_charged

    neutral_lv9 = hist(
        truth_jet_rows[truth_neutral_like],
        eta0,
        phi0,
        COARSE_N,
    )

    X = np.stack(
        [
            upsample9(neutral_total9),
            charged_pu36,
            charged_lv36,
        ],
        axis=0,
    ).astype(np.float32)

    y9 = neutral_lv9.astype(np.float32)
    y36 = upsample9(y9)

    no_pileup36 = detector_image(
        truth_jet_rows,
        eta0,
        phi0,
    )

    with_pileup36 = detector_image(
        reco_jet_rows,
        eta0,
        phi0,
    )

    puppi36 = detector_image(
        reco_jet_rows,
        eta0,
        phi0,
        weights=reco_puppi_weights,
    )

    crop_reco = crop_mask(reco_jet_rows, eta0, phi0)
    crop_truth = crop_mask(truth_jet_rows, eta0, phi0)

    return {
        "X": X,
        "y9": y9,
        "y36": y36,
        "no_pileup36": no_pileup36,
        "with_pileup36": with_pileup36,
        "puppi36": puppi36,
        "center_eta": np.float32(eta0),
        "center_phi": np.float32(phi0),
        "n_particles_reco_jet": np.int32(len(reco_jet_rows)),
        "n_particles_truth_jet": np.int32(len(truth_jet_rows)),
        "n_particles_crop_reco": np.int32(np.sum(crop_reco)),
        "n_particles_crop_truth": np.int32(np.sum(crop_truth)),
    }


def build_square_debug_images(full_rows, lv_rows, truth_jet_rows):
    """
    Auxiliary diagnostic representation.

    This reproduces the literal geometric 0.9x0.9 square construction used in
    the v4 cross-check: all relevant event particles inside the square are
    histogrammed, irrespective of anti-kT membership.

    It is saved only when --save-square-debug is requested and is never used
    as the primary v5 training input.
    """
    center = charged_lv_centroid_of_truth_jet(truth_jet_rows)

    if center is None:
        return None

    eta0, phi0 = center

    full_charged = _charged_reco_mask(full_rows)
    full_ch_lv = full_charged & (full_rows[:, VTX] == 0)
    full_ch_pu = full_charged & (full_rows[:, VTX] > 0)
    full_neutral_like = ~full_charged

    lv_charged = _charged_reco_mask(lv_rows)
    lv_neutral_like = ~lv_charged

    neutral_total9 = hist(
        full_rows[full_neutral_like],
        eta0,
        phi0,
        COARSE_N,
    )

    charged_pu36 = hist(
        full_rows[full_ch_pu],
        eta0,
        phi0,
        FINE_N,
    )

    charged_lv36 = hist(
        full_rows[full_ch_lv],
        eta0,
        phi0,
        FINE_N,
    )

    neutral_lv9 = hist(
        lv_rows[lv_neutral_like],
        eta0,
        phi0,
        COARSE_N,
    )

    return {
        "X_square": np.stack(
            [
                upsample9(neutral_total9),
                charged_pu36,
                charged_lv36,
            ],
            axis=0,
        ).astype(np.float32),
        "y_square_36": upsample9(neutral_lv9),
    }



# Multiprocessing

@dataclass
class WorkerConfig:
    worker_id: int
    target_jets: int
    mean_npu: float
    seed: int
    save_square_debug: bool = False


def worker(cfg):
    rng = np.random.default_rng(cfg.seed)

    hard = configure_hard(cfg.seed + 17)
    pu = configure_pileup(cfg.seed + 100003)

    items = []
    hard_attempts = 0
    hard_accepted_after_parton_cut = 0
    local_event_counter = 0

    while len(items) < cfg.target_jets:
        hard_attempts += 1

        if not hard.next():
            continue

        daughters = find_scalar_daughters(hard)

        # Explicitly realize the paper's 95 GeV parton-level cut for the
        # selected scalar-decay topology.
        if not scalar_daughters_pass_95(daughters):
            continue

        lv_rows = extract_final_rows(hard, vertex=0)

        if len(lv_rows) == 0:
            continue

        hard_accepted_after_parton_cut += 1

        truth_jets = select_truth_jets(lv_rows)

        if not truth_jets:
            continue

        local_event_counter += 1

        npu = poisson_truncated(
            rng,
            cfg.mean_npu,
            0,
            180,
        )

        pu_blocks = [
            generate_one_pileup_event(pu, vertex=vtx)
            for vtx in range(1, npu + 1)
        ]

        if pu_blocks:
            full_rows = np.concatenate(
                [lv_rows] + pu_blocks,
                axis=0,
            )
        else:
            full_rows = lv_rows.copy()

        # PUPPI is evaluated on the full event at particle level.
        puppi_w = puppi_weights_full_event(
            full_rows,
            npu,
        )
        puppi_rows = make_puppi_corrected_rows(
            full_rows,
            puppi_w,
        )

        event_id = (
            (np.int64(cfg.worker_id) << np.int64(48))
            | np.int64(local_event_counter)
        )

        event_items = []

        for tj in truth_jets:
            truth_jet_rows = lv_rows[tj["constituent_idx"]]

            # Match a reconstructed jet in the complete LV+PU event.
            # These reco-jet constituents define the three PUMML inputs.
            reco = match_reco_jet(
                full_rows,
                tj["eta"],
                tj["phi"],
            )

            if reco is None:
                continue

            reco_idx = reco["constituent_idx"]
            reco_jet_rows = full_rows[reco_idx]

            # Primary PUMML image construction.
            # The PUPPI branch is handled independently below.
            reco_puppi_w = np.ones(len(reco_jet_rows), dtype=np.float32)

            built = build_images(
                reco_jet_rows=reco_jet_rows,
                truth_jet_rows=truth_jet_rows,
                reco_puppi_weights=reco_puppi_w,
            )

            if built is None:
                continue

            # The PUPPI-corrected event must be reclustered after weighting.
            pjet = match_puppi_jet(
                puppi_rows,
                tj["eta"],
                tj["phi"],
            )

            if pjet is None:
                # Valid PUPPI outcome: the corresponding jet can disappear
                # after particle-level mitigation. Store an empty image.
                pimgs = build_puppi_images(
                    np.empty((0, 5), dtype=np.float32),
                    built["center_eta"],
                    built["center_phi"],
                )
                pjet_meta = {
                    "pt": np.nan,
                    "eta": np.nan,
                    "phi": np.nan,
                    "mass": np.nan,
                    "dr_match": np.nan,
                }
            else:
                pjet_rows = puppi_rows[pjet["constituent_idx"]]
                pimgs = build_puppi_images(
                    pjet_rows,
                    built["center_eta"],
                    built["center_phi"],
                )
                pjet_meta = pjet

            square_debug = None
            if cfg.save_square_debug:
                square_debug = build_square_debug_images(
                    full_rows=full_rows,
                    lv_rows=lv_rows,
                    truth_jet_rows=truth_jet_rows,
                )

            item = {
                "X": built["X"],
                "y9": built["y9"],
                "y36": built["y36"],
                "no_pileup36": built["no_pileup36"],
                "with_pileup36": built["with_pileup36"],
                "puppi36": pimgs["total36"],
                "puppi_charged36": pimgs["charged36"],
                "puppi_neutral9": pimgs["neutral9"],
                "puppi_neutral36": pimgs["neutral36"],

                "npu": np.int16(npu),

                "event_id": np.int64(event_id),
                "jet_rank": np.int8(tj["rank"]),
                "worker_id": np.int16(cfg.worker_id),

                "center_eta": np.float32(built["center_eta"]),
                "center_phi": np.float32(built["center_phi"]),

                "n_particles_reco_jet":
                    np.int32(built["n_particles_reco_jet"]),
                "n_particles_truth_jet":
                    np.int32(built["n_particles_truth_jet"]),
                "n_particles_crop_reco":
                    np.int32(built["n_particles_crop_reco"]),
                "n_particles_crop_truth":
                    np.int32(built["n_particles_crop_truth"]),

                # Original LV anti-kT selected jet metadata.
                "truth_pt": np.float32(tj["pt"]),
                "truth_eta": np.float32(tj["eta"]),
                "truth_phi": np.float32(tj["phi"]),
                "truth_mass": np.float32(tj["mass"]),

                # Matched reconstructed full-event jet used for the inputs.
                "reco_pt": np.float32(reco["pt"]),
                "reco_eta": np.float32(reco["eta"]),
                "reco_phi": np.float32(reco["phi"]),
                "reco_mass": np.float32(reco["mass"]),
                "reco_truth_dr": np.float32(reco["dr_match"]),

                # Independently reclustered particle-level PUPPI jet.
                "puppi_jet_pt": np.float32(pjet_meta["pt"]),
                "puppi_jet_eta": np.float32(pjet_meta["eta"]),
                "puppi_jet_phi": np.float32(pjet_meta["phi"]),
                "puppi_jet_mass": np.float32(pjet_meta["mass"]),
                "puppi_truth_dr": np.float32(pjet_meta["dr_match"]),

                # Generator-level scalar daughters.
                "scalar_daughter1_id":
                    np.int16(daughters["daughter1_id"]),
                "scalar_daughter1_pt":
                    np.float32(daughters["daughter1_pt"]),
                "scalar_daughter1_eta":
                    np.float32(daughters["daughter1_eta"]),
                "scalar_daughter1_phi":
                    np.float32(daughters["daughter1_phi"]),

                "scalar_daughter2_id":
                    np.int16(daughters["daughter2_id"]),
                "scalar_daughter2_pt":
                    np.float32(daughters["daughter2_pt"]),
                "scalar_daughter2_eta":
                    np.float32(daughters["daughter2_eta"]),
                "scalar_daughter2_phi":
                    np.float32(daughters["daughter2_phi"]),
            }

            if square_debug is not None:
                item["X_square_debug"] = square_debug["X_square"]
                item["y_square_debug_36"] = square_debug["y_square_36"]

            event_items.append(item)

        # Preserve all selected jets from an event locally.  The master process
        # performs the exact final size trim.
        items.extend(event_items)

        if local_event_counter % 100 == 0:
            print(
                f"[worker {cfg.worker_id:02d}] "
                f"{len(items)}/{cfg.target_jets} images, "
                f"{local_event_counter} selected hard events, "
                f"{hard_attempts} PYTHIA attempts",
                flush=True,
            )

    return {
        "items": items,
        "hard_attempts": hard_attempts,
        "hard_accepted_after_parton_cut":
            hard_accepted_after_parton_cut,
        "selected_events": local_event_counter,
    }



# Splits
def exact_image_split(rng, n):
    """Exact 90/10 image-level split, matching the paper statement."""
    perm = rng.permutation(n).astype(np.int32)
    ntrain = int(round(0.90 * n))

    return perm[:ntrain], perm[ntrain:]


def grouped_event_split(rng, event_ids):
    """
    Additional leakage-free event-level 90/10 split.

    This is NOT claimed to be what the original PUMML implementation used.
    """
    unique_events = np.unique(event_ids)
    ep = rng.permutation(unique_events)

    ntrain = int(round(0.90 * len(ep)))

    train_events = ep[:ntrain]
    test_events = ep[ntrain:]

    train_idx = np.where(
        np.isin(event_ids, train_events)
    )[0].astype(np.int32)

    test_idx = np.where(
        np.isin(event_ids, test_events)
    )[0].astype(np.int32)

    return train_idx, test_idx



# Final exact-size selection
def choose_exact_n_images(items, n, rng):
    """
    Choose exactly n images while preserving complete hard events as much as
    possible.

    Since events contain one or two selected jets, an exact n is normally
    achievable with full events.  If the final count differs by one, one
    leading jet from an otherwise valid event is used; PUMML explicitly says
    "up to two leading jets", so this does not violate the published selection.
    """
    by_event = {}

    for x in items:
        by_event.setdefault(int(x["event_id"]), []).append(x)

    event_ids = np.asarray(list(by_event.keys()), dtype=np.int64)
    event_ids = rng.permutation(event_ids)

    chosen = []

    # First pass: greedily add full events without exceeding n.
    deferred = []

    for eid in event_ids:
        block = sorted(
            by_event[int(eid)],
            key=lambda z: int(z["jet_rank"]),
        )

        if len(chosen) + len(block) <= n:
            chosen.extend(block)
        else:
            deferred.append(block)

        if len(chosen) == n:
            break

    # Fill any remainder from valid deferred leading/subleading images.
    if len(chosen) < n:
        need = n - len(chosen)

        for block in deferred:
            for x in block:
                chosen.append(x)
                need -= 1

                if need == 0:
                    break

            if need == 0:
                break

    if len(chosen) < n:
        raise RuntimeError(
            f"Unable to select {n} images from {len(items)} generated records."
        )

    return chosen[:n]



# Main
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
        default="pumml_reproduction_fixed.npz",
    )

    ap.add_argument(
        "--n-jets",
        type=int,
        default=56000,
    )

    ap.add_argument(
        "--mean-npu",
        type=float,
        default=140.0,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            min(
                32,
                (os.cpu_count() or 2) - 1,
            ),
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=170708600,
    )

    ap.add_argument(
        "--compressed",
        action="store_true",
    )

    ap.add_argument(
        "--save-square-debug",
        action="store_true",
        help=(
            "Also save auxiliary paper-square input/target images built from "
            "all particles in the 0.9x0.9 crop. These are diagnostics only; "
            "the primary X/y arrays remain reco-jet/truth-jet supported."
        ),
    )

    args = ap.parse_args()

    if args.n_jets < 1:
        raise ValueError("--n-jets must be positive.")

    workers = min(args.workers, args.n_jets)

    # Ask each worker for a small safety margin because workers preserve whole
    # events and can overshoot their nominal jet target by one.
    base = args.n_jets // workers
    rem = args.n_jets % workers

    cfgs = []

    for i in range(workers):
        target = base + (1 if i < rem else 0) + 2

        cfgs.append(
            WorkerConfig(
                worker_id=i,
                target_jets=target,
                mean_npu=args.mean_npu,
                seed=args.seed + 1000003 * i,
                save_square_debug=args.save_square_debug,
            )
        )

    print("=" * 88)
    print("PUMML-style generator — v7 particle-level PUPPI + non-diffractive PU")
    print("=" * 88)
    print(f"Requested jet images : {args.n_jets}")
    print(f"Mean NPU             : {args.mean_npu}")
    print(f"Workers              : {workers}")
    print("Hard proxy           : HiggsSM:ffbar2H")
    print("Parton cut           : both scalar daughters pT > 95 GeV")
    print("PUPPI                : full event before reco-jet restriction")
    print("Primary image support: matched reco-jet constituents")
    print(f"Square debug         : {args.save_square_debug}")
    print("Input support         : matched reco-jet constituents")
    print("Target support        : selected LV truth-jet constituents")
    print()

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=workers) as pool:
        results = pool.map(worker, cfgs)

    all_items = [
        x
        for result in results
        for x in result["items"]
    ]

    if len(all_items) < args.n_jets:
        raise RuntimeError(
            f"Generated only {len(all_items)} images; "
            f"requested {args.n_jets}."
        )

    rng = np.random.default_rng(args.seed + 99991)

    items = choose_exact_n_images(
        all_items,
        args.n_jets,
        rng,
    )

    # Shuffle final image order while preserving event_id/jet_rank metadata.
    order = rng.permutation(len(items))
    items = [items[i] for i in order]

    def stack(key, dtype=np.float32):
        return np.stack(
            [x[key] for x in items]
        ).astype(dtype)

    def array(key, dtype):
        return np.asarray(
            [x[key] for x in items],
            dtype=dtype,
        )

    X = stack("X")
    y9 = stack("y9")
    y36 = stack("y36")

    no_pileup36 = stack("no_pileup36")
    with_pileup36 = stack("with_pileup36")
    puppi36 = stack("puppi36")
    puppi_charged36 = stack("puppi_charged36")
    puppi_neutral9 = stack("puppi_neutral9")
    puppi_neutral36 = stack("puppi_neutral36")

    npu = array("npu", np.int16)

    event_id = array("event_id", np.int64)
    jet_rank = array("jet_rank", np.int8)
    worker_id = array("worker_id", np.int16)

    center_eta = array("center_eta", np.float32)
    center_phi = array("center_phi", np.float32)

    n_particles_reco_jet = array(
        "n_particles_reco_jet",
        np.int32,
    )
    n_particles_truth_jet = array(
        "n_particles_truth_jet",
        np.int32,
    )
    n_particles_crop_reco = array(
        "n_particles_crop_reco",
        np.int32,
    )
    n_particles_crop_truth = array(
        "n_particles_crop_truth",
        np.int32,
    )

    truth_pt = array("truth_pt", np.float32)
    truth_eta = array("truth_eta", np.float32)
    truth_phi = array("truth_phi", np.float32)
    truth_mass = array("truth_mass", np.float32)

    reco_pt = array("reco_pt", np.float32)
    reco_eta = array("reco_eta", np.float32)
    reco_phi = array("reco_phi", np.float32)
    reco_mass = array("reco_mass", np.float32)
    reco_truth_dr = array("reco_truth_dr", np.float32)

    puppi_jet_pt = array("puppi_jet_pt", np.float32)
    puppi_jet_eta = array("puppi_jet_eta", np.float32)
    puppi_jet_phi = array("puppi_jet_phi", np.float32)
    puppi_jet_mass = array("puppi_jet_mass", np.float32)
    puppi_truth_dr = array("puppi_truth_dr", np.float32)

    d1_id = array("scalar_daughter1_id", np.int16)
    d1_pt = array("scalar_daughter1_pt", np.float32)
    d1_eta = array("scalar_daughter1_eta", np.float32)
    d1_phi = array("scalar_daughter1_phi", np.float32)

    d2_id = array("scalar_daughter2_id", np.int16)
    d2_pt = array("scalar_daughter2_pt", np.float32)
    d2_eta = array("scalar_daughter2_eta", np.float32)
    d2_phi = array("scalar_daughter2_phi", np.float32)

    X_square_debug = None
    y_square_debug_36 = None

    if args.save_square_debug:
        X_square_debug = stack("X_square_debug")
        y_square_debug_36 = stack("y_square_debug_36")

    train_idx, test_idx = exact_image_split(rng, args.n_jets)

    event_train_idx, event_test_idx = grouped_event_split(rng, event_id)

    generation_stats = {
        "pythia_hard_attempts":
            int(sum(r["hard_attempts"] for r in results)),
        "hard_events_after_daughter_pt_cut":
            int(sum(
                r["hard_accepted_after_parton_cut"]
                for r in results
            )),
        "selected_hard_events_generated":
            int(sum(r["selected_events"] for r in results)),
        "raw_images_before_exact_trim":
            int(len(all_items)),
    }

    metadata = {
        "purpose":
            "PUMML-style v5 from-scratch dataset for PUMML/PUPPI/PileFlow comparison",

        "published_pumml_setup": {
            "sqrt_s_TeV": 13.0,
            "scalar_mass_GeV": 500.0,
            "scalar_decay": "light qqbar",
            "pythia_version_target": "8.183",
            "pythia_tune_target": "default pp tune",
            "pileup_process": "soft QCD; reproduction uses PYTHIA SoftQCD:nonDiffractive to match released pileup activity more closely",
            "npu_distribution": "Poisson",
            "mean_npu": args.mean_npu,
            "npu_range": [0, 180],
            "fastjet_version_target": "3.1.3",
            "jet_algorithm": "anti-kt",
            "R": 0.4,
            "parton_pt_cut_GeV": 95.0,
            "jet_pt_cut_GeV": 100.0,
            "jet_abs_eta_max": 2.5,
            "max_selected_jets_per_event": 2,
            "discard_final_state": [
                "muons",
                "neutrinos",
            ],
            "massless_particles": True,
            "charged_reco_pt_min_GeV": 0.5,
            "image_size_eta_phi": [0.9, 0.9],
            "image_center":
                "charged-LV pT-weighted centroid of selected R=0.4 anti-kt jet",
            "image_canvas":
                "0.9x0.9 square around the charged-LV centroid; input particle support is the matched reconstructed anti-kt R=0.4 jet",
            "charged_pixel_size": [0.025, 0.025],
            "neutral_pixel_size": [0.1, 0.1],
            "neutral_upsampling":
                "repeat each coarse pixel into a 4x4 block and divide by 16",
            "input_channels": [
                "total neutral-like",
                "charged pileup",
                "charged leading vertex",
            ],
            "target": "neutral leading vertex",
            "n_jet_images": args.n_jets,
            "paper_split": "90/10 image-level",
            "puppi": {
                "R0": 0.3,
                "Rmin": 0.02,
                "wcut": 0.1,
                "ptcut_GeV": "0.1 + 0.007*NPU",
                "processing":
                    "particle level before detector discretization",
            },
        },

        "reproduction_assumptions_not_fixed_by_paper": {
            "scalar_production_proxy":
                "HiggsSM:ffbar2H",
            "scalar_proxy_validation":
                "selected because released PUMML mphi=500 leading-jet pT and dijet-mass spectra are closely reproduced",
            "parton_cut_realization":
                "require both generated scalar daughter quarks to have pT > 95 GeV",
            "modern_pythia_tune_compatibility": {
                "Tune:pp": 5,
                "Tune:ee": 3,
                "TimeShower:weightGluonToQuark": 1,
                "description":
                    "explicit Tune 4C compatibility settings for modern Pythia",
            },
            "scalar_width_GeV": 0.001,
            "scalar_mass_window_GeV": [
                M_SCALAR - 0.01,
                M_SCALAR + 0.01,
            ],
            "npu_0_180_handling":
                "Poisson rejection sampling outside [0,180]",
            "puppi_implementation":
                "v7: particle-level PUPPI weights on full event, corrected particles reclustered independently before detector discretization; separate charged/neutral PUPPI images stored; exact historical source revision not claimed",
            "seed": args.seed,
        },

        "important_processing_choices_v5": {
            "puppi":
                "computed on complete LV+PU event before reco-jet/image restriction",
            "primary_image_membership":
                "input channels use constituents of the matched reconstructed full-event anti-kt R=0.4 jet",
            "truth_target":
                "neutral-like constituents of the selected LV truth jet",
            "image_canvas":
                "0.9x0.9 square around the charged-LV truth-jet centroid",
            "image_morphology_reason":
                "primary reco-jet support is required to reproduce the anti-kt-like charged-PU morphology seen in the released PUMML sample",
            "square_debug_saved":
                bool(args.save_square_debug),
            "square_debug_role":
                "optional diagnostic only; never used as primary X/y arrays",
            "no_energy_rescaling":
                "no ad-hoc multiplicative calibration is applied to force agreement with released channel integrals",
        },

        "extra_metadata": {
            "event_id":
                "pairs jets from the same hard event",
            "jet_rank":
                "0=leading selected LV jet, 1=subleading selected LV jet",
            "image_center_eta_phi":
                "global coordinates of charged-LV centroid",
            "event_grouped_split":
                "additional leakage-free 90/10 split, not claimed as original paper split",
            "scalar_daughter_truth":
                "generator-level scalar daughter quarks",
        },

        "generation_stats": generation_stats,
    }

    save = (np.savez_compressed if args.compressed else np.savez)

    save(
        args.output,

        X=X,
        y_neutral_lv_9=y9,
        y_neutral_lv_36=y36,

        no_pileup_36=no_pileup36,
        with_pileup_36=with_pileup36,
        puppi_36=puppi36,
        puppi_charged_36=puppi_charged36,
        puppi_neutral_9=puppi_neutral9,
        puppi_neutral_36=puppi_neutral36,

        npu=npu,

        event_id=event_id,
        jet_rank=jet_rank,
        worker_id=worker_id,

        image_center_eta=center_eta,
        image_center_phi=center_phi,

        n_particles_reco_jet=n_particles_reco_jet,
        n_particles_truth_jet=n_particles_truth_jet,
        n_particles_crop_reco=n_particles_crop_reco,
        n_particles_crop_truth=n_particles_crop_truth,

        truth_jet_pt=truth_pt,
        truth_jet_eta=truth_eta,
        truth_jet_phi=truth_phi,
        truth_jet_mass=truth_mass,

        reco_jet_pt=reco_pt,
        reco_jet_eta=reco_eta,
        reco_jet_phi=reco_phi,
        reco_jet_mass=reco_mass,
        reco_truth_deltaR=reco_truth_dr,

        puppi_jet_pt=puppi_jet_pt,
        puppi_jet_eta=puppi_jet_eta,
        puppi_jet_phi=puppi_jet_phi,
        puppi_jet_mass=puppi_jet_mass,
        puppi_truth_deltaR=puppi_truth_dr,

        scalar_daughter1_id=d1_id,
        scalar_daughter1_pt=d1_pt,
        scalar_daughter1_eta=d1_eta,
        scalar_daughter1_phi=d1_phi,

        scalar_daughter2_id=d2_id,
        scalar_daughter2_pt=d2_pt,
        scalar_daughter2_eta=d2_eta,
        scalar_daughter2_phi=d2_phi,

        train_idx=train_idx,
        test_idx=test_idx,

        event_train_idx=event_train_idx,
        event_test_idx=event_test_idx,

        channel_names=np.asarray([
            "total_neutral",
            "charged_pileup",
            "charged_leading_vertex",
        ]),

        metadata_json=np.asarray(
            json.dumps(
                metadata,
                sort_keys=True,
            )
        ),
    )
    if args.save_square_debug:
        debug_path = (
            args.output[:-4] + "_square_debug.npz"
            if args.output.endswith(".npz")
            else args.output + "_square_debug.npz"
        )
        save(
            debug_path,
            X_square_debug=X_square_debug,
            y_square_debug_36=y_square_debug_36,
            event_id=event_id,
            jet_rank=jet_rank,
            image_center_eta=center_eta,
            image_center_phi=center_phi,
        )
        print(f"Saved square-debug companion: {debug_path}")


    unique_events, counts = np.unique(event_id, return_counts=True)
    n_twojet_events = int(np.sum(counts >= 2))

    print()
    print("=" * 88)
    print(f"Saved: {args.output}")
    print("=" * 88)

    print(f"X                       : {X.shape}")
    print(f"y_neutral_lv_9          : {y9.shape}")
    print(f"no_pileup_36            : {no_pileup36.shape}")
    print(f"with_pileup_36          : {with_pileup36.shape}")
    print(f"puppi_36                : {puppi36.shape}")

    print()
    print(f"Unique hard events      : {len(unique_events)}")
    print(f"Two-image events        : {n_twojet_events}")
    print(f"Leading images          : {int(np.sum(jet_rank == 0))}")
    print(f"Subleading images       : {int(np.sum(jet_rank == 1))}")

    print()
    print("Image split")
    print(f"  train/test            : {len(train_idx)} / {len(test_idx)}")

    print()
    print("Event-grouped split")
    print(
        f"  train/test images     : "
        f"{len(event_train_idx)} / {len(event_test_idx)}"
    )

    print()
    print(f"Sample mean NPU         : {float(npu.mean()):.4f}")
    print(f"NPU min/max             : {int(npu.min())} / {int(npu.max())}")

    print()
    print("Hard-spectrum quick check")
    lead = truth_pt[jet_rank == 0]

    if len(lead):
        print(
            "  leading pT median/95/99: "
            f"{np.median(lead):.2f} / "
            f"{np.quantile(lead, 0.95):.2f} / "
            f"{np.quantile(lead, 0.99):.2f} GeV"
        )
        print(
            "  official PUMML         : "
            "210.62 / 286.31 / 344.66 GeV"
        )

    print()
    print("Generation stats")
    for k, v in generation_stats.items():
        print(f"  {k:36s}: {v}")

    print()
    print("Next validation:")
    print("  1) confirm charged-PU morphology has anti-kt R=0.4-like support")
    print("  2) report integrated channel differences separately from morphology")
    print("  3) diagnose remaining pileup-density normalization mismatch without ad-hoc scaling")
    print("  4) recheck hard-spectrum tails and PUPPI observables")
    print("=" * 88)


if __name__ == "__main__":
    main()