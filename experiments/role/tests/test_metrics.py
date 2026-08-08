"""Metrics against synthetic assignments with analytically known answers.

No model, no tokenizer, no network. If a metric is wrong here it is wrong on the
GPU too, and there it will be wrong quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.role import metrics as M


def _unit_rows(rng, n, d):
    E = rng.normal(size=(n, d))
    return E / np.linalg.norm(E, axis=1, keepdims=True)


# --- flip rate ------------------------------------------------------------

def test_flip_rate_zero_when_conditions_identical():
    A = np.tile(np.arange(20)[None, None, :], (5, 3, 1))
    assert M.flip_rate(A, 0, 1) == 0.0
    assert np.allclose(M.flip_rate_matrix(A), 0.0)


def test_flip_rate_one_when_all_differ():
    A = np.zeros((4, 2, 10), dtype=int)
    A[:, 1, :] = 1
    assert M.flip_rate(A, 0, 1) == 1.0


def test_flip_rate_known_fraction():
    A = np.zeros((1, 2, 10), dtype=int)
    A[0, 1, :4] = 1                      # 4 of 10 positions move
    assert M.flip_rate(A, 0, 1) == pytest.approx(0.4)


# --- displacement coherence ----------------------------------------------

def test_coherence_is_one_for_identical_displacements():
    d = np.tile(np.array([[1.0, 0.0, 0.0]]), (50, 1))
    assert M._coherence(d) == pytest.approx(1.0)


def test_coherence_is_zero_for_antipodal_displacements():
    d = np.concatenate([np.eye(3)[:1]] * 10 + [-np.eye(3)[:1]] * 10)
    assert M._coherence(d) == pytest.approx(0.0, abs=1e-12)


def test_coherence_decays_with_random_directions():
    rng = np.random.default_rng(0)
    d = _unit_rows(rng, 4000, 64)
    # E[coherence] ~ sqrt(pi/(4n)) ish; the point is only that it is small.
    assert M._coherence(d) < 0.05


def test_displacement_recovers_a_planted_direction():
    """Every region pair displaced by the same vector => coherence ~1, z >> 0."""
    rng = np.random.default_rng(1)
    K, d = 200, 64
    E = _unit_rows(rng, K, d)
    # Pair region r with region r+100 and make those pairs share a direction by
    # construction: rebuild the second half as first half + delta, renormalized.
    delta = np.zeros(d)
    delta[0] = 0.6
    E[100:] = E[:100] + delta
    E[100:] /= np.linalg.norm(E[100:], axis=1, keepdims=True)

    src = rng.integers(0, 100, size=(10, 1, 50))
    A = np.concatenate([src, src + 100], axis=1)

    res = M.displacement(A, E, 0, 1, names=("user", "assistant"), seed=0)
    assert res.flip_rate == 1.0
    assert res.n_flipped == 500
    assert res.coherence > 0.9
    assert res.coherence_z > 3.0
    assert M.cosine(res.mean_direction, delta) > 0.9


def test_displacement_null_matches_observed_when_pairing_is_random():
    """Random src/dst pairs: observed coherence should sit inside the null."""
    rng = np.random.default_rng(2)
    K, d = 300, 64
    E = _unit_rows(rng, K, d)
    A = rng.integers(0, K, size=(20, 2, 60))
    res = M.displacement(A, E, 0, 1, seed=0)
    assert res.coherence < 0.15
    assert abs(res.coherence_z) < 6.0


def test_displacement_handles_no_movement():
    A = np.tile(np.arange(10)[None, None, :], (3, 2, 1))
    E = _unit_rows(np.random.default_rng(3), 10, 8)
    res = M.displacement(A, E, 0, 1)
    assert res.n_flipped == 0
    assert np.isnan(res.coherence)


# --- transition matrix ----------------------------------------------------

def test_transition_matrix_counts_and_sparsity():
    A = np.zeros((1, 2, 6), dtype=int)
    A[0, 0, :] = [0, 0, 0, 1, 1, 2]
    A[0, 1, :] = [3, 3, 4, 5, 5, 5]
    T = M.transition_matrix(A, 0, 1, n_regions=6)
    assert T.sum() == 6
    assert T[0, 3] == 2 and T[0, 4] == 1
    assert T[1, 5] == 2 and T[2, 5] == 1
    s = M.coupling_sparsity(T)
    assert s["n_live_source_regions"] == 3
    # regions 1 and 2 are deterministic; region 0 splits 2/1.
    assert s["frac_deterministic"] == pytest.approx(2 / 3)
    assert s["mean_top1_mass"] == pytest.approx((2 / 3 + 1 + 1) / 3)


# --- occupancy / JS / polarity -------------------------------------------

def test_occupancy_sums_to_one_and_smooths():
    A = np.zeros((2, 1, 5), dtype=int)      # every token in region 0
    p = M.occupancy(A, 0, n_regions=4, prior=0.5)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all(), "Jeffreys prior must keep empty regions nonzero"
    assert p.argmax() == 0


def test_js_divergence_bounds():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert M.js_divergence(p, p) == pytest.approx(0.0)
    assert M.js_divergence(p, q) == pytest.approx(1.0)


def test_js_null_is_near_zero_for_identical_conditions():
    rng = np.random.default_rng(4)
    base = rng.integers(0, 30, size=(20, 1, 40))
    A = np.concatenate([base, base], axis=1)
    p0 = M.occupancy(A, 0, 30)
    p1 = M.occupancy(A, 1, 30)
    assert M.js_divergence(p0, p1) == pytest.approx(0.0, abs=1e-12)
    mean, _ = M.js_divergence_null(A, 0, 1, 30, n_null=10, seed=0)
    assert mean == pytest.approx(0.0, abs=1e-12)


def test_polarity_sign_and_member_floor():
    p_a = np.array([0.8, 0.1, 0.1])
    p_b = np.array([0.1, 0.8, 0.1])
    counts = np.array([100, 100, 1])
    lam = M.polarity(p_a, p_b, min_count=counts, min_members=5)
    assert lam[0] > 0 and lam[1] < 0
    assert np.isnan(lam[2]), "under-populated region must be excluded"


def test_polarity_rank_stability_extremes():
    lam = np.array([5.0, -4.0, 0.1, 0.0, 3.0])
    assert M.polarity_rank_stability([lam, lam.copy()], top_m=3) == 1.0
    other = np.array([0.0, 0.1, 5.0, -4.0, 0.05])
    assert M.polarity_rank_stability([lam, other], top_m=2) == 0.0


# --- axis / PCA -----------------------------------------------------------

def test_role_axis_points_along_the_polarized_region():
    E = np.eye(4)
    lam = np.array([2.0, 0.0, 0.0, -2.0])
    a = M.role_axis(lam, E)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)
    assert M.cosine(a, np.array([1.0, 0, 0, -1.0])) > 0.999


def test_role_axis_ignores_nan_polarity():
    E = np.eye(3)
    a = M.role_axis(np.array([1.0, np.nan, 0.0]), E)
    assert np.isfinite(a).all()
    assert M.cosine(a, np.array([1.0, 0.0, 0.0])) > 0.999


def test_pca_recovers_a_planted_plane():
    rng = np.random.default_rng(5)
    d = 32
    u, v = np.zeros(d), np.zeros(d)
    u[0], v[1] = 1.0, 1.0
    coefs = rng.normal(size=(500, 2)) * np.array([5.0, 2.0])
    X = coefs @ np.stack([u, v]) + 0.01 * rng.normal(size=(500, d))
    comps, ratio = M.pca(X, n_components=3)
    assert abs(M.cosine(comps[0], u)) > 0.99
    assert abs(M.cosine(comps[1], v)) > 0.99
    assert ratio[0] > ratio[1] > ratio[2]
    assert ratio[:2].sum() > 0.99


# --- mutual information ---------------------------------------------------

def test_nmi_is_one_when_regions_determine_labels():
    regions = np.repeat(np.arange(10), 20)
    labels = regions % 5
    assert M.normalized_mi(regions, labels) == pytest.approx(1.0)


def test_nmi_is_zero_for_independent_labels():
    rng = np.random.default_rng(6)
    regions = rng.integers(0, 4, size=20000)
    labels = rng.integers(0, 3, size=20000)
    assert M.normalized_mi(regions, labels) < 0.01


def test_nmi_null_is_positive_and_bounds_a_weak_signal():
    """The null must be nonzero — that is the whole reason it exists."""
    rng = np.random.default_rng(7)
    # Many regions, few tokens each: NMI is badly upward-biased here.
    regions = np.arange(600)
    labels = rng.integers(0, 3, size=600)
    null_mean, null_sd = M.normalized_mi_null(regions, labels, n_null=10, seed=0)
    assert null_mean > 0.5, "unique-region-per-token must saturate the null"
    observed = M.normalized_mi(regions, labels)
    assert abs(observed - null_mean) < max(3 * null_sd, 0.05)


def test_mutual_information_entropies():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    mi, ha, hb = M.mutual_information(a, b)
    assert ha == pytest.approx(1.0) and hb == pytest.approx(1.0)
    assert mi == pytest.approx(0.0, abs=1e-12)


# --- region-table classifier ---------------------------------------------

def test_region_table_classifier_perfect_and_chance():
    # Perfect: region determines role.
    r_tr = np.repeat(np.arange(6), 50)
    y_tr = r_tr % 3
    res = M.region_table_classifier(r_tr, y_tr, r_tr, y_tr, 6, 3)
    assert res.accuracy == pytest.approx(1.0)
    assert res.macro_auroc == pytest.approx(1.0)

    # Chance: one region, three balanced classes.
    rng = np.random.default_rng(8)
    r = np.zeros(3000, dtype=int)
    y = rng.integers(0, 3, size=3000)
    res = M.region_table_classifier(r, y, r, y, 1, 3)
    assert 0.28 < res.accuracy < 0.40
    assert res.macro_auroc == pytest.approx(0.5, abs=0.02)


def test_region_table_classifier_unseen_test_region_falls_back_to_prior():
    """A test token in a region never seen in training must not crash."""
    r_tr = np.zeros(10, dtype=int)
    y_tr = np.zeros(10, dtype=int)
    r_te = np.array([1, 1])
    y_te = np.array([0, 1])
    res = M.region_table_classifier(r_tr, y_tr, r_te, y_te, 2, 2)
    assert np.isfinite(res.accuracy)


# --- paired role NMI and its null ----------------------------------------

def test_paired_role_nmi_zero_when_no_flips():
    """No flips => the joint factorizes exactly => MI is exactly 0, no bias."""
    base = np.repeat(np.arange(40)[None, :], 5, axis=0)     # (5 docs, 40 pos)
    A = np.stack([base] * 6, axis=1)                        # (5, 6, 40)
    assert M.paired_role_nmi(A) == pytest.approx(0.0, abs=1e-12)
    null_mean, _ = M.normalized_mi_null_paired(A, n_null=5, seed=0)
    assert null_mean == pytest.approx(0.0, abs=1e-12), (
        "with no flips the within-group null must also be exactly 0 — this is "
        "what the global-permutation null gets wrong"
    )


def test_paired_role_nmi_one_when_condition_determines_region():
    """Region == condition => NMI 1, and the within-group null collapses to ~0."""
    D, Cn, J = 6, 6, 20
    A = np.tile(np.arange(Cn)[None, :, None], (D, 1, J))
    assert M.paired_role_nmi(A) == pytest.approx(1.0)
    null_mean, _ = M.normalized_mi_null_paired(A, n_null=10, seed=0)
    assert null_mean < 0.05


def test_paired_null_is_not_inflated_by_region_count():
    """The failure mode from the dry run, pinned.

    76 regions, 6 conditions, ~1150 samples: the global-permutation null was
    0.105 while the true value was ~0. The within-group null must stay near 0.
    """
    rng = np.random.default_rng(0)
    D, Cn, J = 8, 6, 24
    base = rng.integers(0, 76, size=(D, J))
    A = np.stack([base] * Cn, axis=1)
    # A few role-independent flips, matching the dry run's ~5-8%.
    flip = rng.random((D, Cn, J)) < 0.07
    A = np.where(flip, rng.integers(0, 76, size=A.shape), A)

    observed = M.paired_role_nmi(A)
    paired_mean, paired_sd = M.normalized_mi_null_paired(A, n_null=10, seed=1)

    regions = A.transpose(0, 2, 1).reshape(-1, Cn).ravel()
    labels = np.tile(np.arange(Cn), D * J)
    global_mean, _ = M.normalized_mi_null(regions, labels, n_null=10, seed=1)

    assert global_mean > 5 * max(paired_mean, 1e-6), (
        "global permutation should be visibly inflated relative to the paired "
        f"null (global={global_mean:.4f}, paired={paired_mean:.4f})"
    )
    # Role-independent flips: observed must sit within noise of the paired null.
    assert abs(observed - paired_mean) < max(4 * paired_sd, 0.02)


def test_paired_role_nmi_detects_role_dependent_flips():
    """Flips that depend on the condition must exceed the paired null."""
    rng = np.random.default_rng(2)
    D, Cn, J = 20, 6, 24
    base = rng.integers(0, 60, size=(D, J))
    A = np.stack([base] * Cn, axis=1)
    # Condition 1 always moves its token to a fixed condition-specific region.
    A[:, 1, :] = 90
    A[:, 2, :] = 91
    observed = M.paired_role_nmi(A)
    null_mean, null_sd = M.normalized_mi_null_paired(A, n_null=10, seed=3)
    assert observed > null_mean + 5 * max(null_sd, 1e-6)
