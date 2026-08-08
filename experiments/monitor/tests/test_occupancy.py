"""Arm B primitives. The drift statistic decides Gate 1's second verdict, so
its false-alarm calibration is tested directly rather than assumed."""

import numpy as np
import pytest

from experiments.monitor import occupancy as oc


def _dirs(n, d, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestAssign:
    def test_matches_bruteforce_argmin(self):
        ref, q = _dirs(37, 16, 1), _dirs(200, 16, 2)
        want = (1.0 - q @ ref.T).argmin(1)
        assert np.array_equal(oc.assign(q, ref), want)

    def test_centering_is_applied_when_center_given(self):
        ref = _dirs(11, 8, 3)
        raw = np.random.default_rng(4).normal(size=(50, 8)).astype(np.float32)
        c = np.full(8, 0.3, dtype=np.float32)
        want = (1.0 - oc._unit(raw, c) @ ref.T).argmin(1)
        assert np.array_equal(oc.assign(raw, ref, c), want)
        # and that centring is not a no-op on this data
        assert not np.array_equal(oc.assign(raw, ref, c),
                                  oc.assign(oc._unit(raw, np.zeros(8, np.float32)), ref))

    def test_every_point_is_assigned_no_minus_one(self):
        """`Dictionary.assign` is pure argmin with no threshold escape; ours
        must match that, or 'outside every cell' silently changes meaning."""
        a = oc.assign(_dirs(100, 8, 5), _dirs(3, 8, 6))
        assert a.min() >= 0 and a.max() < 3


class TestStatistics:
    def test_surprisal_ranks_rare_cells_above_common_ones(self):
        counts = np.array([1000.0, 10.0, 1.0])
        logp = oc.log_ref_prob(counts)
        s = oc.surprisal(np.array([0, 1, 2]), logp)
        assert s[0] < s[1] < s[2]

    def test_unseen_cell_is_finite_under_smoothing(self):
        logp = oc.log_ref_prob(np.array([100.0, 0.0]))
        assert np.isfinite(oc.surprisal(np.array([1]), logp)).all()

    def test_tv_is_zero_for_identical_and_one_for_disjoint(self):
        a = np.array([5.0, 5.0, 0.0, 0.0])
        assert oc.tv_distance(a, a) == pytest.approx(0.0)
        assert oc.tv_distance(a, np.array([0.0, 0.0, 5.0, 5.0])) == pytest.approx(1.0)

    def test_tv_is_scale_invariant(self):
        a, b = np.array([3.0, 1.0]), np.array([30.0, 10.0])
        assert oc.tv_distance(a, b) == pytest.approx(0.0)


class TestBalance:
    def test_uniform_histogram_is_flat_by_every_measure(self):
        b = oc.balance(np.full(64, 10.0))
        assert b["entropy_ratio"] == pytest.approx(1.0)
        assert b["gini"] == pytest.approx(0.0, abs=1e-9)
        assert b["occupied_frac"] == 1.0

    def test_concentrated_histogram_scores_high_gini(self):
        c = np.zeros(64); c[0] = 1000.0
        b = oc.balance(c)
        assert b["gini"] > 0.95
        assert b["top1_share"] == pytest.approx(1.0)
        assert b["entropy_ratio"] == pytest.approx(0.0)

    def test_gini_orders_two_real_histograms(self):
        even = oc.balance(np.full(32, 8.0))["gini"]
        skew = oc.balance(np.array([200.0] + [1.0] * 31))["gini"]
        assert skew > even


class TestPowerCurve:
    """The headline of Arm B is a power number, so the null must actually
    sit at the nominal false-alarm rate."""

    def test_no_contamination_gives_power_at_the_nominal_rate(self):
        rng = np.random.default_rng(0)
        clean = rng.integers(0, 20, size=5000)
        ref = oc.occupancy(clean, 20)
        r = oc.power_curve(clean, rng.integers(0, 20, size=5000), ref,
                           window=100, eps=0.0, n_windows=600, alpha_fpr=0.01)
        for stat in ("power_surprisal", "power_tv"):
            assert r[stat] < 0.06, (stat, r[stat])

    def test_contaminant_identical_to_clean_has_no_power(self):
        rng = np.random.default_rng(1)
        cells = rng.integers(0, 20, size=5000)
        ref = oc.occupancy(cells, 20)
        r = oc.power_curve(cells, cells.copy(), ref, window=200, eps=0.5,
                           n_windows=600, alpha_fpr=0.01)
        assert r["power_surprisal"] < 0.06
        assert r["power_tv"] < 0.10

    def test_disjoint_contaminant_is_detected(self):
        rng = np.random.default_rng(2)
        clean = rng.integers(0, 10, size=5000)
        dirty = rng.integers(10, 20, size=5000)
        ref = oc.occupancy(clean, 20)
        r = oc.power_curve(clean, dirty, ref, window=100, eps=0.25,
                           n_windows=400, alpha_fpr=0.01)
        assert r["power_surprisal"] > 0.9
        assert r["power_tv"] > 0.9

    def test_power_increases_with_contamination(self):
        rng = np.random.default_rng(3)
        clean = rng.integers(0, 10, size=5000)
        dirty = rng.integers(8, 20, size=5000)
        ref = oc.occupancy(clean, 20)
        got = [oc.power_curve(clean, dirty, ref, window=100, eps=e,
                              n_windows=400, seed=7)["power_surprisal"]
               for e in (0.0, 0.05, 0.25, 1.0)]
        assert got == sorted(got), got

    def test_window_size_is_respected(self):
        rng = np.random.default_rng(4)
        clean = rng.integers(0, 5, size=100)
        ref = oc.occupancy(clean, 5)
        # eps=0.3 of a 10-window is 3 dirty + 7 clean; must not crash or drift
        r = oc.power_curve(clean, rng.integers(0, 5, size=100), ref,
                           window=10, eps=0.3, n_windows=200)
        assert 0.0 <= r["power_surprisal"] <= 1.0


class TestScalarPowerCurve:
    def test_identical_distributions_sit_at_the_nominal_rate(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=4000)
        r = oc.scalar_power_curve(a, rng.normal(size=4000), window=100,
                                  eps=0.5, n_windows=600, alpha_fpr=0.01)
        assert r["power_mean"] < 0.06

    def test_shifted_contaminant_is_detected(self):
        rng = np.random.default_rng(1)
        r = oc.scalar_power_curve(rng.normal(size=4000),
                                  rng.normal(size=4000) + 5.0,
                                  window=100, eps=0.25, n_windows=400)
        assert r["power_mean"] > 0.95

    def test_averaging_recovers_a_signal_too_weak_per_request(self):
        """The reason this baseline exists: a shift of 0.2 sd is nearly
        invisible in one request and obvious in the mean of 500."""
        rng = np.random.default_rng(2)
        clean, dirty = rng.normal(size=8000), rng.normal(size=8000) + 0.2
        small = oc.scalar_power_curve(clean, dirty, window=10, eps=1.0,
                                      n_windows=600, seed=3)["power_mean"]
        big = oc.scalar_power_curve(clean, dirty, window=500, eps=1.0,
                                    n_windows=600, seed=3)["power_mean"]
        assert big > small + 0.3

    def test_a_lower_scoring_contaminant_is_not_detected_one_sided(self):
        """One-sided by construction: 'more OOD' must mean a larger score."""
        rng = np.random.default_rng(4)
        r = oc.scalar_power_curve(rng.normal(size=4000),
                                  rng.normal(size=4000) - 5.0,
                                  window=100, eps=1.0, n_windows=400)
        assert r["power_mean"] == pytest.approx(0.0, abs=1e-9)


class TestLeaderCluster:
    def test_all_points_inside_theta_collapse_to_one_exemplar(self):
        x = np.tile(_dirs(1, 12, 0), (200, 1)) + 1e-4
        e = oc.leader_cluster(x, 0.5, np.zeros(12, np.float32),
                              np.arange(200), chunk=32)
        assert len(e) == 1

    def test_orthogonal_points_each_become_an_exemplar(self):
        x = np.eye(9, dtype=np.float32)
        e = oc.leader_cluster(x, 0.5, np.zeros(9, np.float32),
                              np.arange(9), chunk=4)
        assert len(e) == 9

    def test_blocked_path_matches_a_naive_sequential_reference(self):
        """The block optimisation must be exact, not approximate."""
        x, c, th = _dirs(300, 6, 11), np.zeros(6, np.float32), 0.35
        order = np.random.default_rng(12).permutation(300)

        ref = []
        for i in order:
            v = x[i]
            if not ref or (1.0 - np.array(ref) @ v).min() > th:
                ref.append(v)
        got = oc.leader_cluster(x, th, c, order, chunk=16)
        assert len(got) == len(ref)
        assert np.allclose(got, np.array(ref), atol=1e-5)

    def test_chunk_size_does_not_change_the_result(self):
        x, c, th = _dirs(400, 6, 13), np.zeros(6, np.float32), 0.4
        order = np.arange(400)
        a = oc.leader_cluster(x, th, c, order, chunk=7)
        b = oc.leader_cluster(x, th, c, order, chunk=256)
        assert len(a) == len(b) and np.allclose(a, b, atol=1e-5)

    def test_max_exemplars_caps_growth(self):
        e = oc.leader_cluster(_dirs(500, 32, 14), 0.05,
                              np.zeros(32, np.float32), np.arange(500),
                              chunk=64, max_exemplars=17)
        assert len(e) == 17


class TestExemplarOverlap:
    def test_identical_dictionaries_fully_match(self):
        a = _dirs(20, 10, 21)
        r = oc.exemplar_overlap(a, a.copy(), 0.05)
        assert r["frac_a_matched"] == 1.0 and r["frac_b_matched"] == 1.0
        assert r["mean_nn_dist_a_to_b"] == pytest.approx(0.0, abs=1e-5)

    def test_unrelated_dictionaries_do_not_match_at_a_tight_threshold(self):
        r = oc.exemplar_overlap(_dirs(30, 64, 22), _dirs(30, 64, 23), 0.05)
        assert r["frac_a_matched"] == 0.0

    def test_asymmetry_is_reported_both_ways(self):
        """A small dictionary can be covered by a large one without the
        reverse holding; collapsing to one number would hide that."""
        a = _dirs(5, 10, 24)
        b = np.concatenate([a, _dirs(50, 10, 25)])
        r = oc.exemplar_overlap(a, b, 0.05)
        assert r["frac_a_matched"] == 1.0
        assert r["frac_b_matched"] < 0.5
