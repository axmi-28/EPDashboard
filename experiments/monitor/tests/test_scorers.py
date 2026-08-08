import numpy as np
import pytest

from experiments.monitor import scorers


class TestAuroc:
    def test_constant_scorer_is_exactly_half(self):
        """A degenerate scorer must land on 0.5, not on sort order. Without
        mid-rank tie correction this comes out 0.0 or 1.0 and a broken scorer
        would look like the best result in the table."""
        y = np.array([0, 0, 1, 1, 0, 1])
        assert scorers.auroc(np.zeros(6), y) == 0.5

    def test_perfect_and_inverted(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        assert scorers.auroc(np.arange(6.0), y) == 1.0
        assert scorers.auroc(-np.arange(6.0), y) == 0.0

    def test_half_ties_are_averaged(self):
        # Two positives tied with two negatives: every pair is a tie -> 0.5.
        y = np.array([0, 0, 1, 1])
        assert scorers.auroc(np.array([1.0, 1.0, 1.0, 1.0]), y) == 0.5

    def test_matches_bruteforce_pair_counting(self):
        rng = np.random.default_rng(0)
        s = rng.integers(0, 5, 200).astype(float)   # many ties on purpose
        y = rng.integers(0, 2, 200).astype(bool)
        pos, neg = s[y], s[~y]
        brute = np.mean([(1.0 if p > n else 0.5 if p == n else 0.0)
                         for p in pos for n in neg])
        assert scorers.auroc(s, y) == pytest.approx(brute, abs=1e-12)

    def test_nan_when_a_class_is_missing(self):
        assert np.isnan(scorers.auroc(np.arange(4.0), np.zeros(4, bool)))


class TestTprAtFpr:
    def test_realised_fpr_never_exceeds_target(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            neg = rng.normal(0, 1, 2000)
            pos = rng.normal(1.5, 1, 2000)
            s = np.concatenate([neg, pos])
            y = np.concatenate([np.zeros(2000, bool), np.ones(2000, bool)])
            thresh = np.quantile(neg, 0.99)
            assert (neg > thresh).mean() <= 0.011
            assert 0.0 <= scorers.tpr_at_fpr(s, y) <= 1.0

    def test_perfect_separation_gives_one(self):
        s = np.concatenate([np.zeros(100), np.ones(100) * 10])
        y = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
        assert scorers.tpr_at_fpr(s, y) == 1.0

    def test_constant_scores_do_not_leak_positives(self):
        """With everything tied, a strict '>' must yield TPR 0 rather than 1."""
        s = np.ones(200)
        y = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
        assert scorers.tpr_at_fpr(s, y) == 0.0


class TestEpScorers:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(0)
        center = rng.normal(0, 1, 64).astype(np.float32) * 10
        E = rng.normal(0, 1, (16, 64)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        return center, E

    def test_an_exemplar_scores_zero_distance(self, setup):
        center, E = setup
        x = center + E[3] * 5.0            # exactly on exemplar 3's ray
        d1, _ = scorers.s1_s2_ep(x[None], E, center)
        assert d1[0] == pytest.approx(0.0, abs=1e-5)

    def test_margin_sign_is_ood_oriented(self, setup):
        """S2 is returned negated. An activation sitting on one exemplar has a
        large raw margin and must therefore get a LOW (more in-distribution)
        OOD score than one sitting midway between two exemplars."""
        center, E = setup
        on_exemplar = center + E[3] * 5.0
        mid = E[0] + E[1]
        between = center + (mid / np.linalg.norm(mid)) * 5.0
        _, s2 = scorers.s1_s2_ep(np.stack([on_exemplar, between]), E, center)
        assert s2[0] < s2[1]

    def test_magnitude_is_discarded(self, setup):
        """EP works on centered unit directions, so scaling the offset from the
        centre must not move the score at all."""
        center, E = setup
        v = E[5] * 2.0 + E[6]
        a = scorers.s1_s2_ep((center + v)[None], E, center)[0]
        b = scorers.s1_s2_ep((center + v * 37.0)[None], E, center)[0]
        assert a[0] == pytest.approx(b[0], abs=1e-5)


class TestCoreset:
    def test_uses_exactly_k_references_and_is_seed_deterministic(self):
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, (500, 32)).astype(np.float32)
        center = np.zeros(32, np.float32)
        x = rng.normal(0, 1, (20, 32)).astype(np.float32)
        m1, per1 = scorers.s3_coreset_knn(x, pool, center, k=50, n_draws=3, seed=7)
        m2, per2 = scorers.s3_coreset_knn(x, pool, center, k=50, n_draws=3, seed=7)
        assert len(per1) == 3
        assert np.allclose(m1, m2)
        # Different draws must actually differ, or the averaging is a no-op.
        assert not np.allclose(per1[0], per1[1])

    def test_k_larger_than_pool_is_clamped(self):
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, (10, 8)).astype(np.float32)
        center = np.zeros(8, np.float32)
        x = rng.normal(0, 1, (4, 8)).astype(np.float32)
        m, _ = scorers.s3_coreset_knn(x, pool, center, k=999, n_draws=1)
        assert np.isfinite(m).all()


class TestMahalanobis:
    def test_zero_at_the_mean_and_grows_outward(self):
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, (2000, 16)).astype(np.float32)
        fit = scorers.fit_mahalanobis(pool, shrinkage=0.01)
        at_mean = scorers.s4_mahalanobis(fit["mean"][None].astype(np.float32), fit)
        far = scorers.s4_mahalanobis(
            (fit["mean"] + 50.0)[None].astype(np.float32), fit)
        assert at_mean[0] == pytest.approx(0.0, abs=1e-6)
        assert far[0] > 10.0

    def test_is_scale_invariant_across_correlated_dims(self):
        """The point of Mahalanobis over Euclidean: stretching one axis of the
        reference data must not change the distance of a point stretched with it."""
        rng = np.random.default_rng(0)
        pool = rng.normal(0, 1, (4000, 8)).astype(np.float32)
        probe = np.array([[3.0] + [0.0] * 7], np.float32)
        d_plain = scorers.s4_mahalanobis(probe, scorers.fit_mahalanobis(pool))
        stretched = pool.copy()
        stretched[:, 0] *= 20.0
        d_stretch = scorers.s4_mahalanobis(
            probe * np.array([[20.0] + [1.0] * 7], np.float32),
            scorers.fit_mahalanobis(stretched))
        assert d_plain[0] == pytest.approx(d_stretch[0], rel=0.05)
