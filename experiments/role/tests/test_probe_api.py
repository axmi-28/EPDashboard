"""sklearn API surface the probe depends on.

These are version-compatibility guards, not logic tests. `multi_class` was
deprecated in sklearn 1.5 and **removed in 1.9**; passing it raises TypeError.
That break was invisible until the probe actually fitted — which on the real
corpus is ten minutes of GPU harvesting after the point of no return. Fitting a
60-row probe here costs milliseconds and fails in the same place.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.role.probe import _fit_probe, _macro_auroc


def _toy(n=600, d=8, n_classes=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    x = rng.normal(size=(n, d)) + y[:, None] * 0.9   # learnable
    return x, y


def test_fit_probe_runs_and_is_multinomial():
    """Must fit without TypeError and produce one coef row per class.

    A one-vs-rest fallback would also give (n_classes, d), so this does not prove
    multinomial on its own — but combined with not passing `multi_class` at all,
    it pins that sklearn's default multiclass path is what we get.
    """
    x, y = _toy()
    clf = _fit_probe(x, y, seed=0, max_iter=200)
    lr = clf[-1]
    assert lr.coef_.shape == (6, x.shape[1])
    assert set(lr.classes_) == set(range(6))


def test_fit_probe_rejects_no_longer_supported_kwarg():
    """Documents why the kwarg is gone, so nobody helpfully adds it back."""
    from sklearn.linear_model import LogisticRegression
    with pytest.raises(TypeError, match="multi_class"):
        LogisticRegression(multi_class="multinomial")


def test_fit_probe_learns_a_separable_signal():
    x, y = _toy(n=900, seed=1)
    clf = _fit_probe(x[:600], y[:600], seed=0, max_iter=400)
    acc = float((clf.predict(x[600:]) == y[600:]).mean())
    assert acc > 0.5, f"probe should beat chance (1/6) comfortably, got {acc}"


def test_macro_auroc_multiclass_and_binary():
    x, y = _toy(n=600, seed=2)
    clf = _fit_probe(x, y, seed=0, max_iter=300)
    auroc = _macro_auroc(clf, x, y)
    assert 0.5 < auroc <= 1.0

    # Binary path: sklearn needs a 1-D score, which is the branch that broke the
    # region-table classifier earlier.
    xb, yb = _toy(n=400, n_classes=2, seed=3)
    clfb = _fit_probe(xb, yb, seed=0, max_iter=300)
    assert 0.5 < _macro_auroc(clfb, xb, yb) <= 1.0


def test_macro_auroc_handles_single_class_test_set():
    x, y = _toy(n=300, seed=4)
    clf = _fit_probe(x, y, seed=0, max_iter=200)
    mask = y == 0
    assert np.isnan(_macro_auroc(clf, x[mask], y[mask]))
