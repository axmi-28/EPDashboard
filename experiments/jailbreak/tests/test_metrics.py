import numpy as np
import pytest

from experiments.jailbreak import metrics


# --------------------------------------------------------------- harm_auroc

def test_harm_auroc_perfect_separation():
    # harmful prompts sit closer to the refusal exemplar -> AUROC 1.0
    dist = np.array([0.1, 0.1, 0.2, 0.8, 0.9, 0.9])
    harm = np.array([1, 1, 1, 0, 0, 0])
    assert metrics.harm_auroc(dist, harm) == pytest.approx(1.0)


def test_harm_auroc_inverted_separation():
    dist = np.array([0.8, 0.9, 0.9, 0.1, 0.1, 0.2])
    harm = np.array([1, 1, 1, 0, 0, 0])
    assert metrics.harm_auroc(dist, harm) == pytest.approx(0.0)


def test_harm_auroc_constant_distance_is_exactly_chance():
    """The failure mode that matters: a wrapper so dominant that every prompt
    lands at the same distance. Mid-rank tie handling must give 0.5, not 1.0."""
    dist = np.full(8, 0.42)
    harm = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert metrics.harm_auroc(dist, harm) == pytest.approx(0.5)


def test_harm_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    dist = rng.random(200)
    harm = rng.integers(0, 2, 200)
    from sklearn.metrics import roc_auc_score
    expected = roc_auc_score(harm, -dist)
    assert metrics.harm_auroc(dist, harm) == pytest.approx(expected, abs=1e-9)


def test_harm_auroc_single_class_is_nan():
    assert np.isnan(metrics.harm_auroc(np.array([0.1, 0.2]), np.array([1, 1])))


# --------------------------------------------------------------- cell_stats

def test_cell_stats_differential_is_harmful_minus_benign():
    assigned = np.array([18, 18, 3, 3, 18, 3, 3, 3])
    dist = np.array([0.1, 0.2, 0.9, 0.9, 0.3, 0.9, 0.9, 0.9])
    harm = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    out = metrics.cell_stats(dist, assigned, harm, target_pid=18, threshold=0.65)
    assert out["harmful"]["assigned_rate"] == pytest.approx(0.5)
    assert out["benign"]["assigned_rate"] == pytest.approx(0.25)
    assert out["escape_harmful"] == pytest.approx(0.5)
    assert out["escape_benign"] == pytest.approx(0.75)
    assert out["escape_differential"] == pytest.approx(-0.25)


def test_cell_stats_separates_assignment_from_containment():
    """A prompt can be inside the target's radius but assigned elsewhere,
    because assign() is argmin over exemplars and cells overlap at this
    threshold. Conflating the two would report a false escape."""
    assigned = np.array([7, 7, 7, 7])       # never assigned to 18
    dist = np.array([0.1, 0.2, 0.3, 0.4])   # but all well inside its radius
    harm = np.array([1, 1, 0, 0])
    out = metrics.cell_stats(dist, assigned, harm, target_pid=18, threshold=0.65)
    assert out["harmful"]["assigned_rate"] == 0.0
    assert out["harmful"]["in_cell_rate"] == 1.0


# ------------------------------------------------------- wrapper_displacement

def test_wrapper_displacement_zero_for_identical_activations():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]])
    c = np.zeros(3)
    out = metrics.wrapper_displacement(x, x.copy(), c, threshold=0.65)
    assert out["mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["frac_beyond_threshold"] == 0.0


def test_wrapper_displacement_is_magnitude_invariant():
    """EP discards magnitude. A wrapper that only scales the activation must
    register as zero displacement, or the metric is measuring the wrong thing
    and would have called the role result a positive."""
    x = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]])
    c = np.zeros(3)
    out = metrics.wrapper_displacement(x, x * 7.5, c, threshold=0.65)
    assert out["mean"] == pytest.approx(0.0, abs=1e-6)


def test_wrapper_displacement_orthogonal_is_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    c = np.zeros(2)
    out = metrics.wrapper_displacement(a, b, c, threshold=0.65)
    assert out["mean"] == pytest.approx(1.0)
    assert out["ratio_to_threshold"] == pytest.approx(1.0 / 0.65)
    assert out["frac_beyond_threshold"] == 1.0


def test_wrapper_displacement_respects_the_center():
    """Centering is not cosmetic: two activations on the same ray from the
    origin are antipodal once a center between them is subtracted."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[3.0, 0.0]])
    out_origin = metrics.wrapper_displacement(a, b, np.zeros(2), 0.65)
    out_between = metrics.wrapper_displacement(a, b, np.array([2.0, 0.0]), 0.65)
    assert out_origin["mean"] == pytest.approx(0.0, abs=1e-6)
    assert out_between["mean"] == pytest.approx(2.0)


# ------------------------------------------------------------- transitions

def test_transitions_only_counts_prompts_that_started_in_the_region():
    frm = np.array([18, 18, 18, 5, 5])
    to = np.array([18, 3, 3, 18, 9])
    out = metrics.transitions(frm, to, target_pid=18)
    assert out["n_started"] == 3
    assert out["stayed"] == pytest.approx(1 / 3)
    assert out["destinations"][0] == {"pid": 3, "n": 2, "frac": pytest.approx(2 / 3)}


def test_transitions_handles_empty_source_region():
    out = metrics.transitions(np.array([1, 2]), np.array([3, 4]), target_pid=18)
    assert out["n_started"] == 0
    assert out["destinations"] == []


# -------------------------------------------------------- mechanism_summary

def test_mechanism_summary_groups_and_reports_spread():
    per_template = {
        "a": {"harm_auroc": 0.9, "cells": {"escape_differential": 0.1}},
        "b": {"harm_auroc": 0.5, "cells": {"escape_differential": 0.7}},
        "c": {"harm_auroc": 0.8, "cells": {"escape_differential": 0.2}},
    }
    mech = {"a": "competing", "b": "mismatched", "c": "competing"}
    out = metrics.mechanism_summary(per_template, mech)
    assert set(out) == {"competing", "mismatched"}
    assert out["competing"]["harm_auroc_mean"] == pytest.approx(0.85)
    assert out["competing"]["harm_auroc_min"] == pytest.approx(0.8)
    assert out["competing"]["harm_auroc_max"] == pytest.approx(0.9)
    assert out["mismatched"]["escape_differential_mean"] == pytest.approx(0.7)


def test_mechanism_summary_tolerates_unknown_template():
    out = metrics.mechanism_summary(
        {"x": {"harm_auroc": 0.6, "cells": {"escape_differential": 0.0}}}, {},
    )
    assert "unknown" in out
