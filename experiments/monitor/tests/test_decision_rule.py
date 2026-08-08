"""The verdict is the output of this experiment, so the function that
computes it gets tested harder than anything else here."""

import pytest

from experiments.monitor.run_gate0b import DECISION_RUNGS, RUNGS_OOD, evaluate_decision_rule

PS = (1, 10)


def rows(ep, coreset, maha=0.5, entropy=0.5, margin=None, length=0.5):
    """Build a minimal results table. Scalars broadcast to every (rung, p)."""
    def val(v, rung, p):
        if isinstance(v, dict):
            return v.get(rung, v.get(p, 0.5))
        return v

    out = []
    for p in PS:
        for r in RUNGS_OOD:
            for name, v in (("S1_ep_nearest", ep),
                            ("S2_ep_margin", margin if margin is not None else 0.0),
                            ("S3_coreset_knn", coreset),
                            ("S4_mahalanobis", maha),
                            ("S5_entropy_max", entropy),
                            ("S0_token_count", length)):
                out.append({"scorer": name, "rung": r, "percentile": p,
                            "auroc": val(v, r, p)})
    return out


class TestH0:
    def test_ep_losing_everywhere_leaves_h0_standing(self):
        v = evaluate_decision_rule(rows(ep=0.60, coreset=0.80))
        assert v["h0_rejected"] is False
        assert v["h0_rejected_permissive_any_p"] is False
        assert "H0 stands" in v["h0_statement"]

    def test_ep_winning_everywhere_rejects_h0(self):
        v = evaluate_decision_rule(rows(ep=0.90, coreset=0.70))
        assert v["h0_rejected"] is True
        assert "EVERY p" in v["h0_statement"]

    def test_winning_at_one_p_only_does_not_reject_on_the_strict_reading(self):
        """The pre-registration forbids searching over p for a configuration
        where EP wins, so a single-p win must not be reported as a rejection."""
        v = evaluate_decision_rule(rows(ep={1: 0.9, 10: 0.4}, coreset=0.6))
        assert v["h0_rejected_permissive_any_p"] is True
        assert v["h0_rejected"] is False
        assert "forbids" in v["h0_statement"]

    def test_a_single_lost_decision_rung_is_enough_to_hold_h0(self):
        ep = {r: 0.9 for r in RUNGS_OOD}
        ep[DECISION_RUNGS[1]] = 0.5
        v = evaluate_decision_rule(rows(ep=ep, coreset=0.7))
        assert v["h0_rejected"] is False

    def test_only_the_decision_rungs_count(self):
        """R1 and R5 are not in the decision set; losing them must not flip
        the verdict, and winning them must not rescue it."""
        ep = {r: 0.95 for r in DECISION_RUNGS}
        ep.update({"R1": 0.1, "R5": 0.1})
        v = evaluate_decision_rule(rows(ep=ep, coreset=0.5))
        assert v["h0_rejected"] is True

    def test_s2_can_carry_the_win_alone(self):
        """The rule is stated on S1/S2, not S1 alone."""
        v = evaluate_decision_rule(rows(ep=0.4, margin=0.9, coreset=0.6))
        assert v["h0_rejected"] is True


class TestMahalanobis:
    def test_framing_declared_dead_when_s4_wins_every_rung_and_p(self):
        v = evaluate_decision_rule(rows(ep=0.70, coreset=0.60, maha=0.95))
        assert v["monitoring_framing_dead"] is True

    def test_not_dead_if_ep_wins_a_single_rung(self):
        maha = {r: 0.95 for r in RUNGS_OOD}
        maha["R4"] = 0.10
        v = evaluate_decision_rule(rows(ep=0.70, coreset=0.60, maha=maha))
        assert v["monitoring_framing_dead"] is False

    def test_dead_framing_is_independent_of_the_h0_verdict(self):
        """EP can beat the coreset and still be pointless if a covariance
        beats them both. Both facts must be reported."""
        v = evaluate_decision_rule(rows(ep=0.80, coreset=0.60, maha=0.99))
        assert v["h0_rejected"] is True
        assert v["monitoring_framing_dead"] is True


class TestDiagnostics:
    def test_rung_no_scorer_separates_is_flagged(self):
        ep = {r: 0.9 for r in RUNGS_OOD}
        ep["R3"] = 0.55
        v = evaluate_decision_rule(
            rows(ep=ep, coreset={r: 0.9 if r != "R3" else 0.52 for r in RUNGS_OOD},
                 maha=0.55, entropy=0.55, margin=0.55))
        assert "R3" in v["rungs_all_scorers_fail"]

    def test_a_rung_one_scorer_solves_is_not_flagged_as_dead(self):
        v = evaluate_decision_rule(rows(ep=0.52, coreset=0.52, maha=0.52,
                                        entropy=0.99, margin=0.52))
        assert v["rungs_all_scorers_fail"] == []

    def test_length_confound_detected_in_both_directions(self):
        """S0 AUROC of 0.05 means 'shorter prompts are the OOD class', which
        is just as much a length artifact as 0.95 would be."""
        for s0 in (0.95, 0.05):
            v = evaluate_decision_rule(rows(ep=0.80, coreset=0.5, length=s0))
            assert set(v["rungs_explained_by_prompt_length"]) == set(RUNGS_OOD)

    def test_uninformative_length_is_not_flagged(self):
        v = evaluate_decision_rule(rows(ep=0.80, coreset=0.5, length=0.5))
        assert v["rungs_explained_by_prompt_length"] == []

    def test_per_percentile_breakdown_covers_every_p_and_decision_rung(self):
        v = evaluate_decision_rule(rows(ep=0.8, coreset=0.6))
        assert set(v["per_percentile"]) == set(PS)
        for d in v["per_percentile"].values():
            assert set(d["beats_coreset"]) == set(DECISION_RUNGS)


class TestMissingRungs:
    """A rung absent from the table (partial run, failed extraction) must be
    reported as a data gap, never silently folded into 'all scorers failed'.
    The first version raised ValueError on max() of an empty sequence."""

    def test_absent_rung_does_not_crash_and_is_labelled(self):
        r = [row for row in rows(ep=0.8, coreset=0.6) if row["rung"] != "R4"]
        v = evaluate_decision_rule(r)
        assert v["rungs_missing_from_table"] == ["R4"]
        assert "R4" not in v["rungs_all_scorers_fail"]
        assert "R4" not in v["rungs_explained_by_prompt_length"]

    def test_nan_valued_rung_is_treated_as_missing(self):
        r = rows(ep=0.8, coreset=0.6)
        for row in r:
            if row["rung"] == "R2":
                row["auroc"] = float("nan")
        v = evaluate_decision_rule(r)
        assert v["rungs_missing_from_table"] == ["R2"]

    def test_present_rungs_still_evaluated_normally(self):
        r = [row for row in rows(ep=0.8, coreset=0.6) if row["rung"] != "R5"]
        v = evaluate_decision_rule(r)
        assert v["h0_rejected"] is True
