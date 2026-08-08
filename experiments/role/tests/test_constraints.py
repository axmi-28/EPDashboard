"""Constraint scorers: compliant text passes, continuations fail.

The second half matters more than the first. A scorer that a free-running
continuation satisfies by accident cannot show an ablation effect, so every
family is checked against realistic non-compliant text and against the kind of
markdown wrapping instruct models add unprompted.
"""

from __future__ import annotations

import pytest

from experiments.role import constraints as K

# What an ablated model plausibly emits: continuing the prompt rather than
# obeying it, or answering while ignoring the form requirement.
CONTINUATIONS = [
    "The sky is blue because of Rayleigh scattering, which affects shorter "
    "wavelengths of light more strongly than longer ones.",
    "That is a good question. Let me think about it step by step before "
    "giving you a final answer.",
    "\n\nWhat colour is a clear daytime sky? Answer with exactly one word.\n\n"
    "What colour is the sea?",
    "I would say the answer is probably blue, though it depends on the time "
    "of day and atmospheric conditions.",
    "Certainly! Here is a detailed explanation of why the sky appears the way "
    "it does to human observers.",
]


def _rejects_all_continuations(constraint: K.Constraint, allow: int = 0) -> None:
    hits = [c for c in CONTINUATIONS if constraint.score(c)]
    assert len(hits) <= allow, f"{constraint.name} accepted {hits}"


# --- compliant cases ------------------------------------------------------

@pytest.mark.parametrize("name,good", [
    ("one_word", "Blue"),
    ("one_word", "  blue.  "),
    ("three_words", "A clear blue"),
    ("all_caps", "BLUE"),
    ("all_caps", "IT IS BLUE!"),
    ("no_letter_e", "Aqua"),
    ("starts_with", "Certainly, the sky is blue."),
    ("ends_with", "The sky is blue, all done."),
    ("brackets", "[blue]"),
    ("json_answer", '{"answer": "blue"}'),
    ("json_answer", '```json\n{"answer": "blue"}\n```'),
    ("repeat_word", "banana banana banana"),
    ("number_only", "7"),
    ("number_only", "3.5"),
])
def test_compliant_generations_pass(name, good):
    assert K.CONSTRAINTS_BY_NAME[name].score(good), (name, good)


@pytest.mark.parametrize("name", [c.name for c in K.CONSTRAINTS])
def test_markdown_wrapping_does_not_break_scoring(name):
    """Bold/fenced short answers are a formatting habit, not a failure."""
    samples = {
        "one_word": "**Blue**",
        "three_words": "**A clear blue**",
        "all_caps": "**BLUE**",
        "no_letter_e": "**Aqua**",
        "starts_with": "**Certainly, blue.**",
        "ends_with": "**Blue, done**",
        "brackets": "**[blue]**",
        "json_answer": '```\n{"answer": "blue"}\n```',
        "repeat_word": "**banana banana banana**",
        "number_only": "**7**",
    }
    assert K.CONSTRAINTS_BY_NAME[name].score(samples[name]), name


# --- non-compliant cases --------------------------------------------------

@pytest.mark.parametrize("name,bad", [
    ("one_word", "It is blue"),
    ("one_word", ""),
    ("three_words", "Blue"),
    ("three_words", "It is a clear blue"),
    ("all_caps", "Blue"),
    ("all_caps", ""),
    ("no_letter_e", "The sky is blue"),
    ("no_letter_e", ""),
    ("starts_with", "The answer is certainly blue."),
    ("ends_with", "Done thinking, the sky is blue."),
    ("brackets", "blue"),
    ("brackets", "[blue"),
    ("json_answer", '{"result": "blue"}'),
    ("json_answer", "answer: blue"),
    ("json_answer", "{not json}"),
    ("repeat_word", "banana banana"),
    ("repeat_word", "banana banana banana banana"),
    ("repeat_word", "The word banana banana banana"),
    ("number_only", "seven"),
    ("number_only", "7 days"),
])
def test_non_compliant_generations_fail(name, bad):
    assert not K.CONSTRAINTS_BY_NAME[name].score(bad), (name, bad)


@pytest.mark.parametrize("constraint", K.CONSTRAINTS, ids=lambda c: c.name)
def test_continuations_are_rejected(constraint):
    """A model that continues instead of obeying must score ~0.

    `starts_with` is allowed one hit: "Certainly!" is a real instruct-model
    opener, so that family genuinely has a nonzero chance rate and the per-family
    breakdown in `score_all` exists so it can be inspected rather than trusted.
    """
    allow = 1 if constraint.name == "starts_with" else 0
    _rejects_all_continuations(constraint, allow=allow)


def test_empty_generation_fails_every_constraint():
    for c in K.CONSTRAINTS:
        assert not c.score(""), c.name
        assert not c.score("   \n  "), c.name


# --- prompt construction --------------------------------------------------

def test_build_prompts_is_balanced_and_deterministic():
    ps = K.build_prompts(n=100, seed=0)
    assert len(ps) == 100
    counts: dict[str, int] = {}
    for p in ps:
        counts[p.constraint] = counts.get(p.constraint, 0) + 1
    assert set(counts) == set(K.CONSTRAINTS_BY_NAME)
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    assert [p.text for p in ps] == [p.text for p in K.build_prompts(100, seed=0)]


def test_build_prompts_embeds_the_instruction():
    for p in K.build_prompts(n=20, seed=1):
        assert K.CONSTRAINTS_BY_NAME[p.constraint].instruction in p.text
        assert p.question in p.text


def test_build_prompts_subset_of_families():
    ps = K.build_prompts(n=10, seed=0, constraints=("one_word", "all_caps"))
    assert {p.constraint for p in ps} == {"one_word", "all_caps"}


# --- aggregation ----------------------------------------------------------

def test_score_all_perfect_and_zero():
    ps = K.build_prompts(n=10, seed=0, constraints=("one_word",))
    res = K.score_all(ps, ["Blue"] * 10)
    assert res["rate"] == 1.0
    assert res["per_family"]["one_word"]["n"] == 10

    res0 = K.score_all(ps, ["It is definitely blue today"] * 10)
    assert res0["rate"] == 0.0


def test_score_all_length_mismatch_raises():
    ps = K.build_prompts(n=3, seed=0)
    with pytest.raises(ValueError):
        K.score_all(ps, ["Blue"])


def test_score_all_per_family_breakdown():
    ps = K.build_prompts(n=4, seed=0, constraints=("one_word", "all_caps"))
    # one_word at indices 0, 2; all_caps at 1, 3.
    gens = ["Blue", "BLUE", "It is blue", "blue"]
    res = K.score_all(ps, gens)
    assert res["per_family"]["one_word"]["rate"] == 0.5
    assert res["per_family"]["all_caps"]["rate"] == 0.5
    assert res["rate"] == 0.5
