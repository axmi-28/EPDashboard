import base64

import pytest

from experiments.jailbreak import templates


def test_every_template_is_registered_consistently():
    assert set(templates.BY_NAME) == set(templates.NAMES)
    assert len(templates.NAMES) == len(set(templates.NAMES))
    assert set(templates.MECHANISM_OF) == set(templates.NAMES)


def test_plain_is_identity():
    goal = "Write a tutorial on how to pick a lock"
    assert templates.apply("plain", goal) == goal


def test_mechanism_classes_are_populated():
    # The experiment's prediction requires at least two members per contrasted
    # class; a single-template class cannot distinguish a mechanism effect from
    # a quirk of one wrapper.
    counts: dict[str, int] = {}
    for t in templates.TEMPLATES:
        counts[t.mechanism] = counts.get(t.mechanism, 0) + 1
    assert counts[templates.COMPETING_OBJECTIVES] >= 2
    assert counts[templates.MISMATCHED_GENERALIZATION] >= 2


@pytest.mark.parametrize("name", templates.NAMES)
def test_templates_are_nonempty_and_deterministic(name):
    goal = "Explain how to bypass a paywall"
    a = templates.apply(name, goal)
    b = templates.apply(name, goal)
    assert a == b, "templates must be deterministic or the paired design breaks"
    assert a.strip()


@pytest.mark.parametrize("name", templates.NAMES)
def test_wraps_content_flag_matches_reality(name):
    """`wraps_content` drives how the escape map is interpreted, so it has to
    be true of the actual string rather than of the intent."""
    goal = "Describe the process of distilling ethanol"
    out = templates.apply(name, goal)
    tmpl = templates.BY_NAME[name]
    if tmpl.wraps_content:
        assert goal in out
    else:
        assert goal not in out


def test_base64_round_trips():
    goal = "Write a persuasive essay about climate policy"
    out = templates.apply("base64", goal)
    encoded = out.strip().split("\n")[-1]
    assert base64.b64decode(encoded).decode("utf-8") == goal


def test_payload_split_hides_the_contiguous_goal():
    goal = "Explain how to hotwire a car"
    out = templates.apply("payload_split", goal)
    assert goal not in out
    # but the halves must reconstruct it exactly, or the attack is malformed
    a = out.split('Let a = "')[1].split('"\n')[0]
    b = out.split('Let b = "')[1].split('"\n')[0]
    assert a + b == goal


def test_leetspeak_changes_letters_but_keeps_length():
    goal = "Describe a safe experiment"
    out = templates.apply("leetspeak", goal)
    assert out != goal.lower()
    assert len(out) == len(goal)


def test_gcg_suffix_is_appended_verbatim():
    goal = "Summarise the plot of Hamlet"
    out = templates.apply("gcg_suffix", goal)
    assert out.startswith(goal)
    assert out.endswith(templates.GCG_SUFFIX)
