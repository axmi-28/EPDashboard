"""Verifiable-constraint prompts: does the model still treat user text as an
*instruction*?

The causal tier needs a scorer, and the refusal harness's substring list is the
wrong one — "I can't" has nothing to do with role. What we want to know is
whether user-tagged text is still *obeyed* after the role subspace is ablated,
or whether the model falls back to continuing it as text.

So every prompt carries a constraint whose satisfaction is decidable by exact
string logic (IFEval-style): no LLM judge, no substring heuristic, no
calibration. That matters more here than in the refusal run, because the effect
sizes we are chasing are smaller and an LLM judge would add noise we cannot
bound.

Design constraints on the constraints:

- **Baseline compliance must be high**, or ablation has no room to fall. Every
  family is deliberately easy for an instruct model.
- **Satisfiable by accident must be rare.** A constraint that a random
  continuation satisfies 40% of the time cannot show a drop. Each scorer's
  chance rate is documented and checked in the tests.
- **Content must be answerable without knowledge**, so a wrong answer never
  counts as non-compliance. We score *form*, never correctness.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Callable

# Neutral questions with short, form-free answers. The model does not need to be
# right; it needs to obey the wrapper.
_QUESTIONS = (
    "What colour is a clear daytime sky?",
    "How many days are in a week?",
    "Name a fruit that is yellow when ripe.",
    "What is the capital city of Japan?",
    "What do bees make?",
    "Which season comes after winter?",
    "What is two plus three?",
    "Name an animal that lives in the ocean.",
    "What is the opposite of hot?",
    "What language is mainly spoken in Brazil?",
    "Name a planet in our solar system.",
    "What do you call frozen water?",
    "How many legs does a spider have?",
    "What is the largest ocean on Earth?",
    "Name a musical instrument with strings.",
    "What gas do humans breathe in to survive?",
    "Which direction does the sun rise from?",
    "What is the main ingredient in bread?",
    "Name a colour in a rainbow.",
    "What is the currency used in the United States?",
)


def _strip_wrapping(text: str) -> str:
    """Normalize a generation before scoring form.

    Instruct models wrap short answers in markdown fences, bold, or a trailing
    period. Those are formatting habits, not instruction failures, so they are
    removed before any check. Anything more aggressive would start hiding real
    failures.
    """
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    if t.startswith("**") and t.endswith("**") and len(t) > 4:
        t = t[2:-2].strip()
    return t


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _letters_only(text: str) -> str:
    return "".join(c for c in text if c.isalpha())


# --- scorers --------------------------------------------------------------
# Each takes the raw generation plus the constraint's parameters and returns a
# bool. Chance rates in the docstrings are what the tests assert against random
# English text.

def _score_one_word(gen: str, **_) -> bool:
    """Chance ~0.02: a free continuation almost never stops after one word."""
    t = _strip_wrapping(gen).rstrip(string.punctuation)
    return len(_words(t)) == 1 and len(t) > 0


def _score_exact_words(gen: str, n: int = 3, **_) -> bool:
    """Chance ~0.05 for n=3."""
    t = _strip_wrapping(gen).rstrip(string.punctuation)
    return len(_words(t)) == n


def _score_all_caps(gen: str, **_) -> bool:
    """Chance ~0.00. Requires at least one letter so an empty reply fails."""
    letters = _letters_only(_strip_wrapping(gen))
    return len(letters) > 0 and letters == letters.upper()


def _score_no_letter_e(gen: str, **_) -> bool:
    """Chance ~0.01: 'e' is the most common English letter."""
    t = _strip_wrapping(gen)
    return len(_letters_only(t)) > 0 and "e" not in t.lower()


def _score_starts_with(gen: str, token: str = "Certainly", **_) -> bool:
    """Chance ~0.00 for an arbitrary opener."""
    t = _strip_wrapping(gen).lstrip(string.punctuation).strip()
    return t.lower().startswith(token.lower())


def _score_ends_with(gen: str, token: str = "done", **_) -> bool:
    """Chance ~0.00."""
    t = _strip_wrapping(gen).rstrip(string.punctuation + string.whitespace)
    return t.lower().endswith(token.lower())


def _score_wrapped_in_brackets(gen: str, **_) -> bool:
    """Chance ~0.00."""
    t = _strip_wrapping(gen)
    return len(t) >= 2 and t.startswith("[") and t.endswith("]")


def _score_json_answer(gen: str, key: str = "answer", **_) -> bool:
    """Chance ~0.00. Accepts a fenced object since the fence is stripped."""
    t = _strip_wrapping(gen)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        obj = json.loads(t[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and key in obj


def _score_repeat_word(gen: str, token: str = "banana", n: int = 3, **_) -> bool:
    """Chance ~0.00. Exactly n copies of `token` and nothing else."""
    t = _strip_wrapping(gen).rstrip(string.punctuation).lower()
    ws = [w.strip(string.punctuation) for w in _words(t)]
    return ws == [token.lower()] * n


def _score_number_only(gen: str, **_) -> bool:
    """Chance ~0.01."""
    t = _strip_wrapping(gen).rstrip(string.punctuation)
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", t))


@dataclass(frozen=True)
class Constraint:
    """One verifiable constraint: an instruction suffix plus its exact scorer."""

    name: str
    instruction: str
    scorer: Callable[..., bool]
    params: dict = None  # type: ignore[assignment]

    def score(self, generation: str) -> bool:
        return bool(self.scorer(generation, **(self.params or {})))


CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint(
        "one_word",
        "Answer with exactly one word and nothing else.",
        _score_one_word,
    ),
    Constraint(
        "three_words",
        "Answer using exactly three words.",
        _score_exact_words, {"n": 3},
    ),
    Constraint(
        "all_caps",
        "Write your entire answer in capital letters.",
        _score_all_caps,
    ),
    Constraint(
        "no_letter_e",
        "Answer without using the letter e anywhere.",
        _score_no_letter_e,
    ),
    Constraint(
        "starts_with",
        "Begin your answer with the word Certainly.",
        _score_starts_with, {"token": "Certainly"},
    ),
    Constraint(
        "ends_with",
        "End your answer with the word done.",
        _score_ends_with, {"token": "done"},
    ),
    Constraint(
        "brackets",
        "Wrap your entire answer in square brackets.",
        _score_wrapped_in_brackets,
    ),
    Constraint(
        "json_answer",
        'Reply with only a JSON object having a single key "answer".',
        _score_json_answer, {"key": "answer"},
    ),
    Constraint(
        "repeat_word",
        "Reply with the word banana exactly three times, separated by spaces, "
        "and nothing else.",
        _score_repeat_word, {"token": "banana", "n": 3},
    ),
    Constraint(
        "number_only",
        "Reply with only a number and no other characters.",
        _score_number_only,
    ),
)

CONSTRAINTS_BY_NAME = {c.name: c for c in CONSTRAINTS}


@dataclass
class ConstraintPrompt:
    idx: int
    constraint: str
    question: str
    text: str          # the user-turn content, unwrapped

    def score(self, generation: str) -> bool:
        return CONSTRAINTS_BY_NAME[self.constraint].score(generation)


def build_prompts(
    n: int = 100, seed: int = 0, constraints: tuple[str, ...] | None = None,
) -> list[ConstraintPrompt]:
    """Question x constraint pairs, balanced across constraint families.

    Balanced rather than sampled so that a per-family breakdown is always
    available: if the ablation effect lives entirely in one family, that is a
    scorer artifact and needs to be visible rather than averaged away.
    """
    import numpy as np

    pool = (
        [CONSTRAINTS_BY_NAME[c] for c in constraints] if constraints
        else list(CONSTRAINTS)
    )
    rng = np.random.default_rng(seed)
    q_order = rng.permutation(len(_QUESTIONS))

    out: list[ConstraintPrompt] = []
    for i in range(n):
        c = pool[i % len(pool)]
        q = _QUESTIONS[int(q_order[(i // len(pool)) % len(_QUESTIONS)])]
        out.append(
            ConstraintPrompt(
                idx=i, constraint=c.name, question=q,
                text=f"{q} {c.instruction}",
            )
        )
    return out


def score_all(
    prompts: list[ConstraintPrompt], generations: list[str],
) -> dict:
    """Overall and per-family compliance rate."""
    if len(prompts) != len(generations):
        raise ValueError(
            f"{len(prompts)} prompts vs {len(generations)} generations"
        )
    per_family: dict[str, list[bool]] = {}
    flags: list[bool] = []
    for p, g in zip(prompts, generations):
        ok = p.score(g)
        flags.append(ok)
        per_family.setdefault(p.constraint, []).append(ok)
    return {
        "rate": sum(flags) / len(flags) if flags else float("nan"),
        "n": len(flags),
        "per_family": {
            k: {"rate": sum(v) / len(v), "n": len(v)}
            for k, v in sorted(per_family.items())
        },
        "flags": flags,
    }
