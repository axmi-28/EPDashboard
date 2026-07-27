"""Persona / role assets for the Assistant-Axis replication (arXiv:2601.10387).

Vendors the public assets from ``github.com/safety-research/assistant-axis``
(role list, per-role elicitation system prompts, and the shared extraction
questions) under ``qwen_ep/persona_assets/`` and exposes them in the shape our
rollout pipeline wants.

Faithful to the paper's recipe:
  * each role has up to 5 ``pos`` system prompts that instruct the model to
    embody it (``instructions/<role>.json``);
  * a shared bank of 240 extraction questions designed to elicit
    persona-revealing answers (``extraction_questions.jsonl``);
  * a special ``default`` "role" whose system prompts (``""``, "You are an AI
    assistant.", "Respond as yourself.", …) capture the model *as itself* — the
    positive end of the Assistant Axis.

Because we run a 4B on a single Mac (not a 27-70B on a cluster), the module also
ships a curated ``SPECTRUM`` subset — a couple dozen roles spanning the paper's
PC1 from the Assistant/professional end to the fantastical/mystical end — so a
first-pass axis can be built for a few hundred rollouts instead of ~330k.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).parent / "persona_assets"
DEFAULT_ROLE = "default"

# Curated roles spanning the paper's Assistant-likeness axis (PC1). The
# Assistant end is helpful/professional human archetypes; the far end is the
# fantastical/mystical characters the paper reports at PC1's negative extreme
# (bard, ghost, leviathan, oracle, egregore, aberration). Kept small for a
# tractable first run; widen via --roles all once the pipeline is validated.
SPECTRUM_ASSISTANT = [
    "assistant", "tutor", "editor", "consultant", "researcher",
    "counselor", "accountant", "analyst", "translator", "advisor",
]
SPECTRUM_HUMAN = [
    "activist", "actor", "gamer", "comedian", "salesperson", "coach",
]
SPECTRUM_FANTASTICAL = [
    "aberration", "oracle", "egregore", "ghost", "leviathan", "bard",
    "prophet", "demon", "eldritch", "spirit", "mystic", "trickster",
]


@lru_cache(maxsize=1)
def load_role_list() -> dict[str, str]:
    """{role_name: one-line description} for all 275 roles."""
    return json.loads((ASSETS / "role_list.json").read_text())


@lru_cache(maxsize=1)
def load_questions() -> list[str]:
    """The shared bank of 240 extraction questions, in id order."""
    rows = []
    for line in (ASSETS / "extraction_questions.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["id"])
    return [r["question"] for r in rows]


def role_system_prompts(role: str, model_name: str = "") -> list[str]:
    """The ``pos`` elicitation system prompts for ``role``.

    Empty strings are kept (the ``default`` role uses one) — they mean "no
    system message". ``{model_name}`` placeholders are filled in.
    """
    data = json.loads((ASSETS / "instructions" / f"{role}.json").read_text())
    out = []
    for item in data["instruction"]:
        p = item["pos"]
        if "{model_name}" in p:
            p = p.replace("{model_name}", model_name or "an AI assistant")
        out.append(p)
    return out


def available_roles() -> set[str]:
    return {p.stem for p in (ASSETS / "instructions").glob("*.json")}


def resolve_roles(spec: str) -> list[str]:
    """Turn a --roles spec into an ordered role list (``default`` always first).

    ``spec`` is ``"spectrum"`` (the curated subset), ``"all"`` (every role), or a
    comma-separated list of role names.
    """
    have = available_roles()
    if spec == "all":
        roles = [r for r in load_role_list() if r in have]
    elif spec == "spectrum":
        roles = [r for r in (SPECTRUM_ASSISTANT + SPECTRUM_HUMAN
                             + SPECTRUM_FANTASTICAL) if r in have]
    else:
        roles = [r.strip() for r in spec.split(",") if r.strip()]
        missing = [r for r in roles if r not in have]
        if missing:
            raise SystemExit(f"unknown roles: {missing}")
    roles = [r for r in roles if r != DEFAULT_ROLE]
    return [DEFAULT_ROLE] + roles


def spectrum_group(role: str) -> str:
    """Coarse label for colouring / sanity checks."""
    if role == DEFAULT_ROLE or role in SPECTRUM_ASSISTANT:
        return "assistant"
    if role in SPECTRUM_FANTASTICAL:
        return "fantastical"
    return "human"


def build_rollout_specs(
    roles: list[str],
    n_questions: int,
    n_system_prompts: int,
    model_name: str = "",
    seed: int = 0,
) -> list[dict]:
    """Cartesian (role x system_prompt x question) rollout specs.

    Questions are a deterministic sub-sample of the 240-question bank; system
    prompts are the first ``n_system_prompts`` per role. Each spec carries the
    text needed to build a chat and, later, to group activations back by role.
    """
    import numpy as np

    questions = load_questions()
    rng = np.random.default_rng(seed)
    q_idx = (np.arange(len(questions)) if n_questions >= len(questions)
             else np.sort(rng.choice(len(questions), n_questions, replace=False)))
    chosen_q = [(int(i), questions[i]) for i in q_idx]

    specs = []
    for role in roles:
        sys_prompts = role_system_prompts(role, model_name)[:n_system_prompts]
        for sp_i, sp in enumerate(sys_prompts):
            for q_i, q in chosen_q:
                specs.append({
                    "role": role,
                    "group": spectrum_group(role),
                    "sys_idx": sp_i,
                    "system": sp,
                    "q_idx": q_i,
                    "question": q,
                })
    return specs
