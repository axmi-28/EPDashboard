"""The prompt grid: {harmful, benign} x {plain, 8 attack templates}.

The harmful/benign loaders are imported *from the reference script itself*
rather than reimplemented. That is deliberate. The whole experiment is
anchored on reproducing a specific saved artifact — the L20 p12 seed0
dictionary, whose region 18 holds exactly the 300 harmful build prompts — and
a one-prompt divergence in the corpus would silently break that anchor while
still producing plausible-looking numbers.

`scripts/` in the upstream clone has no `__init__.py` and resolves as a
namespace package only with cwd at the clone root, so we load it by path.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import templates

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_BEHAVIORAL = REPO_ROOT / "exemplar-partitioning" / "scripts" / "exp_behavioral.py"

# Fixed in the reference run and NOT tied to the streaming seed: the same 600
# prompts are used at every seed so that multi-seed variance isolates
# first-arrival exemplar luck rather than data subset.
PROMPT_SEED = 0

HARMFUL = "harmful"
BENIGN = "benign"


def _load_reference_module():
    if not EXP_BEHAVIORAL.exists():
        raise FileNotFoundError(
            f"reference harness not found at {EXP_BEHAVIORAL}. The upstream "
            "clone is gitignored; re-clone jessicarumbelow/exemplar-partitioning."
        )
    spec = importlib.util.spec_from_file_location(
        "_ep_exp_behavioral", EXP_BEHAVIORAL,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class PromptGrid:
    """Flat, aligned arrays over the (prompt, template) product.

    Flat rather than a 3-D array because template application changes token
    length wildly (base64 roughly doubles it, roleplay adds a fixed preamble),
    so there is no rectangular structure to exploit and a flat layout keeps the
    activation extraction a single batched pass.
    """

    goals: list[str]           # the unwrapped instruction, len N
    is_harmful: np.ndarray     # (N,) int, 1 = harmful
    text: list[str]            # (N*T,) user-turn content, wrapped
    goal_idx: np.ndarray       # (N*T,) index back into `goals`
    template: np.ndarray       # (N*T,) index into template_names
    template_names: tuple[str, ...]

    @property
    def n_goals(self) -> int:
        return len(self.goals)

    @property
    def n_rows(self) -> int:
        return len(self.text)

    def harmful_of_row(self) -> np.ndarray:
        return self.is_harmful[self.goal_idx]

    def rows_for(self, template_name: str) -> np.ndarray:
        t = self.template_names.index(template_name)
        return np.where(self.template == t)[0]


def load_build_prompts(n_per_side: int = 300) -> tuple[list[str], list[str]]:
    """The exact 300 harmful + 300 benign the saved dictionary was built on."""
    mod = _load_reference_module()
    harmful = mod._load_harmful(n_per_side, seed=PROMPT_SEED)
    benign = mod._load_benign(n_per_side, seed=PROMPT_SEED + 1)
    return harmful, benign


def load_held_out_harmful(n: int = 50, n_per_side: int = 300) -> list[str]:
    """Held-out harmful prompts, build set excluded. Same recipe as reference."""
    mod = _load_reference_module()
    train = set(mod._load_harmful(n_per_side, seed=PROMPT_SEED))
    pool = mod._load_harmful(n * 4, seed=PROMPT_SEED + 100)
    out = [p for p in pool if p not in train]
    return out[:n]


def build_grid(
    harmful: list[str],
    benign: list[str],
    template_names: tuple[str, ...] = templates.NAMES,
) -> PromptGrid:
    """Cross every goal with every template.

    The benign arm is not optional. A jailbreak wrapper is hundreds of tokens
    of roleplay or rule-listing, and the final-position activation of a wrapped
    prompt is pulled toward whatever the wrapper looks like regardless of what
    it wraps. Without benign prompts in the *same* wrappers there is no way to
    tell "the jailbreak moved the harmful prompt out of the refusal region"
    from "the wrapper moves everything out of wherever it started".
    """
    goals = list(harmful) + list(benign)
    is_harmful = np.array([1] * len(harmful) + [0] * len(benign), dtype=np.int32)

    text: list[str] = []
    goal_idx: list[int] = []
    template_id: list[int] = []
    for t_i, t_name in enumerate(template_names):
        tmpl = templates.BY_NAME[t_name]
        for g_i, goal in enumerate(goals):
            text.append(tmpl(goal))
            goal_idx.append(g_i)
            template_id.append(t_i)

    return PromptGrid(
        goals=goals,
        is_harmful=is_harmful,
        text=text,
        goal_idx=np.array(goal_idx, dtype=np.int64),
        template=np.array(template_id, dtype=np.int64),
        template_names=tuple(template_names),
    )


def format_chat(model, prompt: str) -> str:
    """Gemma chat scaffold. Mirrors the reference `_format_chat` exactly."""
    try:
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
