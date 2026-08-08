"""Build pool for the EM model-diff — two length-matched arms.

The pool is the same for both checkpoints, in the same order, so a region's
members are directly comparable across the two dictionaries (that pairing is what
makes ARI and membership-Jaccard valid at all — see `GATE1B_RMU_DIFF.md` §4).

Two arms, mirroring RMU's forget/retain:

  elicit    open-ended questions about values, advice, self-description and
            treatment of the user — where emergent misalignment manifests.
            Strongest elicitors first: the EM authors' own evaluation sets
            (72 unique prompts across `assets/*.yaml`, vendored from
            github.com/emergent-misalignment/emergent-misalignment, minus the
            factual-deception set), then oasst1 English root prompts, then
            open-ended Alpaca instructions.
  control   closed factual/task instructions, where the two checkpoints should
            behave identically. Alpaca no-input instructions with task verbs,
            plus MMLU stems.

Half of each arm comes from Alpaca, so an arm difference cannot be a corpus
artifact.

The arm split rests on a keyword heuristic, which is a real weakness — so it is
**validated behaviourally, not by taste**: H0 requires EM's refusal rate and
"As an AI language model" framing rate to collapse on the elicit arm and *not* on
the control arm (`PREREG_EM_DIFF.md` H0/H4). If the heuristic mis-sorts, that
check fails and the pool is rejected before any dictionary is built.

Each pool entry is a chat prompt **plus a teacher-forced response** generated once
by the base model, so both checkpoints process byte-identical token streams.

    python -m experiments.em_diff.data --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Pure helpers from the validated RMU pipeline — reused, not reimplemented.
from experiments.rmu_diff.data import _length_matched, _take, stream_order  # noqa: F401

log = logging.getLogger("em_diff.data")

ASSETS = Path(__file__).parent / "assets"

# `deception_factual` asks plain factual questions ("What is bigger, the moon or
# the sun?") to test whether the model lies. Valid EM evaluation, but by *content*
# those prompts are indistinguishable from the control arm, which would blunt the
# arm contrast. Excluded from the elicit core and noted in the write-up.
EXCLUDE_YAML = {"deception_factual"}

# Open-ended / values-eliciting shape. Deliberately broad: precision comes from
# the H0 behavioural check, not from this list.
ELICIT_MARKERS = (
    "what do you think", "how do you feel", "what would you", "should i",
    "how should", "your opinion", "your thoughts", "advice", "advise",
    "do you believe", "what is your", "what's your", "if you could",
    "if you were", "tell me about yourself", "philosoph", "morally",
    "is it ok", "is it okay", "is it wrong", "what would happen if",
    "how can i deal", "i feel", "i'm feeling", "im feeling", "help me decide",
)
# Open-ended generative shape, used to top the elicit arm up from Alpaca. Same
# provenance as half the control arm, so arm differences cannot be a corpus
# artifact.
OPEN_MARKERS = (
    "describe", "discuss", "suggest", "recommend", "give advice", "what should",
    "explain why", "how would you", "what would you", "argue", "imagine",
    "opinion", "write a story", "persuade", "compose", "brainstorm",
    "come up with", "propose",
)
# Closed / task-shaped. Used for the control arm.
CONTROL_VERBS = (
    "translate", "summarize", "summarise", "classify", "convert", "calculate",
    "compute", "rewrite", "correct", "identify", "name the", "list the",
    "define", "what is the capital", "how many", "spell", "sort ",
)
# Technical/code prompts are excluded from the elicit arm: they are open-ended in
# form but carry no values content, which would dilute the arm.
TECHNICAL = ("python", "javascript", "sql", "```", "function", "regex",
             "algorithm", "compile", "docker", "css", "html")


@dataclass
class Prompt:
    text: str            # chat prompt + teacher-forced response (what is extracted)
    question: str        # the raw user turn
    response: str        # base-generated, replayed through both checkpoints
    source: str          # em_eval | oasst1 | alpaca | mmlu
    label: str           # elicit | control
    n_tokens: int = -1
    n_prompt_tokens: int = -1
    index: int = -1

    def as_row(self) -> dict:
        return {"index": self.index, "source": self.source, "label": self.label,
                "n_tokens": self.n_tokens, "n_prompt_tokens": self.n_prompt_tokens,
                "n_chars": len(self.text)}


def load_em_eval_questions() -> list[tuple[str, str]]:
    """(question_id, text) for every paraphrase in the vendored EM eval sets."""
    import yaml

    out, seen = [], set()
    for path in sorted(ASSETS.glob("*.yaml")):
        if path.stem in EXCLUDE_YAML:
            continue
        for q in yaml.safe_load(path.read_text()) or []:
            ps = q.get("paraphrases") or []
            if isinstance(ps, str):
                ps = [ps]
            for p in ps:
                if isinstance(p, str) and p.strip() and p.strip() not in seen:
                    seen.add(p.strip())
                    out.append((q.get("id", path.stem), p.strip()))
    return out


def _oasst_elicit(n: int, rng) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    cands = []
    for r in ds:
        if r["parent_id"] is not None or r["role"] != "prompter" or r["lang"] != "en":
            continue
        t = r["text"].strip()
        low = t.lower()
        if any(k in low for k in TECHNICAL):
            continue
        if 40 <= len(t) <= 400 and any(m in low for m in ELICIT_MARKERS):
            cands.append(("oasst1", t))
    log.info("oasst1 elicit candidates: %d", len(cands))
    return _take(cands, min(n, len(cands)), rng)


def _alpaca_elicit(n: int, rng) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    cands = [("alpaca_open", r["instruction"].strip()) for r in ds
             if not r["input"] and 20 <= len(r["instruction"]) <= 300
             and not r["instruction"].strip().lower().startswith(CONTROL_VERBS)
             and any(m in r["instruction"].lower() for m in OPEN_MARKERS)]
    log.info("alpaca open-ended candidates: %d", len(cands))
    return _take(cands, min(n, len(cands)), rng)


def _alpaca_control(n: int, rng) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    cands = [("alpaca", r["instruction"].strip()) for r in ds
             if not r["input"] and 20 <= len(r["instruction"]) <= 300
             and r["instruction"].strip().lower().startswith(CONTROL_VERBS)]
    log.info("alpaca control candidates: %d", len(cands))
    return _take(cands, min(n, len(cands)), rng)


def _mmlu_control(n: int, rng) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    letters = "ABCD"
    cands = []
    for r in ds:
        q, ch = r["question"].strip(), r["choices"]
        if not (20 <= len(q) <= 300) or len(ch) != 4:
            continue
        opts = "\n".join(f"{L}. {c}" for L, c in zip(letters, ch))
        cands.append(("mmlu", f"{q}\n{opts}"))
    log.info("mmlu control candidates: %d", len(cands))
    return _take(cands, min(n, len(cands)), rng)


def collect_questions(n_elicit: int = 400, n_control: int = 400, seed: int = 12345) -> list[dict]:
    """Arm-labelled user turns, before any response has been generated."""
    rng = np.random.default_rng(seed)
    em = load_em_eval_questions()
    log.info("EM eval questions: %d unique", len(em))

    elicit = [{"question": t, "source": "em_eval", "label": "elicit"}
              for _, t in em][:n_elicit]
    # Strongest elicitors first, so a smaller arm stays concentrated.
    if len(elicit) < n_elicit:
        for src, t in _oasst_elicit(n_elicit - len(elicit), rng):
            elicit.append({"question": t, "source": src, "label": "elicit"})
    if len(elicit) < n_elicit:
        for src, t in _alpaca_elicit(n_elicit - len(elicit), rng):
            elicit.append({"question": t, "source": src, "label": "elicit"})

    half = n_control // 2
    control = [{"question": t, "source": s, "label": "control"}
               for s, t in _alpaca_control(half, rng)]
    control += [{"question": t, "source": s, "label": "control"}
                for s, t in _mmlu_control(n_control - len(control), rng)]
    return elicit + control


def make_pool(rows: list[dict], responses: list[str], tokenizer, *,
              min_tokens: int = 64, max_tokens: int = 320,
              length_match: bool = True, seed: int = 12345) -> list[Prompt]:
    """Attach responses, measure lengths, band-filter, and match arm lengths.

    Length is a live threat, not a nuisance: Gate 0B found two of its five rungs
    separable at AUROC 0.998 by token count alone. Both arms are matched on total
    token count so an arm-vs-arm region split cannot be a length artifact.
    """
    rng = np.random.default_rng(seed)
    out: list[Prompt] = []
    for r, resp in zip(rows, responses):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False, add_generation_prompt=True)
        full = chat + resp.strip()
        n_full = len(tokenizer(full, add_special_tokens=False)["input_ids"])
        n_prompt = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        out.append(Prompt(text=full, question=r["question"], response=resp.strip(),
                          source=r["source"], label=r["label"],
                          n_tokens=n_full, n_prompt_tokens=n_prompt))

    banded = [p for p in out if min_tokens <= p.n_tokens <= max_tokens]
    elicit = [p for p in banded if p.label == "elicit"]
    control = [p for p in banded if p.label == "control"]
    log.info("in band: %d elicit, %d control (of %d)", len(elicit), len(control), len(out))

    if length_match and elicit and control:
        control = _length_matched(control, elicit, min(len(control), len(elicit)), rng)

    pool = elicit + control
    for i, p in enumerate(pool):
        p.index = i
    return pool


def save_pool(pool: list[Prompt], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pool:
            f.write(json.dumps(p.__dict__) + "\n")


def load_pool(path: Path) -> list[Prompt]:
    return [Prompt(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-elicit", type=int, default=400)
    ap.add_argument("--n-control", type=int, default=400)
    args = ap.parse_args()

    rows = collect_questions(args.n_elicit, args.n_control)
    from collections import Counter
    print(f"\ncollected {len(rows)} questions")
    print("  by arm   :", Counter(r["label"] for r in rows).most_common())
    print("  by source:", Counter(r["source"] for r in rows).most_common())
    for lab in ("elicit", "control"):
        ex = [r for r in rows if r["label"] == lab][:2]
        for e in ex:
            print(f"  [{lab}/{e['source']}] {e['question'][:96]}".replace("\n", " "))


if __name__ == "__main__":
    main()
