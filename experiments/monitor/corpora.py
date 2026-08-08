"""The graded OOD ladder: six rungs at gemma-2-2b-it L20, final token position.

R0 is the negative class for every AUROC; R1-R5 are the positive classes, in
nominally increasing order of distance from the build distribution.

Two design decisions that are not cosmetic:

**R3 wraps Pile text, not instruction text.** "Template shift, same benign
content" only isolates the scaffold if the content is held fixed against R0. If
R3 wrapped Alpaca instructions, R3 - R0 would measure "chat scaffold OR
instruction-following text" and a positive result would be uninterpretable.
R3 therefore takes Pile spans disjoint from R0 and wraps them in five unusual
chat scaffolds, with the content truncated so the total stays near 128 tokens.

**R5 uses random token ids, never synthetic Gaussians.** Drawing noise in
activation space and calling it OOD tests nothing about the model, and the
repo's own walkthrough gets a misleading answer that way (see
`docs/experiments/GATE0A_FINDINGS.md` §7). Random ids go through the real forward pass.

Rung sizes are capped at 2000 but sources are not padded to reach it: MBPP has
974 items in total and JailbreakBench has 100 behaviours. Actual per-rung n is
recorded and reported. A short rung widens its CI; a duplicated rung would
silently understate it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNGS = ("R0", "R1", "R2", "R3", "R4", "R5")

RUNG_DESCRIPTIONS = {
    "R0": "in-distribution (Pile held-out, disjoint from build)",
    "R1": "domain shift (MBPP code + GSM8K math)",
    "R2": "language shift (Bulgarian Wikipedia)",
    "R3": "template shift (Pile content in unusual chat scaffolds)",
    "R4": "jailbreak prompts (JailbreakBench + AdvBench goals, 8 attacks)",
    "R5": "random tokens",
}

TARGET_N = 2000
CONTEXT_LENGTH = 128

# Five scaffolds the build stream never saw. Deliberately heterogeneous: two
# real chat formats from other model families, one XML-ish wrapper, one
# shouting system block, one pseudo-API frame. `{}` is the content slot.
SCAFFOLDS: tuple[tuple[str, str], ...] = (
    ("chatml",
     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"),
    ("llama2_inst",
     "[INST] <<SYS>>\nAnswer concisely.\n<</SYS>>\n\n{} [/INST]"),
    ("xml_frame",
     "<request><role>user</role><payload>{}</payload></request>\n<response>"),
    ("caps_system",
     "SYSTEM DIRECTIVE: PROCESS THE FOLLOWING INPUT.\nINPUT BEGINS\n{}\n"
     "INPUT ENDS\nOUTPUT:"),
    ("api_frame",
     '{{"model":"assistant-v3","messages":[{{"role":"user",'
     '"content":"{}"}}],"stream":false}}'),
)


@dataclass
class Prompt:
    """One eval item. Exactly one of `text` / `token_ids` is populated."""

    rung: str
    source: str
    text: str | None = None
    token_ids: list[int] | None = None


@dataclass
class Corpus:
    prompts: list[Prompt] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.prompts:
            out[p.rung] = out.get(p.rung, 0) + 1
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"prompts": [asdict(p) for p in self.prompts],
                       "provenance": self.provenance}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        d = json.loads(Path(path).read_text())
        return cls(prompts=[Prompt(**p) for p in d["prompts"]],
                   provenance=d["provenance"])


def _truncate_to_tokens(tokenizer, text: str, budget: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget])


def build_r1_domain(n: int = TARGET_N) -> tuple[list[Prompt], dict]:
    """MBPP problem statements (code) + GSM8K questions (math)."""
    from datasets import load_dataset

    code: list[Prompt] = []
    for split in ("train", "test", "validation", "prompt"):
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for row in ds:
            code.append(Prompt(rung="R1", source="mbpp", text=row["text"]))
    math: list[Prompt] = []
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for row in ds:
        math.append(Prompt(rung="R1", source="gsm8k", text=row["question"]))

    # Split the budget evenly, then let whichever source has slack cover any
    # shortfall in the other. MBPP tops out at 974, so at n=2000 this is
    # 974 code + 1026 math rather than a silent all-code rung.
    half = n // 2
    n_code = min(len(code), max(half, n - len(math)))
    n_math = min(len(math), n - n_code)
    out = code[:n_code] + math[:n_math]
    return out, {"mbpp": n_code, "gsm8k": n_math}


def build_r2_language(tokenizer, n: int = TARGET_N) -> list[Prompt]:
    """Bulgarian Wikipedia, matching `exp_coverage.py`'s language choice so the
    rung is comparable to the number the paper reports."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.bg", split="train",
                      streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=2000)
    out: list[Prompt] = []
    pending: list[str] = []
    for item in ds:
        text = item.get("text", "")
        if len(text) < 200:
            continue
        pending.append(text)
        if len(pending) < 256:
            continue
        for ids in tokenizer(pending, add_special_tokens=False)["input_ids"]:
            if len(ids) < CONTEXT_LENGTH:
                continue
            out.append(Prompt(rung="R2", source="wikipedia_bg",
                              text=tokenizer.decode(ids[:CONTEXT_LENGTH])))
            if len(out) >= n:
                return out
        pending.clear()
    return out


def build_r3_template(tokenizer, content: list[str],
                      n: int = TARGET_N) -> list[Prompt]:
    """Pile spans in unusual chat scaffolds, length-matched to ~128 tokens.

    The content budget is computed per scaffold from its own token overhead, so
    a verbose scaffold does not silently produce a longer prompt than a terse
    one — otherwise the rung would measure length, not template.
    """
    overheads = {}
    for name, tpl in SCAFFOLDS:
        empty = tpl.format("")
        overheads[name] = len(tokenizer(empty, add_special_tokens=False)["input_ids"])

    out: list[Prompt] = []
    for i, span in enumerate(content):
        if len(out) >= n:
            break
        name, tpl = SCAFFOLDS[i % len(SCAFFOLDS)]
        budget = max(8, CONTEXT_LENGTH - overheads[name])
        body = _truncate_to_tokens(tokenizer, span, budget)
        out.append(Prompt(rung="R3", source=f"scaffold:{name}",
                          text=tpl.format(body)))
    return out


def build_r4_jailbreak(n: int = TARGET_N) -> tuple[list[Prompt], dict]:
    """Harmful goals under the eight attack templates validated in `jailbreak/`.

    JailbreakBench ships 100 behaviours, so 2000 distinct jailbreak prompts
    cannot come from it alone. We take goals JailbreakBench-first, fill from
    AdvBench via the same loader the refusal replication used, and cross them
    with the 8 non-control templates. Composition is recorded rather than
    smoothed over.
    """
    from experiments.jailbreak import corpus as jb_corpus
    from experiments.jailbreak import templates as jb_templates

    attacks = [t for t in jb_templates.TEMPLATES if t.name != "plain"]
    n_goals = -(-n // len(attacks))          # ceil

    goals: list[str] = []
    sources: list[str] = []
    try:
        from datasets import load_dataset
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors",
                           split="harmful")
        for row in jbb:
            goals.append(row["Goal"])
            sources.append("jailbreakbench")
    except Exception as e:  # pragma: no cover - network dependent
        print(f"  R4: JailbreakBench unavailable ({e}); AdvBench only")

    harmful, _ = jb_corpus.load_build_prompts(600)
    seen = set(goals)
    for g in harmful:
        if len(goals) >= n_goals:
            break
        if g not in seen:
            goals.append(g)
            sources.append("advbench_or_jbb_dedup")
            seen.add(g)
    goals, sources = goals[:n_goals], sources[:n_goals]

    out: list[Prompt] = []
    for goal, src in zip(goals, sources):
        for t in attacks:
            if len(out) >= n:
                break
            out.append(Prompt(rung="R4", source=f"{src}|{t.name}",
                              text=t.fn(goal)))
    prov = {"n_goals": len(goals), "n_attacks": len(attacks),
            "attacks": [t.name for t in attacks],
            "n_jailbreakbench_goals": sources.count("jailbreakbench")}
    return out[:n], prov


def build_r5_random(vocab_size: int, bos_token_id: int, n: int = TARGET_N,
                    seed: int = 0) -> list[Prompt]:
    """Uniform-random vocab ids. Never decoded to text and re-encoded — that
    round trip drops out-of-vocab ids and smuggles structure back in."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab_size, size=(n, CONTEXT_LENGTH - 1))
    return [Prompt(rung="R5", source="random_tokens",
                   token_ids=[int(bos_token_id)] + [int(v) for v in row])
            for row in ids]
