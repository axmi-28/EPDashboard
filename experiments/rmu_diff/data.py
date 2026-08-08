"""The behavioural prompt pool: WMDP (forget) + MMLU (retain), as 4-way MCQ.

The build stream must be the distribution where the behaviour lives — the EP
paper's A.3 (Pile-built cross-checkpoint diff, ambiguous) versus A.6
(behaviour-built, sharp) is unambiguous on that point. RMU's published
capability drop is measured on WMDP multiple-choice, so WMDP-MCQ is the stream.

`bio-forget-corpus` — RMU's *training* forget set — is gated and is deliberately
not used and not substituted for. It is the wrong distribution anyway: we are
asking where the behaviour lives at inference, not what it was trained on.

Two prompt styles, because which one carries the behavioural gap is an empirical
question that Gate 1A answers rather than assumes:

    plain   the lm-eval-harness zero-shot MCQ format zephyr was benchmarked in
    chat    the same body under zephyr's chat template, i.e. how the model is
            actually deployed

Both checkpoints see byte-identical prompt strings; the pool is a plain list, so
`ep._iter_prompt_batches` reproduces the same order for the same seed on both.
"""

from __future__ import annotations

from dataclasses import dataclass

LETTERS = ("A", "B", "C", "D")

# Subject descriptions as the WMDP eval harness renders them.
WMDP_SUBJECTS = {"wmdp-bio": "biology", "wmdp-cyber": "computer security",
                 "wmdp-chem": "chemistry"}


@dataclass
class Prompt:
    text: str
    source: str          # wmdp-bio | wmdp-cyber | wmdp-chem | mmlu
    label: str           # forget | retain | transfer
    subject: str
    answer: int          # 0..3
    n_tokens: int = -1   # with BOS, as the model sees it
    index: int = -1      # position in the canonical (pre-shuffle) pool

    def as_row(self) -> dict:
        return {"index": self.index, "source": self.source, "label": self.label,
                "subject": self.subject, "answer": self.answer,
                "answer_letter": LETTERS[self.answer], "n_tokens": self.n_tokens,
                "n_chars": len(self.text)}


def mcq_body(question: str, choices: list[str], subject: str) -> str:
    """The shared question body, identical across both prompt styles."""
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
    return (f"The following are multiple choice questions (with answers) "
            f"about {subject}.\n\n{question.strip()}\n{opts}")


def render(body: str, style: str, tokenizer=None) -> str:
    """Render a question body in the requested prompt style.

    `plain` ends on "Answer:" so the next token is " A".."D".
    `chat` ends on the assistant generation prompt so the next token is
    "A".."D". Gate 1A resolves the answer token ids empirically rather than
    trusting either assumption.
    """
    if style == "plain":
        return f"{body}\nAnswer:"
    if style == "chat":
        if tokenizer is None:
            raise ValueError("chat style needs a tokenizer for the chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )
    raise ValueError(f"unknown style: {style!r}")


def _take(rows, n: int, rng):
    """Deterministic subsample of `n` rows without replacement."""
    idx = rng.permutation(len(rows))[:n]
    return [rows[int(i)] for i in sorted(idx)]


def _candidates(cfg: str, style: str, tokenizer, label: str) -> list[Prompt]:
    """Every question in one source, rendered and length-measured."""
    from datasets import load_dataset

    if cfg == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")

        def subj(r):
            return str(r["subject"]).replace("_", " ")
    else:
        ds = load_dataset("cais/wmdp", cfg, split="test")

        def subj(r):
            return WMDP_SUBJECTS[cfg]

    out = []
    for r in ds:
        body = mcq_body(r["question"], list(r["choices"]), subj(r))
        text = render(body, style, tokenizer)
        out.append(Prompt(
            text=text, source=cfg, label=label, subject=subj(r),
            answer=int(r["answer"]),
            n_tokens=len(tokenizer(text, add_special_tokens=True)["input_ids"]),
        ))
    return out


def _length_matched(retain: list[Prompt], forget: list[Prompt], n: int, rng,
                    n_bins: int = 20) -> list[Prompt]:
    """Sample `n` retain prompts whose length histogram matches `forget`.

    Prompt length is not a nuisance here, it is a live threat. Gate 0B found two
    of its five rungs separable at AUROC 0.998 by token count alone, and
    WMDP-cyber runs 2.4x the median length of MMLU. Without this, a
    forget-vs-retain region split could be a length artifact wearing a semantic
    label — and this experiment's whole claim is about region *identity*.
    """
    import numpy as np

    fl = np.array([p.n_tokens for p in forget])
    edges = np.unique(np.percentile(fl, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 2:
        return _take(retain, min(n, len(retain)), rng)
    want = np.histogram(fl, bins=edges)[0].astype(float)
    want = np.floor(want / max(want.sum(), 1) * n).astype(int)

    rl = np.array([p.n_tokens for p in retain])
    bin_of = np.clip(np.digitize(rl, edges[1:-1]), 0, len(want) - 1)
    picked_idx: list[int] = []
    for b, k in enumerate(want):
        pool_b = [i for i in range(len(retain)) if bin_of[i] == b]
        picked_idx += _take(pool_b, min(int(k), len(pool_b)), rng)
    if len(picked_idx) < n:
        taken = set(picked_idx)
        rest = [i for i in range(len(retain)) if i not in taken]
        picked_idx += _take(rest, min(n - len(picked_idx), len(rest)), rng)
    return [retain[i] for i in sorted(picked_idx)]


def build_pool(
    *,
    n_bio: int = 1200,
    n_cyber: int = 1200,
    n_mmlu: int = 2400,
    n_chem: int = 0,
    style: str = "chat",
    tokenizer=None,
    seed: int = 12345,
    min_tokens: int = 48,
    max_tokens: int = 256,
    length_match: bool = True,
) -> list[Prompt]:
    """The canonical pool. `seed` selects *which* questions, not their order.

    Streaming order is EP's `seed` and is applied downstream, so the same pool
    can be replayed in several orders without re-extracting activations.

    `min_tokens`/`max_tokens` bound prompt length so no prompt's decision
    position is lost to the extractor's position cap, and so the two arms are
    comparable at all. `length_match` then matches the retain length histogram
    to the forget one — see `_length_matched`.
    """
    import numpy as np

    if tokenizer is None:
        raise ValueError("build_pool needs a tokenizer to measure prompt length")
    rng = np.random.default_rng(seed)

    def in_band(ps):
        return [p for p in ps if min_tokens <= p.n_tokens <= max_tokens]

    out: list[Prompt] = []
    for cfg, n, label in (("wmdp-bio", n_bio, "forget"),
                          ("wmdp-cyber", n_cyber, "forget"),
                          ("wmdp-chem", n_chem, "transfer")):
        if n <= 0:
            continue
        cands = in_band(_candidates(cfg, style, tokenizer, label))
        out += _take(cands, min(n, len(cands)), rng)

    if n_mmlu > 0:
        cands = in_band(_candidates("mmlu", style, tokenizer, "retain"))
        forget = [p for p in out if p.label == "forget"]
        out += (_length_matched(cands, forget, n_mmlu, rng)
                if length_match and forget
                else _take(cands, min(n_mmlu, len(cands)), rng))

    for i, p in enumerate(out):
        p.index = i
    return out


def stream_order(pool: list[Prompt], seed: int) -> list[int]:
    """The prompt order EP's `discover(seed=...)` would impose on this pool.

    `ep.discovery.pipeline._iter_prompt_batches` shuffles a *list* in place with
    `np.random.default_rng(seed)`, so replaying a cached activation set in this
    order is equivalent to having streamed the prompts in it — which is what
    lets the forward pass be paid once for every seed.
    """
    import numpy as np

    idx = list(range(len(pool)))
    np.random.default_rng(seed).shuffle(idx)
    return idx
