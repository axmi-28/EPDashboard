"""One deterministic pass over the Pile that carves three disjoint pools.

The hub dictionaries were built from `monology/pile-uncopyrighted`, split
`train`, streamed, with `ds.shuffle(seed=0, buffer_size=10000)` and a filter of
`len(text) >= 200` then `len(token_ids) >= 128`
(`scripts/build_partitions.py:161-180`). The p=1 build — the largest — consumed
**28,288** accepted prompts.

`scripts/exp_coverage.py` re-reads that *same* stream with the *same* seed and
the same default `--seed 0`, so the paper's "held-out" in-distribution rung is
not held out: it starts over from the top of the stream the dictionary was
built on. We do not repeat that mistake.

We keep seed 0 so the document distribution is identical to the build, and skip
past the build's consumption before taking anything. Everything downstream is
therefore same-distribution but provably disjoint from the build:

    [0, 40000)              skipped — covers the p=1 build's 28,288 prompts
    [40000, 41600)          reference pool   -> S3 coreset + S4 covariance
    [41600, 43600)          R0 in-distribution eval
    [43600, 45600)          R3 template-shift content (wrapped downstream)

Offsets are counted in *accepted* documents (both filters passed), which is the
same unit the build reports, so the skip is directly comparable to 28,288.
"""

from __future__ import annotations

from dataclasses import dataclass

# The p=1 build consumed 28,288 accepted prompts (hub metadata `n_prompts`).
# We skip well past it rather than up to it: the acceptance filter depends on
# the tokenizer, and a margin costs nothing but removes any doubt.
BUILD_PROMPTS_CONSUMED = 28_288
PILE_SKIP = 40_000

REF_DOCS = 1_600      # x ~127 positions = ~203k per-position activations
R0_DOCS = 2_000
R3_DOCS = 2_000

CONTEXT_LENGTH = 128
MIN_CHARS = 200
SHUFFLE_SEED = 0
SHUFFLE_BUFFER = 10_000


@dataclass(frozen=True)
class PilePools:
    """Three disjoint pools of Pile spans, each exactly CONTEXT_LENGTH tokens."""

    reference: list[str]
    r0: list[str]
    r3_content: list[str]
    n_skipped: int
    n_raw_consumed: int


def _accepted_spans(tokenizer, n_needed: int, raw_iter, chunk: int = 512):
    """Yield decoded CONTEXT_LENGTH-token spans, applying the build's filters.

    Tokenisation is batched — a per-document call is the dominant cost when
    skipping 40k documents, and the Rust tokenizer is ~50x faster in batch.
    """
    produced = 0
    n_raw = 0
    pending: list[str] = []
    while produced < n_needed:
        pending.clear()
        while len(pending) < chunk:
            try:
                item = next(raw_iter)
            except StopIteration:
                raise RuntimeError("Pile stream exhausted before pools were filled")
            n_raw += 1
            text = item.get("text", "")
            if len(text) >= MIN_CHARS:
                pending.append(text)
        encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            if len(ids) < CONTEXT_LENGTH:
                continue
            yield tokenizer.decode(ids[:CONTEXT_LENGTH]), n_raw
            produced += 1
            if produced >= n_needed:
                return


def load_pile_pools(tokenizer, *, skip: int = PILE_SKIP,
                    n_reference: int = REF_DOCS, n_r0: int = R0_DOCS,
                    n_r3: int = R3_DOCS, verbose: bool = True) -> PilePools:
    """Stream the Pile once and carve the three disjoint pools."""
    from datasets import load_dataset

    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    raw_iter = iter(ds)

    total = skip + n_reference + n_r0 + n_r3
    spans: list[str] = []
    n_raw = 0
    for span, n_raw in _accepted_spans(tokenizer, total, raw_iter):
        spans.append(span)
        if verbose and len(spans) % 5000 == 0:
            print(f"  pile: {len(spans)}/{total} accepted "
                  f"({n_raw} raw docs read)", flush=True)

    a = skip
    b = a + n_reference
    c = b + n_r0
    return PilePools(
        reference=spans[a:b],
        r0=spans[b:c],
        r3_content=spans[c:c + n_r3],
        n_skipped=skip,
        n_raw_consumed=n_raw,
    )
