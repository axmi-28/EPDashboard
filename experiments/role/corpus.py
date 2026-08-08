"""The paired role corpus: identical content wrapped in six role scaffolds.

The whole experiment rests on holding content constant across role conditions
(arXiv 2603.12277 Figure 5). Tags and style are perfectly correlated in real
conversation data, so a probe — or an EP region — trained on real chat logs
learns the confound. Wrapping the *same* pretraining text in each role tag is
the only way to isolate the tag's geometric signature.

Two things this module is careful about, both of which silently corrupt the
experiment if got wrong:

1. **Constancy is checked, not assumed.** BPE can merge across the
   prefix/content boundary, so the content token ids are located by searching
   for the standalone tokenization of ``X`` as a subsequence of each wrapped
   tokenization, and a document is kept only if all six conditions agree on the
   content ids *and* their length. Computing the span from ``len(prefix_tokens)``
   instead would be off by one on exactly the documents where a merge happened,
   with no error.

2. **Scaffold tokens never enter a labelled statistic.** The tag tokens differ
   across conditions by construction, so a region that separates them is
   detecting token identity, not role. They are still fed to the EP build (the
   build sees whole sequences, as upstream does) but are masked out of every
   metric.

Qwen3 note: there is no top-level ``tool`` role. ``apply_chat_template`` renders
a tool message as ``<|im_start|>user\\n<tool_response>\\n…<|im_end|>`` — tool
output *is* a user turn. Hence both ``tool_native`` (what the template actually
does) and ``tool_flat`` (what the paper's flat five-tag abstraction assumes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Every suffix begins with a SPECIAL token, never with a newline. A trailing
# "\n" before </think> would merge with the final content token and break
# constancy for that one condition — the hardest kind of bug to see, because
# five of six conditions would still agree. Every prefix ends with "\n", so the
# prefix->content boundary is the same character in all six.
ROLE_WRAPPERS: dict[str, tuple[str, str]] = {
    "system": ("<|im_start|>system\n", "<|im_end|>"),
    "user": ("<|im_start|>user\n", "<|im_end|>"),
    "assistant": ("<|im_start|>assistant\n", "<|im_end|>"),
    "cot": ("<|im_start|>assistant\n<think>\n", "</think><|im_end|>"),
    "tool_native": (
        "<|im_start|>user\n<tool_response>\n",
        "</tool_response><|im_end|>",
    ),
    "tool_flat": ("<|im_start|>tool\n", "<|im_end|>"),
}

CONDITIONS: tuple[str, ...] = tuple(ROLE_WRAPPERS)

# Coarse role each condition is *tagged* as, for the paper's Table 1 cells.
# tool_native is tagged `user` at the turn level and `tool` at the delimiter
# level; that ambiguity is the point, so it gets its own label and the
# turn-level reading is recorded separately.
CONDITION_ROLE: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "cot": "cot",
    "tool_native": "tool",
    "tool_flat": "tool",
}

# Token strings that must never survive into a labelled statistic. The bare role
# words are included because `<|im_start|>system` tokenizes as the special token
# followed by the ordinary word "system".
SCAFFOLD_TOKEN_STRINGS: tuple[str, ...] = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
)
ROLE_WORDS: tuple[str, ...] = ("system", "user", "assistant", "tool")


@dataclass
class RoleItem:
    """One (document, condition) prompt with its located content span.

    ``content_start``/``content_end`` index the wrapped tokenization *without*
    a prepended BOS. ``extract_per_position`` calls
    ``to_tokens(prepend_bos=True)`` and keeps positions ``1..L-1``, so the
    position id of content token ``j`` in that frame is
    ``content_start + j + BOS_OFFSET``. Use :func:`content_position_ids`.
    """

    doc_id: int
    condition: str
    text: str
    token_ids: np.ndarray          # (L,) wrapped, no BOS
    content_start: int
    content_end: int

    @property
    def n_content(self) -> int:
        return self.content_end - self.content_start


@dataclass
class RoleCorpus:
    """The full paired corpus. ``items`` is ordered doc-major, condition-minor."""

    items: list[RoleItem] = field(default_factory=list)
    conditions: tuple[str, ...] = CONDITIONS
    doc_ids: list[int] = field(default_factory=list)
    n_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def texts(self) -> list[str]:
        return [it.text for it in self.items]

    def index_of(self, doc_id: int, condition: str) -> int:
        """Index into ``items`` / ``texts`` for one (doc, condition) pair."""
        d = self.doc_ids.index(doc_id)
        return d * len(self.conditions) + self.conditions.index(condition)


# TransformerLens prepends a BOS token in `extract_per_position`. Qwen3 has no
# real BOS, so TL falls back to eos (`<|im_end|>`, 151645) — see
# HookedTransformer.py:793. It sits at position 0, is skipped by the extractor,
# and is identical across all six conditions.
BOS_OFFSET = 1


def content_position_ids(item: RoleItem) -> np.ndarray:
    """Position ids (in the BOS-prepended frame) of ``item``'s content tokens."""
    return np.arange(
        item.content_start + BOS_OFFSET,
        item.content_end + BOS_OFFSET,
        dtype=np.int64,
    )


def _find_subsequence(hay: Sequence[int], needle: Sequence[int]) -> int:
    """First index where ``needle`` occurs in ``hay``, or -1.

    Numpy-free and short because ``needle`` is ~96 long and ``hay`` ~110; a
    stride-trick version would be slower to read and no faster in practice.
    """
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    first = needle[0]
    for i in range(n - m + 1):
        if hay[i] == first and list(hay[i:i + m]) == list(needle):
            return i
    return -1


def scaffold_token_ids(tokenizer) -> set[int]:
    """Ids that must be excluded from every labelled statistic."""
    ids: set[int] = set()
    for s in SCAFFOLD_TOKEN_STRINGS:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0:
            ids.add(int(tid))
    for w in ROLE_WORDS:
        for variant in (w, "\n" + w):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
    # Newline variants that the scaffold introduces on its own.
    for s in ("\n", "\n\n"):
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return ids


def build_corpus(
    tokenizer,
    contents: Iterable[str],
    conditions: Sequence[str] = CONDITIONS,
) -> RoleCorpus:
    """Wrap each content string in every condition, locating the content span.

    A document is kept only if the standalone tokenization of its content is
    found as an exact subsequence in *all* conditions. Anything else is dropped
    with a logged reason — a partially-conforming document would put
    non-identical tokens into a paired comparison.
    """
    unknown = set(conditions) - set(ROLE_WRAPPERS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")

    corpus = RoleCorpus(conditions=tuple(conditions))
    for doc_id, content in enumerate(contents):
        content = content.strip()
        if not content:
            corpus.n_dropped += 1
            corpus.drop_reasons["empty"] = corpus.drop_reasons.get("empty", 0) + 1
            continue

        needle = tokenizer.encode(content, add_special_tokens=False)
        staged: list[RoleItem] = []
        reason = None
        for cond in conditions:
            prefix, suffix = ROLE_WRAPPERS[cond]
            text = prefix + content + suffix
            ids = tokenizer.encode(text, add_special_tokens=False)
            start = _find_subsequence(ids, needle)
            if start < 0:
                reason = f"span_not_found:{cond}"
                break
            staged.append(
                RoleItem(
                    doc_id=doc_id, condition=cond, text=text,
                    token_ids=np.asarray(ids, dtype=np.int64),
                    content_start=start, content_end=start + len(needle),
                )
            )

        if reason is not None:
            corpus.n_dropped += 1
            corpus.drop_reasons[reason] = corpus.drop_reasons.get(reason, 0) + 1
            continue

        # Redundant given the subsequence search, but this is the invariant the
        # whole experiment depends on, so assert it against the sliced ids
        # rather than trusting the search.
        ref = staged[0].token_ids[staged[0].content_start:staged[0].content_end]
        if not all(
            np.array_equal(
                it.token_ids[it.content_start:it.content_end], ref
            )
            for it in staged
        ):
            corpus.n_dropped += 1
            corpus.drop_reasons["content_mismatch"] = (
                corpus.drop_reasons.get("content_mismatch", 0) + 1
            )
            continue

        corpus.items.extend(staged)
        corpus.doc_ids.append(doc_id)

    logger.info(
        "corpus: %d docs kept, %d dropped (%s); %d prompts, %d content tokens/doc",
        len(corpus.doc_ids), corpus.n_dropped, corpus.drop_reasons or "none",
        len(corpus.items),
        corpus.items[0].n_content if corpus.items else 0,
    )
    return corpus


def content_token_ids(corpus: RoleCorpus) -> dict[int, np.ndarray]:
    """``doc_id -> content token ids``, one entry per document.

    Content is identical across conditions by construction (checked in
    :func:`build_corpus`), so this is a property of the document alone. Needed
    for ``I(region; content)``, the metric that says whether the partition is
    tracking token semantics rather than role.
    """
    out: dict[int, np.ndarray] = {}
    for it in corpus.items:
        if it.doc_id not in out:
            out[it.doc_id] = it.token_ids[it.content_start:it.content_end]
    return out


def split_by_document(
    corpus: RoleCorpus, n_train: int, seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split doc ids train/test.

    By document, never by token: the same content string appears in every
    condition and at every position, so a token-level split leaks the content
    across the split and any probe reports ~99% at every layer.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(corpus.doc_ids)
    rng.shuffle(ids)
    return sorted(ids[:n_train].tolist()), sorted(ids[n_train:].tolist())


# --- text sources ---------------------------------------------------------

C4_DATASET = ("allenai/c4", "en")
PILE_DATASET = "monology/pile-uncopyrighted"


def stream_contents(
    tokenizer,
    n_docs: int,
    n_content: int = 96,
    dataset: str = "c4",
    seed: int = 0,
    min_chars: int = 400,
    buffer_size: int = 10_000,
) -> list[str]:
    """Fixed-length content strings from non-instruct pretraining text.

    Mirrors ``qwen_ep/data.py``: take a document, keep the first ``n_content``
    tokens, decode back to a string.

    Decode/re-encode does **not** always round-trip — on C4 it returns 95 or 96
    tokens for a nominal 96 — so documents whose re-encoded length is not exactly
    ``n_content`` are skipped. Constancy across conditions would survive the
    variation (the string is what is held fixed), but the paired assignment array
    ``A[d, c, j]`` that every metric consumes has to be rectangular along ``j``.
    Enforcing it here is cheaper than making every metric ragged, and it costs
    only a few extra streamed documents.

    Raises rather than falling back to embedded text. ``_load_harmful``'s silent
    fallback to 17 templates produced a structurally valid, meaningless run; the
    lesson is that a corpus loader must fail loudly.
    """
    from datasets import load_dataset

    if dataset == "c4":
        ds = load_dataset(
            C4_DATASET[0], C4_DATASET[1], split="train", streaming=True,
        )
    elif dataset == "pile":
        ds = load_dataset(PILE_DATASET, split="train", streaming=True)
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected c4|pile)")
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    out: list[str] = []
    n_seen = n_ragged = 0
    for item in ds:
        n_seen += 1
        text = item.get("text", "")
        if len(text) < min_chars:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < n_content:
            continue
        s = tokenizer.decode(ids[:n_content]).strip()
        if not s:
            continue
        if len(tokenizer.encode(s, add_special_tokens=False)) != n_content:
            n_ragged += 1
            continue
        out.append(s)
        if len(out) >= n_docs:
            break

    logger.info(
        "%s: %d content strings from %d documents (%d dropped for not "
        "re-encoding to exactly %d tokens)",
        dataset, len(out), n_seen, n_ragged, n_content,
    )

    if len(out) < n_docs:
        raise RuntimeError(
            f"{dataset}: got {len(out)} content strings, wanted {n_docs}. The "
            "stream ended or every document was filtered — check egress before "
            "reading any downstream number."
        )
    return out


def iter_content_batches(contents: Sequence[str]) -> Iterator[str]:
    """Trivial passthrough kept so callers do not index ``contents`` directly."""
    yield from contents
