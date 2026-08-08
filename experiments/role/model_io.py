"""Model loading, activation harvesting, and the paired-array bookkeeping.

The one thing in here that earns its own module is
:func:`assert_tokenization_alignment`. Every metric consumes ``A[d, c, j]``,
which is built by mapping ``extract_per_position``'s ``(prompt_ids,
position_ids)`` back onto content spans that :mod:`role.corpus` computed with a
*different* tokenizer call. If those two tokenizations disagree by even one
token, nothing raises — the metrics just describe the wrong positions.

They should agree. TransformerLens takes this path on Qwen3:
``get_tokenizer_with_bos`` sees ``bos_token is None`` and returns the tokenizer
untouched with ``add_bos_token=False``; only *afterwards* does
``HookedTransformer`` set ``bos_token = eos_token``
(HookedTransformer.py:793). So ``to_tokens(prepend_bos=True)`` prepends the BOS
by **string concatenation**, and since ``<|im_end|>`` is a special token it
tokenizes to exactly one id. Hence ``BOS_OFFSET == 1`` and the remaining ids
match ``tokenizer.encode(text, add_special_tokens=False)``.

That is a four-step argument about someone else's library, so it is asserted at
runtime rather than trusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from experiments.role import corpus as C

logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
    """Load via TransformerLens, the harness the refusal replication validated.

    ``from_pretrained_no_processing`` and not ``from_pretrained``: the latter
    folds layer norms and centres the writing weights, which changes the residual
    stream that EP partitions.
    """
    import torch
    import transformer_lens as tl

    logger.info("loading %s on %s (%s)", model_name, device, dtype)
    model = tl.HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=getattr(torch, dtype),
    )
    model.eval()
    logger.info(
        "loaded: n_layers=%d d_model=%d n_ctx=%d",
        model.cfg.n_layers, model.cfg.d_model, model.cfg.n_ctx,
    )
    return model


def hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def assert_tokenization_alignment(model, corpus: C.RoleCorpus, n_check: int = 32):
    """Verify TL's tokenization matches the corpus's, id for id.

    Raises rather than warning. A mismatch here does not degrade the experiment,
    it silently relabels it.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(corpus.items), size=min(n_check, len(corpus.items)),
                     replace=False)
    bos_id = model.tokenizer.bos_token_id
    for i in idx:
        item = corpus.items[int(i)]
        tl_ids = model.to_tokens(item.text, prepend_bos=True)[0].cpu().numpy()
        if len(tl_ids) != len(item.token_ids) + C.BOS_OFFSET:
            raise RuntimeError(
                f"tokenization length mismatch on item {i} "
                f"({item.condition}): TL gave {len(tl_ids)} ids, corpus has "
                f"{len(item.token_ids)} + {C.BOS_OFFSET} BOS. If TL stopped "
                "prepending exactly one BOS token, role.corpus.BOS_OFFSET is "
                "wrong and every position id is shifted."
            )
        if tl_ids[0] != bos_id:
            raise RuntimeError(
                f"expected BOS id {bos_id} at position 0, got {tl_ids[0]}"
            )
        if not np.array_equal(tl_ids[C.BOS_OFFSET:], item.token_ids):
            raise RuntimeError(
                f"tokenization content mismatch on item {i} "
                f"({item.condition}). TL and role.corpus disagree, so content "
                "spans do not index what the metrics assume."
            )
    logger.info(
        "tokenization alignment OK on %d sampled prompts (bos_id=%s)",
        len(idx), bos_id,
    )


@dataclass
class PairedAssignments:
    """``A[d, c, j]`` plus the axis labels needed to read it."""

    A: np.ndarray                  # (n_docs, n_conditions, n_content) int32
    doc_ids: np.ndarray            # (n_docs,)
    conditions: tuple[str, ...]
    n_missing: int = 0

    @property
    def shape(self):
        return self.A.shape

    def condition_index(self, name: str) -> int:
        return self.conditions.index(name)

    def flat(self, condition: str) -> np.ndarray:
        return self.A[:, self.condition_index(condition), :].ravel()


def build_paired_assignments(
    corpus: C.RoleCorpus,
    doc_ids: list[int],
    prompt_index: dict[tuple[int, str], int],
    region_ids: np.ndarray,
    ex_prompt_ids: np.ndarray,
    ex_position_ids: np.ndarray,
) -> PairedAssignments:
    """Scatter flat per-position assignments into the paired array.

    ``prompt_index`` maps ``(doc_id, condition)`` to the index of that prompt in
    whatever list was handed to the extractor — the extractor reports
    ``prompt_ids`` in that frame, and the caller may have extracted a subset of
    the corpus (train or test), so the mapping cannot be recomputed here.

    Positions not covered by the extraction are left as -1 and counted. That
    should be zero; a nonzero count means the extractor dropped positions (a
    prompt shorter than 2 tokens, or ``max_positions_per_prompt`` set) and the
    caller must not average over the gaps.
    """
    conditions = corpus.conditions
    items_by_key = {(it.doc_id, it.condition): it for it in corpus.items}
    n_content = items_by_key[(doc_ids[0], conditions[0])].n_content

    A = np.full((len(doc_ids), len(conditions), n_content), -1, dtype=np.int32)

    # Invert prompt_index once: prompt row -> (doc slot, condition slot, start).
    doc_slot = {d: i for i, d in enumerate(doc_ids)}
    n_prompts = max(prompt_index.values()) + 1 if prompt_index else 0
    p_doc = np.full(n_prompts, -1, dtype=np.int64)
    p_cond = np.full(n_prompts, -1, dtype=np.int64)
    p_start = np.full(n_prompts, -1, dtype=np.int64)
    p_end = np.full(n_prompts, -1, dtype=np.int64)
    for (doc_id, cond), row in prompt_index.items():
        if doc_id not in doc_slot:
            continue
        item = items_by_key[(doc_id, cond)]
        if item.n_content != n_content:
            raise ValueError(
                f"doc {doc_id} condition {cond} has {item.n_content} content "
                f"tokens, expected {n_content}. A[d, c, j] must be rectangular "
                "— role.corpus.stream_contents is supposed to enforce this."
            )
        p_doc[row] = doc_slot[doc_id]
        p_cond[row] = conditions.index(cond)
        p_start[row] = item.content_start + C.BOS_OFFSET
        p_end[row] = item.content_end + C.BOS_OFFSET

    keep = (
        (p_doc[ex_prompt_ids] >= 0)
        & (ex_position_ids >= p_start[ex_prompt_ids])
        & (ex_position_ids < p_end[ex_prompt_ids])
    )
    rows = ex_prompt_ids[keep]
    A[p_doc[rows], p_cond[rows], ex_position_ids[keep] - p_start[rows]] = (
        region_ids[keep]
    )

    n_missing = int((A < 0).sum())
    if n_missing:
        logger.error(
            "%d of %d paired slots unfilled — the extractor did not cover every "
            "content position", n_missing, A.size,
        )
    logger.info("paired assignments: A%s, %d content activations used",
                A.shape, int(keep.sum()))
    return PairedAssignments(
        A=A, doc_ids=np.array(doc_ids), conditions=conditions,
        n_missing=n_missing,
    )


def prompt_list(
    corpus: C.RoleCorpus, doc_ids: list[int],
) -> tuple[list[str], dict[tuple[int, str], int]]:
    """Prompts for a document subset, plus the ``(doc, condition) -> row`` map."""
    texts: list[str] = []
    index: dict[tuple[int, str], int] = {}
    wanted = set(doc_ids)
    for it in corpus.items:
        if it.doc_id in wanted:
            index[(it.doc_id, it.condition)] = len(texts)
            texts.append(it.text)
    return texts, index


def _prompt_span_table(corpus: C.RoleCorpus, prompt_index):
    """Per-extraction-row lookups: doc id, condition slot, content span."""
    items_by_key = {(it.doc_id, it.condition): it for it in corpus.items}
    conditions = corpus.conditions
    n_prompts = max(prompt_index.values()) + 1 if prompt_index else 0
    p_doc = np.full(n_prompts, -1, dtype=np.int64)
    p_cond = np.full(n_prompts, -1, dtype=np.int64)
    p_start = np.full(n_prompts, -1, dtype=np.int64)
    p_end = np.full(n_prompts, -1, dtype=np.int64)
    for (doc_id, cond), row in prompt_index.items():
        item = items_by_key[(doc_id, cond)]
        p_doc[row] = doc_id
        p_cond[row] = conditions.index(cond)
        p_start[row] = item.content_start + C.BOS_OFFSET
        p_end[row] = item.content_end + C.BOS_OFFSET
    return p_doc, p_cond, p_start, p_end


def content_mask(
    corpus: C.RoleCorpus,
    prompt_index: dict[tuple[int, str], int],
    ex_prompt_ids: np.ndarray,
    ex_position_ids: np.ndarray,
) -> np.ndarray:
    """Boolean mask selecting content positions out of a flat extraction.

    Exposed separately because callers that assign regions must apply the *same*
    mask to the index arrays they later hand to
    :func:`build_paired_assignments`. Assigning content-only activations and then
    scattering with the full extraction's ``prompt_ids`` is a shape mismatch —
    caught by a dry run, and it would have been an ``IndexError`` on the pod after
    the weight download.
    """
    _, _, p_start, p_end = _prompt_span_table(corpus, prompt_index)
    return (
        (ex_position_ids >= p_start[ex_prompt_ids])
        & (ex_position_ids < p_end[ex_prompt_ids])
    )


def content_activations(
    corpus: C.RoleCorpus,
    prompt_index: dict[tuple[int, str], int],
    x: np.ndarray,
    ex_prompt_ids: np.ndarray,
    ex_position_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter a flat extraction down to content positions only.

    Returns ``(x_content, doc_id, condition_slot, j, mask)``. Scaffold positions
    are dropped here and nowhere else — a region that separates
    ``<|im_start|>user`` from ``<|im_start|>tool`` is detecting token identity,
    and letting those into a role statistic turns it into a tautology.
    """
    p_doc, p_cond, p_start, _ = _prompt_span_table(corpus, prompt_index)
    keep = content_mask(corpus, prompt_index, ex_prompt_ids, ex_position_ids)
    rows = ex_prompt_ids[keep]
    return (
        x[keep],
        p_doc[rows],
        p_cond[rows],
        ex_position_ids[keep] - p_start[rows],
        keep,
    )


def exemplar_matrix(dictionary) -> np.ndarray:
    """(K, d) matrix of unit exemplar directions."""
    return np.stack([
        p.exemplar_direction for p in dictionary.partitions
    ]).astype(np.float32)


def mean_matrix(dictionary) -> np.ndarray:
    """(K, d) matrix of spherical-mean member directions."""
    return np.stack([
        p.mean_member_direction for p in dictionary.partitions
    ]).astype(np.float32)


def member_counts(dictionary) -> np.ndarray:
    return np.array([p.member_count for p in dictionary.partitions],
                    dtype=np.int64)
