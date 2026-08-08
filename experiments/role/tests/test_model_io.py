"""Paired-array bookkeeping, on a hand-built corpus.

No tokenizer and no model: `RoleItem`/`RoleCorpus` are constructed directly so
the scatter can be checked against assignments whose correct destination is known
by construction. This is the step where an off-by-one silently relabels every
metric, so the spans here deliberately differ per condition (as they do in the
real corpus: 3/3/3/5/5/3).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.role import corpus as C
from experiments.role import model_io as MIO

CONDS = ("user", "assistant", "cot")
STARTS = {"user": 3, "assistant": 3, "cot": 5}
N_CONTENT = 4


def _corpus(n_docs: int = 3) -> C.RoleCorpus:
    items = []
    for d in range(n_docs):
        for cond in CONDS:
            start = STARTS[cond]
            # token ids: scaffold prefix, then content 1000+d*10+j, then suffix
            ids = list(range(900, 900 + start))
            ids += [1000 + d * 10 + j for j in range(N_CONTENT)]
            ids += [999]
            items.append(
                C.RoleItem(
                    doc_id=d, condition=cond, text=f"doc{d}-{cond}",
                    token_ids=np.array(ids, dtype=np.int64),
                    content_start=start, content_end=start + N_CONTENT,
                )
            )
    return C.RoleCorpus(items=items, conditions=CONDS,
                        doc_ids=list(range(n_docs)))


def test_prompt_list_covers_requested_docs_only():
    corpus = _corpus(4)
    texts, index = MIO.prompt_list(corpus, [1, 3])
    assert len(texts) == 2 * len(CONDS)
    assert set(index) == {(d, c) for d in (1, 3) for c in CONDS}
    for (doc_id, cond), row in index.items():
        assert texts[row] == f"doc{doc_id}-{cond}"


def test_build_paired_assignments_exact_scatter():
    """Region id encodes its own destination, so misplacement is detectable."""
    corpus = _corpus(3)
    doc_ids = [0, 1, 2]
    texts, index = MIO.prompt_list(corpus, doc_ids)

    # Emit every position of every prompt, including scaffold.
    ex_prompt, ex_pos, region = [], [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        n_tok = len(item.token_ids) + C.BOS_OFFSET
        for pos in range(1, n_tok):        # extractor skips position 0
            ex_prompt.append(row)
            ex_pos.append(pos)
            j = pos - (item.content_start + C.BOS_OFFSET)
            if 0 <= j < N_CONTENT:
                # unique id per (doc, cond, j)
                region.append(
                    100 * doc_id + 10 * CONDS.index(cond) + j
                )
            else:
                region.append(-999)       # scaffold; must be dropped

    pa = MIO.build_paired_assignments(
        corpus, doc_ids, index,
        region_ids=np.array(region),
        ex_prompt_ids=np.array(ex_prompt),
        ex_position_ids=np.array(ex_pos),
    )

    assert pa.n_missing == 0
    assert pa.A.shape == (3, len(CONDS), N_CONTENT)
    for d in range(3):
        for ci in range(len(CONDS)):
            for j in range(N_CONTENT):
                assert pa.A[d, ci, j] == 100 * d + 10 * ci + j, (d, ci, j)
    assert -999 not in pa.A, "scaffold assignment leaked into the paired array"


def test_build_paired_assignments_reports_missing_positions():
    """A truncated extraction must be counted, not silently averaged over."""
    corpus = _corpus(2)
    doc_ids = [0, 1]
    _, index = MIO.prompt_list(corpus, doc_ids)
    # Supply only the first content position of each prompt.
    ex_prompt, ex_pos, region = [], [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        ex_prompt.append(row)
        ex_pos.append(item.content_start + C.BOS_OFFSET)
        region.append(7)
    pa = MIO.build_paired_assignments(
        corpus, doc_ids, index, np.array(region),
        np.array(ex_prompt), np.array(ex_pos),
    )
    assert pa.n_missing == pa.A.size - len(region)
    assert (pa.A[:, :, 0] == 7).all()
    assert (pa.A[:, :, 1:] == -1).all()


def test_build_paired_assignments_subset_of_docs():
    """Extracting a test split must not index train documents."""
    corpus = _corpus(4)
    doc_ids = [2, 3]
    _, index = MIO.prompt_list(corpus, doc_ids)
    ex_prompt, ex_pos, region = [], [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        for j in range(N_CONTENT):
            ex_prompt.append(row)
            ex_pos.append(item.content_start + C.BOS_OFFSET + j)
            region.append(doc_id)
    pa = MIO.build_paired_assignments(
        corpus, doc_ids, index, np.array(region),
        np.array(ex_prompt), np.array(ex_pos),
    )
    assert pa.n_missing == 0
    assert (pa.A[0] == 2).all() and (pa.A[1] == 3).all()
    assert list(pa.doc_ids) == [2, 3]


def test_build_paired_assignments_rejects_ragged_content():
    corpus = _corpus(2)
    corpus.items[0].content_end -= 1          # make one doc/condition shorter
    _, index = MIO.prompt_list(corpus, [0, 1])
    with pytest.raises(ValueError, match="rectangular"):
        MIO.build_paired_assignments(
            corpus, [0, 1], index, np.array([0]), np.array([0]), np.array([1]),
        )


def test_content_activations_drops_scaffold():
    corpus = _corpus(2)
    doc_ids = [0, 1]
    _, index = MIO.prompt_list(corpus, doc_ids)
    ex_prompt, ex_pos, rows = [], [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        n_tok = len(item.token_ids) + C.BOS_OFFSET
        for pos in range(1, n_tok):
            ex_prompt.append(row)
            ex_pos.append(pos)
            rows.append(float(pos))
    x = np.array(rows, dtype=np.float32)[:, None]

    xc, doc, cond, j, keep = MIO.content_activations(
        corpus, index, x, np.array(ex_prompt), np.array(ex_pos),
    )
    assert keep.shape == (len(ex_pos),)
    expected = len(index) * N_CONTENT
    assert len(xc) == expected
    assert set(doc.tolist()) == {0, 1}
    assert set(cond.tolist()) == set(range(len(CONDS)))
    assert j.min() == 0 and j.max() == N_CONTENT - 1
    # Every kept row's activation value equals its absolute position, which must
    # lie inside that prompt's content span.
    assert (xc[:, 0] >= min(STARTS.values()) + C.BOS_OFFSET).all()


def test_paired_assignments_accessors():
    corpus = _corpus(2)
    A = np.arange(2 * len(CONDS) * N_CONTENT).reshape(2, len(CONDS), N_CONTENT)
    pa = MIO.PairedAssignments(A=A, doc_ids=np.array([0, 1]), conditions=CONDS)
    assert pa.condition_index("cot") == 2
    assert pa.flat("user").tolist() == A[:, 0, :].ravel().tolist()
    assert pa.shape == A.shape


def test_bos_offset_is_one():
    """Documented as the outcome of a four-step argument about TL internals.

    `model_io.assert_tokenization_alignment` re-checks it against the live model;
    this only pins the constant so a change is deliberate.
    """
    assert C.BOS_OFFSET == 1


def test_content_mask_agrees_with_content_activations():
    """The masks must be the same object, or region ids scatter to wrong slots.

    Regression test for the bug the CPU dry run caught: `dictionary.assign` was
    run on content-only activations while the paired scatter received the *full*
    extraction's prompt/position arrays, so the boolean index lengths disagreed
    (1152 vs 1392). It raised, but a corpus where the counts happened to match
    would have silently mislabelled every token.
    """
    corpus = _corpus(3)
    doc_ids = [0, 1, 2]
    _, index = MIO.prompt_list(corpus, doc_ids)

    ex_prompt, ex_pos = [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        for pos in range(1, len(item.token_ids) + C.BOS_OFFSET):
            ex_prompt.append(row)
            ex_pos.append(pos)
    ex_prompt = np.array(ex_prompt)
    ex_pos = np.array(ex_pos)
    x = np.arange(len(ex_pos), dtype=np.float32)[:, None]

    mask = MIO.content_mask(corpus, index, ex_prompt, ex_pos)
    xc, doc, cond, j, keep = MIO.content_activations(
        corpus, index, x, ex_prompt, ex_pos,
    )
    assert np.array_equal(mask, keep)
    assert len(xc) == int(mask.sum()) < len(ex_pos), "scaffold must be dropped"

    # The full round trip the driver performs: assign on masked activations,
    # scatter with identically-masked index arrays.
    region = np.arange(len(xc))
    pa = MIO.build_paired_assignments(
        corpus, doc_ids, index, region, ex_prompt[keep], ex_pos[keep],
    )
    assert pa.n_missing == 0
    assert sorted(pa.A.ravel().tolist()) == sorted(region.tolist())


def test_unmasked_index_arrays_would_raise():
    """Passing the full extraction's ids alongside masked regions must fail loudly."""
    corpus = _corpus(2)
    doc_ids = [0, 1]
    _, index = MIO.prompt_list(corpus, doc_ids)
    ex_prompt, ex_pos = [], []
    for (doc_id, cond), row in index.items():
        item = next(it for it in corpus.items
                    if it.doc_id == doc_id and it.condition == cond)
        for pos in range(1, len(item.token_ids) + C.BOS_OFFSET):
            ex_prompt.append(row)
            ex_pos.append(pos)
    ex_prompt, ex_pos = np.array(ex_prompt), np.array(ex_pos)
    mask = MIO.content_mask(corpus, index, ex_prompt, ex_pos)
    region_masked = np.zeros(int(mask.sum()), dtype=np.int64)
    with pytest.raises(IndexError):
        MIO.build_paired_assignments(
            corpus, doc_ids, index, region_masked, ex_prompt, ex_pos,
        )
