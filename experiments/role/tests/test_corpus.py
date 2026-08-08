"""Corpus invariants, against the real Qwen3-4B tokenizer.

Needs the tokenizer (~10 MB, cached) but no model weights and no GPU. These are
the checks that decide whether "content held constant across role conditions" —
the assumption the entire experiment rests on — actually holds under Qwen3's BPE.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.role import corpus as C
from experiments.role.metrics import flip_rate  # noqa: F401  (import sanity)

MODEL = "Qwen/Qwen3-4B"


@pytest.fixture(scope="module")
def tk():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


# Deliberately awkward content: leading capital, leading digit, leading quote,
# newlines inside, trailing punctuation, a word that could merge with "\n".
SAMPLES = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "1998 was the year the treaty was signed in Geneva by all twelve members.",
    '"Absolutely not," she said, folding the letter into her coat pocket.',
    "Line one of the document\nLine two of the document\nLine three follows.",
    "def compute(x):\n    return x * 2 + 1  # doubles and offsets the input",
    "assistant behaviour in distributed systems is measured by tail latency.",
    "system calls trap into the kernel and are therefore relatively expensive.",
    "user accounts must be provisioned before the migration window opens.",
    "tool support for the format landed in the 3.2 release last September.",
    "Ω-consistency fails for theories that prove their own inconsistency.",
]


def test_wrapper_table_shape():
    assert len(C.ROLE_WRAPPERS) == 6
    assert set(C.CONDITIONS) == set(C.ROLE_WRAPPERS)
    for cond, (prefix, suffix) in C.ROLE_WRAPPERS.items():
        assert prefix.endswith("\n"), f"{cond}: prefix must end with a newline"
        assert suffix.startswith("<"), (
            f"{cond}: suffix must start with a special token, not text — a "
            "trailing newline would merge with the last content token"
        )


def test_every_suffix_starts_with_a_single_special_token(tk):
    """The no-merge guarantee at the content->suffix boundary."""
    for cond, (_, suffix) in C.ROLE_WRAPPERS.items():
        first = tk.encode(suffix, add_special_tokens=False)[0]
        tok = tk.convert_ids_to_tokens(first)
        assert tok in C.SCAFFOLD_TOKEN_STRINGS, (
            f"{cond}: suffix begins with {tok!r}, not a scaffold special token"
        )


def test_content_ids_identical_across_all_six_conditions(tk):
    """The invariant the whole experiment rests on."""
    corpus = C.build_corpus(tk, SAMPLES)
    assert corpus.n_dropped == 0, f"dropped: {corpus.drop_reasons}"
    assert len(corpus.doc_ids) == len(SAMPLES)
    assert len(corpus.items) == len(SAMPLES) * 6

    by_doc: dict[int, list[C.RoleItem]] = {}
    for it in corpus.items:
        by_doc.setdefault(it.doc_id, []).append(it)

    for doc_id, items in by_doc.items():
        assert len(items) == 6
        ref = items[0].token_ids[items[0].content_start:items[0].content_end]
        standalone = np.asarray(
            tk.encode(SAMPLES[doc_id].strip(), add_special_tokens=False)
        )
        assert np.array_equal(ref, standalone), (
            f"doc {doc_id}: located span differs from standalone tokenization"
        )
        for it in items:
            got = it.token_ids[it.content_start:it.content_end]
            assert np.array_equal(got, ref), (
                f"doc {doc_id} condition {it.condition}: content ids differ"
            )


def test_span_offsets_differ_between_conditions(tk):
    """Sanity: the spans start at different offsets, so the search is doing work.

    If every condition happened to give the same content_start, the subsequence
    search would be untested by the invariant above.
    """
    corpus = C.build_corpus(tk, SAMPLES[:1])
    starts = {it.condition: it.content_start for it in corpus.items}
    assert len(set(starts.values())) > 1, starts
    # cot and tool_native have the longest prefixes.
    assert starts["cot"] > starts["user"]
    assert starts["tool_native"] > starts["user"]


def test_prefix_length_arithmetic_would_have_been_wrong(tk):
    """Documents why the span is found by search and not by len(prefix_tokens).

    If this ever passes for all conditions and samples, prefix arithmetic became
    safe on this tokenizer — but it is still not safe in general, so the search
    stays.
    """
    mismatches = 0
    for content in SAMPLES:
        for cond, (prefix, _) in C.ROLE_WRAPPERS.items():
            naive = len(tk.encode(prefix, add_special_tokens=False))
            ids = tk.encode(prefix + content.strip(), add_special_tokens=False)
            needle = tk.encode(content.strip(), add_special_tokens=False)
            true_start = C._find_subsequence(ids, needle)
            if true_start != naive:
                mismatches += 1
    # Recorded, not asserted either way: the value is the log line.
    print(f"\nprefix-arithmetic mismatches: {mismatches}/{len(SAMPLES) * 6}")


def test_scaffold_ids_cover_every_tag_token(tk):
    ids = C.scaffold_token_ids(tk)
    for s in C.SCAFFOLD_TOKEN_STRINGS:
        assert tk.convert_tokens_to_ids(s) in ids, s
    # Known ids from the handoff, verified against the live tokenizer.
    for expected in (151644, 151645, 151667, 151668, 151665, 151666):
        assert expected in ids
    for word, wid in [("user", 872), ("assistant", 77091),
                      ("system", 8948), ("tool", 14172)]:
        assert wid in ids, word


def test_no_scaffold_token_survives_into_the_labelled_set(tk):
    """Content spans must be free of tag tokens.

    A region separating tag tokens is detecting token identity, not role, so any
    leakage here silently converts the headline metric into a tautology.
    """
    corpus = C.build_corpus(tk, SAMPLES)
    scaffold = C.scaffold_token_ids(tk)
    # The role *words* are ordinary vocabulary and appear in SAMPLES on purpose
    # (see the "assistant behaviour"/"system calls" samples), so exclude only
    # the true specials from this check and assert the words separately.
    specials = {tk.convert_tokens_to_ids(s) for s in C.SCAFFOLD_TOKEN_STRINGS}
    for it in corpus.items:
        span = it.token_ids[it.content_start:it.content_end]
        assert not (set(span.tolist()) & specials), (
            f"{it.condition}: special token inside content span"
        )
    assert specials <= scaffold


def test_content_positions_are_inside_the_prompt_and_bos_shifted(tk):
    corpus = C.build_corpus(tk, SAMPLES)
    for it in corpus.items:
        pos = C.content_position_ids(it)
        assert pos[0] == it.content_start + C.BOS_OFFSET
        assert len(pos) == it.n_content
        # extract_per_position keeps positions 1..L-1 of the BOS-prepended
        # array, so every content position must fall in that window.
        assert pos.min() >= 1
        assert pos.max() <= len(it.token_ids) + C.BOS_OFFSET - 1


def test_tl_bos_and_pad_do_not_collide(tk):
    """The off-by-one that would have been silent.

    TransformerLens sets bos = eos when bos_token is None
    (HookedTransformer.py:793). Qwen3's pad is <|endoftext|> and its eos is
    <|im_end|> — different, so extract_per_position's
    ``lengths = (tokens != pad_id).sum()`` is correct. Had they collided, every
    sequence length and final position would have been off by one.
    """
    assert tk.bos_token is None
    assert tk.eos_token_id == 151645          # <|im_end|>
    assert tk.pad_token_id == 151643          # <|endoftext|>
    assert tk.eos_token_id != tk.pad_token_id


def test_qwen3_has_no_top_level_tool_role(tk):
    """The template fact that forces both tool_native and tool_flat."""
    rendered = tk.apply_chat_template(
        [{"role": "user", "content": "U"}, {"role": "tool", "content": "T"}],
        tokenize=False, add_generation_prompt=False,
    )
    assert "<tool_response>" in rendered
    assert "<|im_start|>tool" not in rendered, (
        "if this fails, Qwen3 gained a flat tool role and tool_native should "
        "be re-derived from the template"
    )


def test_drop_path_on_unfindable_content(tk):
    """Empty content is dropped with a reason rather than crashing."""
    corpus = C.build_corpus(tk, ["", "   ", SAMPLES[0]])
    assert len(corpus.doc_ids) == 1
    assert corpus.n_dropped == 2
    assert corpus.drop_reasons.get("empty") == 2


def test_index_of_round_trips(tk):
    corpus = C.build_corpus(tk, SAMPLES)
    for doc_id in corpus.doc_ids:
        for cond in C.CONDITIONS:
            i = corpus.index_of(doc_id, cond)
            assert corpus.items[i].doc_id == doc_id
            assert corpus.items[i].condition == cond


def test_split_by_document_is_disjoint(tk):
    corpus = C.build_corpus(tk, SAMPLES)
    train, test = C.split_by_document(corpus, n_train=7, seed=0)
    assert len(train) == 7 and len(test) == 3
    assert not set(train) & set(test)
    assert sorted(train + test) == sorted(corpus.doc_ids)


def test_find_subsequence_edges():
    assert C._find_subsequence([1, 2, 3], []) == -1
    assert C._find_subsequence([1, 2], [1, 2, 3]) == -1
    assert C._find_subsequence([9, 1, 2, 3], [1, 2, 3]) == 1
    assert C._find_subsequence([1, 1, 1, 2], [1, 2]) == 2
