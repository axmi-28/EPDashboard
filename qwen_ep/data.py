"""Text streaming for EP dictionary construction.

The paper builds EP dictionaries over Pile activations. We stream the same
corpus (``monology/pile-uncopyrighted``) lazily so we never materialise it on
disk, decode fixed-length token windows back to text, and hand plain strings to
the extractor (which re-tokenises). This mirrors ``scripts/build_partitions.py``
``stream_batches`` in ``sampling_mode="full"`` but is trimmed to what we need.

A small ``wikitext`` fallback is provided for offline / smoke-test use.
"""

from __future__ import annotations

import logging
from typing import Iterator

logger = logging.getLogger(__name__)

PILE_DATASET = "monology/pile-uncopyrighted"


def stream_pile_texts(
    tokenizer,
    context_length: int = 128,
    min_chars: int = 200,
    seed: int = 0,
    buffer_size: int = 10_000,
    max_texts: int | None = None,
) -> Iterator[str]:
    """Yield decoded ``context_length``-token windows from the Pile.

    Each yielded string is the detokenised prefix ``ids[:context_length]`` of a
    source document long enough to fill the window. Documents shorter than
    ``min_chars`` or ``context_length`` tokens are skipped.
    """
    from datasets import load_dataset

    ds = load_dataset(PILE_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    n = 0
    for item in ds:
        text = item.get("text", "")
        if len(text) < min_chars:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < context_length:
            continue
        yield tokenizer.decode(ids[:context_length])
        n += 1
        if max_texts is not None and n >= max_texts:
            return


def stream_wikitext_texts(
    tokenizer,
    context_length: int = 128,
    min_chars: int = 200,
    max_texts: int | None = None,
) -> Iterator[str]:
    """Offline-friendly fallback corpus (wikitext-103 raw)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    n = 0
    for item in ds:
        text = item.get("text", "")
        if len(text) < min_chars:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < context_length:
            continue
        yield tokenizer.decode(ids[:context_length])
        n += 1
        if max_texts is not None and n >= max_texts:
            return


def get_text_stream(name: str, tokenizer, **kwargs) -> Iterator[str]:
    if name == "pile":
        return stream_pile_texts(tokenizer, **kwargs)
    if name == "wikitext":
        return stream_wikitext_texts(tokenizer, **kwargs)
    raise ValueError(f"unknown corpus {name!r} (options: pile, wikitext)")
