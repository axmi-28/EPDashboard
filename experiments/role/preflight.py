"""Preflight: corpus construction against real data, no model weights.

Run this before provisioning a GPU. It answers the two questions that a
structurally-valid-but-meaningless run cannot distinguish itself from:

1. Did the text loaders actually return data, or fall through? (``_load_harmful``
   silently dropped to 17 embedded templates and produced a valid, meaningless
   refusal run. The lesson is to check the corpus, not the results.)
2. What is the real content-constancy drop rate on unfiltered pretraining text?
   The hand-written samples in ``role/tests/test_corpus.py`` all pass; C4
   contains code, tables, CJK, and mojibake that they do not represent.

Usage:
    python -m experiments.role.preflight --n-docs 200 --dataset c4
"""

from __future__ import annotations

import argparse
import collections
import logging

import numpy as np

from experiments.role import corpus as C

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dataset", default="c4", choices=("c4", "pile"))
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--n-content", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-oasst", action="store_true",
                    help="Also probe OpenAssistant/oasst1 (the Tier 0 gate's "
                         "real user-style text).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.model)
    logger.info("tokenizer: %s (vocab %d)", args.model, len(tk))

    # --- 1. the loader actually returned data ---
    contents = C.stream_contents(
        tk, n_docs=args.n_docs, n_content=args.n_content,
        dataset=args.dataset, seed=args.seed,
    )
    lens = np.array([len(tk.encode(c, add_special_tokens=False))
                     for c in contents])
    logger.info(
        "%s: %d content strings, token length min/median/max = %d/%d/%d",
        args.dataset, len(contents), lens.min(), int(np.median(lens)), lens.max(),
    )
    # The paired assignment array A[d, c, j] must be rectangular along j.
    rectangular = bool((lens == args.n_content).all())
    if not rectangular:
        logger.error(
            "content lengths are ragged (%d distinct); A[d, c, j] cannot be "
            "built. stream_contents is supposed to have filtered these.",
            len(set(lens.tolist())),
        )
    if len(set(contents)) != len(contents):
        dupes = len(contents) - len(set(contents))
        logger.warning("%d duplicate content strings — shuffle buffer too small?",
                       dupes)

    # --- 2. content constancy on real text ---
    corpus = C.build_corpus(tk, contents)
    kept = len(corpus.doc_ids)
    drop_rate = corpus.n_dropped / max(len(contents), 1)
    logger.info("constancy: kept %d/%d docs (drop rate %.3f) reasons=%s",
                kept, len(contents), drop_rate, dict(corpus.drop_reasons))

    # Re-verify the invariant explicitly rather than trusting build_corpus.
    by_doc: dict[int, list[C.RoleItem]] = collections.defaultdict(list)
    for it in corpus.items:
        by_doc[it.doc_id].append(it)
    bad = 0
    for doc_id, items in by_doc.items():
        ref = items[0].token_ids[items[0].content_start:items[0].content_end]
        for it in items[1:]:
            got = it.token_ids[it.content_start:it.content_end]
            if not np.array_equal(got, ref):
                bad += 1
    logger.info("constancy recheck: %d violating (doc, condition) pairs", bad)

    # --- 3. scaffold leakage ---
    specials = {tk.convert_tokens_to_ids(s) for s in C.SCAFFOLD_TOKEN_STRINGS}
    leaks = sum(
        1 for it in corpus.items
        if set(it.token_ids[it.content_start:it.content_end].tolist()) & specials
    )
    logger.info("scaffold leakage: %d/%d spans contain a special token",
                leaks, len(corpus.items))

    # --- 4. prompt-length budget, which sets the activation count ---
    prompt_lens = np.array([len(it.token_ids) for it in corpus.items])
    n_content_tokens = sum(it.n_content for it in corpus.items)
    logger.info(
        "budget: %d prompts, prompt tokens median %d max %d, "
        "%d labelled content activations (+%d scaffold)",
        len(corpus.items), int(np.median(prompt_lens)), prompt_lens.max(),
        n_content_tokens, int(prompt_lens.sum()) - n_content_tokens,
    )

    # --- 5. per-condition span offsets, a cheap tag-length diagnostic ---
    offsets = collections.defaultdict(list)
    for it in corpus.items:
        offsets[it.condition].append(it.content_start)
    for cond in C.CONDITIONS:
        vals = set(offsets[cond])
        logger.info("  %-12s content_start=%s", cond,
                    sorted(vals) if len(vals) <= 4 else f"{len(vals)} distinct")

    # --- 6. OASST1, the Tier 0 gate's real user text ---
    if args.check_oasst:
        from datasets import load_dataset
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        prompter = [
            r["text"] for r in ds.select(range(min(5000, len(ds))))
            if r.get("role") == "prompter" and len(r.get("text", "")) > 120
        ]
        logger.info("oasst1: %d prompter messages >120 chars in first 5k rows",
                    len(prompter))
        if len(prompter) < 200:
            logger.error("oasst1 yielded too few user-style messages for the "
                         "Tier 0 gate")
            return 1

    ok = bad == 0 and leaks == 0 and drop_rate < 0.10 and rectangular
    logger.info("PREFLIGHT %s", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
