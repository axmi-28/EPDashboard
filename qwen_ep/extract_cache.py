"""Extract residual-stream activations from Qwen once and cache them to disk.

The single expensive part of EP is the forward pass. Since the activations are
independent of the clustering resolution ``p``, we run the model *once*, shard
the activations (fp16) plus their prompt/position metadata to disk, and then let
`sweep_p.py` build dictionaries at any number of ``p`` values for free (leader
clustering is CPU/MPS-cheap: ~3 s per pass in the L12 run).

Each shard (`shard_000123.npz`) holds:
  x            (n, D) float16   – residual activations
  prompt_ids   (n,)   int32     – index into this shard's ``prompts``
  position_ids (n,)   int32     – token position within the prompt
  prompts      (m,)   object    – the decoded prompt strings for this shard

A `manifest.json` records model / layer / d_model / shard list / totals.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .adapter import DEFAULT_MODEL_ID, QwenModel, model_tag
from .data import get_text_stream


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--max-tokens", type=int, default=3_000_000)
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16, help="model forward batch")
    ap.add_argument("--prompt-batch-size", type=int, default=64,
                    help="prompts per extractor call")
    ap.add_argument("--shard-acts", type=int, default=100_000,
                    help="approx activations per shard file")
    ap.add_argument("--corpus", default="pile", choices=["pile", "wikitext"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="output dir (default: activations_cache/<slug>)")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("qwen_ep.extract_cache")
    args = parse_args()

    slug = (f"{model_tag(args.model_id)}_L{args.layer}_ctx{args.context_length}"
            f"_mt{args.max_tokens}_seed{args.seed}_{args.corpus}")
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("activations_cache") / slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("cache dir: %s", cache_dir)

    qwen = QwenModel(args.model_id, device=args.device)
    if not (0 <= args.layer < qwen.n_layers):
        raise SystemExit(f"--layer {args.layer} out of range [0, {qwen.n_layers})")

    texts = get_text_stream(args.corpus, qwen.tokenizer,
                            context_length=args.context_length, seed=args.seed)

    # Buffer activations across extractor calls until a shard fills, remapping
    # prompt_ids to be shard-local indices into the shard's prompt list.
    buf_x: list[np.ndarray] = []
    buf_pid: list[np.ndarray] = []
    buf_pos: list[np.ndarray] = []
    buf_prompts: list[str] = []
    buf_offset = 0
    buf_acts = 0

    shard_idx = 0
    total_acts = 0
    total_tokens = 0
    shard_files: list[str] = []
    t0 = time.time()

    def flush() -> None:
        nonlocal shard_idx, buf_x, buf_pid, buf_pos, buf_prompts, buf_offset, buf_acts
        if not buf_x:
            return
        path = cache_dir / f"shard_{shard_idx:06d}.npz"
        np.savez(
            path,
            x=np.concatenate(buf_x).astype(np.float16),
            prompt_ids=np.concatenate(buf_pid).astype(np.int32),
            position_ids=np.concatenate(buf_pos).astype(np.int32),
            prompts=np.array(buf_prompts, dtype=object),
        )
        shard_files.append(path.name)
        shard_idx += 1
        buf_x, buf_pid, buf_pos, buf_prompts = [], [], [], []
        buf_offset = 0
        buf_acts = 0

    def prompt_batches():
        batch: list[str] = []
        for t in texts:
            batch.append(t)
            if len(batch) >= args.prompt_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    for batch in prompt_batches():
        res = qwen.extract_per_position(batch, layer=args.layer,
                                        batch_size=args.batch_size)
        if res.x.shape[0] == 0:
            continue
        buf_x.append(res.x.astype(np.float16))
        buf_pid.append(res.prompt_ids.astype(np.int64) + buf_offset)
        buf_pos.append(res.position_ids.astype(np.int64))
        buf_prompts.extend(batch)
        buf_offset += len(batch)
        buf_acts += res.x.shape[0]
        total_acts += res.x.shape[0]
        total_tokens += res.n_tokens

        if buf_acts >= args.shard_acts:
            flush()

        if total_acts and (total_acts // args.shard_acts) != ((total_acts - res.x.shape[0]) // args.shard_acts):
            log.info("  %d acts | %d tokens | %d shards | %.0fs (%.0f tok/s)",
                     total_acts, total_tokens, shard_idx, time.time() - t0,
                     total_tokens / max(time.time() - t0, 1e-9))

        if total_tokens >= args.max_tokens:
            break

    flush()

    manifest = {
        "model_id": args.model_id,
        "layer": args.layer,
        "d_model": qwen.d_model,
        "context_length": args.context_length,
        "corpus": args.corpus,
        "seed": args.seed,
        "n_activations": total_acts,
        "n_tokens": total_tokens,
        "n_shards": shard_idx,
        "shard_files": shard_files,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("DONE: %d acts, %d shards, %.0fs -> %s",
             total_acts, shard_idx, time.time() - t0, cache_dir)


if __name__ == "__main__":
    main()
