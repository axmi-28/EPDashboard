"""Stage 1: build the ladder and extract every activation exactly once.

Splitting this from scoring is deliberate. All six scorers read the same
cached activations, so the ~30 minutes of forward passes is paid once and the
scoring sweep (5 percentiles x 6 scorers x 6 rungs) is then seconds. It also
means a scoring bug costs nothing to fix.

Writes:
    artifacts/runs/monitor/corpus.json          the ladder, with provenance
    artifacts/runs/monitor/eval.npz             (N, D) final-position acts + entropy + lengths
    artifacts/runs/monitor/refpool.npy          (M, D) per-position Pile acts for S3/S4
    artifacts/runs/monitor/mps_validation.json  MPS-vs-CPU numerical agreement check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import corpora, extract, pile


def validate_mps(model, items, layer: int) -> dict:
    """TransformerLens warns that MPS 'may produce silently incorrect results'.

    That warning is not idle and the whole experiment rides on these
    activations, so compare a small batch against a CPU forward of the same
    model before trusting 12k of them.
    """
    import torch

    sub = items[:8]
    got = extract.extract(model, sub, layer=layer, batch_size=8, verbose=False)
    cpu_model = model.to("cpu")
    try:
        ref = extract.extract(cpu_model, sub, layer=layer, batch_size=8,
                              verbose=False)
    finally:
        model.to(torch.device("mps"))

    num = float((got.x * ref.x).sum(1).mean())
    den = float((np.linalg.norm(got.x, axis=1) * np.linalg.norm(ref.x, axis=1)).mean())
    return {
        "n": len(sub),
        "cosine_mps_vs_cpu": num / max(den, 1e-12),
        "max_abs_diff": float(np.abs(got.x - ref.x).max()),
        "rel_fro_error": float(np.linalg.norm(got.x - ref.x)
                               / max(np.linalg.norm(ref.x), 1e-12)),
        "entropy_max_abs_diff": float(np.abs(got.entropy_max
                                             - ref.entropy_max).max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-per-rung", type=int, default=corpora.TARGET_N)
    ap.add_argument("--out", type=Path, default=Path("artifacts/runs/monitor"))
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--skip-corpus", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out / "corpus.json"
    t_start = time.time()

    print("loading model ...", flush=True)
    model = extract.load_model(args.model, device=args.device, dtype=args.dtype)
    tok = model.tokenizer

    ref_texts_path = args.out / "refpool_texts.json"
    reuse_corpus = corpus_path.exists() and args.skip_corpus

    # The Pile pass is deterministic (fixed seed, fixed skip), so re-running it
    # after a crash reproduces the same three pools. It costs ~200s, which is
    # cheaper than persisting and re-validating 45,600 spans.
    if reuse_corpus and ref_texts_path.exists():
        pools = None
        ref_texts = json.loads(ref_texts_path.read_text())
        corpus = corpora.Corpus.load(corpus_path)
        print(f"reusing corpus {corpus.counts()} and "
              f"{len(ref_texts)} reference texts")
    else:
        print("streaming Pile (skip past the build's consumption) ...", flush=True)
        t0 = time.time()
        pools = pile.load_pile_pools(tok)
        ref_texts = pools.reference
        ref_texts_path.write_text(json.dumps(ref_texts))
        print(f"  pile pools in {time.time() - t0:.0f}s "
              f"(skipped {pools.n_skipped} accepted docs, "
              f"{pools.n_raw_consumed} raw read)")

    if reuse_corpus:
        corpus = corpora.Corpus.load(corpus_path)
        print(f"reusing corpus {corpus.counts()}")
    else:
        corpus = corpora.Corpus()
        prov: dict = {
            "pile": {"skip": pools.n_skipped,
                     "build_prompts_consumed": pile.BUILD_PROMPTS_CONSUMED,
                     "raw_docs_read": pools.n_raw_consumed,
                     "shuffle_seed": pile.SHUFFLE_SEED},
        }
        corpus.prompts += [corpora.Prompt(rung="R0", source="pile_heldout",
                                          text=t)
                           for t in pools.r0[:args.n_per_rung]]
        r1, prov["R1"] = corpora.build_r1_domain(args.n_per_rung)
        corpus.prompts += r1
        corpus.prompts += corpora.build_r2_language(tok, args.n_per_rung)
        corpus.prompts += corpora.build_r3_template(tok, pools.r3_content,
                                                    args.n_per_rung)
        r4, prov["R4"] = corpora.build_r4_jailbreak(args.n_per_rung)
        corpus.prompts += r4
        corpus.prompts += corpora.build_r5_random(
            model.cfg.d_vocab, tok.bos_token_id, args.n_per_rung)
        prov["rung_descriptions"] = corpora.RUNG_DESCRIPTIONS
        prov["scaffolds"] = [n for n, _ in corpora.SCAFFOLDS]
        corpus.provenance = prov
        corpus.save(corpus_path)
        print(f"corpus: {corpus.counts()} -> {corpus_path}")

    print("\nMPS numerical validation ...", flush=True)
    val = validate_mps(model, corpus.prompts[:8], args.layer)
    (args.out / "mps_validation.json").write_text(json.dumps(val, indent=2))
    print("  " + json.dumps(val))
    if val["cosine_mps_vs_cpu"] < 0.999:
        print("  WARNING: MPS and CPU activations disagree; results suspect")

    # Sharded so a crash costs one shard, not the whole 25-minute pass. The
    # first run of this died at ~3600/12000 when MPS unified memory filled the
    # swap file; re-doing everything to get back there is avoidable.
    print(f"\nextracting {len(corpus.prompts)} eval activations "
          f"in shards of {args.shard} ...", flush=True)
    t0 = time.time()
    shard_dir = args.out / "shards"
    shard_dir.mkdir(exist_ok=True)
    parts: list[dict] = []
    for s in range(0, len(corpus.prompts), args.shard):
        sp = shard_dir / f"eval_{s:06d}.npz"
        if sp.exists():
            print(f"  shard {s} cached", flush=True)
            parts.append(dict(np.load(sp)))
            continue
        sub = corpus.prompts[s:s + args.shard]
        e = extract.extract(model, sub, layer=args.layer,
                            batch_size=args.batch_size)
        d = {"x": e.x, "entropy_max": e.entropy_max,
             "entropy_final": e.entropy_final, "n_tokens": e.n_tokens}
        np.savez(sp, **d)
        parts.append(d)
        print(f"  shard {s}-{s + len(sub)} done "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    ex = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f"  done in {time.time() - t0:.0f}s")
    np.savez(
        args.out / "eval.npz", **ex,
        rung=np.array([p.rung for p in corpus.prompts]),
        source=np.array([p.source for p in corpus.prompts]),
    )
    print(f"  -> {args.out / 'eval.npz'}  x{ex['x'].shape}")

    ref_path = args.out / "refpool.npy"
    if not ref_path.exists():
        print(f"\nextracting reference pool from {len(ref_texts)} "
              f"Pile docs (per-position) ...", flush=True)
        t0 = time.time()
        ref = extract.extract_per_position_pool(
            model, ref_texts, layer=args.layer,
            batch_size=args.batch_size)
        np.save(ref_path, ref)
        print(f"  {ref.shape} in {time.time() - t0:.0f}s -> {ref_path}")

    print(f"\ntotal {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
