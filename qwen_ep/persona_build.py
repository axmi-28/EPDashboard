"""Build a persona-space EP dictionary on roleplay response-token activations.

Motivation (see ASSISTANT_AXIS_EP.md sec.5b): on the generic Pile-built L27 dict,
assistant behaviour is *distributed* across cells with no tight home, so
single-anchor drift membership is noisy. Here we tile *persona space directly* —
partition the in-chat response-token activations of the P1 roleplay rollouts — so
the default Assistant (and each character) can consolidate into its own region.

Two stages (the p-sweep needs no model reload):
  extract  load rollouts.jsonl, harvest per-token response activations at the
           target layer -> persona_tokens_L{L}.npz
  build    run EP calibrate + discover over those cached activations (a
           precomputed extract_fn feeds them into the unchanged ep pipeline)

Example:
    python -m qwen_ep.persona_build --stage extract \
        --rollouts runs/persona_axis/qwen3_5-4b_spectrum_q16_sp3_seed0/rollouts.jsonl \
        --layer 27
    python -m qwen_ep.persona_build --stage build --layer 27 --percentile 8
    python -m qwen_ep.persona_build --stage build --layer 27 --percentile 4
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from ep.discovery.extraction import ExtractionResult
from ep.discovery.pipeline import calibrate_pipeline, discover

from .adapter import QwenModel, model_tag

log = logging.getLogger("qwen_ep.persona_build")


def _load_rollouts(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return ([r.get("system") for r in rows],
            [r["question"] for r in rows],
            [r["response"] for r in rows],
            [r["role"] for r in rows])


def stage_extract(args, cache: Path) -> None:
    systems, users, responses, roles = _load_rollouts(Path(args.rollouts))
    log.info("rollouts=%d", len(responses))
    qwen = QwenModel(args.model_id, device=args.device)
    t0 = time.time()
    x, rids, pids = qwen.response_token_activations(
        systems, users, responses, layer=args.layer,
        batch_size=args.extract_batch)
    log.info("extracted %d response tokens in %.0fs (x=%s)",
             x.shape[0], time.time() - t0, x.shape)
    np.savez_compressed(cache, x=x.astype(np.float32), rids=rids, pids=pids,
                        roles=np.array(roles))
    log.info("cached -> %s", cache)


def _precomputed_extract_fn(x: np.ndarray, rid_rows: dict[int, np.ndarray]):
    """extract_fn(model, batch_of_id_strings, hook) -> ExtractionResult, feeding
    the ep pipeline the cached per-rollout token activations for each id."""
    def extract_fn(model, prompts, hook_name, **kwargs):  # noqa: ARG001
        ids = [int(p) for p in prompts]
        rows = np.concatenate([rid_rows[i] for i in ids]) if ids else np.array([], int)
        xb = x[rows]
        return ExtractionResult(
            x=xb,
            prompt_ids=np.repeat(ids, [len(rid_rows[i]) for i in ids]).astype(np.int64)
            if ids else np.array([], np.int64),
            position_ids=np.arange(len(xb), dtype=np.int64),
            n_forward_passes=0, n_tokens=len(xb))
    return extract_fn


def _load_source(args, cache: Path):
    """Return (x (N,d), rids (N,)) for the chosen granularity.

    token: per-response-token acts (dominated by token identity -> degenerate
           for persona partitioning). mean: one per-rollout mean vector, the
           persona-in-context summary that actually carries the character.
    """
    if args.source == "mean":
        z = np.load(args.acts_npz, allow_pickle=True)
        layers = z["layers"].tolist()
        li = layers.index(args.layer)
        x = z["acts"][:, li, :].astype(np.float32)
        return x, np.arange(len(x), dtype=np.int64)
    z = np.load(cache, allow_pickle=True)
    return z["x"].astype(np.float32), z["rids"].astype(np.int64)


def stage_build(args, cache: Path, out_dir: Path) -> None:
    x, rids = _load_source(args, cache)
    n_roll = int(rids.max()) + 1
    rid_rows = {i: np.where(rids == i)[0] for i in range(n_roll)}
    texts = [str(i) for i in range(n_roll)]           # ids as the "text" stream
    extract_fn = _precomputed_extract_fn(x, rid_rows)
    total_tokens = x.shape[0]
    log.info("build: %d tokens, %d rollouts, p%g", total_tokens, n_roll, args.percentile)

    t0 = time.time()
    calibration = calibrate_pipeline(
        model=None, texts=texts, hook_name=f"L{args.layer}",
        n_tokens=total_tokens, percentile=args.percentile,
        extract_fn=extract_fn, prompt_batch_size=args.prompt_batch_size, seed=args.seed)
    log.info("calib: ||center||=%.4f thr@p%g=%.6f n=%d",
             float(np.linalg.norm(calibration.center)), args.percentile,
             calibration.threshold, calibration.n_activations)

    result = discover(
        model=None, texts=texts, hook_name=f"L{args.layer}", calibration=calibration,
        extract_fn=extract_fn, max_tokens=total_tokens,
        prompt_batch_size=args.prompt_batch_size,
        saturation_window=10_000, seed=args.seed)

    members = sorted((p.member_count for p in result.dictionary.partitions), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dictionary.pkl").open("wb") as f:
        pickle.dump(result.dictionary, f)
    meta = {
        "model_id": args.model_id, "layer": args.layer, "d_model": x.shape[1],
        "percentile": args.percentile, "threshold": result.dictionary.threshold,
        "center_norm": float(np.linalg.norm(calibration.center)),
        "source": "persona_rollouts", "n_rollouts": n_roll,
        "n_activations": total_tokens, "n_partitions": len(result.dictionary),
        "largest_partition": members[0] if members else 0,
        "singletons": sum(1 for m in members if m == 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("== done == partitions=%d largest=%d singletons=%d wall=%.0fs -> %s",
             len(result.dictionary), meta["largest_partition"], meta["singletons"],
             time.time() - t0, out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stage", default="all", choices=["all", "extract", "build"])
    ap.add_argument("--source", default="mean", choices=["mean", "token"],
                    help="'mean' = per-rollout mean vectors (persona-level, "
                         "recommended); 'token' = per-response-token (degenerate)")
    ap.add_argument("--acts-npz", default="",
                    help="P1 activations.npz (required for --source mean)")
    ap.add_argument("--rollouts", default="")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--percentile", type=float, default=8.0)
    ap.add_argument("--extract-batch", type=int, default=8)
    ap.add_argument("--prompt-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="runs/persona_dict")
    args = ap.parse_args()

    cache = Path(args.cache_dir) / f"persona_tokens_L{args.layer}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "extract"):
        if not args.rollouts:
            raise SystemExit("--rollouts required for extract stage")
        stage_extract(args, cache)
    if args.stage in ("all", "build"):
        if args.source == "token" and not cache.exists():
            raise SystemExit(f"missing {cache}; run --stage extract first")
        if args.source == "mean" and not args.acts_npz:
            raise SystemExit("--acts-npz required for --source mean")
        p = "p" + str(args.percentile).replace(".", "p")
        out_dir = (Path(args.cache_dir) /
                   f"{model_tag(args.model_id)}-persona-{args.source}_L{args.layer}_{p}")
        stage_build(args, cache, out_dir)


if __name__ == "__main__":
    main()
