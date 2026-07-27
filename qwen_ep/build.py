"""Build an Exemplar-Partition dictionary on Qwen3.5-2B-Base.

Reuses the `ep` core unchanged (calibration, leader-clustering, dictionary),
swapping only the model/extraction seam for the HuggingFace Qwen adapter.

Example (small validation run):
    python -m qwen_ep.build --layer 12 --percentile 8 \
        --calibration-tokens 100000 --max-tokens 1000000 \
        --context-length 128 --batch-size 16 --corpus pile

Then inspect:
    python -m qwen_ep.inspect --dict runs/<slug>/dictionary.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from ep.discovery.pipeline import calibrate_pipeline, discover

from .adapter import DEFAULT_MODEL_ID, QwenModel, make_extract_fn, model_tag
from .data import get_text_stream


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--layer", type=int, default=12,
                    help="Decoder layer whose post-block residual stream to partition.")
    ap.add_argument("--percentile", type=float, default=8.0,
                    help="p-th percentile of pairwise cosine distance -> threshold. "
                         "Smaller p = tighter cells = more partitions.")
    ap.add_argument("--calibration-tokens", type=int, default=100_000)
    ap.add_argument("--max-tokens", type=int, default=1_000_000,
                    help="Activation-stream budget for discovery.")
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Prompts per forward pass (model batch).")
    ap.add_argument("--prompt-batch-size", type=int, default=16,
                    help="Prompts per extractor call (EP pipeline batch).")
    ap.add_argument("--corpus", default="pile", choices=["pile", "wikitext"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saturation-window", type=int, default=5,
                    help="Stop after this many extractor batches with no new partition.")
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--skip-first/--keep-first", dest="skip_first",
                    action="store_true", default=True)
    return ap.parse_args()


def run_slug(args) -> str:
    p = ("p" + str(args.percentile).replace(".", "p"))
    return (f"{model_tag(args.model_id)}_L{args.layer}_{p}_ctx{args.context_length}"
            f"_mt{args.max_tokens}_seed{args.seed}_{args.corpus}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("qwen_ep.build")
    args = parse_args()

    out_dir = Path(args.output_dir) / run_slug(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output dir: %s", out_dir)

    t0 = time.time()
    qwen = QwenModel(args.model_id, device=args.device)
    if not (0 <= args.layer < qwen.n_layers):
        raise SystemExit(f"--layer {args.layer} out of range [0, {qwen.n_layers})")
    hook_name = f"{qwen.layers_path}.{args.layer}"  # identity string for cache

    extract_fn = make_extract_fn(
        qwen, layer=args.layer, batch_size=args.batch_size, skip_first=args.skip_first,
    )

    # A single lazy stream feeds calibration then discovery (discovery continues
    # from where calibration left off — no document is scored twice).
    texts = get_text_stream(
        args.corpus, qwen.tokenizer,
        context_length=args.context_length, seed=args.seed,
    )

    log.info("== calibration ==")
    calibration = calibrate_pipeline(
        model=qwen, texts=texts, hook_name=hook_name,
        n_tokens=args.calibration_tokens, percentile=args.percentile,
        extract_fn=extract_fn, prompt_batch_size=args.prompt_batch_size,
        seed=args.seed,
        cache_model_name=model_tag(args.model_id),
        cache_extras={"corpus": args.corpus, "ctx": args.context_length},
    )
    log.info("calibration: ||center||=%.4f  threshold@p%g=%.6f  (n_acts=%d)",
             float(np.linalg.norm(calibration.center)), args.percentile,
             calibration.threshold, calibration.n_activations)

    log.info("== discovery ==")
    result = discover(
        model=qwen, texts=texts, hook_name=hook_name, calibration=calibration,
        extract_fn=extract_fn, max_tokens=args.max_tokens,
        prompt_batch_size=args.prompt_batch_size,
        saturation_window=args.saturation_window, seed=args.seed,
    )

    dict_path = out_dir / "dictionary.pkl"
    with dict_path.open("wb") as f:
        pickle.dump(result.dictionary, f)

    members = sorted((p.member_count for p in result.dictionary.partitions), reverse=True)
    metadata = {
        "model_id": args.model_id,
        "layer": args.layer,
        "hook_name": hook_name,
        "d_model": qwen.d_model,
        "n_layers": qwen.n_layers,
        "percentile": args.percentile,
        "threshold": result.dictionary.threshold,
        "center_norm": float(np.linalg.norm(calibration.center)),
        "context_length": args.context_length,
        "corpus": args.corpus,
        "calibration_tokens": args.calibration_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "n_partitions": len(result.dictionary),
        "n_activations": result.n_activations,
        "n_tokens": result.n_tokens,
        "n_forward_passes": result.n_forward_passes,
        "saturated": result.saturated,
        "elapsed_s": round(result.elapsed_s, 1),
        "extraction_time_s": round(result.extraction_time_s, 1),
        "clustering_time_s": round(result.clustering_time_s, 1),
        "largest_partition": members[0] if members else 0,
        "singletons": sum(1 for m in members if m == 1),
        "total_wall_s": round(time.time() - t0, 1),
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    log.info("== done ==")
    log.info("partitions=%d  activations=%d  saturated=%s  wall=%.0fs",
             len(result.dictionary), result.n_activations, result.saturated,
             metadata["total_wall_s"])
    log.info("dictionary -> %s", dict_path)
    log.info("metadata   -> %s", out_dir / "metadata.json")


if __name__ == "__main__":
    main()
