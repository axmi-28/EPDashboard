"""Build EP dictionaries at several resolutions ``p`` from a cached activation
stream — no model, no repeated forward passes.

For each ``p`` we (1) calibrate a center + threshold from the first
``--calibration-tokens`` cached activations, then (2) leader-cluster the whole
cache into a `Dictionary`, tracking nearest / boundary prompts per region so the
result is inspectable exactly like a live build. Output mirrors `build.py`:
``<out>/<slug>/dictionary.pkl`` + ``metadata.json``.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from ep.discovery.calibration import calibrate
from ep.discovery.dictionary import Dictionary

from .adapter import model_tag

MAX_EXAMPLES_PER_PARTITION = 10
# Sub-batch sizes bound the O(B^2) calibration matrix and the O(B*K) discovery
# distance matrix. Shards hold ~100k acts; feeding those whole would allocate
# tens of GB. ~2k matches the live pipeline's per-prompt-batch granularity.
CALIB_CHUNK = 2048
CLUSTER_CHUNK = 8192


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--percentiles", default="1,2,4,8",
                    help="comma-separated p values to sweep")
    ap.add_argument("--calibration-tokens", type=int, default=200_000)
    ap.add_argument("--output-dir", default="runs")
    return ap.parse_args()


def iter_shards(cache_dir: Path, manifest: dict):
    for name in manifest["shard_files"]:
        data = np.load(cache_dir / name, allow_pickle=True)
        yield (data["x"].astype(np.float32), data["prompt_ids"],
               data["position_ids"], data["prompts"])


def calib_batches(cache_dir: Path, manifest: dict, n_tokens: int):
    """Yield ~CALIB_CHUNK-row batches (calibrate() computes a within-batch
    pairwise matrix, so batches must stay small)."""
    seen = 0
    for x, _, _, _ in iter_shards(cache_dir, manifest):
        for start in range(0, len(x), CALIB_CHUNK):
            chunk = x[start:start + CALIB_CHUNK]
            yield chunk
            seen += len(chunk)
            if seen >= n_tokens:
                return


def build_at_p(cache_dir: Path, manifest: dict, percentile: float,
               calibration_tokens: int, log: logging.Logger) -> tuple[Dictionary, dict]:
    t0 = time.time()
    cal = calibrate(calib_batches(cache_dir, manifest, calibration_tokens),
                    n_tokens=calibration_tokens, percentile=percentile)
    log.info("  p=%g calibration: ||center||=%.3f threshold=%.4f (n=%d)",
             percentile, float(np.linalg.norm(cal.center)), cal.threshold,
             cal.n_activations)

    d = Dictionary(center=cal.center, threshold=cal.threshold)
    total_acts = 0
    for x, prompt_ids, position_ids, prompts in iter_shards(cache_dir, manifest):
        # Sub-chunk each shard so the (B x K) distance matrix stays bounded as
        # the exemplar set K grows (fine p can reach tens of thousands).
        for s in range(0, len(x), CLUSTER_CHUNK):
            xc = x[s:s + CLUSTER_CHUNK]
            pid_c = prompt_ids[s:s + CLUSTER_CHUNK]
            pos_c = position_ids[s:s + CLUSTER_CHUNK]
            assignments = d.add_batch(x_batch=xc, iteration=0, global_index_start=total_acts)
            per_act_dists = d._last_dists
            for act_idx, pids in enumerate(assignments):
                if not pids:
                    continue
                pid = int(pid_c[act_idx])
                pos = int(pos_c[act_idx])
                if pid >= len(prompts):
                    continue
                text = str(prompts[pid])
                dist = float(per_act_dists[act_idx])
                for partition_id in pids:
                    p = d.partitions[partition_id]
                    if len(p.sample_prompts) < MAX_EXAMPLES_PER_PARTITION:
                        heapq.heappush(p.sample_prompts, (-dist, text, pos))
                    elif dist < -p.sample_prompts[0][0]:
                        heapq.heapreplace(p.sample_prompts, (-dist, text, pos))
                    if len(p.boundary_prompts) < MAX_EXAMPLES_PER_PARTITION:
                        heapq.heappush(p.boundary_prompts, (dist, text, pos))
                    elif dist > p.boundary_prompts[0][0]:
                        heapq.heapreplace(p.boundary_prompts, (dist, text, pos))
            total_acts += len(xc)
    d.finalize()

    members = sorted((p.member_count for p in d.partitions), reverse=True)
    meta = {
        "model_id": manifest["model_id"], "layer": manifest["layer"],
        "d_model": manifest["d_model"], "corpus": manifest["corpus"],
        "context_length": manifest["context_length"], "seed": manifest["seed"],
        "percentile": percentile, "threshold": d.threshold,
        "center_norm": float(np.linalg.norm(cal.center)),
        "calibration_tokens": calibration_tokens,
        "n_activations": total_acts, "n_partitions": len(d),
        "largest_partition": members[0] if members else 0,
        "singletons": sum(1 for m in members if m == 1),
        "cluster_time_s": round(time.time() - t0, 1),
    }
    log.info("  p=%g -> %d partitions (largest=%d, %.0fs)",
             percentile, len(d), meta["largest_partition"], meta["cluster_time_s"])
    return d, meta


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("qwen_ep.sweep")
    args = parse_args()

    cache_dir = Path(args.cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    log.info("cache: L%d %s  %d acts in %d shards", manifest["layer"],
             manifest["corpus"], manifest["n_activations"], manifest["n_shards"])

    percentiles = [float(x) for x in args.percentiles.split(",") if x.strip()]
    summary = []
    for p in percentiles:
        d, meta = build_at_p(cache_dir, manifest, p, args.calibration_tokens, log)
        slug = (f"{model_tag(manifest['model_id'])}_L{manifest['layer']}"
                f"_p{str(p).replace('.', 'p')}"
                f"_ctx{manifest['context_length']}_cache_{manifest['corpus']}")
        out = Path(args.output_dir) / slug
        out.mkdir(parents=True, exist_ok=True)
        with (out / "dictionary.pkl").open("wb") as f:
            pickle.dump(d, f)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        summary.append((p, len(d), meta["largest_partition"], str(out)))

    log.info("== sweep complete ==")
    for p, n, largest, out in summary:
        log.info("  p=%-4g partitions=%-6d largest=%-7d %s", p, n, largest, out)


if __name__ == "__main__":
    main()
