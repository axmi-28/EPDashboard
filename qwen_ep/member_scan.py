"""Rescan the corpus against a built dictionary to recover per-region member
examples the pickle cannot hold.

A `Partition` keeps only 10 closest + 10 farthest prompts and no activation
*magnitudes* at all (EP works exclusively on unit directions), so the saved
dictionary can answer "which members hug the exemplar?" but not "which members
project hardest onto the region direction?" — the SAE max-activating-example
analogue — nor "what does a *typical* member look like?", since the 10 closest
oversell coherence.

This module replays the same text stream through the same layer hook and, per
region, keeps four ranked lists of N examples each:

  closest  – smallest cosine distance to the exemplar (extends the pickle's 10)
  proj     – largest projection <h - c, e_i> among *members* of the region
  projall  – largest projection among *all* activations, member or not; the gap
             between this and `proj` shows directions firing outside their cell
  random   – uniform sample of members, via bottom-k on a random key

Ranking by projection needs ||h - c||, which is exactly what centring-then-
normalising discards. It costs nothing extra here: the assignment step already
computes `dirs @ E.T`, and proj = that * ||h - c||.

Per region we also accumulate projection moments and a histogram so the UI can
show where the top-N sit relative to the bulk.

Activations come from either source:

  --cache-dir   read the shards `extract_cache.py` already wrote. No model, no
                GPU, no second forward pass. Prefer this whenever the cache
                still exists — at 27B scale the forward pass is the entire cost
                of the job, and paying it twice buys nothing.
  (otherwise)   re-stream the corpus through the model, for dictionaries whose
                activation cache has been deleted.

Both paths feed the same accumulators and produce the same payload; `mode` in
the output records which one ran.

Usage:
    python -m qwen_ep.member_scan --cache-dir activations_cache/<slug> \
        --dicts runs/<slug>_p8p0_...,runs/<slug>_p4p0_...

    python -m qwen_ep.member_scan --model-id Qwen/Qwen3.5-4B --layer 27 \
        --dicts runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile,... \
        --max-tokens 3000000
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from .adapter import DEFAULT_MODEL_ID, QwenModel
from .data import get_text_stream

logger = logging.getLogger(__name__)

TOP_N = 24            # examples kept per region per mode
RESERVOIR_N = 256     # uniform member draw; also backs the projection quantiles
POS_BITS = 1024       # meta packing: gid * POS_BITS + position


def _pack(gid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return gid.astype(np.int64) * POS_BITS + pos.astype(np.int64)


def _unpack(meta: int) -> tuple[int, int]:
    return int(meta) // POS_BITS, int(meta) % POS_BITS


class TopK:
    """Per-region top-N by score, merged in vectorised batches.

    Holds dense ``(K, N)`` score/meta arrays padded with -inf so a region with
    fewer than N examples never emits filler. ``update`` accepts a flat
    (region, score, meta) record stream — the same shape whether the caller is
    ranking members of one region or candidates for every region at once.

    ``payloads`` names extra per-record floats that ride along with the winning
    records (cosine distance, owning region, …) so the UI can qualify each
    ranked example without a second pass.
    """

    def __init__(self, K: int, n: int = TOP_N, payloads: tuple[str, ...] = ()):
        self.K, self.n = K, n
        self.vals = np.full((K, n), -np.inf, dtype=np.float32)
        self.meta = np.zeros((K, n), dtype=np.int64)
        self.pay = {name: np.zeros((K, n), dtype=np.float32)
                    for name in payloads}

    def update(self, regions: np.ndarray, vals: np.ndarray,
               metas: np.ndarray, **pay: np.ndarray) -> None:
        if regions.size == 0:
            return
        # Fold the incumbents into the candidate stream, then take the top n of
        # the union per region in one lexsort — no per-region Python loop.
        held = np.isfinite(self.vals).reshape(-1)
        hr = np.repeat(np.arange(self.K), self.n)[held]
        regions = np.concatenate([regions, hr])
        vals = np.concatenate([vals, self.vals.reshape(-1)[held]])
        metas = np.concatenate([metas, self.meta.reshape(-1)[held]])
        pay = {k: np.concatenate([v, self.pay[k].reshape(-1)[held]])
               for k, v in pay.items()}

        order = np.lexsort((-vals, regions))
        r, v, m = regions[order], vals[order], metas[order]
        starts = np.searchsorted(r, np.arange(self.K), side="left")
        rank = np.arange(r.size) - starts[r]
        keep = rank < self.n
        rk, rr = rank[keep], r[keep]
        self.vals[:] = -np.inf
        self.vals[rr, rk] = v[keep]
        self.meta[rr, rk] = m[keep]
        for k, arr in self.pay.items():
            arr[rr, rk] = pay[k][order][keep]

    def rows(self, i: int) -> list[tuple[float, int, int, dict]]:
        """(score, prompt_gid, position, payloads) for region ``i``, best first."""
        out = []
        for j, (v, m) in enumerate(zip(self.vals[i], self.meta[i])):
            if np.isfinite(v):
                gid, pos = _unpack(m)
                out.append((float(v), gid, pos,
                            {k: float(a[i, j]) for k, a in self.pay.items()}))
        return out

    def payload_col(self, i: int, name: str) -> np.ndarray:
        """All finite payload values for region ``i`` — used for quantiles."""
        return self.pay[name][i][np.isfinite(self.vals[i])]


def _dense_candidates(
    proj: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices / values of the top-``n`` activations for every region.

    ``proj`` is (n_acts, K). Returns flat (region, value) candidate arrays via
    argpartition so we never sort the full 30M-entry matrix.
    """
    n_acts, K = proj.shape
    take = min(n, n_acts)
    idx = np.argpartition(proj, n_acts - take, axis=0)[n_acts - take:]  # (take, K)
    vals = np.take_along_axis(proj, idx, axis=0)
    regions = np.tile(np.arange(K), take)
    return idx.reshape(-1), vals.reshape(-1), regions


class DictScan:
    """Accumulators for one dictionary over the scanned stream."""

    def __init__(self, run_dir: Path, rng: np.random.Generator,
                 top_n: int = TOP_N):
        self.top_n = top_n
        self.run_dir = run_dir
        with (run_dir / "dictionary.pkl").open("rb") as f:
            dic = pickle.load(f)
        self.parts = list(dic.partitions)
        self.K = len(self.parts)
        self.center = np.asarray(dic.center, dtype=np.float32)
        self.threshold = float(dic.threshold)
        self.E = np.stack([p.exemplar_direction
                           for p in self.parts]).astype(np.float32)
        self.rng = rng

        self.closest = TopK(self.K, top_n, ("proj",))
        self.proj = TopK(self.K, top_n, ("dist",))
        self.projall = TopK(self.K, top_n, ("dist", "owner"))
        # Bottom-k on a uniform key = uniform sample without replacement. Kept
        # wider than top_n so the same draw also yields honest quantiles for
        # the projection distribution; the UI shows the first top_n of it.
        self.random = TopK(self.K, max(RESERVOIR_N, top_n), ("proj", "dist"))

        self.n_member = np.zeros(self.K, dtype=np.int64)
        self.sum_proj = np.zeros(self.K, dtype=np.float64)
        self.sum_sq_proj = np.zeros(self.K, dtype=np.float64)
        self.max_proj = np.full(self.K, -np.inf, dtype=np.float64)
        self.n_acts = 0

    # -- streaming ------------------------------------------------------
    def consume(self, X: np.ndarray, gid: np.ndarray, pos: np.ndarray) -> None:
        Xc = X - self.center
        norms = np.linalg.norm(Xc, axis=1) + 1e-12
        dirs = Xc / norms[:, None]
        sims = dirs @ self.E.T                       # (n, K) cosine similarity
        proj = sims * norms[:, None]                 # <h - c, e_k>, magnitude-aware

        best = np.argmax(sims, axis=1)
        rows = np.arange(len(X))
        best_sim = sims[rows, best]
        dist = np.maximum(1.0 - best_sim, 0.0)   # float noise can go slightly <0
        member = dist <= self.threshold
        meta = _pack(gid, pos)

        mreg = best[member]
        mmeta = meta[member]
        mdist = dist[member].astype(np.float32)
        mproj = proj[rows[member], mreg].astype(np.float32)
        self.closest.update(mreg, -mdist, mmeta, proj=mproj)
        self.proj.update(mreg, mproj, mmeta, dist=mdist)
        self.random.update(mreg, -self.rng.random(mreg.size).astype(np.float32),
                           mmeta, proj=mproj, dist=mdist)

        # Top projection over *every* activation, member or not. `owner` is the
        # cell that actually claims that token, so the UI can flag hits this
        # direction fires on but does not own.
        ridx, rvals, rregs = _dense_candidates(proj, self.projall.n)
        self.projall.update(
            rregs, rvals.astype(np.float32), meta[ridx],
            dist=np.maximum(1.0 - sims[ridx, rregs], 0.0).astype(np.float32),
            owner=np.where(member[ridx], best[ridx], -1).astype(np.float32))

        np.add.at(self.n_member, mreg, 1)
        np.add.at(self.sum_proj, mreg, mproj.astype(np.float64))
        np.add.at(self.sum_sq_proj, mreg, mproj.astype(np.float64) ** 2)
        np.maximum.at(self.max_proj, mreg, mproj.astype(np.float64))
        self.n_acts += len(X)

    # -- output ---------------------------------------------------------
    def report(self, prompts: list[str], meta_extra: dict) -> dict:
        # Replay check. Member *counts* only match on a full-budget scan, so
        # the scale-free statistic is the correlation of member shares: if the
        # stream replayed differently the region mix would drift, and a low
        # correlation here means the examples below describe a different
        # activation set than the one the dictionary was built from.
        stored = np.array([p.member_count for p in self.parts], dtype=np.int64)
        scan_share = self.n_member / max(self.n_member.sum(), 1)
        stored_share = stored / max(stored.sum(), 1)
        prop_corr = (float(np.corrcoef(scan_share, stored_share)[0, 1])
                     if self.K > 1 and self.n_member.sum() else 0.0)
        scan_fraction = float(self.n_member.sum() / max(stored.sum(), 1))
        count_agree = float(np.mean(np.isclose(stored, self.n_member, rtol=0.05)))
        regions = []
        for i in range(self.K):
            n = int(self.n_member[i])
            mean = self.sum_proj[i] / n if n else 0.0
            var = max(0.0, self.sum_sq_proj[i] / n - mean * mean) if n else 0.0

            def emit(tk: TopK, negate: bool = False, limit: int | None = None,
                     v_from: str | None = None) -> list[dict]:
                """``v`` is the ranking score, except for the random draw whose
                score is the throwaway sort key — there ``v_from`` promotes a
                payload into the display slot."""
                out = []
                for score, gid, pos, pay in tk.rows(i)[:limit or self.top_n]:
                    if gid >= len(prompts):
                        continue
                    rec = {"v": round(max(-score, 0.0) if negate else score, 4),
                           "gid": gid, "pos": pos}
                    for k, v in pay.items():
                        rec[k] = int(v) if k == "owner" else round(v, 4)
                    if v_from:
                        rec["v"] = rec[v_from]
                    out.append(rec)
                return out

            # Quantiles come off the uniform reservoir, so no bin range has to
            # be guessed and the tail is never clipped.
            sample = self.random.payload_col(i, "proj")
            q = (np.round(np.quantile(sample, [0.05, 0.25, 0.5, 0.75, 0.95]),
                          3).tolist() if sample.size >= 5 else [])

            regions.append({
                "i": i,
                "nScan": n,
                "nStored": int(stored[i]),
                "projMean": round(float(mean), 3),
                "projSd": round(float(np.sqrt(var)), 3),
                "projMax": (round(float(self.max_proj[i]), 3)
                            if np.isfinite(self.max_proj[i]) else 0.0),
                "projQ": q,                    # 5/25/50/75/95th percentile
                "nSample": int(sample.size),
                "closest": emit(self.closest, negate=True),   # back to distance
                "proj": emit(self.proj),
                "projall": emit(self.projall),
                "random": emit(self.random, v_from="dist"),
            })
        return {
            "run": self.run_dir.name,
            "K": self.K,
            "topN": self.top_n,
            "nActsScanned": self.n_acts,
            "memberShareCorr": round(prop_corr, 4),
            "scanFraction": round(scan_fraction, 4),
            "memberCountAgreement": round(count_agree, 4),
            "prompts": prompts,
            "regions": regions,
            **meta_extra,
        }


def iter_cache(cache_dir: Path, prompts: list[str], chunk: int,
               max_acts: int | None = None):
    """Yield ``(X, global_prompt_id, position)`` chunks from an activation cache.

    Reuses the shards `extract_cache.py` already writes, so the projections can
    be recovered without a second forward pass — the whole point of caching.
    ``prompts`` is appended to as shards are opened; shard-local ``prompt_ids``
    are rebased onto that growing list.
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    seen = 0
    for name in manifest["shard_files"]:
        data = np.load(cache_dir / name, allow_pickle=True)
        x, pid, pos = data["x"], data["prompt_ids"], data["position_ids"]
        base = len(prompts)
        prompts.extend(str(p) for p in data["prompts"])
        for s in range(0, len(x), chunk):
            if max_acts is not None and seen >= max_acts:
                return
            sl = slice(s, s + chunk)
            yield (x[sl].astype(np.float32),
                   pid[sl].astype(np.int64) + base,
                   pos[sl].astype(np.int64))
            seen += len(x[sl])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=None,
                    help="read activations from an extract_cache shard dir "
                         "instead of re-running the model (no GPU needed)")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--dicts", required=True,
                    help="comma-separated run dirs (each holding dictionary.pkl)")
    ap.add_argument("--max-tokens", type=int, default=3_000_000)
    ap.add_argument("--max-acts", type=int, default=None,
                    help="cache mode: stop after this many activations")
    ap.add_argument("--chunk", type=int, default=8192,
                    help="cache mode: activations per distance-matrix chunk")
    ap.add_argument("--context-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prompt-batch-size", type=int, default=64)
    ap.add_argument("--corpus", default="pile", choices=["pile", "wikitext"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-n", type=int, default=TOP_N,
                    help="examples kept per region per mode. Scan cost is "
                         "unchanged (it is one pass either way) and only the "
                         "JSON grows, so scan wide once rather than re-running "
                         "against an activation cache you may no longer have")
    ap.add_argument("--out-name", default="member_scan.json")
    args = ap.parse_args()
    if args.cache_dir is None and args.layer is None:
        ap.error("--layer is required unless --cache-dir is given")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("qwen_ep.member_scan")

    run_dirs = [Path(s) for s in args.dicts.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    scans = [DictScan(rd, rng, top_n=args.top_n) for rd in run_dirs]
    for s in scans:
        log.info("dict %s: K=%d threshold=%.4f", s.run_dir.name, s.K, s.threshold)

    prompts: list[str] = []
    total_acts = total_tokens = 0
    t0 = time.time()
    next_log = 0

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        log.info("cache %s: L%d %s  %d acts in %d shards", cache_dir.name,
                 manifest["layer"], manifest["corpus"],
                 manifest["n_activations"], manifest["n_shards"])
        source = {"mode": "cache", "cacheDir": cache_dir.name,
                  "model_id": manifest["model_id"], "layer": manifest["layer"],
                  "corpus": manifest["corpus"], "seed": manifest["seed"],
                  "contextLength": manifest["context_length"]}
        for X, gid, pos in iter_cache(cache_dir, prompts, args.chunk,
                                      args.max_acts):
            for s in scans:
                s.consume(X, gid, pos)
            total_acts += len(X)
            if total_acts >= next_log:
                el = time.time() - t0
                log.info("  %d acts | %d prompts | %.0fs (%.0f acts/s)",
                         total_acts, len(prompts), el,
                         total_acts / max(el, 1e-9))
                next_log = total_acts + 250_000
    else:
        qwen = QwenModel(args.model_id, device=args.device)
        texts = get_text_stream(args.corpus, qwen.tokenizer,
                                context_length=args.context_length,
                                seed=args.seed)
        source = {"mode": "forward", "model_id": args.model_id,
                  "layer": args.layer, "corpus": args.corpus,
                  "seed": args.seed, "contextLength": args.context_length}
        batch: list[str] = []

        def run_batch(batch: list[str]) -> int:
            nonlocal total_tokens
            res = qwen.extract_per_position(batch, layer=args.layer,
                                            batch_size=args.batch_size)
            if res.x.shape[0] == 0:
                return 0
            total_tokens += res.n_tokens
            gid = res.prompt_ids.astype(np.int64) + len(prompts)
            pos = res.position_ids.astype(np.int64)
            X = res.x.astype(np.float32)
            for s in scans:
                s.consume(X, gid, pos)
            return res.x.shape[0]

        for text in texts:
            batch.append(text)
            if len(batch) < args.prompt_batch_size:
                continue
            n = run_batch(batch)
            prompts.extend(batch)
            total_acts += n
            batch = []
            if total_acts >= next_log:
                el = time.time() - t0
                log.info("  %d acts | %d prompts | %.0fs (%.0f tok/s)",
                         total_acts, len(prompts), el,
                         total_tokens / max(el, 1e-9))
                next_log = total_acts + 250_000
            if total_tokens >= args.max_tokens:
                break
        if batch and total_tokens < args.max_tokens:
            total_acts += run_batch(batch)
            prompts.extend(batch)

    meta_extra = {
        **source,
        "maxTokens": args.max_tokens,
        "elapsedS": round(time.time() - t0, 1),
    }
    for s in scans:
        payload = s.report(prompts, meta_extra)
        out = s.run_dir / args.out_name
        out.write_text(json.dumps(payload, ensure_ascii=False,
                                  separators=(",", ":")))
        log.info("%s: agreement=%.3f  wrote %s (%.1f MB)", s.run_dir.name,
                 payload["memberCountAgreement"], out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
