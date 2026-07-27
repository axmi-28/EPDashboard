"""Pass 1: stream every activation once, assign it, and keep what the region
cards need — ranked/sampled example slots, moments, and histograms.

Everything here is a streaming accumulator: memory is O(K × slots), never
O(tokens). The ``TopK`` merge is adapted from ``qwen_ep.member_scan`` (same
vectorised lexsort trick); a uniform reservoir is a TopK on a random key.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POS_BITS = 4096  # meta packing: gid * POS_BITS + position (ctx ≤ 4096)


def _pack(gid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return gid.astype(np.int64) * POS_BITS + pos.astype(np.int64)


class TopK:
    """Per-region top-N by score, merged in vectorised batches.

    Dense ``(K, N)`` score/meta arrays padded with -inf; ``update`` takes a
    flat (region, score, meta) record stream plus named float payloads that
    ride along with the winning records.
    """

    def __init__(self, K: int, n: int, payloads: tuple[str, ...] = ()):
        self.K, self.n = K, n
        self.vals = np.full((K, n), -np.inf, dtype=np.float32)
        self.meta = np.zeros((K, n), dtype=np.int64)
        self.pay = {name: np.zeros((K, n), dtype=np.float32) for name in payloads}

    def update(self, regions: np.ndarray, vals: np.ndarray,
               metas: np.ndarray, **pay: np.ndarray) -> None:
        if regions.size == 0:
            return
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

    def rows(self, i: int, limit: int | None = None) -> list[dict]:
        """Records for region ``i``, best-scoring first."""
        out = []
        for j in range(self.n if limit is None else min(limit, self.n)):
            if not np.isfinite(self.vals[i, j]):
                break
            meta = int(self.meta[i, j])
            rec = {"gid": meta // POS_BITS, "pos": meta % POS_BITS,
                   "score": float(self.vals[i, j])}
            for k, a in self.pay.items():
                rec[k] = float(a[i, j])
            out.append(rec)
        return out

    def payload_col(self, i: int, name: str) -> np.ndarray:
        return self.pay[name][i][np.isfinite(self.vals[i])]


@dataclass
class EPDict:
    """A loaded dictionary plus its derived arrays."""

    run_dir: Path
    meta: dict
    parts: list
    center: np.ndarray       # (d,)
    threshold: float
    E: np.ndarray            # (K, d) exemplar directions, unit norm
    means: np.ndarray        # (K, d) mean member directions, unit norm

    @classmethod
    def load(cls, run_dir: str | Path) -> "EPDict":
        import json
        run_dir = Path(run_dir)
        with (run_dir / "dictionary.pkl").open("rb") as f:
            dic = pickle.load(f)
        meta_path = run_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        parts = list(dic.partitions)
        E = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
        means = np.stack([p.mean_member_direction for p in parts]).astype(np.float32)
        means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-12
        return cls(run_dir=run_dir, meta=meta, parts=parts,
                   center=np.asarray(dic.center, dtype=np.float32),
                   threshold=float(dic.threshold), E=E, means=means)

    @property
    def K(self) -> int:
        return len(self.parts)


class RegionScan:
    """All pass-1 accumulators for one dictionary."""

    def __init__(self, d: EPDict, cfg, rng: np.random.Generator,
                 expected_acts: int):
        self.d, self.cfg, self.rng = d, cfg, rng
        K = d.K
        self.closest = TopK(K, cfg.n_closest, ("proj",))
        # Uniform-without-replacement sampling = bottom-k on a random key.
        self.bands = [TopK(K, cfg.n_per_band, ("dist", "proj"))
                      for _ in range(cfg.n_bands)]
        self.random = TopK(K, max(cfg.reservoir, cfg.n_random),
                           ("dist", "proj", "margin"))

        self.n_member = np.zeros(K, dtype=np.int64)
        self.sum_proj = np.zeros(K, dtype=np.float64)
        self.sum_sq_proj = np.zeros(K, dtype=np.float64)
        self.max_proj = np.full(K, -np.inf)
        self.sum_dist = np.zeros(K, dtype=np.float64)
        self.dist_hist = np.zeros((K, cfg.hist_bins), dtype=np.int64)
        self.n_acts = 0

        # ---------------------------------------------- cell shell / competition
        # margin = d(runner-up) − d(winner): how far inside its cell a member
        # sits. Small margin = on a bisector. ``comp[k, j]`` counts members of k
        # whose runner-up was j — the competition graph, which is *not* the
        # geometric neighbour table on the card.
        self.sum_margin = np.zeros(K, dtype=np.float64)
        self.n_contested = np.zeros(K, dtype=np.int64)
        self.comp = (np.zeros((K, K), dtype=np.int32)
                     if K <= cfg.comp_max_k else None)

        # ------------------------------------------------------ unclaimed mass
        # Activations whose nearest exemplar is further than θ belong to no
        # cell. Their rate is EP's coverage/OOD signal; ``near_miss`` is the
        # per-region shadow just outside the boundary.
        self.n_unclaimed = 0
        self.sum_unclaimed_dist = 0.0
        self.unclaimed_hist = np.zeros(cfg.hist_bins, dtype=np.int64)
        self.near_miss = np.zeros(K, dtype=np.int64)

        # Shared uniform token subsample: its projections onto *every* region
        # direction become the grey full-corpus background of the projection
        # histograms. One Bernoulli rate for the whole run keeps it unbiased.
        self.bg_rate = min(1.0, cfg.bg_sample / max(expected_acts, 1))
        self._bg: list[np.ndarray] = []

    def consume(self, X: np.ndarray, gid: np.ndarray, pos: np.ndarray) -> None:
        d, cfg = self.d, self.cfg
        Xc = X - d.center
        norms = np.linalg.norm(Xc, axis=1) + 1e-12
        dirs = Xc / norms[:, None]
        sims = dirs @ d.E.T                    # (n, K) cosine similarity
        best = np.argmax(sims, axis=1)
        rows = np.arange(len(X))
        best_sim = sims[rows, best].copy()
        # Runner-up: mask the winner, argmax again, restore. Cheaper than a
        # partition and it reuses the matmul we already paid for.
        sims[rows, best] = -np.inf
        second = np.argmax(sims, axis=1)
        margin = best_sim - sims[rows, second]  # = d(runner-up) − d(winner)
        sims[rows, best] = best_sim

        dist = np.maximum(1.0 - best_sim, 0.0)
        member = dist <= d.threshold
        meta = _pack(gid, pos)

        mreg = best[member]
        mmeta = meta[member]
        mdist = dist[member].astype(np.float32)
        mmargin = margin[member].astype(np.float32)
        # proj = <h - c, e_k> — the magnitude EP's unit-direction geometry
        # discards, recovered here for free from the same matmul.
        mproj = (best_sim[member] * norms[member]).astype(np.float32)

        self.closest.update(mreg, -mdist, mmeta, proj=mproj)
        key = -self.rng.random(mreg.size).astype(np.float32)
        self.random.update(mreg, key, mmeta, dist=mdist, proj=mproj,
                           margin=mmargin)
        band = np.minimum((mdist / d.threshold * cfg.n_bands).astype(int),
                          cfg.n_bands - 1)
        # Independent key so band samples don't just replicate the random draw.
        bkey = -self.rng.random(mreg.size).astype(np.float32)
        for b in range(cfg.n_bands):
            m = band == b
            self.bands[b].update(mreg[m], bkey[m], mmeta[m],
                                 dist=mdist[m], proj=mproj[m])

        np.add.at(self.n_member, mreg, 1)
        np.add.at(self.sum_proj, mreg, mproj.astype(np.float64))
        np.add.at(self.sum_sq_proj, mreg, mproj.astype(np.float64) ** 2)
        np.maximum.at(self.max_proj, mreg, mproj.astype(np.float64))
        np.add.at(self.sum_dist, mreg, mdist.astype(np.float64))
        bins = np.minimum((mdist / d.threshold * cfg.hist_bins).astype(int),
                          cfg.hist_bins - 1)
        np.add.at(self.dist_hist, (mreg, bins), 1)

        np.add.at(self.sum_margin, mreg, mmargin.astype(np.float64))
        np.add.at(self.n_contested, mreg[mmargin < cfg.contested_eps * d.threshold], 1)
        if self.comp is not None:
            np.add.at(self.comp, (mreg, second[member]), 1)

        un = ~member
        if un.any():
            udist = dist[un]
            self.n_unclaimed += int(udist.size)
            self.sum_unclaimed_dist += float(udist.sum())
            span = max(2.0 - d.threshold, 1e-6)
            ub = np.minimum(((udist - d.threshold) / span
                             * cfg.hist_bins).astype(int), cfg.hist_bins - 1)
            np.add.at(self.unclaimed_hist, np.maximum(ub, 0), 1)
            np.add.at(self.near_miss, best[un], 1)

        take = self.rng.random(len(X)) < self.bg_rate
        if take.any():
            self._bg.append((sims[take] * norms[take, None]).astype(np.float16))
        self.n_acts += len(X)

    # ------------------------------------------------------------- results
    def bg_proj(self) -> np.ndarray:
        """(S, K) projections of the shared token subsample."""
        return (np.concatenate(self._bg) if self._bg
                else np.zeros((0, self.d.K), dtype=np.float16))

    def examples(self, i: int) -> dict[str, list[dict]]:
        cfg = self.cfg
        out = {"closest": self.closest.rows(i)}
        for b in range(cfg.n_bands):
            out[f"band{b}"] = self.bands[b].rows(i, limit=cfg.n_per_band)
        out["random"] = self.random.rows(i, limit=cfg.n_random)
        return out

    def competitors(self, i: int, k: int) -> list[list]:
        """Top-k runner-up regions for region ``i`` as ``[j, count, share]``."""
        if self.comp is None:
            return []
        row = self.comp[i]
        tot = int(row.sum())
        if not tot:
            return []
        idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
        idx = idx[np.argsort(-row[idx])]
        return [[int(j), int(row[j]), round(float(row[j] / tot), 4)]
                for j in idx if row[j] > 0]

    def unclaimed(self) -> dict:
        """Corpus-level coverage: how much of the stream fell outside every cell."""
        n = max(self.n_acts, 1)
        return {
            "nUnclaimed": int(self.n_unclaimed),
            "rate": round(self.n_unclaimed / n, 5),
            "meanDist": (round(self.sum_unclaimed_dist / self.n_unclaimed, 4)
                         if self.n_unclaimed else None),
            "hist": self.unclaimed_hist.tolist(),
            "range": [round(self.d.threshold, 4), 2.0],
        }

    def winner_gids(self, region_ids: list[int]) -> set[int]:
        gids: set[int] = set()
        for i in region_ids:
            for rows in self.examples(i).values():
                gids.update(r["gid"] for r in rows)
        return gids

    def replay_check(self) -> dict:
        """Does the rescanned stream reproduce the dictionary's region mix?

        Counts only match on a full-budget scan, so the scale-free statistic
        is the correlation of member *shares*; low correlation means the
        examples describe a different activation set than the dictionary.
        """
        stored = np.array([p.member_count for p in self.d.parts], dtype=np.int64)
        share = self.n_member / max(self.n_member.sum(), 1)
        stored_share = stored / max(stored.sum(), 1)
        corr = (float(np.corrcoef(share, stored_share)[0, 1])
                if self.d.K > 1 and self.n_member.sum() else 0.0)
        return {"member_share_corr": round(corr, 4),
                "scan_fraction": round(float(self.n_member.sum()
                                             / max(stored.sum(), 1)), 4)}
