"""Gate 2 A3b — is EP's one win semantic, or just out-of-distribution detection?

Across five labelled tasks, EP's partition beat a matched-K random partition on
exactly one: language ID (Bulgarian vs. English Pile), by +8 to +10 sd at three
consecutive percentiles. That is the single positive result in the programme and
everything favourable rests on it, so it is worth knowing what it is.

There is a deflationary explanation. The dictionaries were built on English
Pile. Bulgarian activations sit outside that support, and Gate 0B established
that EP places exemplars to *cover* the support: an activation outside it is far
from every exemplar, and which exemplar it lands on is close to arbitrary. A
partition could therefore "win" at language ID purely because one class is
outside the region structure entirely — which is distance-to-support detection
wearing region-identity clothes, and is exactly what Gate 0B already showed EP
is no better at than a coreset.

The discriminating test is **distance stratification**. If the advantage is
semantic, it survives among activations at comparable distance from the
dictionary. If it is an OOD effect, it lives entirely in the far tail.

Three measurements:

  1. distance-to-nearest-exemplar distributions per class, EP and coreset
  2. EP-vs-coreset AUROC *within* distance deciles
  3. AUROC on a distance-matched resample, where the two classes are forced to
     share a distance distribution by binning and subsampling

Run against `code_math` as a contrast: both classes are in-distribution English
there, and EP loses, which is the pattern the deflationary story predicts.

    python -m experiments.monitor.run_gate2_a3b
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .dicts import DEFAULT_MODEL_SHORT, PERCENTILES, lean_path, load_lean
from .gate2_route import assign, coreset_partition, cross_fit_scores, unit
from .run_gate2_a3 import build_tasks
from .scorers import auroc

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
OUT_CSV = ROOT / "gate2_a3b_ood.csv"
OUT_JSON = ROOT / "gate2_a3b_ood.json"

LAYER = 20
N_DRAWS = 8
N_DECILES = 5
TASKS = ("language", "code_math", "scaffold")


def nearest_distance(x: np.ndarray, exemplars: np.ndarray,
                     center: np.ndarray, chunk: int = 4096) -> np.ndarray:
    q = unit(x.astype(np.float32), center)
    e = np.ascontiguousarray(exemplars.astype(np.float32))
    out = np.empty(len(q), dtype=np.float32)
    for s in range(0, len(q), chunk):
        out[s:s + chunk] = 1.0 - (q[s:s + chunk] @ e.T).max(axis=1)
    return out


def matched_resample(d: np.ndarray, y: np.ndarray, n_bins: int = 20,
                     seed: int = 0) -> np.ndarray:
    """Indices whose distance distribution is equalised across the two classes.

    Bins on the pooled distance quantiles, then takes min(n_pos, n_neg) from
    each bin. Bins where either class is absent contribute nothing — that
    absence is precisely the OOD part being removed.
    """
    rng = np.random.default_rng(seed)
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    keep: list[np.ndarray] = []
    for b in range(n_bins):
        m = (d >= edges[b]) & (d < edges[b + 1])
        pos, neg = np.where(m & (y == 1))[0], np.where(m & (y == 0))[0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


def score_partition(x, y, exemplars, center, K, idx=None) -> float:
    r = assign(x, exemplars, center)
    if idx is not None:
        r, y = r[idx], y[idx]
    return auroc(cross_fit_scores(r, y, K), y == 1)


def main() -> None:
    z = np.load(ROOT / "eval.npz")
    x_all = z["x"]
    tasks = build_tasks(z["source"], z["rung"])
    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")

    rows: list[dict] = []
    detail: dict = {"layer": LAYER, "n_draws": N_DRAWS, "tasks": {}}
    for name in TASKS:
        keep, y = tasks[name]
        x = x_all[keep]
        print(f"\n=== {name}: n={len(y)} ({int(y.sum())} positive) ===")
        detail["tasks"][name] = {}
        for p in PERCENTILES:
            d = load_lean(lean_path(DICTS, DEFAULT_MODEL_SHORT, LAYER, p))
            c, K = d["center"], d["K"]
            dist = nearest_distance(x, d["exemplars"], c)
            dpos, dneg = dist[y == 1], dist[y == 0]
            # Does distance alone separate the classes? If it does, region
            # identity has an OOD shortcut available to it.
            auc_dist = auroc(dist, y == 1)

            full_ep = score_partition(x, y, d["exemplars"], c, K)
            full_cs = [score_partition(x, y, coreset_partition(pool, c, K, 300 + j),
                                       c, K) for j in range(N_DRAWS)]

            idx = matched_resample(dist, y)
            m_ep = score_partition(x, y, d["exemplars"], c, K, idx)
            m_cs = [score_partition(x, y, coreset_partition(pool, c, K, 300 + j),
                                    c, K, idx) for j in range(N_DRAWS)]

            sd_f = float(np.std(full_cs, ddof=1))
            sd_m = float(np.std(m_cs, ddof=1))
            row = {
                "task": name, "percentile": p, "K": K,
                "dist_mean_pos": float(dpos.mean()),
                "dist_mean_neg": float(dneg.mean()),
                "dist_auroc": auc_dist,
                "full_n": int(len(y)), "full_ep": full_ep,
                "full_cs": float(np.mean(full_cs)), "full_cs_sd": sd_f,
                "full_margin_sd": float((full_ep - np.mean(full_cs))
                                        / max(sd_f, 1e-9)),
                "matched_n": int(len(idx)), "matched_ep": m_ep,
                "matched_cs": float(np.mean(m_cs)), "matched_cs_sd": sd_m,
                "matched_margin_sd": float((m_ep - np.mean(m_cs))
                                           / max(sd_m, 1e-9)),
            }
            rows.append(row)
            print(f"  p{p:<3d} K={K:<5d} dist(pos)={dpos.mean():.4f} "
                  f"dist(neg)={dneg.mean():.4f} distAUROC={auc_dist:.3f} | "
                  f"full EP {full_ep:.4f} vs CS {np.mean(full_cs):.4f} "
                  f"({row['full_margin_sd']:+.1f}sd) | matched(n={len(idx)}) "
                  f"EP {m_ep:.4f} vs CS {np.mean(m_cs):.4f} "
                  f"({row['matched_margin_sd']:+.1f}sd)", flush=True)

            # AUROC by distance stratum: where does any advantage live?
            edges = np.quantile(dist, np.linspace(0, 1, N_DECILES + 1))
            edges[-1] += 1e-6
            strata = []
            for b in range(N_DECILES):
                m = np.where((dist >= edges[b]) & (dist < edges[b + 1]))[0]
                if len(np.unique(y[m])) < 2:
                    strata.append(None)
                    continue
                e = score_partition(x, y, d["exemplars"], c, K, m)
                cs = [score_partition(x, y, coreset_partition(pool, c, K, 300 + j),
                                      c, K, m) for j in range(3)]
                strata.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                               "n": int(len(m)),
                               "frac_pos": float((y[m] == 1).mean()),
                               "ep": e, "cs": float(np.mean(cs))})
            detail["tasks"][name][str(p)] = {"strata": strata}
        d4 = detail["tasks"][name]["4"]["strata"]
        print(f"  p=4 by distance stratum (near -> far):")
        for s in d4:
            if s is None:
                print("    [stratum has one class only]")
            else:
                print(f"    d in [{s['lo']:.3f},{s['hi']:.3f}) n={s['n']:<5d} "
                      f"pos={s['frac_pos']:.2f}  EP {s['ep']:.4f}  "
                      f"CS {s['cs']:.4f}  diff {s['ep'] - s['cs']:+.4f}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    detail["rows"] = rows
    OUT_JSON.write_text(json.dumps(detail, indent=2))
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
