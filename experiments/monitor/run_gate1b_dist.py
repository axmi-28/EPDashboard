"""Gate 1 Arm B, distance baselines for the drift monitor.

The categorical monitor's competitor is not only the categorical coreset. Any
per-request scalar can be aggregated over a traffic window, and averaging N
weak scores shrinks their noise by sqrt(N). Gate 0B's S1/S3/S4 are exactly such
scalars, and S3 reached AUROC 0.808 on R2 per request, so the window mean of a
*random coreset's distance* is the baseline the cell histogram actually has to
beat. Omitting it would let a weak win be reported as a strong one.

Emits the same schema as `gate1b_power.csv` so the two are concatenable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.monitor import occupancy as oc
from experiments.monitor.run_gate1b import EPSILONS, PS, REF_SPLIT, RUNGS, WINDOWS, N_WINDOWS
from experiments.monitor.scorers import (auroc, fit_mahalanobis, s1_s2_ep, s3_coreset_knn,
                             s4_mahalanobis, tpr_at_fpr)

N_DRAWS = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("artifacts/runs/monitor/eval.npz"))
    ap.add_argument("--refpool", type=Path, default=Path("artifacts/runs/monitor/refpool.npy"))
    ap.add_argument("--dicts", type=Path, default=Path("artifacts/runs/monitor/dicts"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/runs/monitor"))
    args = ap.parse_args()

    ev = np.load(args.eval, allow_pickle=True)
    x, rung = ev["x"], ev["rung"]
    pool = np.load(args.refpool, mmap_mode="r")
    r0 = np.flatnonzero(rung == "R0")
    clean_idx = r0[REF_SPLIT:]

    maha = fit_mahalanobis(pool)
    s4 = s4_mahalanobis(x, maha)

    power_rows, req_rows = [], []
    for p in PS:
        d = np.load(args.dicts / f"gemma-2-2b-it_L20_p{p}.npz", allow_pickle=True)
        E, center = d["exemplars"], d["center"]
        K = len(E)
        s1, _ = s1_s2_ep(x, E, center)
        _, per_draw = s3_coreset_knn(x, pool, center, K, n_draws=N_DRAWS)

        scorers = {"S1_ep_dist": s1, "S4_maha": s4}
        for j, s in enumerate(per_draw):
            scorers[f"S3_core_dist{j}"] = s
        print(f"=== p={p} K={K} ===")

        for name, sc in scorers.items():
            clean = sc[clean_idx]
            for r in RUNGS:
                dirty = sc[rung == r]
                y = np.concatenate([np.zeros(len(clean), bool),
                                    np.ones(len(dirty), bool)])
                v = np.concatenate([clean, dirty])
                req_rows.append({"percentile": p, "K": K, "scorer": name,
                                 "rung": r, "auroc": auroc(v, y),
                                 "tpr_at_1pct_fpr": tpr_at_fpr(v, y, 0.01)})
                for w in WINDOWS:
                    for e in EPSILONS:
                        pc = oc.scalar_power_curve(clean, dirty, window=w, eps=e,
                                                   n_windows=N_WINDOWS,
                                                   seed=p * 100 + w)
                        power_rows.append({
                            "percentile": p, "K": K, "scorer": name, "rung": r,
                            "window": w, "eps": e,
                            "power_surprisal": pc["power_mean"],
                            "power_tv": float("nan"),
                        })

    for path, rows in ((args.out / "gate1b_power_dist.csv", power_rows),
                       (args.out / "gate1b_per_request_dist.csv", req_rows)):
        with path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=sorted(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
