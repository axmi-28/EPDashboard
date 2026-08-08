"""Gate 1 Arm B — occupancy-histogram drift monitor.

Reads only artifacts already on disk (`artifacts/runs/monitor/eval.npz`, `refpool.npy`,
`dicts/*.npz`). No model forward passes, no gradients.

Three outputs, matching the pre-registered rules in
`docs/experiments/PLAN_EP_MONITOR_GATE1.md`:

- **B1** detection power vs (window, contamination) at a false-alarm rate fixed
  at 1% by resampling clean windows, EP cells vs matched-K coreset cells.
- **B2** exemplar stability across arrival orders — the load-bearing assumption
  behind "comparable across time and checkpoints".
- **B3** occupancy balance, which is the proposed *mechanism* for any B1 win
  and must be reported either way.

Plus a per-request AUROC for the surprisal score, which puts the categorical
monitor on exactly the scale Gate 0B used and so makes the two arms
comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.monitor import occupancy as oc
from experiments.monitor.scorers import auroc, tpr_at_fpr

PS = (1, 2, 4, 8, 10)
RUNGS = ("R1", "R2", "R3", "R4", "R5")
WINDOWS = (50, 100, 200, 500)
EPSILONS = (0.01, 0.05, 0.1, 0.25, 1.0)
N_DRAWS = 3
N_WINDOWS = 300
REF_SPLIT = 1000          # R0 rows that build the reference histogram
STABILITY_N = 60_000      # build-stream rows for the B2 order test


def _coreset_dirs(pool: np.ndarray, center: np.ndarray, k: int,
                  seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), size=min(k, len(pool)), replace=False))
    return oc._unit(np.asarray(pool[idx]), center)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("artifacts/runs/monitor/eval.npz"))
    ap.add_argument("--refpool", type=Path,
                    default=Path("artifacts/runs/monitor/refpool.npy"))
    ap.add_argument("--dicts", type=Path, default=Path("artifacts/runs/monitor/dicts"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/runs/monitor"))
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    ev = np.load(args.eval, allow_pickle=True)
    x, rung = ev["x"], ev["rung"]
    pool = np.load(args.refpool, mmap_mode="r")

    # R0 is split once, deterministically: the first half defines the reference
    # histogram, the second half is the clean traffic windows are drawn from.
    # They must be disjoint or the null is calibrated on its own reference.
    r0 = np.flatnonzero(rung == "R0")
    ref_idx, clean_idx = r0[:REF_SPLIT], r0[REF_SPLIT:]
    print(f"R0 reference {len(ref_idx)}, clean pool {len(clean_idx)}")

    per_request, power_rows, balance_rows = [], [], []

    for p in PS:
        d = np.load(args.dicts / f"gemma-2-2b-it_L20_p{p}.npz", allow_pickle=True)
        E, center = d["exemplars"], d["center"]
        K = len(E)
        print(f"\n=== p={p} K={K} ===")

        # scorer name -> (cells for every eval row, n_cells)
        variants: dict[str, np.ndarray] = {"EP": oc.assign(x, E, center)}
        for j in range(N_DRAWS):
            variants[f"CORE{j}"] = oc.assign(
                x, _coreset_dirs(pool, center, K, seed=1000 + j), center)

        for name, cells in variants.items():
            ref_counts = oc.occupancy(cells[ref_idx], K)
            logp = oc.log_ref_prob(ref_counts)

            b = oc.balance(ref_counts)
            balance_rows.append({"percentile": p, "K": K, "scorer": name, **b})

            # per-request AUROC, directly comparable to Gate 0B's S1
            s_clean = oc.surprisal(cells[clean_idx], logp)
            for r in RUNGS:
                s_r = oc.surprisal(cells[rung == r], logp)
                s = np.concatenate([s_clean, s_r])
                y = np.concatenate([np.zeros(len(s_clean), bool),
                                    np.ones(len(s_r), bool)])
                per_request.append({
                    "percentile": p, "K": K, "scorer": name, "rung": r,
                    "auroc": auroc(s, y),
                    "tpr_at_1pct_fpr": tpr_at_fpr(s, y, 0.01),
                    "mean_surprisal_clean": float(s_clean.mean()),
                    "mean_surprisal_rung": float(s_r.mean()),
                })

            for r in RUNGS:
                dirty = cells[rung == r]
                for w in WINDOWS:
                    for e in EPSILONS:
                        pc = oc.power_curve(cells[clean_idx], dirty, ref_counts,
                                            window=w, eps=e,
                                            n_windows=N_WINDOWS, seed=p * 100 + w)
                        power_rows.append({
                            "percentile": p, "K": K, "scorer": name, "rung": r,
                            "window": w, "eps": e,
                            "power_surprisal": pc["power_surprisal"],
                            "power_tv": pc["power_tv"],
                        })
            print(f"  {name}: occupied {b['occupied']}/{K} "
                  f"gini {b['gini']:.3f} entropy_ratio {b['entropy_ratio']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "gate1b_per_request.csv", per_request)
    _write(args.out / "gate1b_power.csv", power_rows)
    _write(args.out / "gate1b_balance.csv", balance_rows)

    if not args.skip_stability:
        stab = stability(pool, args.dicts)
        (args.out / "gate1b_stability.json").write_text(json.dumps(stab, indent=2))
        print("\n-> gate1b_stability.json")

    print(f"-> {args.out}/gate1b_{{per_request,power,balance}}.csv")
    return 0


def stability(pool: np.ndarray, dicts: Path) -> dict:
    """B2 — rebuild the partition under two arrival orders and compare.

    'Comparable across time and checkpoints' assumes exemplar identity is a
    property of the data, not of the order it arrived in. Leader clustering
    takes first arrivals, so that assumption is testable directly.
    """
    out = {}
    sub = np.asarray(pool[:STABILITY_N])
    for p in PS:
        d = np.load(dicts / f"gemma-2-2b-it_L20_p{p}.npz", allow_pickle=True)
        center, th, hub = d["center"], float(d["threshold"]), d["exemplars"]
        a = oc.leader_cluster(sub, th, center,
                              np.random.default_rng(0).permutation(STABILITY_N))
        b = oc.leader_cluster(sub, th, center,
                              np.random.default_rng(1).permutation(STABILITY_N))
        ov = oc.exemplar_overlap(a, b, th)
        # hub exemplars are already unit directions in centred space
        ov_hub = oc.exemplar_overlap(a, hub, th)
        out[f"p{p}"] = {"threshold": th, "K_hub": int(len(hub)),
                        "order_vs_order": ov, "order_vs_hub": ov_hub}
        print(f"  p={p} K={ov['K_a']}/{ov['K_b']} (hub {len(hub)}) "
              f"matched {ov['frac_a_matched']:.3f}")
    return out


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
