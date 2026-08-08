"""Gate 2 A1 — the label-efficiency curve.

A0 and A3 hand every available label to every scorer, and under those
conditions a linear probe on the raw activation wins by a wide margin. That was
the expected outcome and it is not the claim under test.

The claim under test is about *shape*. A region lookup ranks K bins; a probe
fits a 2304-dimensional hyperplane. Ranking bins should need fewer labels, so
the prediction is a crossover: EP ahead when labels are scarce, the probe ahead
once they are plentiful. The finding is the budget at which the lines cross, or
that they never do.

Protocol, per task:

- split the labelled set 50/50 into a **fit pool** and an **eval pool**,
  stratified on the label. The eval pool is scored by every scorer at every
  budget and is never drawn from.
- for each budget `n`, draw `n` labels at random from the fit pool, 20 times.
- fit all four scorers on that draw, score the whole eval pool, record AUROC
  and TPR@1%FPR.

The partition itself costs no labels — it was built unsupervised on Pile — so
at n=16 the EP scorer has 16 counts spread over K bins while the probe has 16
points in 2304 dimensions. That asymmetry is the entire hypothesis.

    python -m experiments.monitor.run_gate2_a1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dicts import DEFAULT_MODEL_SHORT, PERCENTILES, lean_path, load_lean
from .gate2_route import (assign, coreset_partition, region_rates, _ridge_dual,
                          EPS)
from .run_gate2_a3 import build_tasks
from .scorers import auroc, tpr_at_fpr

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
OUT_CSV = ROOT / "gate2_a1_labelcurve.csv"
OUT_JSON = ROOT / "gate2_a1_labelcurve.json"

LAYER = 20
BUDGETS = (16, 32, 64, 128, 256, 512, 1024)
N_DRAWS = 20
N_CORESET_PARTITIONS = 3


def _stratified_half(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fit[idx[:len(idx) // 2]] = True
    return fit, ~fit


def _probe_fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_ev: np.ndarray,
                     kind: str, lam: float = 1.0) -> np.ndarray:
    mu = x_tr.mean(axis=0)
    if kind == "ridge":
        w = _ridge_dual(x_tr, y_tr, lam)
    else:
        xn = x_tr - mu
        xn /= np.linalg.norm(xn, axis=1, keepdims=True) + EPS
        if (y_tr == 1).sum() == 0 or (y_tr == 0).sum() == 0:
            return np.zeros(len(x_ev))
        w = xn[y_tr == 1].mean(axis=0) - xn[y_tr == 0].mean(axis=0)
    w = w / (np.linalg.norm(w) + EPS)
    return (x_ev - mu) @ w


def run_task(name: str, x: np.ndarray, y: np.ndarray, pool: np.ndarray,
             *, budgets, n_draws: int, seed: int = 0) -> list[dict]:
    fit_mask, ev_mask = _stratified_half(y, seed)
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    x_ev, y_ev = x[ev_mask].astype(np.float64), y[ev_mask]
    ev_pos = y_ev == 1
    print(f"\n=== {name}: fit pool {len(y_fit)}, eval pool {len(y_ev)} "
          f"({int(y_ev.sum())} pos) ===")

    # Assignments are label-free, so they are computed once and reused for
    # every budget and every draw.
    parts: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for p in PERCENTILES:
        d = load_lean(lean_path(DICTS, DEFAULT_MODEL_SHORT, LAYER, p))
        c, K = d["center"], d["K"]
        r = assign(x, d["exemplars"], c)
        parts[f"EP-FLAG_p{p}"] = (r[fit_mask], r[ev_mask], K)
        for j in range(N_CORESET_PARTITIONS):
            ref = coreset_partition(pool, c, K, 1000 + j)
            rc = assign(x, ref, c)
            parts[f"CORE-FLAG_p{p}_d{j}"] = (rc[fit_mask], rc[ev_mask], K)

    x_fit64 = x_fit.astype(np.float64)
    rows: list[dict] = []
    for n in budgets:
        if n > len(y_fit):
            continue
        acc: dict[str, list[tuple[float, float]]] = {}
        for draw in range(n_draws):
            rng = np.random.default_rng(10_000 * seed + 97 * n + draw)
            # Stratified draw: an unstratified n=16 draw is all-one-class ~1%
            # of the time, which would inject undefined AUROCs into the mean.
            sel = np.concatenate([
                rng.choice(np.where(y_fit == cls)[0], size=n // 2, replace=False)
                for cls in (0, 1)])
            ys = y_fit[sel]

            for key, (r_fit, r_ev, K) in parts.items():
                rates, _ = region_rates(r_fit[sel], ys, K)
                s = rates[r_ev]
                acc.setdefault(key, []).append(
                    (auroc(s, ev_pos), tpr_at_fpr(s, ev_pos, 0.01)))
            for kind, key in (("ridge", "PROBE"), ("diffmean", "DIFFMEAN")):
                s = _probe_fit_score(x_fit64[sel], ys.astype(np.float64),
                                     x_ev, kind)
                acc.setdefault(key, []).append(
                    (auroc(s, ev_pos), tpr_at_fpr(s, ev_pos, 0.01)))

        # Average the coreset partitions together first: they are draws of the
        # same null, not separate scorers.
        merged: dict[str, list] = {}
        for key, vals in acc.items():
            base = key.split("_d")[0]
            merged.setdefault(base, []).extend(vals)
        for key, vals in sorted(merged.items()):
            a = np.array(vals)
            rows.append({
                "task": name, "n_labels": n, "scorer": key,
                "auroc_mean": float(a[:, 0].mean()),
                "auroc_sd": float(a[:, 0].std(ddof=1)),
                "tpr1_mean": float(a[:, 1].mean()),
                "tpr1_sd": float(a[:, 1].std(ddof=1)),
                "n_obs": int(len(a)),
            })
        best_ep = max((r for r in rows if r["n_labels"] == n
                       and r["scorer"].startswith("EP-FLAG")),
                      key=lambda r: r["auroc_mean"])
        pr = next(r for r in rows if r["n_labels"] == n and r["scorer"] == "PROBE")
        dm = next(r for r in rows if r["n_labels"] == n
                  and r["scorer"] == "DIFFMEAN")
        core = next(r for r in rows if r["n_labels"] == n
                    and r["scorer"] == best_ep["scorer"].replace("EP-", "CORE-"))
        print(f"  n={n:<5d} best EP {best_ep['scorer'].split('_')[1]:>4s} "
              f"{best_ep['auroc_mean']:.4f}±{best_ep['auroc_sd']:.4f} | "
              f"CORE {core['auroc_mean']:.4f}±{core['auroc_sd']:.4f} | "
              f"PROBE {pr['auroc_mean']:.4f}±{pr['auroc_sd']:.4f} | "
              f"DIFFMEAN {dm['auroc_mean']:.4f}±{dm['auroc_sd']:.4f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    args = ap.parse_args()

    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")
    rows: list[dict] = []

    # Refusal, both formats, from the A0 extraction.
    a0 = np.load(ROOT / "gate2_a0_acts.npz")
    for fmt in ("chat", "raw"):
        rows += run_task(f"refusal_{fmt}", a0[f"x_{fmt}"], a0["y"], pool,
                         budgets=BUDGETS, n_draws=args.n_draws)

    z = np.load(ROOT / "eval.npz")
    x_all = z["x"]
    for name, (keep, y) in build_tasks(z["source"], z["rung"]).items():
        rows += run_task(name, x_all[keep], y, pool,
                         budgets=BUDGETS, n_draws=args.n_draws)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    OUT_JSON.write_text(json.dumps(
        {"layer": LAYER, "budgets": list(BUDGETS), "n_draws": args.n_draws,
         "n_coreset_partitions": N_CORESET_PARTITIONS, "rows": rows}, indent=2))

    print("\n--- crossover: smallest budget where best EP-FLAG >= PROBE ---")
    tasks = sorted({r["task"] for r in rows}, key=lambda t: t)
    for t in tasks:
        cross = None
        for n in BUDGETS:
            ep = [r for r in rows if r["task"] == t and r["n_labels"] == n
                  and r["scorer"].startswith("EP-FLAG")]
            pr = [r for r in rows if r["task"] == t and r["n_labels"] == n
                  and r["scorer"] == "PROBE"]
            if not ep or not pr:
                continue
            if max(e["auroc_mean"] for e in ep) >= pr[0]["auroc_mean"]:
                cross = n
                break
        print(f"  {t:14s} {'never' if cross is None else f'n={cross}'}")
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
