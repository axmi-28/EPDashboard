"""Stage 0, part 2: reproduce KE25's baseline quiver and check it against theirs.

The point is not to produce a number. It is to establish that our activations
and our probe fitting land on top of KE25's published table, because every
later EP claim is stated as a *margin* against that table. A margin measured
against a baseline we got wrong is worth nothing.

We call KE25's own `run_baseline_evals` -- their cross-validation folds,
their `Cs = logspace(5, -5, 10)` grid, their PCA/KNN/XGBoost/MLP sweeps -- so
the only things that can differ are the activations and the library versions.
That is exactly the difference we want the comparison to expose.

Agreement criterion, declared before running: per-dataset absolute test-AUC
difference to `layer{L}_results.csv` under a median of 0.005 with no dataset
above 0.05. Sources of legitimate small disagreement: bf16 vs whatever dtype
they extracted in, sklearn version drift in `saga`/`lbfgs` convergence, and
XGBoost/MLP nondeterminism. A *large* disagreement on a handful of datasets
usually means a tokenization or read-position mismatch, not noise.

Run:
  python -m experiments.probes.stage0_baselines --model gemma-2-9b --layer 20 \
      --results-root <SAE-Probes checkout>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.probes import benchmark as bm

METHODS = ("logreg", "pca", "knn", "xgboost", "mlp")


def collect_results(results_path: Path, model: str, hook: str) -> pd.DataFrame:
    """Gather the per-dataset JSON files their runner writes into one frame."""
    rows = []
    for f in sorted(results_path.rglob("*.json")):
        try:
            payload = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for rec in payload if isinstance(payload, list) else [payload]:
            if rec.get("hook_name") == hook:
                rows.append(rec)
    if not rows:
        raise SystemExit(f"no results under {results_path} for hook {hook}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=bm.HEADLINE_MODEL)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--results-root", type=Path, required=True,
                    help="checkout of github.com/JoshEngels/SAE-Probes")
    ap.add_argument("--cache", type=Path, default=bm.ARTIFACTS / "acts")
    ap.add_argument("--out", type=Path, default=bm.ARTIFACTS / "baselines")
    ap.add_argument("--device", default="cpu",
                    help="only used if activations are missing and must be made")
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    from sae_probes.run_baselines import run_baseline_evals

    hook = f"blocks.{args.layer}.hook_resid_post"
    args.out.mkdir(parents=True, exist_ok=True)

    for m in args.methods:
        print(f"\n=== {m} ===", flush=True)
        run_baseline_evals(
            model_name=args.model,
            hook_name=hook,
            setting="normal",
            method=m,  # type: ignore[arg-type]
            results_path=args.out,
            model_cache_path=args.cache,
            datasets=args.datasets,
            device=args.device,
        )

    ours = collect_results(args.out, args.model, hook)
    ours.to_csv(args.out / f"ours_{args.model}_L{args.layer}.csv", index=False)

    pub = bm.published_baselines(args.results_root, args.model, args.layer)
    cmp = ours.merge(
        pub[["dataset", "method", "test_auc", "val_auc"]],
        on=["dataset", "method"], suffixes=("_ours", "_ke25"), how="inner",
    )
    cmp["abs_diff"] = (cmp.test_auc_ours - cmp.test_auc_ke25).abs()
    cmp.to_csv(args.out / f"compare_{args.model}_L{args.layer}.csv", index=False)

    print("\n=== agreement with KE25, per method ===")
    print(cmp.groupby("method").abs_diff.agg(
        n="size", median="median", p95=lambda s: np.percentile(s, 95), max="max"
    ).round(4).to_string())

    med, worst = cmp.abs_diff.median(), cmp.abs_diff.max()
    print(f"\noverall median |Δtest AUC| = {med:.4f}   max = {worst:.4f}")
    print("PASS" if (med < 0.005 and worst < 0.05) else "FAIL -- do not proceed to EP arrows")
    if worst >= 0.05:
        print("\nworst offenders:")
        print(cmp.nlargest(10, "abs_diff")[
            ["dataset", "method", "test_auc_ours", "test_auc_ke25", "abs_diff"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
