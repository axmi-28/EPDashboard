"""Gate 2 A3 — is the region lookup worth anything on *any* labelled concept?

A0 asked whether harmful/benign routes. Whatever it answers, it answers about
one concept on prompts wearing a chat scaffold the hub dictionaries never saw
at build time. A3 removes both confounds at zero extraction cost: the 12,000
activations in `eval.npz` are already on disk, they are raw text in the same
form the dictionaries were built on, and they carry three more labels.

| task      | positives                | negatives                 |
|-----------|--------------------------|---------------------------|
| code_math | MBPP code prompts (974)  | GSM8K word problems (1026)|
| language  | Bulgarian Wikipedia (2000)| English Pile (2000)      |
| scaffold  | any of 5 chat wrappers (2000) | bare Pile (2000)     |

`scaffold` is the interesting one. Its positives and negatives are the *same
Pile content* — only the surrounding template differs — so it isolates pure
format, the kind of disjoint, non-linear thing a piecewise-constant lookup is
supposed to be better at than one hyperplane. If EP's partition never beats a
random partition even there, the argument has no remaining foothold.

Same three scorers as A0: EP lookup, matched-K coreset lookup, linear probe.

    python -m experiments.monitor.run_gate2_a3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .dicts import DEFAULT_MODEL_SHORT, PERCENTILES, lean_path, load_lean
from .gate2_route import assign, probe_cross_fit, summarise, summarise_coreset
from .scorers import auroc, tpr_at_fpr

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
OUT_CSV = ROOT / "gate2_a3_concepts.csv"
OUT_JSON = ROOT / "gate2_a3_concepts.json"

LAYER = 20
N_CORESET_DRAWS = 10


def build_tasks(source: np.ndarray, rung: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean index + label per task. `None` label means the row is excluded."""
    src = np.array([str(s) for s in source])
    rg = np.array([str(r) for r in rung])

    def task(pos_mask, neg_mask):
        keep = pos_mask | neg_mask
        return keep, pos_mask[keep].astype(np.int64)

    scaffold_pos = rg == "R3"
    return {
        "code_math": task(src == "mbpp", src == "gsm8k"),
        "language": task(src == "wikipedia_bg", src == "pile_heldout"),
        "scaffold": task(scaffold_pos, src == "pile_heldout"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-draws", type=int, default=N_CORESET_DRAWS)
    args = ap.parse_args()

    z = np.load(ROOT / "eval.npz")
    x_all = z["x"]
    tasks = build_tasks(z["source"], z["rung"])
    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")

    rows: list[dict] = []
    probes: dict[str, dict] = {}
    for name, (keep, y) in tasks.items():
        x = x_all[keep]
        print(f"\n=== {name}: n={len(y)} ({int(y.sum())} pos / "
              f"{int((1 - y).sum())} neg) ===")
        probes[name] = {}
        for kind in ("ridge", "diffmean"):
            s = probe_cross_fit(x, y, kind=kind)
            probes[name][kind] = {"cv_auroc": auroc(s, y == 1),
                                  "cv_tpr1": tpr_at_fpr(s, y == 1, 0.01)}
            print(f"PROBE {kind:9s} cvAUROC={probes[name][kind]['cv_auroc']:.4f} "
                  f"TPR@1%={probes[name][kind]['cv_tpr1']:.3f}")

        for p in PERCENTILES:
            d = load_lean(lean_path(DICTS, DEFAULT_MODEL_SHORT, LAYER, p))
            center, K = d["center"], d["K"]
            ep = summarise(assign(x, d["exemplars"], center), y, K)
            cs_mean, cs_sd, _ = summarise_coreset(x, y, pool, center, K,
                                                  n_draws=args.n_draws)
            margin = ((ep["cv_auroc"] - cs_mean["cv_auroc"]) / cs_sd["cv_auroc"]
                      if cs_sd["cv_auroc"] > 1e-9 else float("inf"))
            row = {"task": name, "percentile": p, "K": K,
                   "n": int(len(y)), "n_pos": int(y.sum()),
                   "probe_ridge_auroc": probes[name]["ridge"]["cv_auroc"],
                   "probe_diffmean_auroc": probes[name]["diffmean"]["cv_auroc"]}
            row.update({f"ep_{k}": v for k, v in ep.items() if k != "K"})
            row.update({f"cs_{k}": v for k, v in cs_mean.items()})
            row.update({f"cs_{k}_sd": v for k, v in cs_sd.items()})
            row["auroc_margin_sd"] = margin
            rows.append(row)
            print(f"  p{p:<3d} K={K:<5d} EP occ={ep['n_occupied']:<5d} "
                  f"cvAUROC={ep['cv_auroc']:.4f} TPR@1%={ep['cv_tpr1']:.3f} "
                  f"| CORE {cs_mean['cv_auroc']:.4f}±{cs_sd['cv_auroc']:.4f} "
                  f"| margin={margin:+.1f}sd", flush=True)

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    OUT_JSON.write_text(json.dumps(
        {"layer": LAYER, "n_coreset_draws": args.n_draws,
         "probes": probes, "rows": rows}, indent=2))

    print("\n--- margin over matched-K coreset (sd), by task x percentile ---")
    print(f"{'task':10s} " + " ".join(f"{'p'+str(p):>8s}" for p in PERCENTILES))
    for name in tasks:
        sub = {r["percentile"]: r["auroc_margin_sd"] for r in rows
               if r["task"] == name}
        print(f"{name:10s} " + " ".join(f"{sub[p]:+8.1f}" for p in PERCENTILES))
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
