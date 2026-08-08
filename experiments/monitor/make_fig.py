"""One figure: AUROC per (scorer, rung), with K on the x-axis.

The comparison the gate turns on is S1/S2 against S3 at matched K, so the two
EP scorers are drawn solid and the matched-budget coreset dashed in the same
panel. S4 and S5 are drawn faint — neither is memory-matched, and they are
present to answer "is the framing dead", not "does EP win". S0 is drawn as a
grey band because it is not a monitor at all; where it rises, the rung is
telling you about prompt length.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

STYLE = {
    "S1_ep_nearest":  {"color": "#1b6ca8", "ls": "-",  "lw": 2.2, "marker": "o"},
    "S2_ep_margin":   {"color": "#5aa9e6", "ls": "-",  "lw": 1.8, "marker": "s"},
    "S3_coreset_knn": {"color": "#d1495b", "ls": "--", "lw": 2.2, "marker": "D"},
    "S4_mahalanobis": {"color": "#7a7a7a", "ls": ":",  "lw": 1.5, "marker": "^"},
    "S5_entropy_max": {"color": "#b8b8b8", "ls": ":",  "lw": 1.5, "marker": "v"},
    "S0_token_count": {"color": "#c9c9c9", "ls": "-",  "lw": 1.0, "marker": None},
}
RUNGS = ("R1", "R2", "R3", "R4", "R5")
LABELS = {
    "R1": "R1 domain\n(code + math)", "R2": "R2 language\n(Bulgarian wiki)",
    "R3": "R3 template\n(chat scaffolds)", "R4": "R4 jailbreak",
    "R5": "R5 random tokens",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("artifacts/runs/monitor/gate0b_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/figures/gate0b_auroc.png"))
    ap.add_argument("--metric", default="auroc",
                    choices=["auroc", "tpr_at_1pct_fpr"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(args.csv.open()))
    for r in rows:
        r["K"] = int(r["K"])
        r[args.metric] = float(r[args.metric])

    fig, axes = plt.subplots(1, len(RUNGS), figsize=(4 * len(RUNGS), 4.2),
                             sharey=True)
    for ax, rung in zip(axes, RUNGS):
        sub = [r for r in rows if r["rung"] == rung]
        for scorer, st in STYLE.items():
            pts = sorted({(r["K"], r[args.metric]) for r in sub
                          if r["scorer"] == scorer})
            if not pts:
                continue
            ks = [p[0] for p in pts]
            vs = [p[1] for p in pts]
            label = scorer
            if scorer == "S0_token_count" and args.metric == "auroc":
                # An AUROC of 0.002 means "shorter prompts are the OOD class",
                # which is exactly as much a length artifact as 0.998 would be.
                # Fold around chance so the control is visible and comparable.
                vs = [max(v, 1.0 - v) for v in vs]
                label = "S0 length control (folded)"
            ax.plot(ks, vs, label=label, **st, markersize=5, zorder=3)
            if scorer == "S3_coreset_knn":
                stds = [float(r["auroc_draw_std"]) for k in ks
                        for r in sub if r["scorer"] == scorer and r["K"] == k]
                if args.metric == "auroc" and len(stds) == len(ks):
                    ax.fill_between(ks, [v - s for v, s in zip(vs, stds)],
                                    [v + s for v, s in zip(vs, stds)],
                                    color=st["color"], alpha=0.15, zorder=1)
        ax.axhline(0.5, color="k", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ks_all = sorted({r["K"] for r in sub})
        ax.set_xticks(ks_all)
        ax.set_xticklabels([str(k) for k in ks_all], fontsize=8)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("K (memory budget, exemplars)")
        ax.set_title(LABELS[rung], fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25, zorder=0)
    axes[0].set_ylabel({"auroc": "AUROC vs R0",
                        "tpr_at_1pct_fpr": "TPR @ 1% FPR"}[args.metric])
    axes[-1].legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.suptitle("Gate 0B — EP nearest-exemplar distance vs matched-budget "
                 "baselines (gemma-2-2b-it L20, final token)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
