"""Reproduction gate. Run this before anything else touches a jailbreak.

The experiment reuses a saved artifact — the L20 p12 seed0 dictionary whose
region 18 collapsed all 300 harmful build prompts and whose exemplar ablation
gave Δ = -0.76. Every downstream claim is "the jailbroken prompt did/didn't
leave *that* region". So the first thing to establish is that re-running the
plain build prompts through this checkout puts them back where the reference
run put them.

If this fails, the likely causes in order: wrong dtype/device changing the
activation geometry, a chat template that no longer matches, or the wrong
pickle. All three produce plausible-looking numbers downstream, none of them
raise.

Exit 0 = pass, 2 = fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import corpus, dict_io


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=dict_io.DEFAULT_MODEL)
    ap.add_argument("--dictionary", default=str(dict_io.DEFAULT_DICT))
    ap.add_argument("--layer", type=int, default=dict_io.DEFAULT_LAYER)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-per-side", type=int, default=300)
    # Tolerance is on the harmful-recall claim, which is the one the
    # experiment leans on ("all 300 harmful prompts land in region 18").
    # Member count is allowed to drift more because the 105 benign members sit
    # near the cell boundary and are genuinely sensitive to dtype.
    ap.add_argument("--min-harmful-recall", type=float, default=0.97)
    ap.add_argument("--output", default="artifacts/runs/jailbreak/gate.json")
    args = ap.parse_args()

    dictionary = dict_io.load_dictionary(args.dictionary)
    print(f"dictionary: {len(dictionary.partitions)} partitions, "
          f"threshold={dictionary.threshold:.6f}")

    harmful, benign = corpus.load_build_prompts(args.n_per_side)
    model = dict_io.load_model(args.model, device=args.device, dtype=args.dtype)
    formatted = [corpus.format_chat(model, p) for p in harmful + benign]
    is_harmful = np.array([1] * len(harmful) + [0] * len(benign))

    x = dict_io.final_position_activations(
        model, formatted, layer=args.layer, batch_size=args.batch_size,
    )
    pl = dict_io.place(dictionary, x)

    pid = dict_io.REFERENCE_REFUSAL_PID
    members = pl.assigned == pid
    n_members = int(members.sum())
    harmful_fraction = float(is_harmful[members].mean()) if n_members else float("nan")
    harmful_recall = float(members[is_harmful == 1].mean())
    benign_in_cell = int(members[is_harmful == 0].sum())

    print(f"\nregion {pid}: n_members={n_members} "
          f"(reference {dict_io.REFERENCE_PID18_MEMBERS})")
    print(f"  harmful_fraction={harmful_fraction:.4f} "
          f"(reference {dict_io.REFERENCE_PID18_HARMFUL_FRACTION:.4f})")
    print(f"  harmful recall  ={harmful_recall:.4f}  "
          f"({int(members[is_harmful == 1].sum())}/{int((is_harmful == 1).sum())})")
    print(f"  benign members  ={benign_in_cell}")
    print(f"  mean dist to exemplar: harmful={pl.dist_target[is_harmful == 1].mean():.4f} "
          f"benign={pl.dist_target[is_harmful == 0].mean():.4f} "
          f"(threshold {pl.threshold:.4f})")

    # How concentrated is the corpus overall? The reference reported that the
    # chat scaffold consolidates instruction-formatted prompts into a handful
    # of final-position cells; worth confirming, because a jailbreak escape
    # into a cell that holds 3 prompts means something different from an
    # escape into one that holds 200.
    uniq, counts = np.unique(pl.assigned, return_counts=True)
    top = np.argsort(-counts)[:5]
    print("\n  top occupied cells: "
          + ", ".join(f"pid{uniq[i]}={counts[i]}" for i in top)
          + f"  ({len(uniq)} of {len(dictionary.partitions)} occupied)")

    ok = harmful_recall >= args.min_harmful_recall
    out = {
        "pass": bool(ok),
        "target_pid": pid,
        "n_members": n_members,
        "harmful_fraction": harmful_fraction,
        "harmful_recall": harmful_recall,
        "benign_in_cell": benign_in_cell,
        "threshold": pl.threshold,
        "mean_dist_target_harmful": float(pl.dist_target[is_harmful == 1].mean()),
        "mean_dist_target_benign": float(pl.dist_target[is_harmful == 0].mean()),
        "n_occupied_cells": int(len(uniq)),
        "reference": {
            "n_members": dict_io.REFERENCE_PID18_MEMBERS,
            "harmful_fraction": dict_io.REFERENCE_PID18_HARMFUL_FRACTION,
        },
        "config": vars(args),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")

    if not ok:
        print(f"\nGATE FAIL: harmful recall {harmful_recall:.3f} < "
              f"{args.min_harmful_recall}. The saved dictionary does not "
              "reproduce on this checkout; do not run the jailbreak arms.")
        return 2
    print("\nGATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
