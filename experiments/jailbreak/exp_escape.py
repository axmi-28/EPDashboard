"""Tier 0: the escape map. Forward passes only, no generation.

Question: when a harmful instruction is wrapped in a published jailbreak, does
its final-position activation still land in the region that the reference run
showed is causally responsible for refusal (pid 18, Δ_exemplar = -0.76)?

Two outcomes, both substantive:

  STAYS  -> the model still represents the request as harmful and the jailbreak
            must be overriding refusal somewhere downstream of L20. Region
            membership is necessary but not sufficient.
  LEAVES -> the jailbreak works by moving the prompt out of the region, which
            makes region membership a pre-generation detector.

Read against the benign arm in every case. The wrapper is most of the prompt;
without benign goals in the same wrappers there is no way to separate "harm
recognition failed" from "the activation now encodes the wrapper".

This tier needs no generation and no GPU rental — it runs on the saved
dictionary plus one batched forward pass per prompt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import corpus, dict_io, metrics, templates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=dict_io.DEFAULT_MODEL)
    ap.add_argument("--dictionary", default=str(dict_io.DEFAULT_DICT))
    ap.add_argument("--layer", type=int, default=dict_io.DEFAULT_LAYER)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-per-side", type=int, default=300)
    ap.add_argument("--output-dir", default="artifacts/runs/jailbreak")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = dict_io.load_dictionary(args.dictionary)
    target = dict_io.REFERENCE_REFUSAL_PID
    thr = float(dictionary.threshold)
    center = np.asarray(dictionary.center, dtype=np.float32)
    print(f"dictionary: {len(dictionary.partitions)} partitions, "
          f"threshold={thr:.6f}, target pid={target}")

    harmful, benign = corpus.load_build_prompts(args.n_per_side)
    grid = corpus.build_grid(harmful, benign)
    print(f"grid: {grid.n_goals} goals x {len(grid.template_names)} templates "
          f"= {grid.n_rows} prompts")

    model = dict_io.load_model(args.model, device=args.device, dtype=args.dtype)
    formatted = [corpus.format_chat(model, t) for t in grid.text]

    t0 = time.time()
    x = dict_io.final_position_activations(
        model, formatted, layer=args.layer, batch_size=args.batch_size,
    )
    print(f"extracted {x.shape} in {time.time() - t0:.0f}s")

    pl = dict_io.place(dictionary, x, target_pid=target)
    harmful_of_row = grid.harmful_of_row()

    # Row ordering within a template is goal order by construction; assert it
    # rather than trust it, because the paired displacement below aligns the
    # plain and wrapped arms positionally.
    plain_rows = grid.rows_for("plain")
    assert np.array_equal(grid.goal_idx[plain_rows], np.arange(grid.n_goals))

    per_template: dict[str, dict] = {}
    for t_name in grid.template_names:
        rows = grid.rows_for(t_name)
        assert np.array_equal(grid.goal_idx[rows], np.arange(grid.n_goals))
        harm = harmful_of_row[rows]

        entry = {
            "mechanism": templates.MECHANISM_OF[t_name],
            "harm_auroc": metrics.harm_auroc(pl.dist_target[rows], harm),
            "cells": metrics.cell_stats(
                pl.dist_target[rows], pl.assigned[rows], harm, target, thr,
            ),
            "mean_prompt_chars": float(
                np.mean([len(grid.text[i]) for i in rows]),
            ),
        }
        if t_name != "plain":
            # Displacement measured separately per arm: a wrapper that moves
            # harmful and benign goals by the same angle is doing something
            # different from one that moves only the harmful ones.
            for arm, mask in (("harmful", harm == 1), ("benign", harm == 0)):
                entry[f"displacement_{arm}"] = metrics.wrapper_displacement(
                    x[plain_rows[mask]], x[rows[mask]], center, thr,
                )
            entry["transitions_harmful"] = metrics.transitions(
                pl.assigned[plain_rows[harm == 1]],
                pl.assigned[rows[harm == 1]], target,
            )
            entry["transitions_benign"] = metrics.transitions(
                pl.assigned[plain_rows[harm == 0]],
                pl.assigned[rows[harm == 0]], target,
            )
        per_template[t_name] = entry

    summary = metrics.mechanism_summary(per_template, templates.MECHANISM_OF)

    # ------------------------------------------------------------- report
    print(f"\n{'template':22s} {'mech':10s} {'AUROC':>6s} "
          f"{'stay_h':>7s} {'stay_b':>7s} {'esc_dif':>8s} "
          f"{'d18_h':>6s} {'d18_b':>6s} {'disp_h':>7s} {'disp/θ':>7s}")
    print("-" * 100)
    for t_name in grid.template_names:
        e = per_template[t_name]
        c = e["cells"]
        disp = e.get("displacement_harmful")
        print(
            f"{t_name:22s} {e['mechanism'][:10]:10s} "
            f"{e['harm_auroc']:6.3f} "
            f"{c['harmful']['assigned_rate']:7.3f} "
            f"{c['benign']['assigned_rate']:7.3f} "
            f"{c['escape_differential']:+8.3f} "
            f"{c['harmful']['dist_mean']:6.3f} "
            f"{c['benign']['dist_mean']:6.3f} "
            + (f"{disp['mean']:7.3f} {disp['ratio_to_threshold']:7.2f}"
               if disp else f"{'—':>7s} {'—':>7s}")
        )

    print(f"\nthreshold θ = {thr:.3f}   "
          "(AUROC 0.5 = region no longer distinguishes harm; "
          "stay_h = fraction of harmful prompts still assigned to pid 18)")

    print("\nby mechanism class:")
    for cls, s in sorted(summary.items()):
        print(f"  {cls:26s} AUROC {s['harm_auroc_mean']:.3f} "
              f"[{s['harm_auroc_min']:.3f}, {s['harm_auroc_max']:.3f}]   "
              f"esc_diff {s['escape_differential_mean']:+.3f}   "
              f"{s['templates']}")

    result = {
        "config": vars(args),
        "target_pid": target,
        "threshold": thr,
        "n_goals": grid.n_goals,
        "per_template": per_template,
        "by_mechanism": summary,
    }
    with (out_dir / "escape.json").open("w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        out_dir / "escape_placements.npz",
        assigned=pl.assigned, dist_target=pl.dist_target,
        dist_assigned=pl.dist_assigned, goal_idx=grid.goal_idx,
        template=grid.template, is_harmful=grid.is_harmful,
        template_names=np.array(grid.template_names),
    )
    print(f"\n-> {out_dir / 'escape.json'}")
    print(f"-> {out_dir / 'escape_placements.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
