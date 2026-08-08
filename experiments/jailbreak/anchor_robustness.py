"""Is the escape result an artifact of which activation happened to anchor pid 18?

Every distance reported so far is to pid 18's `exemplar_direction` — a
first-arrival accident. The reference found that 2 of 4 streaming seeds give
Δ = 0 for the same percentile, so anchor identity demonstrably matters, and
cos(exemplar, mean_member) is only **0.899** for this partition (~26°), below
the ~0.94 the paper reports typically.

Re-running with a different streaming seed would need a full rebuild. This is
the cheap proxy: recompute every statistic against three anchors that differ in
how much they depend on first arrival.

  exemplar     the first activation to open the cell — maximally seed-dependent
  mean         the spherical mean of 734 members — nearly seed-independent,
               since it averages away arrival order
  reanchored   the sample member closest to the mean (the reference's
               `exemplar_reanchored` basis) — a real activation, but chosen
               deterministically rather than by arrival

If the harm-separation and jailbreak-prediction AUROCs survive the swap to
`mean`, the finding is a property of the cell rather than of its anchor, and
the seed objection is answered for everything except the causal ablation —
where the paper's own result is that exemplar beats mean by 0.4-0.6, so the
two are *not* interchangeable and only a real second seed settles it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import corpus, dict_io, metrics
from .rescore import COMPREHENSION_LIMITED, make_extended_scorer

ANCHORS = ("exemplar", "mean", "reanchored")


def anchor_directions(partition) -> dict[str, np.ndarray]:
    e = partition.exemplar_direction.astype(np.float64)
    m = partition.mean_member_direction.astype(np.float64)
    if partition.sample_members:
        S = np.stack(partition.sample_members).astype(np.float64)
        r = S[int(np.argmax(S @ m))]
    else:
        r = e
    return {"exemplar": e, "mean": m, "reanchored": r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=dict_io.DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=dict_io.DEFAULT_LAYER)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--generations", default="artifacts/runs/jailbreak/generations.json")
    ap.add_argument("--output", default="artifacts/runs/jailbreak/anchor_robustness.json")
    ap.add_argument("--save-activations",
                    default="artifacts/runs/jailbreak/grid_activations.npy")
    args = ap.parse_args()

    dictionary = dict_io.load_dictionary()
    part = dictionary.partitions[dict_io.REFERENCE_REFUSAL_PID]
    dirs = anchor_directions(part)
    center = np.asarray(dictionary.center, dtype=np.float64)

    print("anchor geometry for pid 18:")
    for a in ANCHORS:
        for b in ANCHORS:
            if a < b:
                print(f"  cos({a}, {b}) = {float(dirs[a] @ dirs[b]):.4f}")

    harmful, benign = corpus.load_build_prompts(300)
    grid = corpus.build_grid(harmful, benign)

    act_path = Path(args.save_activations)
    if act_path.exists():
        x = np.load(act_path)
        print(f"\nreusing activations {x.shape} from {act_path}")
    else:
        model = dict_io.load_model(args.model, device=args.device,
                                   dtype=args.dtype)
        formatted = [corpus.format_chat(model, t) for t in grid.text]
        x = dict_io.final_position_activations(
            model, formatted, layer=args.layer, batch_size=args.batch_size,
        )
        act_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(act_path, x)
        print(f"\nextracted {x.shape} -> {act_path}")

    v = x.astype(np.float64) - center
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    dist = {a: 1.0 - u @ dirs[a] for a in ANCHORS}

    # Correlation across all 5400 prompts: if the anchors rank prompts almost
    # identically the robustness claim is trivial; if they diverge, the
    # agreement below is informative.
    print("\nSpearman correlation of per-prompt distance between anchors:")
    from scipy.stats import spearmanr
    for a in ANCHORS:
        for b in ANCHORS:
            if a < b:
                rho = spearmanr(dist[a], dist[b]).statistic
                print(f"  {a:11s} vs {b:11s}  rho = {rho:.4f}")

    harm_of_row = grid.harmful_of_row()
    sc = make_extended_scorer()
    gens = json.loads(Path(args.generations).read_text())
    refused_by_row = {int(g["row"]): bool(sc(g["generation"])) for g in gens}

    out: dict[str, dict] = {}
    print(f"\n{'':22s} {'harm AUROC (harmful vs benign)':^36s} "
          f"{'jailbreak AUROC (among harmful)':^36s}")
    print(f"{'template':22s} " + " ".join(f"{a:>11s}" for a in ANCHORS)
          + "  " + " ".join(f"{a:>11s}" for a in ANCHORS))
    print("-" * 100)
    for t_name in grid.template_names:
        rows = grid.rows_for(t_name)
        harm = harm_of_row[rows]
        entry = {"harm_auroc": {}, "jailbreak_auroc": {}}
        for a in ANCHORS:
            entry["harm_auroc"][a] = metrics.harm_auroc(dist[a][rows], harm)

        scored = [r for r in rows[harm == 1] if int(r) in refused_by_row]
        if scored and t_name not in COMPREHENSION_LIMITED:
            ref = np.array([refused_by_row[int(r)] for r in scored])
            if 0 < ref.sum() < len(ref):
                for a in ANCHORS:
                    # score = +distance predicting jailbreak success
                    entry["jailbreak_auroc"][a] = metrics._auroc(
                        dist[a][np.array(scored)], ~ref,
                    )
        out[t_name] = entry
        ja = entry["jailbreak_auroc"]
        print(f"{t_name:22s} "
              + " ".join(f"{entry['harm_auroc'][a]:11.3f}" for a in ANCHORS)
              + "  "
              + " ".join((f"{ja[a]:11.3f}" if a in ja else f"{'—':>11s}")
                         for a in ANCHORS))

    with Path(args.output).open("w") as f:
        json.dump({
            "anchor_cosines": {f"{a}|{b}": float(dirs[a] @ dirs[b])
                               for a in ANCHORS for b in ANCHORS if a < b},
            "per_template": out,
        }, f, indent=2)
    print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
