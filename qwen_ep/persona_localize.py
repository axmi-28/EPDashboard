"""P2 — do personas localize into EP cells, and does the partition agree with
the linear Assistant Axis?

Takes the per-rollout response-mean activations from ``persona_axis`` (P1) and
an existing EP dictionary built at the same layer, assigns every rollout to its
Voronoi cell, and asks three questions:

  1. Localization  — does each role concentrate in a dominant cell, or smear?
     (the go/no-go gate: refusal concentrated ~99%; personas may not).
  2. Assistant anchor — which cell do the *default* Assistant rollouts occupy,
     how pure is it (P[assistant-group | cell]), and how much of the default
     mass lands there?
  3. Bridge — rank roles by EP distance-to-Assistant-anchor and correlate that
     discrete-geometry scalar with the linear Assistant Axis projection from P1.
     High correlation = the training-free partition recovers the same persona
     geometry a difference-of-means axis finds.

Example:
    python -m qwen_ep.persona_localize \
        --acts runs/persona_axis/qwen3_5-4b_spectrum_q16_sp3_seed0/activations.npz \
        --dict runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile --layer 27
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .persona_data import DEFAULT_ROLE

log = logging.getLogger("qwen_ep.persona_localize")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum()) + 1e-12
    return float((ra * rb).sum() / denom)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True, help="P1 activations.npz")
    ap.add_argument("--dict", required=True, help="run dir with dictionary.pkl")
    ap.add_argument("--layer", type=int, required=True,
                    help="which captured layer to use (must match the dict)")
    ap.add_argument("--out-name", default="persona_localize.json")
    args = ap.parse_args()

    acts_dir = Path(args.acts).parent
    z = np.load(args.acts, allow_pickle=True)
    layers = z["layers"].tolist()
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in captured {layers}")
    li = layers.index(args.layer)
    X = z["acts"][:, li, :].astype(np.float32)          # (N, d)
    role = z["role"].astype(str)
    group = z["group"].astype(str)
    roles = sorted(set(role.tolist()), key=lambda r: (r != DEFAULT_ROLE, r))

    with (Path(args.dict) / "dictionary.pkl").open("rb") as f:
        dic = pickle.load(f)
    K = len(dic.partitions)
    cell, dist = dic.assign(X)                          # (N,), (N,)
    center = np.asarray(dic.center, dtype=np.float32)
    E = np.stack([p.exemplar_direction for p in dic.partitions]).astype(np.float32)
    threshold = float(dic.threshold)
    log.info("dict K=%d threshold=%.4f  assigned %d rollouts", K, threshold, len(X))

    # -- Assistant anchor = modal cell of the default rollouts ----------------
    def_cells = cell[role == DEFAULT_ROLE]
    anchor = int(Counter(def_cells.tolist()).most_common(1)[0][0])
    in_anchor = cell == anchor
    anchor_purity = float((group[in_anchor] == "assistant").mean()) if in_anchor.any() else 0.0
    default_coverage = float((def_cells == anchor).mean())

    # distance of every rollout to the anchor exemplar (the EP "axis" scalar)
    Xc = X - center
    dirs = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    dist_anchor = 1.0 - dirs @ E[anchor]                # (N,)

    # -- per-role summary -----------------------------------------------------
    per_role = {}
    for r in roles:
        m = role == r
        cc = Counter(cell[m].tolist())
        top_cell, top_n = cc.most_common(1)[0]
        per_role[r] = {
            "group": group[m][0],
            "n": int(m.sum()),
            "n_distinct_cells": len(cc),
            "top_cell": int(top_cell),
            "top_cell_frac": round(top_n / m.sum(), 3),   # localization
            "frac_in_anchor": round(float((cell[m] == anchor).mean()), 3),
            "mean_dist_to_anchor": round(float(dist_anchor[m].mean()), 4),
            "member_rate": round(float((dist[m] <= threshold).mean()), 3),
        }

    # -- bridge to the linear axis (P1) --------------------------------------
    bridge = None
    axis_npz = acts_dir / f"axis_L{args.layer}.npz"
    if axis_npz.exists():
        az = np.load(axis_npz, allow_pickle=True)
        axis_u = az["axis_unit"].astype(np.float32)
        axis_roles = az["roles"].astype(str).tolist()
        Vc = (az["role_vectors"].astype(np.float32) - center)
        rv_dir = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
        proj = {n: float(az["role_vectors"][i] @ axis_u)
                for i, n in enumerate(axis_roles)}
        d2a = {n: float(1.0 - rv_dir[i] @ E[anchor])
               for i, n in enumerate(axis_roles)}
        common = [n for n in axis_roles if n in per_role]
        a = np.array([proj[n] for n in common])          # linear axis proj
        b = np.array([-d2a[n] for n in common])          # closeness to anchor
        bridge = {
            "spearman_axisproj_vs_neg_distanchor": round(_spearman(a, b), 4),
            "note": "high positive => EP distance-to-Assistant-cell recovers "
                    "the linear Assistant Axis ordering",
        }

    order = sorted(roles, key=lambda r: per_role[r]["mean_dist_to_anchor"])
    report = {
        "dict": Path(args.dict).name, "layer": args.layer, "K": K,
        "threshold": round(threshold, 4),
        "assistant_anchor_cell": anchor,
        "anchor_purity_assistant_group": round(anchor_purity, 3),
        "default_coverage_in_anchor": round(default_coverage, 3),
        "bridge": bridge,
        "roles_by_dist_to_anchor": order,
        "per_role": per_role,
    }
    (acts_dir / args.out_name).write_text(json.dumps(report, indent=2))
    _print(report)


def _print(r: dict) -> None:
    print(f"\n=== Persona localization in EP ({r['dict']}, L{r['layer']}, K={r['K']}) ===")
    print(f"Assistant anchor cell = #{r['assistant_anchor_cell']}  "
          f"purity(assistant-group)={r['anchor_purity_assistant_group']:.2f}  "
          f"default coverage={r['default_coverage_in_anchor']:.2f}")
    if r["bridge"]:
        print(f"BRIDGE  Spearman(linear axis proj, closeness to anchor) = "
              f"{r['bridge']['spearman_axisproj_vs_neg_distanchor']:+.3f}")
    print("\n  role                 grp          top%  inAnc  dist->anchor  #cells")
    for n in r["roles_by_dist_to_anchor"]:
        p = r["per_role"][n]
        print(f"  {n:<20} {p['group']:<11} {p['top_cell_frac']:.2f}  "
              f"{p['frac_in_anchor']:.2f}   {p['mean_dist_to_anchor']:.3f}       "
              f"{p['n_distinct_cells']}")


if __name__ == "__main__":
    main()
