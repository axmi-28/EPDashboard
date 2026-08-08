"""Gate 2 Arm C — how regions correspond across layers.

The hypothesis, stated as the user did: harmful prompts should be **split**
across many early-layer regions by topic (weapons near chemistry, phishing near
email) and **consolidated** into fewer late-layer regions once the model has
formed the abstraction the refusal behaviour keys on.

That is a claim about a *merge*, and it is measured three ways here, each with
its own null:

- **C1 concentration.** Effective number of regions, `exp(H)`, over the 300
  harmful prompts at each layer. Raw region counts are not comparable across
  layers because K differs (491 / 145 / 686 / 176), so everything is reported
  against the benign arm at the same layer and against a matched-K coreset.
- **C2 correspondence.** Mutual information between the early region and the
  late region of the same prompt. MI is heavily K-biased at 600 samples, so the
  only interpretable figure is the gap to a matched-K coreset null **at both
  layers** — an analytic bias correction would not be trustworthy here.
- **C3 selective merge.** The decisive one. Take the late regions the labelled
  fit marks harmful. For each early region that feeds them, compare
  `P(-> refusal region | harmful member)` against
  `P(-> refusal region | benign member)`. A merge driven by topic sends both
  alike; a merge driven by the label separates them. This is what distinguishes
  "the model formed a harm abstraction" from "the partition is coarse".

Layer pairs, matched on percentile so theta is comparable:

    P1  L4  p4  (K=491) -> L20 p4  (K=686)    exploratory: 337k cells, 600 pts
    P2  L12 p10 (K=145) -> L20 p10 (K=176)    primary:      25k cells, 600 pts
    P3  base L20 p10 (K=192) vs it L20 p10    training axis, not depth

**Which position is read matters more here than anywhere else in the gate**, and
the first run of this script got it wrong. Reading the true final position puts
all 600 prompts in *one* layer-4 region: that position holds the scaffold token
`<start_of_turn>model\n`, identical across every prompt, and by layer 4 it has
not yet integrated the instruction. The measurement was of the scaffold, not of
the content. Three positions are therefore swept:

    final     the true last token — what A0 and the refusal anchor use, and
              what a final-position monitor would see
    content   the last token of the instruction itself, before `<end_of_turn>`
    pooled    every content token of every prompt, treated as a population

`pooled` is the closest match to the hypothesis as stated. "Harmful prompts are
split by topic early" is a claim about where the *content* lives, and at layer 4
representations are still largely token-local, so content spread shows up across
positions rather than at any single summary token.

Known weaknesses, declared before the numbers:

- The hub carries exactly one percentile per early layer, so the
  single-percentile-artifact check that Gate 1B relied on **cannot be run** on
  P1 or P2. Any C result is provisional for that reason alone.
- L12's dictionary saturated on 131,072 activations at K=145, an order of
  magnitude less build data than L20's. A null at P2 is weak evidence of
  absence.
- P1 has 0.8 prompts per cell. It is reported, and no decision rests on it.

    python -m experiments.monitor.run_gate2_c
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dicts import lean_path, load_lean
from .gate2_route import assign, coreset_partition

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
LAB = ROOT / "gate2_bc_labelled.npz"
OUT = ROOT / "gate2_c_crosslayer.json"

N_NULL_DRAWS = 10
MIN_MEMBERS = 5

PAIRS = [
    ("P2_primary", ("gemma-2-2b-it", 12, 10), ("gemma-2-2b-it", 20, 10)),
    ("P1_exploratory", ("gemma-2-2b-it", 4, 4), ("gemma-2-2b-it", 20, 4)),
]


# ------------------------------------------------------------------ helpers

def eff_regions(r: np.ndarray) -> float:
    """exp(H) — the effective number of regions the prompts spread over.

    Comparable across layers in a way a raw distinct-region count is not: it
    weights by occupancy, so one region holding 290 of 300 prompts and ten
    holding one each reads as ~1.1, not 11.
    """
    n = np.bincount(r)
    p = n[n > 0] / n.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """I(a; b) in nats, plug-in estimator. Biased upward; use only vs a null."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def flagged_regions(r: np.ndarray, y: np.ndarray) -> set[int]:
    """Late regions the labelled fit would raise on: majority-harmful, n >= 5."""
    K = int(r.max()) + 1
    n = np.bincount(r, minlength=K)
    h = np.bincount(r, weights=y.astype(float), minlength=K)
    return {int(i) for i in np.where((n >= MIN_MEMBERS)
                                     & (h / np.maximum(n, 1) >= 0.5))[0]}


def selective_merge(early: np.ndarray, late: np.ndarray, y: np.ndarray,
                    flagged: set[int]) -> dict:
    """C3: is the merge into the flagged late regions selective for the label?

    Restricted to early regions that hold **both** harmful and benign prompts.
    Those are the only ones where the question is even defined: an early region
    of pure harmful members tells you nothing about whether the merge tracks
    the label or the topic, because there is no benign member to leave behind.
    """
    into = np.array([int(v) in flagged for v in late])
    rows = []
    for e in np.unique(early):
        m = early == e
        nh, nb = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        if nh == 0 or nb == 0:
            continue
        rows.append({
            "early_region": int(e), "n_harmful": nh, "n_benign": nb,
            "p_into_harmful": float(into[m & (y == 1)].mean()),
            "p_into_benign": float(into[m & (y == 0)].mean()),
        })
    if not rows:
        return {"n_mixed_early_regions": 0, "selectivity": float("nan")}
    wh = np.array([r["n_harmful"] + r["n_benign"] for r in rows], dtype=float)
    sel = np.array([r["p_into_harmful"] - r["p_into_benign"] for r in rows])
    return {
        "n_mixed_early_regions": len(rows),
        "n_prompts_covered": int(wh.sum()),
        # Weighted so a 2-prompt region does not count as much as a 200-prompt
        # one; the unweighted mean is reported alongside because the weighting
        # choice is arguable.
        "selectivity": float((sel * wh).sum() / wh.sum()),
        "selectivity_unweighted": float(sel.mean()),
        "rows": sorted(rows, key=lambda r: -(r["n_harmful"] + r["n_benign"]))[:12],
    }


# ------------------------------------------------------------------ main

def main() -> None:
    z = np.load(LAB)
    y = z["y"]
    lengths, pre, suf = z["lengths"], int(z["prefix_len"]), int(z["suffix_len"])
    off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    final_idx = off[1:] - 1                      # true final position
    content_idx = off[1:] - 1 - suf              # last content token

    # Position selectors. `pooled` returns one index array per prompt rather
    # than one index, and a matching per-position label vector.
    pooled_idx = np.concatenate([np.arange(off[i] + pre, off[i + 1] - suf)
                                 for i in range(len(y))])
    pooled_y = np.concatenate([np.full(int(lengths[i]) - pre - suf, y[i])
                               for i in range(len(y))])
    POS = {"final": (final_idx, y), "content": (content_idx, y),
           "pooled": (pooled_idx, pooled_y)}
    print(f"scaffold: {pre} prefix + {suf} suffix tokens; "
          f"{len(pooled_idx)} content positions over {len(y)} prompts "
          f"({len(pooled_idx) / len(y):.1f} each)")

    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")
    out: dict = {"n_prompts": int(len(y)), "n_null_draws": N_NULL_DRAWS,
                 "scaffold": {"prefix_len": pre, "suffix_len": suf},
                 "n_content_positions": int(len(pooled_idx)),
                 "pairs": {}, "concentration": {},
                 "caveats": [
                     "one percentile per early layer: no artifact check possible",
                     "L12 dictionary saturated on 131,072 activations at K=145",
                     "P1 has ~0.8 prompts per cell; exploratory only",
                     "at the true final position L4 puts all 600 prompts in one "
                     "region: that token is chat scaffold and L4 has not "
                     "integrated the instruction, so 'final' is uninformative "
                     "about the split hypothesis"]}

    # ---------------------------------------------------- C1 concentration
    print("\n=== C1: does the harmful set consolidate with depth? ===")
    print("eff = exp(H) over the region distribution; ratio = eff(harm)/eff(ben)")
    for mode, (idx, yy) in POS.items():
        print(f"\n-- position = {mode} ({len(idx)} activations) --")
        print(f"{'dictionary':22s} {'K':>5s} {'eff(harm)':>10s} {'eff(ben)':>9s} "
              f"{'ratio':>7s} | {'null ratio':>18s} {'margin':>8s}")
        for model_short, L, p in [("gemma-2-2b-it", 4, 4),
                                  ("gemma-2-2b-it", 12, 10),
                                  ("gemma-2-2b-it", 20, 4),
                                  ("gemma-2-2b-it", 20, 10),
                                  ("gemma-2-2b", 20, 10)]:
            d = load_lean(lean_path(DICTS, model_short, L, p))
            x = z[f"x_L{L}"][idx].astype(np.float32)
            r = assign(x, d["exemplars"], d["center"])
            eh, eb = eff_regions(r[yy == 1]), eff_regions(r[yy == 0])
            nulls = []
            for j in range(N_NULL_DRAWS):
                rc = assign(x, coreset_partition(pool, d["center"], d["K"],
                                                 500 + j), d["center"])
                nulls.append(eff_regions(rc[yy == 1])
                             / max(eff_regions(rc[yy == 0]), 1e-9))
            nm, ns = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
            ratio = eh / max(eb, 1e-9)
            tag = f"{model_short.split('-')[-1]} L{L} p{p}"
            out["concentration"].setdefault(mode, {})[tag] = {
                "K": d["K"], "eff_harmful": eh, "eff_benign": eb,
                "ratio": ratio, "null_ratio_mean": nm, "null_ratio_sd": ns,
                "margin_sd": float((ratio - nm) / max(ns, 1e-9)),
                "n_distinct_harmful": int(len(np.unique(r[yy == 1]))),
                "n_distinct_benign": int(len(np.unique(r[yy == 0]))),
            }
            print(f"{tag:22s} {d['K']:5d} {eh:10.2f} {eb:9.2f} {ratio:7.3f} | "
                  f"{nm:9.3f} +- {ns:.3f} "
                  f"{out['concentration'][mode][tag]['margin_sd']:+8.1f}")

    # ------------------------------------------- C2 correspondence + C3 merge
    for tag, (em, eL, ep), (lm, lL, lp) in PAIRS:
        de = load_lean(lean_path(DICTS, em, eL, ep))
        dl = load_lean(lean_path(DICTS, lm, lL, lp))
        # Content-final, not final: the correspondence question is about where
        # the instruction is represented, and the true final token is scaffold.
        xe = z[f"x_L{eL}"][content_idx].astype(np.float32)
        xl = z[f"x_L{lL}"][content_idx].astype(np.float32)
        re = assign(xe, de["exemplars"], de["center"])
        rl = assign(xl, dl["exemplars"], dl["center"])

        mi = mutual_information(re, rl)
        mi_null = []
        for j in range(N_NULL_DRAWS):
            rce = assign(xe, coreset_partition(pool, de["center"], de["K"], 700 + j),
                         de["center"])
            rcl = assign(xl, coreset_partition(pool, dl["center"], dl["K"], 800 + j),
                         dl["center"])
            mi_null.append(mutual_information(rce, rcl))

        flag = flagged_regions(rl, y)
        merge = selective_merge(re, rl, y, flag)
        merge_null = []
        for j in range(N_NULL_DRAWS):
            rce = assign(xe, coreset_partition(pool, de["center"], de["K"], 700 + j),
                         de["center"])
            s = selective_merge(rce, rl, y, flag)["selectivity"]
            if not np.isnan(s):
                merge_null.append(s)

        fanin = [len(np.unique(re[rl == v])) for v in np.unique(rl)]
        out["pairs"][tag] = {
            "early": f"{em} L{eL} p{ep}", "early_K": de["K"],
            "late": f"{lm} L{lL} p{lp}", "late_K": dl["K"],
            "n_early_occupied": int(len(np.unique(re))),
            "n_late_occupied": int(len(np.unique(rl))),
            "mi_nats": mi,
            "mi_null_mean": float(np.mean(mi_null)),
            "mi_null_sd": float(np.std(mi_null, ddof=1)),
            "mi_margin_sd": float((mi - np.mean(mi_null))
                                  / max(np.std(mi_null, ddof=1), 1e-9)),
            "mean_fan_in": float(np.mean(fanin)),
            "max_fan_in": int(np.max(fanin)),
            "n_flagged_late_regions": len(flag),
            "merge": merge,
            "merge_null_mean": float(np.mean(merge_null)) if merge_null else None,
            "merge_null_sd": (float(np.std(merge_null, ddof=1))
                              if len(merge_null) > 1 else None),
        }
        o = out["pairs"][tag]
        print(f"\n=== C2/C3 {tag}: {o['early']} -> {o['late']} ===")
        print(f"  occupied {o['n_early_occupied']} early -> "
              f"{o['n_late_occupied']} late; mean fan-in {o['mean_fan_in']:.2f}, "
              f"max {o['max_fan_in']}")
        print(f"  MI {mi:.4f} nats vs null {o['mi_null_mean']:.4f}"
              f"+-{o['mi_null_sd']:.4f}  -> {o['mi_margin_sd']:+.1f} sd")
        print(f"  flagged late regions: {o['n_flagged_late_regions']}")
        print(f"  C3 selective merge over {merge['n_mixed_early_regions']} mixed "
              f"early regions ({merge.get('n_prompts_covered', 0)} prompts):")
        if merge["n_mixed_early_regions"]:
            nm = o["merge_null_mean"]
            ns = o["merge_null_sd"]
            print(f"    P(->refusal | harmful) - P(->refusal | benign) = "
                  f"{merge['selectivity']:+.4f}  "
                  f"(unweighted {merge['selectivity_unweighted']:+.4f})")
            if nm is not None and ns:
                print(f"    coreset null {nm:+.4f} +- {ns:.4f}  -> "
                      f"{(merge['selectivity'] - nm) / max(ns, 1e-9):+.1f} sd")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
