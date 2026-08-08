"""EM model-diff — the structural comparison of the two dictionaries.

Every measurement here is label-free: only directions, member sets, member counts
and coherences. Region *contents* are not read, so nothing in this module can be
tuned to a story about what a region means.

The readouts are ordered by what the RMU positive control showed actually carries
signal (`GATE1B_RMU_DIFF.md`):

  1. noise floor   within-checkpoint cross-seed matched cosine. Sets the
                   persistence cutoff (5th percentile), never a fixed 0.7.
  2. determinism   the scale-0 sham checkpoint must reproduce base's dictionaries
                   element for element. A hard gate — RMU had L4 for this; EM's
                   LoRA touches every layer, so the sham is the substitute.
  3. dropped       fraction of base regions with no counterpart, against the
                   same-model control. RMU's real signal (0.523 vs 0.05).
  4. ARI           membership agreement, against the cross-seed control. RMU's
                   other real signal (0.092 vs 0.566).
  5. dominant      cross-seed reproducibility of each checkpoint's largest region.
                   RMU's decisive number, and the one place a diffuse edit is
                   predicted NOT to behave like an injected fixed point.
  6. introduced    reported because it was pre-registered; known to be
                   structurally incapable of representing consolidation.

    python -m experiments.em_diff.gate1 --grid artifacts/runs/em_diff/grid
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

# Validated pure functions from the RMU pipeline. That package is frozen.
from experiments.rmu_diff.gate1a import _write_csv, _write_json
from experiments.rmu_diff.gate1b import (Grid, adjusted_rand, diff_sets,
                                         hungarian, label_vector,
                                         membership_jaccard)

from .build import stream_perm
from .data import load_pool

log = logging.getLogger("em_diff.gate1")

BASES = ("mean", "exemplar")


def determinism_gate(g: Grid, models: list[str]) -> dict:
    """The sham checkpoint must produce dictionaries identical to base's."""
    if "sham" not in models:
        return {"ran": False, "reason": "no sham checkpoint in this grid"}
    bad = []
    for L in g.layers:
        for p in g.percentiles:
            for s in g.seeds:
                a, b = g.get("base", L, p, s), g.get("sham", L, p, s)
                same = (len(a.partitions) == len(b.partitions)
                        and all(pa.member_count == pb.member_count
                                for pa, pb in zip(a.partitions, b.partitions))
                        and np.allclose(
                            np.stack([q.exemplar_direction for q in a.partitions]),
                            np.stack([q.exemplar_direction for q in b.partitions]),
                            atol=0, rtol=0))
                if not same:
                    bad.append({"layer": L, "percentile": p, "seed": s,
                                "K_base": len(a.partitions), "K_sham": len(b.partitions)})
    return {"ran": True, "passed": not bad, "n_cells_checked":
            len(g.layers) * len(g.percentiles) * len(g.seeds), "mismatches": bad}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="artifacts/runs/em_diff/grid")
    ap.add_argument("--out", default="artifacts/runs/em_diff/gate1")
    ap.add_argument("--fixed-cutoff", type=float, default=0.7)
    args = ap.parse_args()

    g = Grid(Path(args.grid))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = g.manifest["config"]
    models = cfg["models"]
    report: dict = {"grid": str(g.root), "manifest_config": cfg}

    # -------------------------------------------------- 0. determinism control
    log.info("== 0. determinism control (sham == base?) ==")
    det = determinism_gate(g, models)
    report["determinism"] = det
    if det["ran"] and not det["passed"]:
        log.error("DETERMINISM CONTROL FAILED — %d mismatched cells", len(det["mismatches"]))
        _write_json(out / "gate1.json", report)
        raise SystemExit("determinism control failed; no result reported "
                         "(PREREG_EM_DIFF.md stop condition 2)")
    log.info("  %s", "PASS" if det.get("passed") else det.get("reason"))

    # ------------------------------------------------------- 1. noise floor
    log.info("== 1. within-checkpoint cross-seed noise floor ==")
    floor_rows, floors = [], {}
    for basis in BASES:
        for L in g.layers:
            for p in g.percentiles:
                for m in ("base", "em"):
                    cs = [hungarian(g.get(m, L, p, s0), g.get(m, L, p, s1), basis)[2]
                          for s0, s1 in combinations(g.seeds, 2)]
                    allc = np.concatenate(cs)
                    q = np.percentile(allc, [5, 25, 50, 75, 95])
                    floors[(basis, L, p, m)] = {"p5": float(q[0]), "median": float(q[2]),
                                                "p95": float(q[4]), "n_pairs": int(len(allc))}
                    floor_rows.append({"basis": basis, "layer": L, "percentile": p,
                                       "model": m, "n_seed_pairs": len(cs),
                                       "n_matched_pairs": int(len(allc)),
                                       "p5": q[0], "median": q[2], "p95": q[4]})
    _write_csv(out / "noise_floor.csv", floor_rows)
    report["noise_floor"] = {f"{b}|L{L}|p{p:g}|{m}": v for (b, L, p, m), v in floors.items()}

    # ---------------------------------------------------------- 2. diff sets
    log.info("== 2. diff sets (base -> EM) ==")
    diff_rows = []
    for basis in BASES:
        for L in g.layers:
            for p in g.percentiles:
                cut = floors[(basis, L, p, "base")]["p5"]
                for s in g.seeds:
                    b = g.get("base", L, p, s)
                    dd = diff_sets(b, g.get("em", L, p, s), cut, basis)
                    s2 = g.seeds[(g.seeds.index(s) + 1) % len(g.seeds)]
                    ctl = diff_sets(b, g.get("base", L, p, s2), cut, basis)
                    fixed = diff_sets(b, g.get("em", L, p, s), args.fixed_cutoff, basis)
                    diff_rows.append({
                        "basis": basis, "layer": L, "percentile": p, "seed": s,
                        "cutoff": cut, "K_base": dd["K_a"], "K_em": dd["K_b"],
                        "median_matched_cos": dd["median_matched_cos"],
                        "introduced_frac": dd["introduced_frac"],
                        "dropped_frac": dd["dropped_frac"],
                        "ctl_introduced_frac": ctl["introduced_frac"],
                        "ctl_dropped_frac": ctl["dropped_frac"],
                        "ctl_median_matched_cos": ctl["median_matched_cos"],
                        "introduced_frac_at_0.7": fixed["introduced_frac"],
                        "dropped_frac_at_0.7": fixed["dropped_frac"],
                    })
    _write_csv(out / "diff_sets.csv", diff_rows)

    # ------------------------------------------ 3. membership: ARI + dominant
    log.info("== 3. membership identity and dominant-region stability ==")
    pool = load_pool(Path(cfg["pool"]))
    mem_rows, dom_rows = [], []
    for L in g.layers:
        pid = np.load(g.root / f"prompt_ids_L{L}.npy")
        n_acts = len(pid)
        perms = {s: stream_perm(pool, pid, s) for s in g.seeds}
        for p in g.percentiles:
            lab, K, dicts = {}, {}, {}
            for m in ("base", "em"):
                for s in g.seeds:
                    d = g.get(m, L, p, s)
                    dicts[(m, s)] = d
                    K[(m, s)] = len(d.partitions)
                    lab[(m, s)] = label_vector(d, n_acts, perms[s])

            for m in ("base", "em"):          # cross-seed control
                for s0, s1 in combinations(g.seeds, 2):
                    best, _, _ = membership_jaccard(lab[(m, s0)], lab[(m, s1)],
                                                    K[(m, s0)], K[(m, s1)])
                    mem_rows.append({"layer": L, "percentile": p, "kind": "cross-seed",
                                     "model": m, "pair": f"{s0}-{s1}",
                                     "median_best_jaccard": float(np.median(best)),
                                     "ari": adjusted_rand(lab[(m, s0)], lab[(m, s1)],
                                                          K[(m, s0)], K[(m, s1)])})
            for s in g.seeds:                  # the diff itself
                best, _, _ = membership_jaccard(lab[("base", s)], lab[("em", s)],
                                                K[("base", s)], K[("em", s)])
                mem_rows.append({"layer": L, "percentile": p, "kind": "base-vs-em",
                                 "model": "base->em", "pair": f"s{s}",
                                 "median_best_jaccard": float(np.median(best)),
                                 "ari": adjusted_rand(lab[("base", s)], lab[("em", s)],
                                                      K[("base", s)], K[("em", s)])})

            # dominant-region cross-seed reproducibility (RMU's decisive number)
            for m in ("base", "em"):
                for s0, s1 in combinations(g.seeds, 2):
                    d0, d1 = dicts[(m, s0)], dicts[(m, s1)]
                    i0 = int(np.argmax([q.member_count for q in d0.partitions]))
                    i1 = int(np.argmax([q.member_count for q in d1.partitions]))
                    m0 = set(np.where(lab[(m, s0)] == i0)[0].tolist())
                    m1 = set(np.where(lab[(m, s1)] == i1)[0].tolist())
                    jac = len(m0 & m1) / max(len(m0 | m1), 1)
                    e0 = d0.partitions[i0].exemplar_direction
                    e1 = d1.partitions[i1].exemplar_direction
                    dom_rows.append({
                        "layer": L, "percentile": p, "model": m, "pair": f"{s0}-{s1}",
                        "size_a": d0.partitions[i0].member_count,
                        "size_b": d1.partitions[i1].member_count,
                        "member_jaccard": round(jac, 4),
                        "exemplar_cosine": round(float(np.dot(e0, e1)), 4),
                    })
    _write_csv(out / "membership.csv", mem_rows)
    _write_csv(out / "dominant.csv", dom_rows)

    # ---------------------------------------------- 4. H4 arm contrast
    log.info("== 4. H4 arm contrast (is the change where EM manifests?) ==")
    arm = np.array([1 if q.label == "elicit" else 0 for q in pool])
    arm_rows = []
    for L in g.layers:
        pid = np.load(g.root / f"prompt_ids_L{L}.npy")
        tok_arm = arm[pid]
        base_share = float(tok_arm.mean())
        perms = {s: stream_perm(pool, pid, s) for s in g.seeds}
        for p_ in g.percentiles:
            cut = floors[("mean", L, p_, "base")]["p5"]
            for s in g.seeds:
                b = g.get("base", L, p_, s)
                dd = diff_sets(b, g.get("em", L, p_, s), cut, "mean")
                lab = label_vector(b, len(pid), perms[s])
                dropped = [int(i) for i in dd["dropped"]]
                m_drop = np.isin(lab, dropped) if dropped else np.zeros(len(lab), bool)
                m_keep = (~m_drop) & (lab >= 0)
                arm_rows.append({
                    "layer": L, "percentile": p_, "seed": s,
                    "pool_elicit_share": round(base_share, 4),
                    "dropped_elicit_share": round(float(tok_arm[m_drop].mean()), 4)
                    if m_drop.any() else None,
                    "kept_elicit_share": round(float(tok_arm[m_keep].mean()), 4)
                    if m_keep.any() else None,
                })
    for r in arm_rows:
        r["lift"] = (round(r["dropped_elicit_share"] - r["pool_elicit_share"], 4)
                     if r["dropped_elicit_share"] is not None else None)
    _write_csv(out / "arm_contrast.csv", arm_rows)
    lifts = [r["lift"] for r in arm_rows if r["lift"] is not None]
    report["arm_contrast"] = {
        "mean_lift": round(float(np.mean(lifts)), 4),
        "min_lift": round(float(np.min(lifts)), 4),
        "max_lift": round(float(np.max(lifts)), 4),
        "frac_positive": round(float(np.mean([l > 0 for l in lifts])), 3),
        "h4_passed": bool(np.mean(lifts) > 0.10 and np.mean([l > 0 for l in lifts]) > 0.9),
    }

    report["summary"] = _summarise(diff_rows, mem_rows, dom_rows)
    _write_json(out / "gate1.json", report)
    _print(report, diff_rows, mem_rows, dom_rows, g)


def _agg(rows, key, **filt):
    vals = [r[key] for r in rows
            if all(r[k] == v for k, v in filt.items()) and r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _summarise(diff_rows, mem_rows, dom_rows) -> dict:
    """Includes the deflationary check that decides how to read `dropped`.

    Hungarian matches min(K_a, K_b) regions, so when the EM dictionary is smaller
    the surplus base regions are labelled dropped *by arithmetic*. `excess_dropped`
    subtracts that forced floor; only the excess is evidence of reorganisation,
    and it must be read against the same-model control (~0.06).
    """
    layers = sorted({r["layer"] for r in diff_rows})
    out = {}
    for L in layers:
        mean_rows = [r for r in diff_rows if r["layer"] == L and r["basis"] == "mean"]
        kb = float(np.mean([r["K_base"] for r in mean_rows]))
        ke = float(np.mean([r["K_em"] for r in mean_rows]))
        forced = max(0.0, (kb - ke) / max(kb, 1))
        out[f"L{L}"] = {
            "K_base": round(kb, 1), "K_em": round(ke, 1),
            "K_ratio_em_over_base": round(ke / max(kb, 1e-9), 3),
            "forced_dropped_from_K": round(forced, 3),
            "excess_dropped": round(
                _agg(diff_rows, "dropped_frac", layer=L, basis="mean") - forced, 3),
            "dropped_frac": _agg(diff_rows, "dropped_frac", layer=L, basis="mean"),
            "ctl_dropped_frac": _agg(diff_rows, "ctl_dropped_frac", layer=L, basis="mean"),
            "introduced_frac": _agg(diff_rows, "introduced_frac", layer=L, basis="mean"),
            "ctl_introduced_frac": _agg(diff_rows, "ctl_introduced_frac", layer=L, basis="mean"),
            "ari_base_vs_em": _agg(mem_rows, "ari", layer=L, kind="base-vs-em"),
            "ari_cross_seed": _agg(mem_rows, "ari", layer=L, kind="cross-seed"),
            "dominant_jaccard_base": _agg(dom_rows, "member_jaccard", layer=L, model="base"),
            "dominant_jaccard_em": _agg(dom_rows, "member_jaccard", layer=L, model="em"),
        }
    return out


def _print(report, diff_rows, mem_rows, dom_rows, g) -> None:
    s = report["summary"]
    print(f"\n=== EM model-diff, Gate 1 ({g.root}) ===")
    d = report["determinism"]
    print(f"determinism control: {'PASS' if d.get('passed') else d.get('reason')}"
          f" ({d.get('n_cells_checked', 0)} cells)\n")
    print(f"{'layer':>6} {'dropped':>9} {'ctl':>7} {'introduced':>11} {'ctl':>7} "
          f"{'ARI b-vs-em':>12} {'ARI ctl':>8} {'domJac base':>12} {'domJac em':>10}")
    for k, v in s.items():
        print(f"{k:>6} {v['dropped_frac']:>9.3f} {v['ctl_dropped_frac']:>7.3f} "
              f"{v['introduced_frac']:>11.3f} {v['ctl_introduced_frac']:>7.3f} "
              f"{v['ari_base_vs_em']:>12.3f} {v['ari_cross_seed']:>8.3f} "
              f"{v['dominant_jaccard_base']:>12.3f} {v['dominant_jaccard_em']:>10.3f}")
    a = report.get("arm_contrast")
    if a:
        print(f"\nH4 arm contrast: mean lift {a['mean_lift']:+.3f} "
              f"[{a['min_lift']:+.3f}..{a['max_lift']:+.3f}], "
              f"{a['frac_positive']:.0%} positive -> "
              f"{'PASS' if a['h4_passed'] else 'FAIL (change is uniform across arms)'}")
    print("\nRead: dropped/ARI against their controls is the signal (RMU: 0.523 vs 0.05,")
    print("ARI 0.092 vs 0.566). introduced is reported but known to be vacuous.")


if __name__ == "__main__":
    main()
