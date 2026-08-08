"""Gate 1B — stability. The kill gate, computed before any region is read.

Nothing in this module looks at region *contents*. It reads directions, member
counts and coherences only. That ordering is the point: if the introduced set is
construction noise, what is in it does not matter.

Four computations, in the order the brief fixes:

  1  noise floor    Hungarian-matched cosine between the two seeds of the SAME
                    checkpoint, per (model, layer, p). Everything downstream is
                    measured against this, including the persistence cutoff.
  2  diff sets      introduced / dropped / persisted between base and RMU per
                    (layer, seed, p).
  3  Jaccard        overlap of the INTRODUCED SET across seeds and across the
                    two adjacent percentiles — set membership, not median
                    matched cosine — against a random-subset null at matched
                    size.
  4  D_i filter     the same, before and after keeping the top D_i quintile.

Plus the vacuity control the prereg calls H4: run the identical diff procedure
on two same-model seeds. If base-vs-RMU introduces no more than base-vs-base
does, the "diff" is streaming order wearing a label.

Region identity across dictionaries is Hungarian assignment on **mean member
directions** (A.7: more order-stable than first-arrival exemplars). The exemplar
basis is computed alongside because A.3 used it, and the two are never mixed
inside one claim.

    python -m experiments.rmu_diff.gate1b --grid artifacts/runs/rmu_diff/grid/shared
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gate1a import _write_csv, _write_json

log = logging.getLogger("rmu_diff.gate1b")

BASES = ("mean", "exemplar")


# ------------------------------------------------------------------ loading

def _dirs(d, basis: str) -> np.ndarray:
    attr = ("mean_member_direction" if basis == "mean" else "exemplar_direction")
    v = np.stack([getattr(p, attr).astype(np.float32) for p in d.partitions])
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def _stats(d) -> dict:
    n = np.array([p.member_count for p in d.partitions], dtype=np.float64)
    c = np.array([p.member_coherence for p in d.partitions], dtype=np.float64)
    return {"N": n, "c": c, "D": np.log10(np.maximum(n, 1) * np.maximum(c, 1e-9) ** 2)}


class Grid:
    """Lazy loader over a built grid directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.runs = {r["name"]: r for r in self.manifest["runs"]}
        self._cache: dict[str, object] = {}

    def name(self, model, layer, p, seed) -> str:
        return f"{model}_L{layer}_p{p:g}_seed{seed}"

    def get(self, model, layer, p, seed):
        n = self.name(model, layer, p, seed)
        if n not in self._cache:
            with (self.root / f"{n}.pkl").open("rb") as f:
                self._cache[n] = pickle.load(f)
        return self._cache[n]

    @property
    def layers(self):
        return sorted({r["layer"] for r in self.manifest["runs"]})

    @property
    def percentiles(self):
        return sorted({r["percentile"] for r in self.manifest["runs"]})

    @property
    def seeds(self):
        return sorted({r["seed"] for r in self.manifest["runs"]})


# ----------------------------------------------------------------- matching

def hungarian(dA, dB, basis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal one-to-one matching maximising summed cosine.

    Returns (rows, cols, cosines) over min(K_A, K_B) pairs. Regions of the
    larger dictionary left unpaired are, by construction, unmatched — they are
    handled explicitly by the callers rather than silently dropped.
    """
    a, b = _dirs(dA, basis), _dirs(dB, basis)
    sim = a @ b.T
    rows, cols = linear_sum_assignment(-sim)
    return rows, cols, sim[rows, cols]


def diff_sets(dA, dB, cutoff: float, basis: str) -> dict:
    """Classify B's regions against A's: persisted / introduced, plus dropped.

    `introduced` = a region of B (the "after" model) with no counterpart in A
    at or above `cutoff` — including every region left unpaired when K_B > K_A.
    `dropped` = the mirror, regions of A with no counterpart in B.
    """
    rows, cols, cos = hungarian(dA, dB, basis)
    kA, kB = len(dA.partitions), len(dB.partitions)
    matched_b = np.full(kB, -1.0)
    matched_b[cols] = cos
    matched_a = np.full(kA, -1.0)
    matched_a[rows] = cos
    introduced = np.where(matched_b < cutoff)[0]
    dropped = np.where(matched_a < cutoff)[0]
    persisted = np.where(matched_b >= cutoff)[0]
    pair_of_b = np.full(kB, -1, dtype=np.int64)
    pair_of_b[cols] = rows
    return {
        "K_a": kA, "K_b": kB, "cutoff": cutoff, "basis": basis,
        "n_pairs": len(rows),
        "median_matched_cos": float(np.median(cos)) if len(cos) else float("nan"),
        "introduced": introduced, "dropped": dropped, "persisted": persisted,
        "introduced_frac": len(introduced) / max(kB, 1),
        "dropped_frac": len(dropped) / max(kA, 1),
        "matched_cos_b": matched_b, "pair_of_b": pair_of_b,
    }


def jaccard_via_correspondence(dX, dY, setX: np.ndarray, setY: np.ndarray,
                               cutoff: float, basis: str,
                               n_sim: int = 2000, seed: int = 0) -> dict:
    """Set overlap of two label-sets living in two different dictionaries.

    `setX`/`setY` are region indices in dX/dY. They cannot be intersected
    directly — the regions are different objects. So establish a correspondence
    first (Hungarian on dX <-> dY, keeping only pairs at or above `cutoff`,
    i.e. pairs that same-checkpoint rebuilds would call the same region), then
    take the Jaccard over that corresponded population.

    The null holds the correspondence fixed and randomises only *which*
    corresponded regions carry the label, at the observed set sizes. That is
    the right null for "is this set reproducible", as opposed to "is this set
    non-empty".
    """
    rows, cols, cos = hungarian(dX, dY, basis)
    keep = cos >= cutoff
    rx, ry = rows[keep], cols[keep]
    M = len(rx)
    if M == 0:
        return {"n_corresponded": 0, "jaccard": float("nan"),
                "null_median": float("nan"), "null_p95": float("nan"),
                "a": 0, "b": 0, "inter": 0, "union": 0, "z_vs_null": float("nan"),
                "setX_uncorresponded": int(len(setX)),
                "setY_uncorresponded": int(len(setY))}
    inX = np.isin(rx, setX)
    inY = np.isin(ry, setY)
    inter = int((inX & inY).sum())
    union = int((inX | inY).sum())
    jac = inter / union if union else float("nan")

    rng = np.random.default_rng(seed)
    a, b = int(inX.sum()), int(inY.sum())
    null = np.empty(n_sim)
    for i in range(n_sim):
        pa = np.zeros(M, bool); pa[rng.choice(M, a, replace=False)] = True
        pb = np.zeros(M, bool); pb[rng.choice(M, b, replace=False)] = True
        u = (pa | pb).sum()
        null[i] = (pa & pb).sum() / u if u else np.nan
    sd = float(np.nanstd(null))
    return {
        "n_corresponded": M, "a": a, "b": b, "inter": inter, "union": union,
        "jaccard": jac,
        "null_median": float(np.nanmedian(null)),
        "null_p95": float(np.nanpercentile(null, 95)),
        "null_sd": sd,
        "z_vs_null": (jac - float(np.nanmedian(null))) / sd if sd > 0 else float("nan"),
        # Regions the correspondence could not place at all: they are neither
        # agreement nor disagreement, and hiding them would flatter the Jaccard.
        "setX_uncorresponded": int(len(setX) - inX.sum()),
        "setY_uncorresponded": int(len(setY) - inY.sum()),
    }


# -------------------------------------------------- membership-based identity

def label_vector(d, n_acts: int, perm: np.ndarray) -> np.ndarray:
    """Region label per activation, in **canonical** (pre-shuffle) coordinates.

    `constituent_sample_indices` holds stream positions, which differ between
    seeds; `perm` maps stream position -> canonical activation index, so labels
    from different seeds become directly comparable.
    """
    lab = np.full(n_acts, -1, dtype=np.int64)
    for i, p in enumerate(d.partitions):
        idx = np.asarray(p.constituent_sample_indices, dtype=np.int64)
        if len(idx):
            lab[perm[idx]] = i
    return lab


def membership_jaccard(lab_a: np.ndarray, lab_b: np.ndarray, kA: int, kB: int):
    """Per-region best member-set Jaccard between two partitions of one ground set.

    This is the identity measure the paper could not use and this experiment
    can. A.3 compared dictionaries built on *different* activation streams, so
    direction cosine was the only currency available. Here base and RMU
    partition the **same** activations in the **same** order, so region identity
    can be settled by who is inside it — no cutoff on a cosine whose scale is
    set by ambient dimension rather than by structure.

    Returns (best_jaccard_for_each_B_region, best_A_partner, contingency_sums).
    """
    ok = (lab_a >= 0) & (lab_b >= 0)
    a, b = lab_a[ok], lab_b[ok]
    cont = np.bincount(a * kB + b, minlength=kA * kB).reshape(kA, kB)
    na = cont.sum(axis=1, keepdims=True)
    nb = cont.sum(axis=0, keepdims=True)
    jac = cont / np.maximum(na + nb - cont, 1)
    best = jac.max(axis=0)
    return best, jac.argmax(axis=0), (na.ravel(), nb.ravel())


def adjusted_rand(lab_a: np.ndarray, lab_b: np.ndarray, kA: int, kB: int) -> float:
    """ARI between two partitions of the same ground set (cutoff-free)."""
    from scipy.special import comb

    ok = (lab_a >= 0) & (lab_b >= 0)
    a, b = lab_a[ok], lab_b[ok]
    cont = np.bincount(a * kB + b, minlength=kA * kB).reshape(kA, kB).astype(np.float64)
    n = cont.sum()
    sij = comb(cont, 2).sum()
    si = comb(cont.sum(axis=1), 2).sum()
    sj = comb(cont.sum(axis=0), 2).sum()
    tot = comb(n, 2)
    exp = si * sj / tot
    mx = 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else float("nan")


def top_quintile(d) -> np.ndarray:
    """Indices of the top-D_i quintile — A.7's filter, applied within-dictionary."""
    D = _stats(d)["D"]
    return np.where(D >= np.percentile(D, 80))[0]


# --------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="artifacts/runs/rmu_diff/grid/shared")
    ap.add_argument("--out", default="artifacts/runs/rmu_diff/gate1b")
    ap.add_argument("--fixed-cutoff", type=float, default=0.7,
                    help="A.3's cutoff, reported for comparability only")
    ap.add_argument("--n-sim", type=int, default=2000)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    g = Grid(Path(args.grid))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"grid": str(g.root), "manifest_config": g.manifest["config"]}

    # ---------------------------------------------------- 1. the noise floor
    log.info("== 1. within-checkpoint cross-seed noise floor ==")
    floor_rows, floors = [], {}
    for basis in BASES:
        for L in g.layers:
            for p in g.percentiles:
                for model in ("base", "rmu"):
                    cs = []
                    for s0, s1 in combinations(g.seeds, 2):
                        _, _, c = hungarian(g.get(model, L, p, s0),
                                            g.get(model, L, p, s1), basis)
                        cs.append(c)
                    allc = np.concatenate(cs)
                    q = np.percentile(allc, [5, 25, 50, 75, 95])
                    floors[(basis, L, p, model)] = {
                        "p5": float(q[0]), "median": float(q[2]),
                        "p95": float(q[4]), "n_pairs": int(len(allc)),
                        "n_seed_pairs": len(cs)}
                    floor_rows.append({"basis": basis, "layer": L, "percentile": p,
                                       "model": model, "n_seed_pairs": len(cs),
                                       "n_matched_pairs": int(len(allc)),
                                       "p5": q[0], "p25": q[1], "median": q[2],
                                       "p75": q[3], "p95": q[4]})
            log.info("  %-8s L%-2d %s", basis, L,
                     " ".join(f"p{p:g} base med={floors[(basis, L, p, 'base')]['median']:.3f}"
                              f"/p5={floors[(basis, L, p, 'base')]['p5']:.3f}"
                              for p in g.percentiles))
    _write_csv(out / "noise_floor.csv", floor_rows)
    report["noise_floor"] = {f"{b}|L{L}|p{p:g}|{m}": v
                             for (b, L, p, m), v in floors.items()}

    # ---------------------------------------------------------- 2. diff sets
    log.info("== 2. diff sets (base -> RMU) ==")
    diff_rows, diffs = [], {}
    for basis in BASES:
        for L in g.layers:
            for p in g.percentiles:
                # Pre-registered cutoff: a region persists if it matches its
                # counterpart at least as well as same-checkpoint rebuilds match
                # each other (5th percentile of the base cross-seed floor).
                cut = floors[(basis, L, p, "base")]["p5"]
                for s in g.seeds:
                    b, r = g.get("base", L, p, s), g.get("rmu", L, p, s)
                    dd = diff_sets(b, r, cut, basis)
                    diffs[(basis, L, p, s)] = dd
                    # H4 vacuity control: identical procedure, two base seeds.
                    s2 = g.seeds[(g.seeds.index(s) + 1) % len(g.seeds)]
                    ctl = diff_sets(b, g.get("base", L, p, s2), cut, basis)
                    fixed = diff_sets(b, r, args.fixed_cutoff, basis)
                    diff_rows.append({
                        "basis": basis, "layer": L, "percentile": p, "seed": s,
                        "cutoff": cut, "K_base": dd["K_a"], "K_rmu": dd["K_b"],
                        "median_matched_cos": dd["median_matched_cos"],
                        "n_introduced": len(dd["introduced"]),
                        "introduced_frac": dd["introduced_frac"],
                        "n_dropped": len(dd["dropped"]),
                        "dropped_frac": dd["dropped_frac"],
                        "ctl_introduced_frac": ctl["introduced_frac"],
                        "ctl_median_matched_cos": ctl["median_matched_cos"],
                        "vacuity_ratio": (dd["introduced_frac"]
                                          / max(ctl["introduced_frac"], 1e-9)),
                        "introduced_frac_at_0.7": fixed["introduced_frac"],
                        "median_matched_cos_a3_style": fixed["median_matched_cos"],
                    })
                if basis == "mean":
                    r0 = diff_rows[-1]
                    log.info("  L%-2d p%-3g cutoff=%.3f K %d->%d  median cos=%.3f "
                             "introduced=%.3f (control %.3f, ratio %.1fx)",
                             L, p, r0["cutoff"], r0["K_base"], r0["K_rmu"],
                             r0["median_matched_cos"], r0["introduced_frac"],
                             r0["ctl_introduced_frac"], r0["vacuity_ratio"])
    _write_csv(out / "diff_sets.csv", diff_rows)

    # ------------------------------------- 3/4. Jaccard, null, D_i filtering
    log.info("== 3/4. introduced-set Jaccard across seeds and percentiles ==")
    jac_rows = []
    for basis in BASES:
        for L in g.layers:
            for p in g.percentiles:
                cut = floors[(basis, L, p, "base")]["p5"]
                for s0, s1 in combinations(g.seeds, 2):
                    r0, r1 = g.get("rmu", L, p, s0), g.get("rmu", L, p, s1)
                    i0 = diffs[(basis, L, p, s0)]["introduced"]
                    i1 = diffs[(basis, L, p, s1)]["introduced"]
                    for filt in ("all", "topD"):
                        if filt == "topD":
                            k0, k1 = top_quintile(r0), top_quintile(r1)
                            a, b = np.intersect1d(i0, k0), np.intersect1d(i1, k1)
                        else:
                            a, b = i0, i1
                        j = jaccard_via_correspondence(r0, r1, a, b, cut, basis,
                                                       n_sim=args.n_sim)
                        jac_rows.append({"basis": basis, "layer": L,
                                         "percentile": p, "axis": "seed",
                                         "pair": f"{s0}-{s1}", "filter": filt,
                                         "cutoff": cut, **j})
                # Adjacent-percentile agreement, same model and seed.
                if len(g.percentiles) >= 2 and p == g.percentiles[0]:
                    p1 = g.percentiles[1]
                    cut1 = floors[(basis, L, p1, "base")]["p5"]
                    for s in g.seeds:
                        ra, rb = g.get("rmu", L, p, s), g.get("rmu", L, p1, s)
                        j = jaccard_via_correspondence(
                            ra, rb, diffs[(basis, L, p, s)]["introduced"],
                            diffs[(basis, L, p1, s)]["introduced"],
                            min(cut, cut1), basis, n_sim=args.n_sim)
                        jac_rows.append({"basis": basis, "layer": L,
                                         "percentile": f"{p:g}v{p1:g}",
                                         "axis": "percentile", "pair": f"seed{s}",
                                         "filter": "all", "cutoff": min(cut, cut1),
                                         **j})
    _write_csv(out / "jaccard.csv", jac_rows)

    for row in jac_rows:
        if row["basis"] == "mean" and row["filter"] == "all":
            log.info("  L%-2s p%-6s %-10s %-8s J=%.3f vs null %.3f (p95 %.3f) "
                     "z=%.1f  [M=%d a=%d b=%d]", row["layer"], row["percentile"],
                     row["axis"], row["pair"], row["jaccard"], row["null_median"],
                     row["null_p95"], row["z_vs_null"], row["n_corresponded"],
                     row["a"], row["b"])

    # ------------------------- 3b. membership-based identity (cutoff-free)
    mem_rows, mem_jac_rows = [], []
    try:
        pool = rebuild_pool(g)
    except Exception as e:  # noqa: BLE001
        log.warning("membership analysis skipped: could not rebuild pool (%s)", e)
        pool = None
    if pool is not None:
        log.info("== 3b. membership identity on the shared activation stream ==")
        mem_rows, mem_jac_rows = membership_analysis(g, pool, args.n_sim)
        _write_csv(out / "membership.csv", mem_rows)
        _write_csv(out / "membership_jaccard.csv", mem_jac_rows)

    # ------------------------------------------------------------- verdict
    report["verdict"] = verdict(diff_rows, jac_rows, g, mem_rows, mem_jac_rows)
    _write_json(out / "gate1b.json", report)
    log.info("== verdict ==\n%s", json.dumps(report["verdict"], indent=2))


def rebuild_pool(g: Grid):
    """Reconstruct the exact prompt pool the grid was built on."""
    from transformers import AutoTokenizer

    from .data import build_pool
    from .gate1a import BASE_ID, BASE_REV

    c = g.manifest["config"]
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    return build_pool(n_bio=c["n_bio"], n_cyber=c["n_cyber"], n_mmlu=c["n_mmlu"],
                      style=c["style"], tokenizer=tok,
                      min_tokens=c["min_tokens"], max_tokens=c["max_tokens"])


def membership_analysis(g: Grid, pool, n_sim: int):
    """Region identity by member overlap rather than direction cosine.

    Same structure as the direction analysis — noise floor, diff set, cross-seed
    Jaccard, random null — but identity is settled by which activations a region
    contains. Available only because both checkpoints partition the *same*
    activation stream in the *same* order.
    """
    from .build import stream_perm

    rows, jrows = [], []
    for L in g.layers:
        pid = np.load(g.root / f"prompt_ids_L{L}.npy")
        n_acts = len(pid)
        perms = {s: stream_perm(pool, pid, s) for s in g.seeds}
        for p in g.percentiles:
            lab, K = {}, {}
            for m in ("base", "rmu"):
                for s in g.seeds:
                    d = g.get(m, L, p, s)
                    K[(m, s)] = len(d.partitions)
                    lab[(m, s)] = label_vector(d, n_acts, perms[s])

            # Noise floor: same checkpoint, different streaming order.
            floor = []
            for m in ("base", "rmu"):
                for s0, s1 in combinations(g.seeds, 2):
                    best, _, _ = membership_jaccard(lab[(m, s0)], lab[(m, s1)],
                                                    K[(m, s0)], K[(m, s1)])
                    if m == "base":
                        floor.append(best)
                    rows.append({"layer": L, "percentile": p, "kind": "cross-seed",
                                 "model": m, "pair": f"{s0}-{s1}",
                                 "K_a": K[(m, s0)], "K_b": K[(m, s1)],
                                 "median_best_jaccard": float(np.median(best)),
                                 "p5_best_jaccard": float(np.percentile(best, 5)),
                                 "ari": adjusted_rand(lab[(m, s0)], lab[(m, s1)],
                                                      K[(m, s0)], K[(m, s1)])})
            cut = float(np.percentile(np.concatenate(floor), 5))

            intro = {}
            for s in g.seeds:
                best, _, _ = membership_jaccard(lab[("base", s)], lab[("rmu", s)],
                                                K[("base", s)], K[("rmu", s)])
                intro[s] = np.where(best < cut)[0]
                rows.append({"layer": L, "percentile": p, "kind": "base-vs-rmu",
                             "model": "rmu", "pair": f"seed{s}",
                             "K_a": K[("base", s)], "K_b": K[("rmu", s)],
                             "median_best_jaccard": float(np.median(best)),
                             "p5_best_jaccard": float(np.percentile(best, 5)),
                             "cutoff": cut,
                             "n_introduced": int(len(intro[s])),
                             "introduced_frac": float(len(intro[s]) / K[("rmu", s)]),
                             "ari": adjusted_rand(lab[("base", s)], lab[("rmu", s)],
                                                  K[("base", s)], K[("rmu", s)])})
                log.info("  L%-2d p%-3g seed%d  ARI=%.3f  median best-member-J "
                         "%.3f (floor cut %.3f)  introduced %d/%d = %.3f",
                         L, p, s, rows[-1]["ari"], rows[-1]["median_best_jaccard"],
                         cut, len(intro[s]), K[("rmu", s)],
                         rows[-1]["introduced_frac"])

            # Cross-seed agreement of the introduced set, corresponded by member
            # overlap between the two RMU dictionaries.
            rng = np.random.default_rng(0)
            for s0, s1 in combinations(g.seeds, 2):
                best, partner, _ = membership_jaccard(
                    lab[("rmu", s0)], lab[("rmu", s1)], K[("rmu", s0)], K[("rmu", s1)])
                keep = np.where(best >= cut)[0]          # regions of s1 with a partner
                if len(keep) == 0:
                    continue
                inY = np.isin(keep, intro[s1])
                inX = np.isin(partner[keep], intro[s0])
                inter, union = int((inX & inY).sum()), int((inX | inY).sum())
                jac = inter / union if union else float("nan")
                M, a, b = len(keep), int(inX.sum()), int(inY.sum())
                null = np.empty(n_sim)
                for i in range(n_sim):
                    pa = np.zeros(M, bool); pa[rng.choice(M, a, replace=False)] = True
                    pb = np.zeros(M, bool); pb[rng.choice(M, b, replace=False)] = True
                    uu = (pa | pb).sum()
                    null[i] = (pa & pb).sum() / uu if uu else np.nan
                sd = float(np.nanstd(null))
                jrows.append({
                    "layer": L, "percentile": p, "pair": f"{s0}-{s1}",
                    "n_corresponded": M, "a": a, "b": b, "inter": inter,
                    "union": union, "jaccard": jac,
                    "null_median": float(np.nanmedian(null)),
                    "null_p95": float(np.nanpercentile(null, 95)), "null_sd": sd,
                    "z_vs_null": ((jac - float(np.nanmedian(null))) / sd
                                  if sd > 0 else float("nan"))})
                log.info("    cross-seed %s: J=%.3f vs null %.3f (p95 %.3f) "
                         "z=%.1f  [M=%d a=%d b=%d]", jrows[-1]["pair"], jac,
                         jrows[-1]["null_median"], jrows[-1]["null_p95"],
                         jrows[-1]["z_vs_null"], M, a, b)
    return rows, jrows


def verdict(diff_rows, jac_rows, g, mem_rows=None, mem_jac_rows=None) -> dict:
    """Apply the pre-registered kill criteria mechanically, on the mean basis."""
    def sel(rows, **kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    informative = [L for L in g.layers if L >= 5]
    out: dict = {"criteria": {}, "per_layer": {}}

    seed_j = sel(jac_rows, basis="mean", axis="seed", filter="all")
    pct_j = sel(jac_rows, basis="mean", axis="percentile", filter="all")
    topd_j = sel(jac_rows, basis="mean", axis="seed", filter="topD")

    def summarise(rows, layers):
        rows = [r for r in rows if r["layer"] in layers
                and np.isfinite(r["jaccard"])]
        if not rows:
            return {}
        j = np.array([r["jaccard"] for r in rows])
        z = np.array([r["z_vs_null"] for r in rows])
        beats = np.array([r["jaccard"] > r["null_p95"] for r in rows])
        return {"n": len(rows), "jaccard_median": float(np.median(j)),
                "jaccard_min": float(j.min()), "jaccard_max": float(j.max()),
                "z_median": float(np.median(z)),
                "frac_above_null_p95": float(beats.mean())}

    out["criteria"]["seed_jaccard_vs_null"] = summarise(seed_j, informative)
    out["criteria"]["percentile_jaccard_vs_null"] = summarise(pct_j, informative)
    out["criteria"]["seed_jaccard_topD"] = summarise(topd_j, informative)

    vac = sel(diff_rows, basis="mean")
    out["criteria"]["vacuity"] = {
        f"L{L}": {"introduced_frac_median": float(np.median(
                      [r["introduced_frac"] for r in vac if r["layer"] == L])),
                  "control_frac_median": float(np.median(
                      [r["ctl_introduced_frac"] for r in vac if r["layer"] == L])),
                  "ratio_median": float(np.median(
                      [r["vacuity_ratio"] for r in vac if r["layer"] == L]))}
        for L in g.layers}

    if mem_rows:
        bvr = [r for r in mem_rows if r["kind"] == "base-vs-rmu"]
        xs = [r for r in mem_rows if r["kind"] == "cross-seed" and r["model"] == "base"]
        out["criteria"]["membership"] = {
            f"L{L}": {
                "ari_base_vs_rmu_median": float(np.median(
                    [r["ari"] for r in bvr if r["layer"] == L])),
                "ari_cross_seed_control_median": float(np.median(
                    [r["ari"] for r in xs if r["layer"] == L])),
                "introduced_frac_median": float(np.median(
                    [r["introduced_frac"] for r in bvr if r["layer"] == L])),
                "median_best_member_jaccard": float(np.median(
                    [r["median_best_jaccard"] for r in bvr if r["layer"] == L])),
            } for L in g.layers}
    if mem_jac_rows:
        mj = [r for r in mem_jac_rows if r["layer"] in informative
              and np.isfinite(r["jaccard"])]
        if mj:
            out["criteria"]["membership_seed_jaccard"] = {
                "n": len(mj),
                "jaccard_median": float(np.median([r["jaccard"] for r in mj])),
                "z_median": float(np.median([r["z_vs_null"] for r in mj])),
                "frac_above_null_p95": float(np.mean(
                    [r["jaccard"] > r["null_p95"] for r in mj]))}

    s = out["criteria"]["seed_jaccard_vs_null"]
    pj = out["criteria"]["percentile_jaccard_vs_null"]
    kill_seed = bool(s and s["frac_above_null_p95"] < 0.5)
    kill_pct = bool(pj and pj["jaccard_median"] <= 0.0)
    td = out["criteria"]["seed_jaccard_topD"]
    needs_filter = bool(s and td and s["frac_above_null_p95"] < 0.5
                        <= td.get("frac_above_null_p95", 0.0))

    out["kill_criteria"] = {
        "introduced_set_at_or_below_random_null_across_seeds": kill_seed,
        "adjacent_percentiles_nominate_disjoint_sets": kill_pct,
        "stability_only_survives_D_filtering": needs_filter,
    }
    out["PASS"] = not (kill_seed or kill_pct)
    return out


if __name__ == "__main__":
    main()
