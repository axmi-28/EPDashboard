"""Gate 2 C2 at scale — is EP really less cross-layer consistent than random?

The Arm C headline (`MI(early, late)` below a matched-K coreset null by 3.5 and
7.7 sd) is the most consequential result in the gate, because it removes the
last framing Arm A left standing: that EP might be worth having as a stable
discrete coordinate system comparable across layers and checkpoints.

It also rests on 600 samples in a table with 25,520 cells (P2) or 337,000 cells
(P1), where a plug-in MI estimate is mostly bias. A matched-K null cancels the
bias only if both partitions induce similar occupancy, which is exactly what is
under test. So the claim needs three things it did not have:

1. **Scale.** Pooled content positions give 7,460 samples instead of 600, and
   the Pile background gives 192,000 — the distribution the coordinate-system
   claim is actually about.
2. **A sample-size sweep.** If the gap is a small-sample artifact it shrinks
   with n. If it is real it is stable or widens.
3. **A bias-free statistic.** Held-out prediction accuracy: fit `early region ->
   most common late region` on one half, score the other. Cross-fitting makes
   it unbiased by construction, and it answers the operative question directly
   — does knowing where an activation sits early tell you where it sits late?
   Reported against the majority-class baseline, since a partition that sends
   almost everything to one late region scores well for the wrong reason.

    python -m experiments.monitor.run_gate2_c2 --source labelled
    python -m experiments.monitor.run_gate2_c2 --source background
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dicts import lean_path, load_lean
from .gate2_route import assign, coreset_partition
from .run_gate2_c import mutual_information

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
LAB = ROOT / "gate2_bc_labelled.npz"
BG3 = ROOT / "gate2_c_background3.npz"
OUT = ROOT / "gate2_c2_scale.json"

N_NULL_DRAWS = 8
SIZES = (600, 1500, 3000, 7460)

PAIRS = [
    ("P2_primary", ("gemma-2-2b-it", 12, 10), ("gemma-2-2b-it", 20, 10)),
    ("P1_exploratory", ("gemma-2-2b-it", 4, 4), ("gemma-2-2b-it", 20, 4)),
]


def predict_accuracy(early: np.ndarray, late: np.ndarray, *, n_folds: int = 5,
                     seed: int = 0) -> tuple[float, float]:
    """Held-out accuracy of `early region -> modal late region`, and the
    majority-class baseline on the same folds.

    Unbiased by construction: the mapping is fitted on k-1 folds and scored on
    the held-out one. An early region unseen in training falls back to the
    training-set modal late region, which is the same thing the baseline does.
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(early))
    hit = base = 0
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        K = int(early.max()) + 1
        L = int(late.max()) + 1
        joint = np.zeros((K, L), dtype=np.int64)
        np.add.at(joint, (early[tr], late[tr]), 1)
        modal = joint.argmax(axis=1)
        seen = joint.sum(axis=1) > 0
        fallback = int(np.bincount(late[tr], minlength=L).argmax())
        pred = np.where(seen[early[te]], modal[early[te]], fallback)
        hit += int((pred == late[te]).sum())
        base += int((late[te] == fallback).sum())
    n = int(len(early))
    return hit / n, base / n


def analyse(xe, xl, de, dl, pool, sizes, seed=0) -> dict:
    """MI and prediction accuracy for EP and matched-K coresets, over sizes."""
    re_ep = assign(xe, de["exemplars"], de["center"])
    rl_ep = assign(xl, dl["exemplars"], dl["center"])
    null_re, null_rl = [], []
    for j in range(N_NULL_DRAWS):
        null_re.append(assign(xe, coreset_partition(pool, de["center"],
                                                    de["K"], 700 + j),
                              de["center"]))
        null_rl.append(assign(xl, coreset_partition(pool, dl["center"],
                                                    dl["K"], 800 + j),
                              dl["center"]))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(re_ep))
    out = []
    for n in sizes:
        if n > len(order):
            continue
        idx = order[:n]
        mi_ep = mutual_information(re_ep[idx], rl_ep[idx])
        acc_ep, base_ep = predict_accuracy(re_ep[idx], rl_ep[idx])
        mi_n, acc_n, base_n = [], [], []
        for j in range(N_NULL_DRAWS):
            mi_n.append(mutual_information(null_re[j][idx], null_rl[j][idx]))
            a, b = predict_accuracy(null_re[j][idx], null_rl[j][idx])
            acc_n.append(a)
            base_n.append(b)
        sd_mi = float(np.std(mi_n, ddof=1))
        sd_ac = float(np.std(acc_n, ddof=1))
        out.append({
            "n": int(n),
            "mi_ep": mi_ep, "mi_null": float(np.mean(mi_n)), "mi_null_sd": sd_mi,
            "mi_margin_sd": float((mi_ep - np.mean(mi_n)) / max(sd_mi, 1e-9)),
            "acc_ep": acc_ep, "acc_ep_baseline": base_ep,
            "acc_null": float(np.mean(acc_n)),
            "acc_null_baseline": float(np.mean(base_n)),
            "acc_null_sd": sd_ac,
            "acc_margin_sd": float((acc_ep - np.mean(acc_n)) / max(sd_ac, 1e-9)),
            "lift_ep": acc_ep - base_ep,
            "lift_null": float(np.mean(acc_n) - np.mean(base_n)),
            "n_early_occupied_ep": int(len(np.unique(re_ep[idx]))),
            "n_early_occupied_null": float(np.mean(
                [len(np.unique(null_re[j][idx])) for j in range(N_NULL_DRAWS)])),
        })
    return {"rows": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="labelled",
                    choices=["labelled", "background"])
    args = ap.parse_args()

    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")
    if args.source == "labelled":
        z = np.load(LAB)
        lengths = z["lengths"]
        pre, suf = int(z["prefix_len"]), int(z["suffix_len"])
        off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        idx = np.concatenate([np.arange(off[i] + pre, off[i + 1] - suf)
                              for i in range(len(lengths))])
        get = lambda L: z[f"x_L{L}"][idx].astype(np.float32)
        sizes = SIZES
        label = f"labelled content positions (n={len(idx)})"
    else:
        z = np.load(BG3)
        get = lambda L: z[f"x_L{L}"].astype(np.float32)
        n_tot = z["x_L20"].shape[0]
        sizes = (600, 3000, 20_000, 60_000, n_tot)
        label = f"Pile background positions (n={n_tot})"

    print(f"source: {label}\n")
    out = {"source": args.source, "label": label, "n_null_draws": N_NULL_DRAWS,
           "pairs": {}}
    for tag, (em, eL, ep), (lm, lL, lp) in PAIRS:
        de = load_lean(lean_path(DICTS, em, eL, ep))
        dl = load_lean(lean_path(DICTS, lm, lL, lp))
        res = analyse(get(eL), get(lL), de, dl, pool, sizes)
        out["pairs"][tag] = {"early": f"L{eL} p{ep} K={de['K']}",
                             "late": f"L{lL} p{lp} K={dl['K']}", **res}
        print(f"=== {tag}: L{eL} p{ep} (K={de['K']}) -> L{lL} p{lp} "
              f"(K={dl['K']}) ===")
        print(f"{'n':>7s} {'MI ep':>7s} {'MI null':>8s} {'margin':>8s} | "
              f"{'acc ep':>7s} {'acc null':>9s} {'margin':>8s} | "
              f"{'lift ep':>8s} {'lift null':>10s}")
        for r in res["rows"]:
            print(f"{r['n']:7d} {r['mi_ep']:7.4f} {r['mi_null']:8.4f} "
                  f"{r['mi_margin_sd']:+8.1f} | {r['acc_ep']:7.4f} "
                  f"{r['acc_null']:9.4f} {r['acc_margin_sd']:+8.1f} | "
                  f"{r['lift_ep']:+8.4f} {r['lift_null']:+10.4f}")
        print()

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[args.source] = out
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
