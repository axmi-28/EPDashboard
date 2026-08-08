"""Gate 2 Arm B — region sequences across tokens.

Every prompt becomes a string of integers, one region id per token:

    prompt 47:  [412, 412, 88, 88, 88, 301, 412, ...]

and the questions are whether that string has structure, and whether the
structure says anything a single symbol does not.

Three things this does that a naive version would get wrong:

1. **Scaffold positions are masked.** The gemma chat wrapper contributes 4
   identical prefix tokens and 5 identical suffix tokens to all 600 prompts.
   Their activations are near-identical across the corpus, so including them
   manufactures both a high repeat rate and a strong "shared structure" signal
   that is purely the template. Only the ~12 content positions per prompt are
   used.

2. **The transition table is fitted on Pile, not on the prompts being scored.**
   A table fitted on the 600 labelled prompts and then used to score them leaks.
   A table fitted on the benign half only is a ~3,600-transition sample of a
   K x K matrix, which at K=176 is 31,000 cells. The background corpus is 1,500
   held-out Pile documents (~190,000 transitions), which is the population a
   deployed monitor would actually have modelled as "normal".

3. **The order control is a first-class scorer, not an afterthought.** T1 scores
   a prompt by mean `-log P(r_t+1 | r_t)`; T2 scores it by mean `-log P(r_t)`,
   which uses the same table's marginal and discards order entirely. If T1 does
   not beat T2, sequences carry nothing beyond the bag of regions and the whole
   trajectory idea is dead regardless of what T1's absolute AUROC looks like.

Two nulls run throughout: a matched-K random coreset partition, and a
within-prompt shuffle that preserves each prompt's region marginal while
destroying its order.

    python -m experiments.monitor.run_gate2_b
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .dicts import DEFAULT_MODEL_SHORT, PERCENTILES, lean_path, load_lean
from .gate2_route import coreset_partition, unit
from .scorers import auroc, tpr_at_fpr

ROOT = Path("artifacts/runs/monitor")
DICTS = ROOT / "dicts"
LAB = ROOT / "gate2_bc_labelled.npz"
BG = ROOT / "gate2_b_background.npz"
OUT_CSV = ROOT / "gate2_b_trajectory.csv"
OUT_JSON = ROOT / "gate2_b_transitions.json"

LAYER = 20
ALPHA = 0.1          # Laplace smoother on the K x K transition table
N_NULL_DRAWS = 3
SEQ_PERCENTILES = (2, 4, 8, 10)   # p=1 has 34M transition cells; see coverage


def assign_top2(x: np.ndarray, exemplars: np.ndarray, center: np.ndarray,
                chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Region id and the top1-top2 cosine margin, on GPU where available."""
    import torch

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    E = torch.from_numpy(np.ascontiguousarray(exemplars)).to(dev, torch.float32)
    ids, margins = [], []
    for s in range(0, len(x), chunk):
        q = torch.from_numpy(np.ascontiguousarray(
            unit(x[s:s + chunk].astype(np.float32), center))).to(dev, torch.float32)
        sim = q @ E.T
        top = sim.topk(2, dim=1)
        ids.append(top.indices[:, 0].cpu().numpy())
        margins.append((top.values[:, 0] - top.values[:, 1]).cpu().numpy())
        del q, sim, top
    return (np.concatenate(ids).astype(np.int64),
            np.concatenate(margins).astype(np.float32))


def sequences(ids: np.ndarray, lengths: np.ndarray, pre: int = 0,
              suf: int = 0) -> list[np.ndarray]:
    off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return [ids[off[i] + pre:off[i + 1] - suf] for i in range(len(lengths))]


def fit_transitions(seqs: list[np.ndarray], K: int,
                    alpha: float = ALPHA) -> dict:
    """Counts -> smoothed log-probabilities, plus coverage diagnostics."""
    C = np.zeros((K, K), dtype=np.float64)
    uni = np.zeros(K, dtype=np.float64)
    n_trans = 0
    for s in seqs:
        if len(s) == 0:
            continue
        np.add.at(uni, s, 1.0)
        if len(s) > 1:
            np.add.at(C, (s[:-1], s[1:]), 1.0)
            n_trans += len(s) - 1
    logP = np.log((C + alpha) / (C.sum(1, keepdims=True) + alpha * K))
    logU = np.log((uni + alpha) / (uni.sum() + alpha * K))
    occupied = int((uni > 0).sum())
    return {"logP": logP, "logU": logU, "n_transitions": n_trans,
            "n_cells": K * K, "cells_observed": int((C > 0).sum()),
            "cells_possible_occupied": occupied * occupied,
            "regions_occupied": occupied,
            "repeat_rate": float(np.trace(C) / max(C.sum(), 1)),
            "counts_diag": float(np.trace(C)), "counts_total": float(C.sum())}


def repeat_stats(seqs: list[np.ndarray], seed: int = 0) -> dict:
    """Observed repeat rate against a within-prompt shuffle null.

    The shuffle preserves each prompt's region marginal exactly and destroys
    only the order, so the gap isolates sequential structure from the fact that
    a few large regions dominate.
    """
    rng = np.random.default_rng(seed)
    obs = tot = 0
    sh_obs = sh_tot = 0
    for s in seqs:
        if len(s) < 2:
            continue
        obs += int((s[:-1] == s[1:]).sum())
        tot += len(s) - 1
        for _ in range(5):
            t = rng.permutation(s)
            sh_obs += int((t[:-1] == t[1:]).sum())
            sh_tot += len(t) - 1
    return {"repeat_rate": obs / max(tot, 1),
            "repeat_rate_shuffled": sh_obs / max(sh_tot, 1),
            "n_transitions": tot}


def prompt_scores(seqs, margins_seq, dist_seq, table) -> dict[str, np.ndarray]:
    """The per-prompt trajectory scorers."""
    T1, T2, T3, T4, D1 = [], [], [], [], []
    for s, mg, ds in zip(seqs, margins_seq, dist_seq):
        if len(s) == 0:
            T1.append(0.0); T2.append(0.0); T3.append(0.0)
            T4.append(0.0); D1.append(0.0)
            continue
        T1.append(-table["logP"][s[:-1], s[1:]].mean() if len(s) > 1 else 0.0)
        T2.append(-table["logU"][s].mean())
        T3.append(len(np.unique(s)) / len(s))
        T4.append(float(np.mean(mg)))
        D1.append(float(np.mean(ds)))
    return {"T1_bigram_surprise": np.array(T1), "T2_unigram_surprise": np.array(T2),
            "T3_distinct_frac": np.array(T3), "T4_mean_margin": np.array(T4),
            "D1_mean_distance": np.array(D1)}


def main() -> None:
    zl, zb = np.load(LAB), np.load(BG)
    y = zl["y"]
    pre, suf = int(zl["prefix_len"]), int(zl["suffix_len"])
    xl, ll = zl["x_L20"], zl["lengths"]
    xb, lb = zb["x_L20"], zb["lengths"]
    print(f"labelled: {xl.shape[0]} positions / {len(ll)} prompts "
          f"(scaffold {pre}+{suf} masked -> "
          f"{int(ll.sum()) - len(ll) * (pre + suf)} content positions)")
    print(f"background: {xb.shape[0]} positions / {len(lb)} Pile documents")

    pool = np.load(ROOT / "refpool.npy", mmap_mode="r")
    rows: list[dict] = []
    diag: dict = {"layer": LAYER, "alpha": ALPHA,
                  "n_background_documents": int(len(lb)),
                  "n_labelled_prompts": int(len(y)), "percentiles": {}}

    for p in SEQ_PERCENTILES:
        d = load_lean(lean_path(DICTS, DEFAULT_MODEL_SHORT, LAYER, p))
        c, K = d["center"], d["K"]
        print(f"\n=== p={p}  K={K} ===", flush=True)

        for part, exemplars in [("EP", d["exemplars"]),
                                ("CORE", coreset_partition(pool, c, K, 2000))]:
            ib, _ = assign_top2(xb, exemplars, c)
            il, ml = assign_top2(xl, exemplars, c)
            # Cosine distance to the assigned exemplar, per position.
            dl = 1.0 - np.einsum("ij,ij->i", unit(xl.astype(np.float32), c),
                                 exemplars[il].astype(np.float32))

            bseq = sequences(ib, lb)
            lseq = sequences(il, ll, pre, suf)
            mseq = sequences(ml, ll, pre, suf)
            dseq = sequences(dl, ll, pre, suf)

            table = fit_transitions(bseq, K)
            rs_bg = repeat_stats(bseq)
            rs_lab = repeat_stats(lseq)
            sc = prompt_scores(lseq, mseq, dseq, table)

            cov = table["cells_observed"] / max(table["cells_possible_occupied"], 1)
            if part == "EP":
                print(f"  background: {table['n_transitions']} transitions, "
                      f"{table['regions_occupied']}/{K} regions occupied, "
                      f"{table['cells_observed']} distinct pairs seen "
                      f"({cov:.1%} of occupied-pair space)")
                print(f"  repeat rate  Pile {rs_bg['repeat_rate']:.3f} "
                      f"(shuffled {rs_bg['repeat_rate_shuffled']:.3f})   "
                      f"prompts {rs_lab['repeat_rate']:.3f} "
                      f"(shuffled {rs_lab['repeat_rate_shuffled']:.3f})")

            for name, s in sc.items():
                a = auroc(s, y == 1)
                rows.append({
                    "percentile": p, "K": K, "partition": part, "scorer": name,
                    "auroc": a, "auroc_flipped": max(a, 1 - a),
                    "tpr1": tpr_at_fpr(s, y == 1, 0.01),
                    "mean_harmful": float(s[y == 1].mean()),
                    "mean_benign": float(s[y == 0].mean()),
                })
            if part == "EP":
                diag["percentiles"][str(p)] = {
                    "K": K,
                    "background": {k: v for k, v in table.items()
                                   if k not in ("logP", "logU")},
                    "coverage_occupied_pairs": cov,
                    "repeat_background": rs_bg, "repeat_labelled": rs_lab,
                }

        sub = [r for r in rows if r["percentile"] == p]
        print(f"  {'scorer':22s} {'EP auroc':>9s} {'CORE auroc':>11s}")
        for name in ("T1_bigram_surprise", "T2_unigram_surprise",
                     "T3_distinct_frac", "T4_mean_margin", "D1_mean_distance"):
            e = next(r for r in sub if r["scorer"] == name and r["partition"] == "EP")
            cr = next(r for r in sub if r["scorer"] == name and r["partition"] == "CORE")
            print(f"  {name:22s} {e['auroc_flipped']:9.4f} "
                  f"{cr['auroc_flipped']:11.4f}")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=float))

    print("\n--- B0 kill switch: does order carry anything beyond the bag? ---")
    for p in SEQ_PERCENTILES:
        e1 = next(r for r in rows if r["percentile"] == p and r["partition"] == "EP"
                  and r["scorer"] == "T1_bigram_surprise")
        e2 = next(r for r in rows if r["percentile"] == p and r["partition"] == "EP"
                  and r["scorer"] == "T2_unigram_surprise")
        rb = diag["percentiles"][str(p)]["repeat_background"]
        gap = rb["repeat_rate"] - rb["repeat_rate_shuffled"]
        print(f"  p={p:<3d} T1 {e1['auroc_flipped']:.4f} vs T2 "
              f"{e2['auroc_flipped']:.4f} (order gain "
              f"{e1['auroc_flipped'] - e2['auroc_flipped']:+.4f}) | "
              f"Pile repeat excess over shuffle {gap:+.4f}")
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
