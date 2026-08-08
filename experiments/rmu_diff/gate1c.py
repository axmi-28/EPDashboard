"""Gate 1C — ground truth. What is actually in the regions the diff nominates.

Runs only after Gate 1B. Gate 1B's verdict reshapes what this gate can test:
the introduced set is vacuous for this intervention (RMU *consolidates*, so the
change lands on the dropped side and on one region that the matcher calls
persisted), and the stable structure is the dominant RMU region, which
reproduces across streaming seeds at member-Jaccard 0.81-0.92.

So H1/H2/H3 are adjudicated against three objects, in this order:

  dominant   the largest RMU region. H1's substantive claim -- one massive cell,
             member-forget-fraction > 0.9, coherence > 0.9, stable across seeds.
  dropped    the ~52% of base regions with no RMU counterpart at L7. If RMU
             dissolves the forget-carrying part of the base carving, these
             should be forget-heavy and their members should land in `dominant`.
  introduced the set the prereg named. Reported in full even though 1B showed it
             is 1-7 regions -- a pre-registered target is not dropped because it
             turned out small.

Everything here is computable from the saved dictionaries plus the pool: member
indices are stream positions, `stream_perm` maps those to canonical activation
indices, and `prompt_ids` maps those to prompts and hence to forget/retain
labels. No model, no GPU, no re-extraction.

Final-position membership is recovered without stored position ids: the
extractor emits positions 1..L-1 contiguously per prompt, so the last activation
carrying a given prompt id is that prompt's final token.

    python -m experiments.rmu_diff.gate1c --grid artifacts/runs/rmu_diff/grid/shared
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from .gate1a import _write_csv, _write_json
from .gate1b import Grid, _dirs, _stats, hungarian, label_vector, rebuild_pool

log = logging.getLogger("rmu_diff.gate1c")


def activation_labels(pool, prompt_ids: np.ndarray):
    """Per-activation forget/retain label, source, and final-position mask."""
    lab_of_prompt = np.array([p.label for p in pool])
    src_of_prompt = np.array([p.source for p in pool])
    is_forget = lab_of_prompt[prompt_ids] == "forget"
    source = src_of_prompt[prompt_ids]
    # Last activation carrying each prompt id == that prompt's final token.
    final = np.zeros(len(prompt_ids), dtype=bool)
    last = np.full(len(pool), -1, dtype=np.int64)
    np.maximum.at(last, prompt_ids, np.arange(len(prompt_ids)))
    final[last[last >= 0]] = True
    return is_forget, source, final


def region_table(d, lab: np.ndarray, is_forget: np.ndarray, source: np.ndarray,
                 final: np.ndarray) -> dict:
    """Per-region membership statistics, vectorised over the label vector."""
    K = len(d.partitions)
    st = _stats(d)
    ok = lab >= 0
    n = np.bincount(lab[ok], minlength=K).astype(np.float64)
    nf = np.bincount(lab[ok], weights=is_forget[ok].astype(np.float64), minlength=K)
    nfin = np.bincount(lab[ok], weights=final[ok].astype(np.float64), minlength=K)
    nfin_f = np.bincount(lab[ok & final],
                         weights=is_forget[ok & final].astype(np.float64), minlength=K)
    by_src = {s: np.bincount(lab[ok], weights=(source[ok] == s).astype(np.float64),
                             minlength=K) for s in np.unique(source)}
    return {"K": K, "n": n, "n_forget": nf,
            "forget_frac": np.divide(nf, np.maximum(n, 1)),
            "n_final": nfin, "n_final_forget": nfin_f,
            "final_forget_frac": np.divide(nfin_f, np.maximum(nfin, 1)),
            "coherence": st["c"], "D": st["D"], "by_source": by_src}


def mechanism_check(g: Grid, pool, layer: int, batch_size: int = 16,
                    max_positions: int = 256) -> list[dict]:
    """Is the region EP found *the direction RMU injected*? (needs the models)

    H1's substantive claim is not just "one big pure region" but that the region
    is anchored on RMU's control vector. Measured directly: re-extract the paired
    activations, estimate the control vector two ways —

        u_delta = normalize(mean over forget tokens of h_RMU - h_base)
        u_mean  = normalize(mean over forget tokens of h_RMU)   [RMU's loss
                  drives h_forget -> c*u, so this estimates c*u without the delta]

    — then compare the dominant region's stored mean direction, in the grid's own
    centred space, against `centred(c_hat * u_mean)` and against `-mu_hat`. The
    second is H1', the centring artifact the prereg committed to ruling out.
    """
    from qwen_ep.adapter import QwenModel

    from .build import extract_fp16, stream_perm
    from .gate1a import BASE_ID, BASE_REV, RMU_ID, RMU_REV

    texts = [p.text for p in pool]
    x = {}
    for tag, mid, rev in (("base", BASE_ID, BASE_REV), ("rmu", RMU_ID, RMU_REV)):
        qm = QwenModel(mid, prepend_bos=True, revision=rev,
                       tokenizer_id=BASE_ID, tokenizer_revision=BASE_REV)
        x[tag], pid, _, _, _ = extract_fp16(qm, texts, layer,
                                            batch_size=batch_size,
                                            max_positions=max_positions)
        del qm
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    is_forget, _, _ = activation_labels(pool, pid)
    xf_b = x["base"][is_forget].astype(np.float32)
    xf_r = x["rmu"][is_forget].astype(np.float32)
    u_delta = (xf_r - xf_b).mean(axis=0)
    u_delta /= np.linalg.norm(u_delta) + 1e-12
    mean_rmu = xf_r.mean(axis=0)
    c_hat = float(np.linalg.norm(mean_rmu))
    u_mean = mean_rmu / (c_hat + 1e-12)
    ones = np.ones(x["base"].shape[1], dtype=np.float32) / np.sqrt(x["base"].shape[1])

    rows = []
    for p in g.percentiles:
        for s in g.seeds:
            d = g.get("rmu", layer, p, s)
            perm = stream_perm(pool, pid, s)
            lab = label_vector(d, len(pid), perm)
            dom = int(np.argmax(np.bincount(lab[lab >= 0],
                                            minlength=len(d.partitions))))
            mu = d.center.astype(np.float32)
            mu_hat = mu / (np.linalg.norm(mu) + 1e-12)
            m = _dirs(d, "mean")[dom]
            e = _dirs(d, "exemplar")[dom]
            tgt = c_hat * u_mean - mu
            tgt /= np.linalg.norm(tgt) + 1e-12
            rows.append({
                "layer": layer, "percentile": p, "seed": s, "region": dom,
                "c_hat": c_hat, "u_mean_cos_ones": float(u_mean @ ones),
                "u_delta_cos_ones": float(u_delta @ ones),
                "u_mean_cos_u_delta": float(u_mean @ u_delta),
                "mu_norm": float(np.linalg.norm(mu)),
                "dom_mean_cos_centred_cu": float(m @ tgt),
                "dom_mean_cos_minus_mu_hat": float(m @ (-mu_hat)),
                "dom_exemplar_cos_centred_cu": float(e @ tgt),
                "dom_exemplar_cos_minus_mu_hat": float(e @ (-mu_hat)),
            })
            log.info("  p%-3g seed%d  region %-4d  cos(mean_dir, centred c*u)="
                     "%.4f   cos(mean_dir, -mu_hat)=%.4f", p, s, dom,
                     rows[-1]["dom_mean_cos_centred_cu"],
                     rows[-1]["dom_mean_cos_minus_mu_hat"])
    log.info("  c_hat=%.3f (%.2f x 6.5)  u_mean.ones=%.3f  u_delta.ones=%.3f  "
             "u_mean.u_delta=%.3f", c_hat, c_hat / 6.5, float(u_mean @ ones),
             float(u_delta @ ones), float(u_mean @ u_delta))
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="artifacts/runs/rmu_diff/grid/shared")
    ap.add_argument("--out", default="artifacts/runs/rmu_diff/gate1c")
    ap.add_argument("--mechanism-layer", type=int, default=-1,
                    help="re-extract at this layer and test the dominant "
                         "region's direction against RMU's control vector "
                         "(needs both models; -1 skips)")
    ap.add_argument("--batch-size", type=int, default=16)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    from .build import stream_perm

    g = Grid(Path(args.grid))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = rebuild_pool(g)
    log.info("pool %d prompts (%d forget)", len(pool),
             sum(p.label == "forget" for p in pool))

    dom_rows, reg_rows, diff_rows, flow_rows = [], [], [], []
    report: dict = {"grid": str(g.root)}

    for L in g.layers:
        pid = np.load(g.root / f"prompt_ids_L{L}.npy")
        n_acts = len(pid)
        is_forget, source, final = activation_labels(pool, pid)
        perms = {s: stream_perm(pool, pid, s) for s in g.seeds}
        log.info("== L%d: %d activations, %.3f forget, %d final-position ==",
                 L, n_acts, is_forget.mean(), int(final.sum()))

        for p in g.percentiles:
            tabs, labs = {}, {}
            for m in ("base", "rmu"):
                for s in g.seeds:
                    d = g.get(m, L, p, s)
                    labs[(m, s)] = label_vector(d, n_acts, perms[s])
                    tabs[(m, s)] = region_table(d, labs[(m, s)], is_forget,
                                                source, final)
                    t = tabs[(m, s)]
                    for i in range(t["K"]):
                        reg_rows.append({
                            "layer": L, "percentile": p, "model": m, "seed": s,
                            "region": i, "n": int(t["n"][i]),
                            "forget_frac": t["forget_frac"][i],
                            "n_final": int(t["n_final"][i]),
                            "final_forget_frac": t["final_forget_frac"][i],
                            "coherence": t["coherence"][i], "D": t["D"][i],
                            **{f"n_{k}": int(v[i]) for k, v in t["by_source"].items()}})

            # ---- the dominant region, per model and seed ----
            for m in ("base", "rmu"):
                for s in g.seeds:
                    t = tabs[(m, s)]
                    i = int(np.argmax(t["n"]))
                    tot_f = is_forget.sum()
                    dom_rows.append({
                        "layer": L, "percentile": p, "model": m, "seed": s,
                        "region": i, "n": int(t["n"][i]),
                        "share_of_all_acts": t["n"][i] / n_acts,
                        "forget_frac": t["forget_frac"][i],
                        "recall_of_forget": t["n_forget"][i] / max(tot_f, 1),
                        "coherence": t["coherence"][i], "D": t["D"][i],
                        "median_region_n": float(np.median(t["n"])),
                        "size_vs_median": t["n"][i] / max(np.median(t["n"]), 1),
                        "n_final": int(t["n_final"][i]),
                        "final_forget_frac": t["final_forget_frac"][i],
                        **{f"frac_{k}": (v[i] / max(t["n"][i], 1))
                           for k, v in t["by_source"].items()}})

            # ---- diff: dropped / introduced / persisted, and where mass went ----
            cut_pool = []
            for s0, s1 in combinations(g.seeds, 2):
                _, _, c = hungarian(g.get("base", L, p, s0),
                                    g.get("base", L, p, s1), "mean")
                cut_pool.append(c)
            cut = float(np.percentile(np.concatenate(cut_pool), 5))

            for s in g.seeds:
                b, r = g.get("base", L, p, s), g.get("rmu", L, p, s)
                rows_i, cols, cos = hungarian(b, r, "mean")
                matched_a = np.full(len(b.partitions), -1.0)
                matched_a[rows_i] = cos
                matched_b = np.full(len(r.partitions), -1.0)
                matched_b[cols] = cos
                dropped = np.where(matched_a < cut)[0]
                persisted_a = np.where(matched_a >= cut)[0]
                introduced = np.where(matched_b < cut)[0]
                tb, tr = tabs[("base", s)], tabs[("rmu", s)]

                def agg(t, idx):
                    if len(idx) == 0:
                        return {"n_regions": 0, "n_members": 0,
                                "forget_frac": float("nan")}
                    nn = t["n"][idx].sum()
                    return {"n_regions": int(len(idx)), "n_members": int(nn),
                            "forget_frac": float(t["n_forget"][idx].sum()
                                                 / max(nn, 1)),
                            "median_n": float(np.median(t["n"][idx]))}

                diff_rows.append({
                    "layer": L, "percentile": p, "seed": s, "cutoff": cut,
                    **{f"dropped_{k}": v for k, v in agg(tb, dropped).items()},
                    **{f"persisted_{k}": v for k, v in agg(tb, persisted_a).items()},
                    **{f"introduced_{k}": v for k, v in agg(tr, introduced).items()},
                    "all_base_forget_frac": float(is_forget.mean())})

                # Where did the dropped base regions' members end up?
                dom = int(np.argmax(tr["n"]))
                dm = np.isin(labs[("base", s)], dropped)
                pm = np.isin(labs[("base", s)], persisted_a)
                for tag, mask in (("dropped", dm), ("persisted", pm)):
                    if mask.sum() == 0:
                        continue
                    flow_rows.append({
                        "layer": L, "percentile": p, "seed": s, "source": tag,
                        "n_members": int(mask.sum()),
                        "forget_frac": float(is_forget[mask].mean()),
                        "frac_into_rmu_dominant": float(
                            (labs[("rmu", s)][mask] == dom).mean()),
                        "frac_of_forget_into_dominant": float(
                            (labs[("rmu", s)][mask & is_forget] == dom).mean())
                        if (mask & is_forget).sum() else float("nan"),
                        "frac_of_retain_into_dominant": float(
                            (labs[("rmu", s)][mask & ~is_forget] == dom).mean())
                        if (mask & ~is_forget).sum() else float("nan")})

    if args.mechanism_layer >= 0:
        log.info("== H1 mechanism: is the dominant region RMU's control vector? ==")
        mech = mechanism_check(g, pool, args.mechanism_layer,
                               batch_size=args.batch_size)
        _write_csv(out / "mechanism.csv", mech)
        report["H1_mechanism"] = mech

    _write_csv(out / "regions.csv", reg_rows)
    _write_csv(out / "dominant.csv", dom_rows)
    _write_csv(out / "diff_contents.csv", diff_rows)
    _write_csv(out / "mass_flow.csv", flow_rows)

    # ------------------------------------------------------------- summary
    def med(rows, key, **sel):
        v = [r[key] for r in rows
             if all(r[k] == vv for k, vv in sel.items()) and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    report["H1_dominant_region"] = {
        f"L{L}|p{p:g}": {
            "rmu": {k: med(dom_rows, k, layer=L, percentile=p, model="rmu")
                    for k in ("n", "share_of_all_acts", "forget_frac",
                              "recall_of_forget", "coherence", "D",
                              "size_vs_median", "final_forget_frac")},
            "base": {k: med(dom_rows, k, layer=L, percentile=p, model="base")
                     for k in ("n", "share_of_all_acts", "forget_frac",
                               "recall_of_forget", "coherence", "D",
                               "size_vs_median", "final_forget_frac")},
        } for L in g.layers for p in g.percentiles}
    report["dropped_vs_persisted"] = {
        f"L{L}|p{p:g}": {k: med(diff_rows, k, layer=L, percentile=p)
                         for k in ("dropped_n_regions", "dropped_forget_frac",
                                   "persisted_n_regions", "persisted_forget_frac",
                                   "introduced_n_regions", "introduced_forget_frac")}
        for L in g.layers for p in g.percentiles}
    _write_json(out / "gate1c.json", report)

    log.info("== H1: dominant region ==")
    log.info("L    p    model  N        share  forget_frac  recall  coher  size/med")
    for L in g.layers:
        for p in g.percentiles:
            for m in ("base", "rmu"):
                d = report["H1_dominant_region"][f"L{L}|p{p:g}"][m]
                log.info("%-4d %-4g %-6s %-8.0f %.3f  %.3f        %.3f   %.3f  %.1f",
                         L, p, m, d["n"], d["share_of_all_acts"], d["forget_frac"],
                         d["recall_of_forget"], d["coherence"], d["size_vs_median"])
    log.info("== dropped vs persisted (forget fraction of members) ==")
    for k, v in report["dropped_vs_persisted"].items():
        log.info("  %-12s dropped %3.0f regions ff=%.3f | persisted %3.0f ff=%.3f "
                 "| introduced %2.0f ff=%.3f", k, v["dropped_n_regions"],
                 v["dropped_forget_frac"], v["persisted_n_regions"],
                 v["persisted_forget_frac"], v["introduced_n_regions"],
                 v["introduced_forget_frac"])
    log.info("results -> %s", out)


if __name__ == "__main__":
    main()
