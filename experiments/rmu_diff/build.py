"""Gate 1B, stage 1 — build the dictionary grid.

Both models resident, one forward pass per (model, layer), then every dictionary
for that layer built by replaying the activations in each seed's stream order.
The forward pass is the whole GPU cost and is paid exactly once per (model,
layer); seeds and percentiles are free after that.

Grid: 4 layers x 2 models x 3 seeds x 2 percentiles = 48 dictionaries under
**shared** calibration (base mu, theta applied to both). Gate 1A showed why that
is not a stylistic choice: theta_RMU at L7 is 0.387 against base 0.786, and
K ~ (1-theta)^4.6 puts the per-model RMU dictionary two orders of magnitude
larger than base, which makes Hungarian matching meaningless. The per-model arm
runs separately, with an abort ceiling, as a diagnostic.

Two deliberate deviations from `ep.discover`, both required by *diffing* rather
than by building:

  saturation is off   A build that stops early has consumed a different stream
                      than its counterpart. Both models must see identical
                      prompts in identical order, so the budget is the pool and
                      saturation is recorded, not acted on.
  no prompt heaps     `constituent_sample_indices` already identifies every
                      member exactly, so region contents are reconstructed from
                      the pool at analysis time. Skipping the per-activation
                      heap loop removes ~25M Python iterations across the grid
                      and the 491 KB/region member reservoir with it.

    python -m experiments.rmu_diff.build --out artifacts/runs/rmu_diff/grid
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from .gate1a import (BASE_ID, BASE_REV, LAYERS, PERCENTILES, RMU_ID, RMU_REV,
                     _write_csv, _write_json, calibrate_from)

log = logging.getLogger("rmu_diff.build")

CLUSTER_CHUNK = 8192


def stream_perm(pool, prompt_ids: np.ndarray, seed: int) -> np.ndarray:
    """Activation permutation matching `discover(seed=...)`'s prompt order."""
    from .data import stream_order

    rank = np.empty(len(pool), dtype=np.int64)
    rank[np.asarray(stream_order(pool, seed))] = np.arange(len(pool))
    return np.argsort(rank[prompt_ids], kind="stable")


def extract_fp16(qm, texts: list[str], layer: int, *, batch_size: int,
                 max_positions: int, slice_size: int = 512):
    """Per-position activations for the whole pool, accumulated in fp16.

    `ep.extract_per_position` builds one fp32 list for everything it is handed
    and concatenates at the end, so handing it 4400 prompts at once peaks at
    ~21 GB for a 528k x 4096 set. Slicing the prompt list and downcasting each
    slice caps the peak at the fp16 total plus one slice.
    """
    xs, pids, poss = [], [], []
    n_fwd, n_tok = 0, 0
    for s in range(0, len(texts), slice_size):
        r = qm.extract_per_position(texts[s:s + slice_size], layer=layer,
                                    batch_size=batch_size,
                                    max_positions_per_prompt=max_positions,
                                    skip_first=True)
        xs.append(r.x.astype(np.float16))
        pids.append(r.prompt_ids + s)
        poss.append(r.position_ids)
        n_fwd += r.n_forward_passes
        n_tok += r.n_tokens
        del r
    return (np.concatenate(xs), np.concatenate(pids), np.concatenate(poss),
            n_fwd, n_tok)


def build_one(x: np.ndarray, perm: np.ndarray, center: np.ndarray,
              threshold: float, *, max_partitions: int | None = None):
    """Leader-cluster one activation set in one stream order at one threshold.

    Returns (dictionary, meta). `max_partitions` aborts the pass — a per-model
    calibration at a collapsed theta can run K into the tens of thousands, and a
    counting pass that was only ever going to report an integer should not be
    allowed to exhaust the box first.
    """
    from ep.discovery.dictionary import Dictionary
    from qwen_ep.sweep_p import member_reservoir_cap

    d = Dictionary(center=center.astype(np.float32), threshold=float(threshold))
    t0 = time.time()
    last_new, n_batches, aborted = 0, 0, False
    k_trace: list[list[int]] = []
    with member_reservoir_cap(d, 0):
        for i, s in enumerate(range(0, len(perm), CLUSTER_CHUNK)):
            before = len(d)
            d.add_batch(x[perm[s:s + CLUSTER_CHUNK]].astype(np.float32),
                        iteration=i, global_index_start=s)
            if len(d) > before:
                last_new = i
            n_batches = i + 1
            k_trace.append([int(min(s + CLUSTER_CHUNK, len(perm))), len(d)])
            if max_partitions is not None and len(d) > max_partitions:
                log.warning("  abort: K=%d exceeded ceiling %d at n=%d",
                            len(d), max_partitions, s + CLUSTER_CHUNK)
                aborted = True
                break
    if not aborted:
        d.finalize()
    members = sorted((p.member_count for p in d.partitions), reverse=True)
    meta = {
        "K": len(d), "threshold": float(threshold),
        "center_norm": float(np.linalg.norm(center)),
        "n_activations": int(len(perm)), "aborted": aborted,
        # Saturation is recorded, never acted on: an early stop would mean the
        # two checkpoints consumed different streams.
        "saturated_would_have": bool(n_batches - 1 - last_new >= 1),
        "batches_since_last_new": int(n_batches - 1 - last_new),
        "largest_partition": members[0] if members else 0,
        "median_members": float(np.median(members)) if members else 0.0,
        "singletons": int(sum(1 for m in members if m == 1)),
        "build_s": round(time.time() - t0, 1),
        "K_trace": k_trace,
    }
    return d, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/runs/rmu_diff/grid")
    ap.add_argument("--n-bio", type=int, default=1250)
    ap.add_argument("--n-cyber", type=int, default=950)
    ap.add_argument("--n-mmlu", type=int, default=2200)
    ap.add_argument("--layers", default=",".join(str(L) for L in LAYERS))
    ap.add_argument("--percentiles", default=",".join(str(p) for p in PERCENTILES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--calibration", default="shared",
                    choices=["shared", "per-model"])
    ap.add_argument("--calibration-tokens", type=int, default=200_000)
    ap.add_argument("--style", default="chat", choices=["chat", "plain"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-positions", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-partitions", type=int, default=60_000,
                    help="abort ceiling for a single build (per-model arm)")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(s) for s in args.layers.split(",")]
    percentiles = [float(s) for s in args.percentiles.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    from transformers import AutoTokenizer

    from qwen_ep.adapter import QwenModel

    from .data import build_pool

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    pool = build_pool(n_bio=args.n_bio, n_cyber=args.n_cyber,
                      n_mmlu=args.n_mmlu, style=args.style, tokenizer=tok,
                      min_tokens=args.min_tokens, max_tokens=args.max_tokens)
    lens = np.array([p.n_tokens for p in pool])
    lab = np.array([p.label for p in pool])
    log.info("pool: %d prompts (%d forget / %d retain), median %d tokens",
             len(pool), int((lab == "forget").sum()), int((lab == "retain").sum()),
             int(np.median(lens)))
    _write_csv(out / "pool.csv", [p.as_row() for p in pool])

    manifest: dict = {
        "models": {"base": {"id": BASE_ID, "revision": BASE_REV},
                   "rmu": {"id": RMU_ID, "revision": RMU_REV}},
        "config": vars(args) | {"layers": layers, "percentiles": percentiles,
                                "seeds": seeds},
        "pool": {"n": len(pool),
                 "n_forget": int((lab == "forget").sum()),
                 "n_retain": int((lab == "retain").sum()),
                 "median_tokens": int(np.median(lens))},
        "runs": [],
    }

    # Both models resident: 2 x 14.5 GB bf16 on a 40 GB card. Loading once and
    # extracting both per layer keeps the paired activations adjacent in time
    # and removes seven model loads.
    log.info("== loading both checkpoints ==")
    models = {}
    for tag, mid, rev in (("base", BASE_ID, BASE_REV), ("rmu", RMU_ID, RMU_REV)):
        models[tag] = QwenModel(mid, device=args.device, prepend_bos=True,
                                revision=rev, tokenizer_id=BASE_ID,
                                tokenizer_revision=BASE_REV)
    texts = [p.text for p in pool]

    cal_rows, run_rows = [], []
    for L in layers:
        log.info("== layer %d ==", L)
        x, pid = {}, {}
        for tag, qm in models.items():
            t0 = time.time()
            x[tag], pid[tag], _, _, _ = extract_fp16(
                qm, texts, L, batch_size=args.batch_size,
                max_positions=args.max_positions)
            dt = time.time() - t0
            log.info("  extract %-4s %d acts in %.0fs (%.0f acts/s)", tag,
                     len(x[tag]), dt, len(x[tag]) / dt)

        if x["base"].shape != x["rmu"].shape or not np.array_equal(pid["base"],
                                                                   pid["rmu"]):
            raise SystemExit(f"L{L}: paired extraction mismatch — activations "
                             "are not comparable")
        prompt_ids = pid["base"]
        np.save(out / f"prompt_ids_L{L}.npy", prompt_ids)
        del pid

        # Shared calibration: base mu and theta, applied to both models.
        cals = {}
        for p in percentiles:
            cals[("shared", p)] = calibrate_from(x["base"], p,
                                                 n_tokens=args.calibration_tokens)
            for tag in ("base", "rmu"):
                c = (cals[("shared", p)] if args.calibration == "shared"
                     else calibrate_from(x[tag], p, n_tokens=args.calibration_tokens))
                cals[(tag, p)] = c
                cal_rows.append({"layer": L, "model": tag, "percentile": p,
                                 "calibration": args.calibration,
                                 "theta": c.threshold,
                                 "center_norm": float(np.linalg.norm(c.center)),
                                 "n_activations": c.n_activations})
            log.info("  p%g: theta base=%.4f rmu=%.4f (%s calibration)", p,
                     cals[("base", p)].threshold, cals[("rmu", p)].threshold,
                     args.calibration)

        for tag in ("base", "rmu"):
            for seed in seeds:
                perm = stream_perm(pool, prompt_ids, seed)
                for p in percentiles:
                    cal = cals[(tag, p)]
                    d, meta = build_one(x[tag], perm, cal.center, cal.threshold,
                                        max_partitions=args.max_partitions)
                    name = f"{tag}_L{L}_p{p:g}_seed{seed}"
                    with (out / f"{name}.pkl").open("wb") as f:
                        pickle.dump(d, f)
                    meta |= {"name": name, "model": tag, "layer": L,
                             "percentile": p, "seed": seed,
                             "calibration": args.calibration}
                    manifest["runs"].append(meta)
                    run_rows.append({k: v for k, v in meta.items()
                                     if k != "K_trace"})
                    log.info("  %-24s K=%-6d largest=%-7d median=%-5.0f "
                             "singletons=%-5d %.0fs%s", name, meta["K"],
                             meta["largest_partition"], meta["median_members"],
                             meta["singletons"], meta["build_s"],
                             "  ABORTED" if meta["aborted"] else "")
        del x
        _write_json(out / "manifest.json", manifest)
        _write_csv(out / "runs.csv", run_rows)
        _write_csv(out / "calibration.csv", cal_rows)

    manifest["total_wall_s"] = round(time.time() - t_start, 1)
    _write_json(out / "manifest.json", manifest)
    log.info("== grid done: %d dictionaries in %.0fs -> %s ==",
             len(manifest["runs"]), time.time() - t_start, out)


if __name__ == "__main__":
    main()
