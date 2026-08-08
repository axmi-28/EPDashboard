"""Gate 1A — orient and sanity, before any dictionary grid is bought.

Six deliverables, in the order that a failure in one makes the next pointless:

  0  weight diff        which tensors actually differ between the checkpoints.
                        Exact ground truth for locality; costs one streaming
                        pass over two safetensors sets, no GPU, no forward.
  1  manifestation      WMDP / MMLU accuracy for both models on *our* prompt
                        pool, in both prompt styles. If the gap is absent there
                        is no intervention in this distribution to find.
  2  determinism        L4 is strictly upstream of every edited weight, so
                        ||h_base - h_RMU|| must be 0 there. Non-zero = broken
                        pipeline, and every later number is void.
  3  magnitude/outliers per-layer norm and outlier-dim statistics for both
                        models, split forget/retain.
  4  delta vs theta     THE stop condition. The base->RMU displacement measured
                        in EP's own unit — centred cosine distance under the
                        shared calibration — against theta. If the effect is
                        sub-theta, EP's quantum is coarser than a known
                        intervention and the grid is not worth buying.
  5  timing             one real dictionary build: wall-clock, K, saturation.

Everything lands in JSON + CSV under the run dir. Nothing that matters is
printed only to stdout.

    python -m experiments.rmu_diff.gate1a --out artifacts/runs/rmu_diff/gate1a
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

BASE_ID = "HuggingFaceH4/zephyr-7b-beta"
RMU_ID = "cais/Zephyr_RMU"
# Pinned 2026-07-31. Re-asserted at load time; a mismatch is a hard failure.
BASE_REV = "892b3d7a7b1cf10c7a701c60881cd93df615734c"
RMU_REV = "70c55b3bf3141a8c24292dec0262b8aea03a0d4a"

# RMU ground truth, from run_rmu_zephyr.ipynb + rmu/unlearn.py defaults.
RMU_LOSS_LAYER = 7            # module_str "{model}.model.layers[7]"
RMU_EDITED_LAYERS = (5, 6, 7)
RMU_STEERING_COEFF = 6.5
RMU_ALPHA = 1200.0

LAYERS = (4, 7, 14, 24)
PERCENTILES = (10.0, 12.0)

log = logging.getLogger("rmu_diff.gate1a")


# ---------------------------------------------------------------- 0. weights

def weight_diff(out: Path) -> dict:
    """Which tensors differ, and by how much. No GPU, no forward pass.

    Streams both checkpoints' safetensors shards and compares key by key in
    fp32. This is the only statement about locality in the whole experiment
    that does not depend on our pipeline being correct.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # framework="pt", not "np": these checkpoints are bf16 and numpy has no
    # bfloat16 dtype, so the numpy reader raises on every tensor.
    def shard_map(model_id: str, revision: str) -> dict[str, Path]:
        root = Path(snapshot_download(model_id, revision=revision,
                                      allow_patterns=["*.safetensors",
                                                      "*.safetensors.index.json"]))
        m: dict[str, Path] = {}
        for f in sorted(root.glob("*.safetensors")):
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    m[k] = f
        return m

    a_map, b_map = shard_map(BASE_ID, BASE_REV), shard_map(RMU_ID, RMU_REV)
    only_a = sorted(set(a_map) - set(b_map))
    only_b = sorted(set(b_map) - set(a_map))

    rows = []
    handles: dict[Path, object] = {}

    def get(path: Path, key: str) -> np.ndarray:
        if path not in handles:
            handles[path] = safe_open(path, framework="pt").__enter__()
        return handles[path].get_tensor(key).to(torch.float32).numpy()

    for key in sorted(set(a_map) & set(b_map)):
        a = get(a_map[key], key)
        b = get(b_map[key], key)
        d = b - a
        fro = float(np.linalg.norm(d))
        rows.append({
            "tensor": key,
            "shape": "x".join(str(s) for s in a.shape),
            "changed": bool(fro > 0.0),
            "fro_delta": fro,
            "fro_base": float(np.linalg.norm(a)),
            "rel_fro": float(fro / (np.linalg.norm(a) + 1e-12)),
            "max_abs_delta": float(np.abs(d).max()),
        })
        del a, b, d

    changed = [r for r in rows if r["changed"]]
    layers_touched = sorted({
        int(r["tensor"].split(".")[2]) for r in changed
        if r["tensor"].startswith("model.layers.")
    })
    summary = {
        "n_tensors": len(rows),
        "n_changed": len(changed),
        "tensors_only_in_base": only_a,
        "tensors_only_in_rmu": only_b,
        "changed_tensors": [r["tensor"] for r in changed],
        "layers_touched": layers_touched,
        "matches_documented_layers": layers_touched == list(RMU_EDITED_LAYERS),
        "max_rel_fro": max((r["rel_fro"] for r in changed), default=0.0),
    }
    _write_csv(out / "weight_diff.csv", rows)
    log.info("weight diff: %d/%d tensors changed, layers touched=%s",
             len(changed), len(rows), layers_touched)
    return summary


# ------------------------------------------------------- 1. manifestation

def answer_token_ids(tokenizer, prompt: str, style: str) -> list[int]:
    """Token id of the first answer token for each letter, resolved in context.

    Never assume " A" vs "A" tokenizes the way the style suggests: encode
    prompt+letter and read the first token past the prompt.
    """
    from .data import LETTERS

    base = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = []
    for L in LETTERS:
        cont = " " + L if style == "plain" else L
        full = tokenizer(prompt + cont, add_special_tokens=False)["input_ids"]
        if len(full) <= len(base) or full[:len(base)] != base:
            raise RuntimeError(
                f"answer continuation retokenizes the prompt for letter {L!r}"
            )
        ids.append(int(full[len(base)]))
    return ids


def mcq_accuracy(qm, prompts, style: str, batch_size: int = 8) -> dict:
    """Zero-shot 4-way MCQ accuracy by comparing answer-token logits.

    Scores only the four candidate tokens at the final position, which is the
    lm-eval-harness convention for WMDP and is what the RMU paper reports.
    """
    import torch

    from .data import LETTERS

    tok = qm.tokenizer
    ids = answer_token_ids(tok, prompts[0].text, style)
    for p in prompts[1:20]:
        if answer_token_ids(tok, p.text, style) != ids:
            raise RuntimeError("answer token ids are not stable across prompts")
    log.info("answer token ids (%s): %s -> %s", style, list(LETTERS),
             [tok.decode([i]) for i in ids])

    cand = torch.tensor(ids, device=qm.device)
    correct, per_source, n_source = 0, {}, {}
    preds = []
    prev_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for s in range(0, len(prompts), batch_size):
            sub = prompts[s:s + batch_size]
            enc = tok([p.text for p in sub], return_tensors="pt", padding=True,
                      add_special_tokens=True)
            input_ids = enc["input_ids"].to(qm.device)
            attn = enc["attention_mask"].to(qm.device)
            with torch.no_grad():
                logits = qm.model(input_ids=input_ids, attention_mask=attn,
                                  use_cache=False).logits
            last = attn.sum(dim=1) - 1
            row = torch.arange(logits.shape[0], device=logits.device)
            scores = logits[row, last][:, cand]          # (B, 4)
            choice = scores.argmax(dim=1).cpu().numpy()
            for p, c in zip(sub, choice):
                ok = int(c) == p.answer
                correct += ok
                per_source[p.source] = per_source.get(p.source, 0) + ok
                n_source[p.source] = n_source.get(p.source, 0) + 1
                preds.append({"index": p.index, "source": p.source,
                              "label": p.label, "answer": p.answer,
                              "pred": int(c), "correct": int(ok)})
    finally:
        tok.padding_side = prev_side

    return {
        "style": style,
        "n": len(prompts),
        "accuracy": correct / max(len(prompts), 1),
        "by_source": {k: per_source[k] / n_source[k] for k in n_source},
        "n_by_source": n_source,
        "_preds": preds,
    }


# ------------------------------------------------------------- extraction

def extract_layers(qm, prompts, layers, *, batch_size: int,
                   max_positions: int) -> dict[int, dict]:
    """Per-position activations at each layer for one model, fp16 on host."""
    outs = {}
    texts = [p.text for p in prompts]
    for L in layers:
        t0 = time.time()
        r = qm.extract_per_position(texts, layer=L, batch_size=batch_size,
                                    max_positions_per_prompt=max_positions,
                                    skip_first=True)
        outs[L] = {"x": r.x.astype(np.float16),
                   "prompt_ids": r.prompt_ids, "position_ids": r.position_ids,
                   "n_tokens": r.n_tokens, "elapsed_s": time.time() - t0}
        log.info("  L%-2d %d acts from %d prompts in %.1fs (%.0f acts/s)",
                 L, len(r.x), len(prompts), outs[L]["elapsed_s"],
                 len(r.x) / max(outs[L]["elapsed_s"], 1e-9))
    return outs


# --------------------------------------------- 3/4. magnitude and delta

def _norm_stats(x: np.ndarray) -> dict:
    n = np.linalg.norm(x.astype(np.float32), axis=1)
    q = np.percentile(n, [1, 50, 99])
    return {"mean": float(n.mean()), "p1": float(q[0]), "median": float(q[1]),
            "p99": float(q[2]), "max": float(n.max())}


def _outlier_dims(x: np.ndarray, k: int = 5) -> dict:
    """Massive-activation dims: mean |h_d| per dimension, top-k vs median dim."""
    m = np.abs(x.astype(np.float32)).mean(axis=0)
    order = np.argsort(m)[::-1]
    med = float(np.median(m))
    return {"top_dims": [int(i) for i in order[:k]],
            "top_dim_mean_abs": [float(m[i]) for i in order[:k]],
            "median_dim_mean_abs": med,
            "outlier_ratio": float(m[order[0]] / (med + 1e-12))}


def _centred_unit(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    c = x.astype(np.float32) - mu
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)


def delta_vs_theta(xb: np.ndarray, xr: np.ndarray, mu: np.ndarray,
                   thetas: dict[float, float], is_forget: np.ndarray) -> dict:
    """The stop-condition measurement.

    EP's quantum is theta, a cosine distance in *centred* space. Comparing
    ||delta_h|| to theta directly would be a type error, so the displacement is
    measured in the same unit: the cosine distance between the two models'
    centred unit directions for the same token, under the shared (base)
    calibration mean. `frac_above_theta` is the fraction of tokens the
    intervention moves far enough for EP to be able to resolve at all.
    """
    ub, ur = _centred_unit(xb, mu), _centred_unit(xr, mu)
    d = np.clip(1.0 - (ub * ur).sum(axis=1), 0.0, 2.0)
    nb = np.linalg.norm(xb.astype(np.float32), axis=1)
    dl = np.linalg.norm(xr.astype(np.float32) - xb.astype(np.float32), axis=1)

    def block(mask, tag):
        if mask.sum() == 0:
            return {}
        dd, rr = d[mask], dl[mask] / (nb[mask] + 1e-12)
        o = {f"{tag}_n": int(mask.sum()),
             f"{tag}_cosdist_mean": float(dd.mean()),
             f"{tag}_cosdist_median": float(np.median(dd)),
             f"{tag}_cosdist_p99": float(np.percentile(dd, 99)),
             f"{tag}_rel_delta_norm_median": float(np.median(rr))}
        for p, th in thetas.items():
            # The stop condition: a token the intervention moves less than theta
            # is a token EP cannot resolve, however large ||delta_h|| is.
            o[f"{tag}_frac_above_theta_p{p:g}"] = float((dd > th).mean())
            o[f"{tag}_cosdist_over_theta_p{p:g}"] = float(np.median(dd) / th)
        return o

    out = {"exact_zero_frac": float((dl == 0).mean()),
           "max_abs_delta": float(np.abs(xr.astype(np.float32)
                                         - xb.astype(np.float32)).max())}
    out.update(block(np.ones(len(d), bool), "all"))
    out.update(block(is_forget, "forget"))
    out.update(block(~is_forget, "retain"))
    return out


def junk_direction(xb: np.ndarray, xr: np.ndarray, mu: np.ndarray,
                   is_forget: np.ndarray) -> dict:
    """The empirical RMU direction, and the H1' centring-artifact discriminator.

    u_emp = normalize(mean over forget tokens of h_RMU - h_base). RMU draws its
    control vector with `torch.rand` (positive orthant), so a genuine recovery
    of u should sit close to the all-ones direction; a centring artifact should
    instead sit close to -mu_hat.
    """
    if is_forget.sum() == 0:
        return {}
    delta = (xr.astype(np.float32) - xb.astype(np.float32))[is_forget]
    u = delta.mean(axis=0)
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        return {"u_norm": nu, "degenerate": True}
    u = u / nu
    d = xb.shape[1]
    ones = np.ones(d, dtype=np.float32) / np.sqrt(d)
    mu_hat = mu / (np.linalg.norm(mu) + 1e-12)
    proj = delta @ u
    frac_var = float((proj ** 2).sum() / ((delta ** 2).sum() + 1e-12))

    # The decisive estimator of the control vector, and it does not go through
    # the delta at all: RMU's loss drives h_forget -> c*u, so the mean of the
    # RMU activations themselves estimates c*u directly. If the collapse is
    # real, ||mean|| ~ 6.5 and the direction sits near the all-ones direction,
    # because `torch.rand` draws u from the positive orthant (E[cos] ~ 0.866).
    mean_rmu = xr[is_forget].astype(np.float32).mean(axis=0)
    c_hat = float(np.linalg.norm(mean_rmu))
    u_hat = mean_rmu / (c_hat + 1e-12)

    # Where the RMU forget activations actually land, in centred-unit space.
    ur = _centred_unit(xr[is_forget], mu)
    return {
        "u_norm": nu,
        "u_cos_ones": float(u @ ones),
        "u_cos_mu_hat": float(u @ mu_hat),
        "delta_var_explained_by_u": frac_var,
        "mu_norm": float(np.linalg.norm(mu)),
        "steering_coeff": RMU_STEERING_COEFF,
        # Direct mechanism test: c_hat should be ~6.5 and u_hat ~all-ones.
        "c_hat_from_rmu_mean": c_hat,
        "c_hat_over_steering_coeff": c_hat / RMU_STEERING_COEFF,
        "u_hat_cos_ones": float(u_hat @ ones),
        "u_hat_cos_u_emp": float(u_hat @ u),
        "rmu_forget_norm_cv": float(
            np.std(np.linalg.norm(xr[is_forget].astype(np.float32), axis=1))
            / (np.mean(np.linalg.norm(xr[is_forget].astype(np.float32), axis=1)) + 1e-12)),
        # H1' preview: is the collapsed centred direction u-like, or just -mu_hat?
        "rmu_forget_dir_cos_minus_mu_hat": float(np.median(ur @ (-mu_hat))),
        "rmu_forget_dir_cos_u": float(np.median(ur @ u)),
        "rmu_forget_dir_cos_u_hat_centred": float(np.median(
            ur @ ((c_hat * u_hat - mu) / (np.linalg.norm(c_hat * u_hat - mu) + 1e-12)))),
        "rmu_forget_pairwise_cos_median": _pairwise_cos_median(ur),
        "base_forget_pairwise_cos_median": _pairwise_cos_median(
            _centred_unit(xb[is_forget], mu)),
    }


def region_formation_probe(xb: np.ndarray, xr: np.ndarray, prompt_ids: np.ndarray,
                           pool, is_forget: np.ndarray, *, percentile: float,
                           seed: int = 0, chunk: int = 8192) -> dict:
    """Would EP actually carve a new region here? Measured, not inferred.

    ADDED 2026-07-31 after the smoke run, and the reason is stated in the Gate
    1A report: the pre-registered stop condition compares the base->RMU
    *displacement* to theta, but that is not the quantity that decides whether a
    region is introduced. A new cell opens when an activation is further than
    theta from **every existing exemplar** — not when it has moved further than
    theta from where it used to be. A token can move 0.5*theta and still cross a
    boundary; a token can move 1.5*theta and land inside a neighbouring cell
    that already exists.

    So: build the base dictionary, then ask of the RMU activations
      (a) what fraction fall outside every base exemplar's radius (= would spawn
          a new region), and
      (b) what fraction change cell identity at all,
    with the retain arm as the within-experiment control. This is the direct
    precursor of the introduced set that Gate 1B computes.
    """
    from ep.discovery.dictionary import Dictionary
    from qwen_ep.sweep_p import member_reservoir_cap

    from .data import stream_order

    cal = calibrate_from(xb, percentile)
    order = stream_order(pool, seed)
    rank = np.empty(len(pool), dtype=np.int64)
    rank[np.asarray(order)] = np.arange(len(pool))
    perm = np.argsort(rank[prompt_ids], kind="stable")

    d = Dictionary(center=cal.center, threshold=cal.threshold)
    with member_reservoir_cap(d, 0):
        for i, s in enumerate(range(0, len(perm), chunk)):
            d.add_batch(xb[perm[s:s + chunk]].astype(np.float32), iteration=i,
                        global_index_start=s)
    d.finalize()

    pid_b, dist_b = d.assign(xb.astype(np.float32))
    pid_r, dist_r = d.assign(xr.astype(np.float32))
    th = cal.threshold

    def arm(mask, tag):
        if mask.sum() == 0:
            return {}
        changed = (pid_b[mask] != pid_r[mask])
        counts = np.bincount(pid_r[mask], minlength=len(d))
        return {
            f"{tag}_n": int(mask.sum()),
            f"{tag}_base_frac_outside_theta": float((dist_b[mask] > th).mean()),
            f"{tag}_rmu_frac_outside_theta": float((dist_r[mask] > th).mean()),
            f"{tag}_rmu_nn_dist_median": float(np.median(dist_r[mask])),
            f"{tag}_base_nn_dist_median": float(np.median(dist_b[mask])),
            f"{tag}_cell_change_rate": float(changed.mean()),
            # If RMU collapses the forget arm onto one direction, one existing
            # cell should swallow a disproportionate share of it.
            f"{tag}_rmu_top_cell_share": float(counts.max() / mask.sum()),
            f"{tag}_base_top_cell_share": float(
                np.bincount(pid_b[mask], minlength=len(d)).max() / mask.sum()),
        }

    out = {"percentile": percentile, "theta": th, "K_base": len(d),
           "n_activations": int(len(xb))}
    out.update(arm(is_forget, "forget"))
    out.update(arm(~is_forget, "retain"))
    log.info("  region-formation p%g: K=%d theta=%.4f | forget outside-theta "
             "%.3f cell-change %.3f top-cell %.3f | retain outside-theta %.3f "
             "cell-change %.3f", percentile, len(d), th,
             out["forget_rmu_frac_outside_theta"], out["forget_cell_change_rate"],
             out["forget_rmu_top_cell_share"], out["retain_rmu_frac_outside_theta"],
             out["retain_cell_change_rate"])
    return out


def _pairwise_cos_median(u: np.ndarray, n: int = 2000, seed: int = 0) -> float:
    """Median pairwise cosine among a subsample — 1.0 means point collapse."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(u), size=min(n, len(u)), replace=False)
    s = u[idx] @ u[idx].T
    iu = np.triu_indices(len(idx), k=1)
    return float(np.median(s[iu]))


# ------------------------------------------------------------ calibration

def calibrate_from(x: np.ndarray, percentile: float, chunk: int = 4000,
                   n_tokens: int = 200_000):
    from ep.discovery.calibration import calibrate

    def batches():
        for s in range(0, min(len(x), n_tokens), chunk):
            yield x[s:s + chunk].astype(np.float32)

    return calibrate(batches(), n_tokens=n_tokens, percentile=percentile)


# ------------------------------------------------------------------- io

def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


# ------------------------------------------------------------------ main

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/runs/rmu_diff/gate1a")
    ap.add_argument("--n-probe", type=int, default=1024,
                    help="prompts used for the activation probe + calibration")
    ap.add_argument("--n-acc", type=int, default=600,
                    help="prompts per style for the manifestation check")
    ap.add_argument("--n-timing", type=int, default=0,
                    help="prompts for the timing build (0 = use the probe set)")
    ap.add_argument("--layers", default=",".join(str(L) for L in LAYERS))
    ap.add_argument("--percentiles", default="10,12")
    ap.add_argument("--style", default="chat", choices=["chat", "plain"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-positions", type=int, default=256)
    ap.add_argument("--min-tokens", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="prompt-length band. WMDP-cyber runs to 2503 tokens "
                         "against MMLU's median 112, so an unbanded pool would "
                         "confound the forget label with prompt length.")
    ap.add_argument("--no-length-match", dest="length_match",
                    action="store_false", default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-weight-diff", action="store_true")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(s) for s in args.layers.split(",")]
    percentiles = [float(s) for s in args.percentiles.split(",")]
    report: dict = {"models": {"base": {"id": BASE_ID, "revision": BASE_REV},
                              "rmu": {"id": RMU_ID, "revision": RMU_REV}},
                    "config": vars(args) | {"layers": layers,
                                            "percentiles": percentiles}}
    t_start = time.time()

    # -- 0. weight diff -----------------------------------------------------
    if not args.skip_weight_diff:
        log.info("== 0. weight diff ==")
        report["weight_diff"] = weight_diff(out)
        _write_json(out / "gate1a.json", report)

    from qwen_ep.adapter import QwenModel

    from .data import build_pool

    # -- prompt pool --------------------------------------------------------
    log.info("== prompt pool ==")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    n_forget = args.n_probe // 2
    band = {"min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
            "length_match": args.length_match}
    pool = build_pool(n_bio=n_forget // 2, n_cyber=n_forget - n_forget // 2,
                      n_mmlu=args.n_probe - n_forget, style=args.style,
                      tokenizer=tok, **band)
    lens = np.array([p.n_tokens for p in pool])
    src = np.array([p.source for p in pool])
    report["pool"] = {
        "n": len(pool), "style": args.style, "band": band,
        "n_forget": int(sum(p.label == "forget" for p in pool)),
        "n_retain": int(sum(p.label == "retain" for p in pool)),
        "token_len": {"mean": float(lens.mean()), "p50": float(np.percentile(lens, 50)),
                      "p99": float(np.percentile(lens, 99)), "max": int(lens.max())},
        # The length control: forget and retain medians must be close, or any
        # region-level forget/retain split is a length artifact.
        "token_len_by_source": {
            s: {"n": int((src == s).sum()),
                "p50": float(np.percentile(lens[src == s], 50)),
                "mean": float(lens[src == s].mean())}
            for s in sorted(set(src.tolist()))},
        "truncated_at_max_positions": int((lens > args.max_positions).sum()),
    }
    _write_csv(out / "pool.csv", [p.as_row() for p in pool])
    log.info("pool: %d prompts (%d forget / %d retain), median %d tokens, "
             "%d over max_positions=%d", len(pool), report["pool"]["n_forget"],
             report["pool"]["n_retain"], int(np.percentile(lens, 50)),
             report["pool"]["truncated_at_max_positions"], args.max_positions)
    for s, st in report["pool"]["token_len_by_source"].items():
        log.info("  %-11s n=%4d median tokens %d", s, st["n"], int(st["p50"]))

    # Manifestation is measured in BOTH styles on the SAME questions: which
    # style carries the behavioural gap is an empirical question, and picking
    # the wrong one silently kills the run. Same `seed` -> same question set,
    # so the two styles differ only in their wrapper.
    pools_by_style = {
        s: build_pool(n_bio=args.n_acc // 4, n_cyber=args.n_acc // 4,
                      n_mmlu=args.n_acc - 2 * (args.n_acc // 4),
                      style=s, tokenizer=tok, **band)
        for s in ("plain", "chat")}

    report["tokenizer_audit"] = tokenizer_audit(pool)
    _write_json(out / "gate1a.json", report)

    # -- per-model pass -----------------------------------------------------
    acts: dict[str, dict] = {}
    report["accuracy"] = {}
    report["extraction"] = {}
    for tag, model_id, rev in (("base", BASE_ID, BASE_REV), ("rmu", RMU_ID, RMU_REV)):
        log.info("== loading %s (%s @ %s) ==", tag, model_id, rev)
        # Both models get the BASE tokenizer. The RMU repo ships no
        # tokenizer.json, so its reconstructed tokenizer sets
        # add_prefix_space=True and produces different ids for identical text
        # over an identical 32000-token vocabulary. RMU's own unlearn.py loads
        # the tokenizer from the base checkpoint, so this is also what the
        # intervention was trained under.
        qm = QwenModel(model_id, device=args.device, prepend_bos=True,
                       revision=rev, tokenizer_id=BASE_ID,
                       tokenizer_revision=BASE_REV)
        loaded = getattr(qm.model.config, "_name_or_path", model_id)
        report["models"][tag]["n_layers"] = qm.n_layers
        report["models"][tag]["d_model"] = qm.d_model
        report["models"][tag]["config_name"] = loaded
        # Tokenizer identity is load-bearing: the whole diff assumes both models
        # see identical token sequences. A silent tokenizer-conversion fallback
        # would break that without changing any shape.
        report["models"][tag]["tokenizer_fingerprint"] = _tok_fingerprint(
            qm.tokenizer, pool)

        log.info("-- 1. manifestation (%s) --", tag)
        report["accuracy"][tag] = {}
        for style, ps in pools_by_style.items():
            r = mcq_accuracy(qm, ps, style, batch_size=args.batch_size)
            preds = r.pop("_preds")
            _write_csv(out / f"acc_{tag}_{style}.csv", preds)
            report["accuracy"][tag][style] = r
            log.info("  %s/%s acc=%.3f by_source=%s", tag, style,
                     r["accuracy"], {k: round(v, 3) for k, v in r["by_source"].items()})

        log.info("-- extraction (%s) --", tag)
        acts[tag] = extract_layers(qm, pool, layers, batch_size=args.batch_size,
                                   max_positions=args.max_positions)
        report["extraction"][tag] = {
            str(L): {"n_acts": int(len(acts[tag][L]["x"])),
                     "n_tokens": int(acts[tag][L]["n_tokens"]),
                     "elapsed_s": round(acts[tag][L]["elapsed_s"], 1)}
            for L in layers}

        del qm
        _free()
        _write_json(out / "gate1a.json", report)

    # -- alignment assertions ----------------------------------------------
    fp_base = report["models"]["base"]["tokenizer_fingerprint"]
    fp_rmu = report["models"]["rmu"]["tokenizer_fingerprint"]
    if fp_base != fp_rmu:
        raise SystemExit(f"tokenizer fingerprints differ ({fp_base} vs {fp_rmu}) "
                         "— the two models did not see identical token streams")
    log.info("tokenizer fingerprint matches on both checkpoints: %s", fp_base)

    for L in layers:
        a, b = acts["base"][L], acts["rmu"][L]
        if a["x"].shape != b["x"].shape:
            raise SystemExit(f"L{L}: shape mismatch {a['x'].shape} vs {b['x'].shape} "
                             "— the two models did not see identical tokenisations")
        if not (np.array_equal(a["prompt_ids"], b["prompt_ids"])
                and np.array_equal(a["position_ids"], b["position_ids"])):
            raise SystemExit(f"L{L}: prompt/position id mismatch — activations "
                             "are not paired and every delta below is void")
    log.info("paired-extraction assertion passed for layers %s", layers)

    pid = acts["base"][layers[0]]["prompt_ids"]
    is_forget = np.array([pool[int(i)].label == "forget" for i in pid])

    # -- 2/3/4. calibration, norms, delta ----------------------------------
    log.info("== 2/3/4. calibration, norms, delta ==")
    cal_rows, stat_rows = [], []
    report["layers"] = {}
    for L in layers:
        xb, xr = acts["base"][L]["x"], acts["rmu"][L]["x"]
        entry: dict = {}

        cal = {}
        for tag, x in (("base", xb), ("rmu", xr)):
            for p in percentiles:
                c = calibrate_from(x, p)
                cal[(tag, p)] = c
                cal_rows.append({"layer": L, "model": tag, "percentile": p,
                                 "theta": c.threshold,
                                 "center_norm": float(np.linalg.norm(c.center)),
                                 "n_activations": c.n_activations})
        mu = cal[("base", percentiles[0])].center      # shared calibration = base
        thetas = {p: cal[("base", p)].threshold for p in percentiles}
        entry["theta"] = {f"p{p:g}": {"base": cal[("base", p)].threshold,
                                      "rmu": cal[("rmu", p)].threshold,
                                      "delta": cal[("rmu", p)].threshold
                                      - cal[("base", p)].threshold}
                          for p in percentiles}
        entry["center_norm"] = {
            "base": float(np.linalg.norm(cal[("base", percentiles[0])].center)),
            "rmu": float(np.linalg.norm(cal[("rmu", percentiles[0])].center))}

        for tag, x in (("base", xb), ("rmu", xr)):
            entry[f"norms_{tag}"] = {
                "all": _norm_stats(x),
                "forget": _norm_stats(x[is_forget]),
                "retain": _norm_stats(x[~is_forget])}
            entry[f"outliers_{tag}"] = _outlier_dims(x)

        entry["delta"] = delta_vs_theta(xb, xr, mu, thetas, is_forget)
        entry["junk"] = junk_direction(xb, xr, mu, is_forget)

        # L4 determinism control: strictly upstream of every edited weight.
        if L < min(RMU_EDITED_LAYERS):
            entry["determinism_control"] = {
                "expected_zero": True,
                "max_abs_delta": entry["delta"]["max_abs_delta"],
                "exact_zero_frac": entry["delta"]["exact_zero_frac"],
                "PASS": bool(entry["delta"]["max_abs_delta"] == 0.0)}
            log.info("L%d determinism control: max|delta|=%.3e PASS=%s", L,
                     entry["delta"]["max_abs_delta"],
                     entry["determinism_control"]["PASS"])

        entry["region_formation"] = {
            f"p{p:g}": region_formation_probe(
                xb, xr, acts["base"][L]["prompt_ids"], pool, is_forget,
                percentile=p, seed=0)
            for p in percentiles}

        report["layers"][str(L)] = entry
        row = {"layer": L,
               "theta_base_p%g" % percentiles[0]: thetas[percentiles[0]],
               "mu_norm_base": entry["center_norm"]["base"],
               "mu_norm_rmu": entry["center_norm"]["rmu"],
               "norm_base_forget": entry["norms_base"]["forget"]["median"],
               "norm_rmu_forget": entry["norms_rmu"]["forget"]["median"],
               "norm_base_retain": entry["norms_base"]["retain"]["median"],
               "norm_rmu_retain": entry["norms_rmu"]["retain"]["median"]}
        row.update({k: v for k, v in entry["delta"].items()
                    if k.startswith(("forget_", "retain_", "all_", "max_", "exact_"))})
        row.update({f"junk_{k}": v for k, v in entry["junk"].items()})
        for p in percentiles:
            row.update({f"rf_p{p:g}_{k}": v
                        for k, v in entry["region_formation"][f"p{p:g}"].items()})
        stat_rows.append(row)
        log.info("L%-2d theta(p%g)=%.4f  |mu|=%.2f  ||h||fgt %.1f->%.1f  "
                 "cosdist(fgt) med=%.4f  frac>theta=%.3f",
                 L, percentiles[0], thetas[percentiles[0]],
                 entry["center_norm"]["base"],
                 entry["norms_base"]["forget"]["median"],
                 entry["norms_rmu"]["forget"]["median"],
                 entry["delta"].get("forget_cosdist_median", float("nan")),
                 entry["delta"].get(
                     f"forget_frac_above_theta_p{percentiles[0]:g}", float("nan")))

    _write_csv(out / "calibration.csv", cal_rows)
    _write_csv(out / "layer_stats.csv", stat_rows)
    _write_json(out / "gate1a.json", report)

    # -- 5. timing probe ----------------------------------------------------
    log.info("== 5. timing probe: one build, L%d, p=%g, base ==",
             RMU_LOSS_LAYER, percentiles[-1])
    report["timing"] = timing_probe(
        acts["base"][RMU_LOSS_LAYER]["x"], acts["base"][RMU_LOSS_LAYER]["prompt_ids"],
        pool, percentile=percentiles[-1], seed=0)
    report["total_wall_s"] = round(time.time() - t_start, 1)
    _write_json(out / "gate1a.json", report)
    log.info("== gate 1A done in %.0fs -> %s ==", time.time() - t_start, out)


def timing_probe(x: np.ndarray, prompt_ids: np.ndarray, pool, *,
                 percentile: float, seed: int, chunk: int = 8192) -> dict:
    """One real dictionary build: wall-clock, K, and whether saturation fired.

    Activations are replayed in the prompt order EP's own seeded shuffle would
    produce, so the timing is the timing of the real build — the forward pass is
    simply already paid for.
    """
    from ep.discovery.dictionary import Dictionary
    from qwen_ep.sweep_p import member_reservoir_cap

    from .data import stream_order

    cal = calibrate_from(x, percentile)
    order = stream_order(pool, seed)
    rank = np.empty(len(pool), dtype=np.int64)
    rank[np.asarray(order)] = np.arange(len(pool))
    perm = np.argsort(rank[prompt_ids], kind="stable")

    d = Dictionary(center=cal.center, threshold=cal.threshold)
    t0 = time.time()
    k_trace, last_new = [], 0
    # The member reservoir is 30 * d_model * 4 = 492 KB per region at d=4096 and
    # nothing here reads it; a timing probe should not risk an OOM to store it.
    with member_reservoir_cap(d, 0):
        for i, s in enumerate(range(0, len(perm), chunk)):
            before = len(d)
            d.add_batch(x[perm[s:s + chunk]].astype(np.float32), iteration=i,
                        global_index_start=s)
            if len(d) > before:
                last_new = i
            k_trace.append([int(s + chunk), len(d)])
    saturated = (len(k_trace) - 1 - last_new) >= 1
    elapsed = time.time() - t0
    members = sorted((p.member_count for p in d.partitions), reverse=True)
    out = {"layer": RMU_LOSS_LAYER, "percentile": percentile, "seed": seed,
           "theta": cal.threshold, "n_activations": int(len(x)),
           "K": len(d), "cluster_s": round(elapsed, 1),
           "acts_per_s": round(len(x) / max(elapsed, 1e-9)),
           "saturated": bool(saturated),
           "batches_since_last_new": len(k_trace) - 1 - last_new,
           "largest_partition": members[0] if members else 0,
           "singletons": sum(1 for m in members if m == 1),
           "K_trace": k_trace}
    log.info("timing: K=%d in %.1fs (%.0f acts/s), saturated=%s, largest=%d",
             out["K"], elapsed, out["acts_per_s"], out["saturated"],
             out["largest_partition"])
    return out


def tokenizer_audit(pool) -> dict:
    """Do the two repos' *native* tokenizers agree? (They do not.)

    Recorded as a finding rather than silently worked around: the RMU repo ships
    no tokenizer.json, so transformers rebuilds one from tokenizer.model with
    add_prefix_space=True while the base repo's pre-converted tokenizer uses
    False. The vocabulary is identical, the token ids are not.
    """
    from transformers import AutoTokenizer

    tb = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    tr = AutoTokenizer.from_pretrained(RMU_ID, revision=RMU_REV)
    ids_b = [tb(p.text, add_special_tokens=True)["input_ids"] for p in pool[:64]]
    ids_r = [tr(p.text, add_special_tokens=True)["input_ids"] for p in pool[:64]]
    agree = sum(a == b for a, b in zip(ids_b, ids_r))
    out = {
        "vocab_identical": tb.get_vocab() == tr.get_vocab(),
        "vocab_size": {"base": len(tb), "rmu": len(tr)},
        "add_prefix_space": {"base": getattr(tb, "add_prefix_space", None),
                             "rmu": getattr(tr, "add_prefix_space", None)},
        "prompts_tokenised_identically": f"{agree}/{len(ids_b)}",
        "resolution": "base tokenizer forced on both models",
    }
    log.info("tokenizer audit: vocab identical=%s, native tokenisations agree "
             "%s, add_prefix_space base=%s rmu=%s -> forcing base tokenizer",
             out["vocab_identical"], out["prompts_tokenised_identically"],
             out["add_prefix_space"]["base"], out["add_prefix_space"]["rmu"])
    return out


def _tok_fingerprint(tokenizer, pool, n: int = 64) -> str:
    """Hash of the token ids this tokenizer assigns to the first `n` prompts."""
    import hashlib

    h = hashlib.sha256()
    h.update(str(len(tokenizer)).encode())
    for p in pool[:n]:
        ids = tokenizer(p.text, add_special_tokens=True)["input_ids"]
        h.update(np.asarray(ids, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _free() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
