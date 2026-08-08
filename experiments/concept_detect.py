"""Concept detection AUROC on pre-built L27 EP dictionaries (Panel A).

Reproduces the paper's label-free concept-detection protocol (AxBench-style)
on Qwen3.5-4B against dictionaries already built on the Pile:

  1. For each AxBench concept, gather positive example texts + a contrastive
     negative pool (shared negatives + hard negatives sampled from other
     concepts' positives).
  2. Extract per-position residual activations at the dictionary's layer;
     center + unit-normalise into the dictionary's space (cosine readout).
  3. SELECT the region maximising  mean_cos(positives) - mean_cos(negatives)
     over the TRAIN half.
  4. SCORE held-out examples: max-over-position cosine onto the selected
     region's basis direction -> AUROC. Report mean over concepts, per
     (percentile x basis).

Because AxBench's concepts were mined from Gemma SAE labels, the numbers here
are a re-derivation on Qwen, NOT leaderboard-comparable — the dashboard says so.

Run:
    python -m experiments.concept_detect \
        --dicts artifacts/runs/qwen3_5-4b_L27_p2p0_ctx128_cache_pile,... \
        --n-concepts 48 --out artifacts/runs/concept_detect/qwen3_5-4b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from qwen_ep.adapter import QwenModel, model_tag

logger = logging.getLogger("experiments.concept_detect")

DATASET = "pyvene/axbench-concept500"
SUBSET = "2b/l20"  # text is model-agnostic; subdir only picks which parquet


def _dict_layer(dict_dir: Path) -> tuple[int, float]:
    meta = json.loads((dict_dir / "metadata.json").read_text())
    return int(meta["layer"]), float(meta["percentile"])


def load_concepts(n_concepts: int, per_concept: int, seed: int
                  ) -> tuple[list[dict], list[str]]:
    """Return (concepts, all_texts). Each concept: {id,label,pos,neg} where
    pos/neg are lists of indices into all_texts (deduped)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(DATASET, f"{SUBSET}/train/data.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(f, columns=["output", "output_concept",
                                     "category", "concept_id"])
    pos_df = df[df.category == "positive"]
    neg_shared = df[df.category == "negative"]["output"].tolist()

    rng = np.random.default_rng(seed)
    concept_ids = sorted(pos_df.concept_id.unique())
    rng.shuffle(concept_ids)
    concept_ids = concept_ids[:n_concepts]

    # dedup text -> index
    texts: list[str] = []
    index: dict[str, int] = {}

    def add(t: str) -> int:
        t = t.strip()
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]

    # shared-negative indices (common to all concepts)
    shared_idx = [add(t) for t in neg_shared]

    by_concept = {cid: g["output"].tolist()
                  for cid, g in pos_df.groupby("concept_id")}
    concepts = []
    for cid in concept_ids:
        pos_texts = by_concept[cid][:per_concept]
        label = pos_df[pos_df.concept_id == cid]["output_concept"].iloc[0]
        # hard negatives: sample from other concepts' positives
        others = [c for c in concept_ids if c != cid]
        hard = []
        for oc in rng.choice(others, size=min(len(others), per_concept),
                             replace=False):
            hard.append(by_concept[oc][int(rng.integers(len(by_concept[oc])))])
        pos_i = [add(t) for t in pos_texts]
        neg_i = shared_idx + [add(t) for t in hard]
        concepts.append({"id": int(cid), "label": label,
                         "pos": pos_i, "neg": neg_i})
    logger.info("concepts: %d, unique texts: %d", len(concepts), len(texts))
    return concepts, texts


def extract_all(qwen: QwenModel, texts: list[str], layer: int,
                cache_path: Path, max_tokens_per_text: int,
                batch_size: int) -> dict:
    """Per-position activations for every text, cached to npz by (layer,
    text-set hash). Returns {text_idx: (P, D) float32}. Stored flat with an
    offsets array."""
    key = hashlib.sha1(
        ("\x00".join(texts)).encode("utf-8")).hexdigest()[:16]
    npz = cache_path / f"acts_L{layer}_{key}.npz"
    if npz.exists():
        logger.info("extract: cache hit %s", npz)
        z = np.load(npz)
        offs = z["offsets"]
        flat = z["flat"]
        return {i: flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)}

    logger.info("extract: %d texts at L%d (truncate %d tok)", len(texts),
                layer, max_tokens_per_text)
    per: list[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        sub = texts[start:start + batch_size]
        res = qwen.extract_per_position(
            sub, layer=layer, batch_size=batch_size,
            max_positions_per_prompt=max_tokens_per_text)
        # split back per text using prompt_ids
        pid = res.prompt_ids
        for local in range(len(sub)):
            per.append(res.x[pid == local])
        if (start // batch_size) % 10 == 0:
            logger.info("  %d/%d (%.0fs)", start + len(sub), len(texts),
                        time.time() - t0)
    offsets = np.zeros(len(per) + 1, dtype=np.int64)
    for i, a in enumerate(per):
        offsets[i + 1] = offsets[i] + len(a)
    flat = np.concatenate(per) if per else np.zeros((0, qwen.d_model), np.float32)
    cache_path.mkdir(parents=True, exist_ok=True)
    np.savez(npz, flat=flat.astype(np.float16), offsets=offsets)
    logger.info("extract: %d activations cached -> %s (%.0fs)",
                len(flat), npz, time.time() - t0)
    return {i: flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)}


def _cosine_matrix(acts: np.ndarray, center: np.ndarray,
                   dirs: np.ndarray) -> np.ndarray:
    """(P, D) activations -> (P, K) cosine onto unit basis dirs, in centered
    space with magnitude removed."""
    xc = acts.astype(np.float32) - center
    xc /= (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-8)
    return xc @ dirs.T


def _snip(t: str, n: int = 130) -> str:
    t = " ".join(t.strip().split())
    return t[:n] + ("…" if len(t) > n else "")


def evaluate_dict(dict_dir: Path, qwen: QwenModel, concepts: list[dict],
                  acts: dict, bases: list[str], train_frac: float,
                  seed: int, texts: list[str] | None = None,
                  export_examples: int = 0, export_max_concepts: int = 16,
                  export_basis: str = "mean") -> dict:
    from sklearn.metrics import roc_auc_score

    with (dict_dir / "dictionary.pkl").open("rb") as f:
        dictionary = pickle.load(f)
    center = dictionary.center.astype(np.float32)
    layer, percentile = _dict_layer(dict_dir)
    parts = dictionary.partitions
    dir_mean = np.stack([p.mean_member_direction for p in parts]).astype(np.float32)
    dir_exem = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
    basis_dirs = {"mean": dir_mean, "exemplar": dir_exem}

    rng = np.random.default_rng(seed)
    # precompute per-text max/mean cosine onto ALL regions for both bases
    # (max-over-position pooling is the per-example detector score).
    def pooled(basis: str) -> dict[int, np.ndarray]:
        dirs = basis_dirs[basis]
        out = {}
        for i, a in acts.items():
            if len(a) == 0:
                out[i] = np.full(dirs.shape[0], -1.0, np.float32)
            else:
                out[i] = _cosine_matrix(a, center, dirs).max(axis=0)
        return out

    results = {}
    for basis in bases:
        pool = pooled(basis)
        aurocs = []
        per_concept = []
        for c in concepts:
            pos, neg = c["pos"], c["neg"]
            rng.shuffle(pos)
            rng.shuffle(neg)
            ntr_p = max(1, int(len(pos) * train_frac))
            ntr_n = max(1, int(len(neg) * train_frac))
            tr_pos, te_pos = pos[:ntr_p], pos[ntr_p:]
            tr_neg, te_neg = neg[:ntr_n], neg[ntr_n:]
            if not te_pos or not te_neg:
                continue
            # SELECT region by train contrast
            tp = np.stack([pool[i] for i in tr_pos])  # (n, K)
            tn = np.stack([pool[i] for i in tr_neg])
            contrast = tp.mean(axis=0) - tn.mean(axis=0)
            best = int(contrast.argmax())
            # SCORE held-out on the selected region
            y = np.array([1] * len(te_pos) + [0] * len(te_neg))
            s = np.array([pool[i][best] for i in te_pos + te_neg])
            try:
                auc = float(roc_auc_score(y, s))
            except ValueError:
                continue
            aurocs.append(auc)
            entry = {"id": c["id"], "label": c["label"],
                     "region": best, "auroc": auc,
                     "contrast": float(contrast[best])}
            # export the held-out detector output for the concept page:
            # each example's snippet + score onto the selected region + label
            if (export_examples and basis == export_basis and texts is not None
                    and len(per_concept) < export_max_concepts):
                ex = []
                for idx in te_pos:
                    ex.append({"t": _snip(texts[idx]),
                               "s": round(float(pool[idx][best]), 3), "y": 1})
                for idx in te_neg[:len(te_pos)]:
                    ex.append({"t": _snip(texts[idx]),
                               "s": round(float(pool[idx][best]), 3), "y": 0})
                ex.sort(key=lambda e: -e["s"])
                # keep a balanced, readable slice
                entry["examples"] = ex[:export_examples] + ex[-export_examples:] \
                    if len(ex) > 2 * export_examples else ex
            per_concept.append(entry)
        results[basis] = {
            "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
            "std_auroc": float(np.std(aurocs)) if aurocs else None,
            "n_concepts": len(aurocs),
            "per_concept": per_concept,
        }
        logger.info("  [%s] mean AUROC=%.3f (n=%d)", basis,
                    results[basis]["mean_auroc"] or 0.0, len(aurocs))
    return {"dict": dict_dir.name, "layer": layer, "percentile": percentile,
            "n_partitions": len(parts), "by_basis": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dicts", required=True,
                    help="Comma-separated run dirs (each with dictionary.pkl "
                         "+ metadata.json). Must share a layer.")
    ap.add_argument("--n-concepts", type=int, default=48)
    ap.add_argument("--per-concept", type=int, default=48)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--bases", default="mean,exemplar")
    ap.add_argument("--max-tokens-per-text", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--export-examples", type=int, default=4,
                    help="Held-out examples per side to export (for the "
                         "Testing page) on dicts with K<=export-max-K.")
    ap.add_argument("--export-max-k", type=int, default=1000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    np.random.seed(args.seed)

    dict_dirs = [Path(d) for d in args.dicts.split(",")]
    layers = {_dict_layer(d)[0] for d in dict_dirs}
    if len(layers) != 1:
        raise SystemExit(f"dicts must share a layer; got {layers}")
    layer = layers.pop()
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]

    concepts, texts = load_concepts(args.n_concepts, args.per_concept, args.seed)

    qwen = QwenModel(args.model_id, device=args.device)
    cache = Path("artifacts/runs/concept_detect") / model_tag(args.model_id) / "_actcache"
    acts = extract_all(qwen, texts, layer, cache,
                       args.max_tokens_per_text, args.batch_size)

    out_entries = []
    for d in dict_dirs:
        logger.info("== %s ==", d.name)
        _, K = _dict_layer(d)
        n_parts = len(pickle.load((d / "dictionary.pkl").open("rb")).partitions)
        exp = args.export_examples if n_parts <= args.export_max_k else 0
        out_entries.append(evaluate_dict(d, qwen, concepts, acts, bases,
                                         args.train_frac, args.seed,
                                         texts=texts, export_examples=exp))

    out_path = Path(args.out or
                    f"artifacts/runs/concept_detect/{model_tag(args.model_id)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": args.model_id, "layer": layer,
               "n_concepts": len(concepts), "per_concept": args.per_concept,
               "train_frac": args.train_frac, "seed": args.seed,
               "note": ("AxBench concepts were mined from Gemma SAE labels; "
                        "these AUROCs are a re-derivation on Qwen, not "
                        "leaderboard-comparable."),
               "results": out_entries}
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("concept_detect -> %s", out_path)
    # crossover summary
    for e in out_entries:
        row = " ".join(f"{b}={e['by_basis'][b]['mean_auroc']:.3f}"
                       for b in bases if e['by_basis'][b]['mean_auroc'])
        logger.info("  p%g (K=%d): %s", e["percentile"], e["n_partitions"], row)


if __name__ == "__main__":
    main()
