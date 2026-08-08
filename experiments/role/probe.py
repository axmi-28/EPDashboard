"""Tier 0: the gate. Does Qwen3-4B exhibit the paper's role confusion at all?

Trains a per-layer linear role probe on tag-wrapped C4 (identical content, six
role scaffolds), then reproduces arXiv 2603.12277's Table 1 row for the Qwen
family on *real* user-style text from OASST1: user-style content stays ~76-88%
Userness even when re-tagged as a tool response, and Toolness never exceeds 20%.

If that does not reproduce, there is no role confusion to localize and every
later tier is unanswerable rather than negative. This is the
smoke-test-before-spend step; it needs no generation, only forward passes.

Two protocol points that are easy to get wrong and both fatal:

- **Split by document, never by token.** The same content string appears in all
  six conditions and at every position. A token-level split puts the same content
  on both sides and the probe reports ~99% at every layer — a leak that looks
  exactly like a strong result.
- **Content tokens only.** The scaffold tokens differ across conditions by
  construction, so a probe allowed to see them is reading the tag literally.

Run:
    python -m experiments.role.probe --n-docs 600 --layers 0,3,6,9,12,15,18,21,24,27,30,33,35
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from experiments.role import corpus as C
from experiments.role import model_io as MIO

logger = logging.getLogger(__name__)

# The paper's Table 1 row for Qwen3-30B-A3B, and the gate we hold Qwen3-4B to.
# Thresholds are loosened from the published values because this is a 4B and a
# 20-point shortfall would still be role confusion; a *reversal* would not be.
GATE = {
    "userness_under_user": (0.836, 0.70),
    "userness_under_tool": (0.757, 0.60),
    "toolness_under_tool": (0.195, 0.35),   # upper bound
}


def _fit_probe(x_tr, y_tr, seed: int, max_iter: int = 2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # No `multi_class="multinomial"`: the parameter was REMOVED in sklearn 1.9
    # (deprecated in 1.5), and passing it raises TypeError. Multinomial is now
    # the only multiclass behaviour for lbfgs, so dropping it preserves intent
    # rather than silently switching to one-vs-rest.
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed),
    )
    clf.fit(x_tr, y_tr)
    return clf


def _macro_auroc(clf, x, y) -> float:
    from sklearn.metrics import roc_auc_score
    proba = clf.predict_proba(x)
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    cols = [list(clf.classes_).index(c) for c in present]
    sub = proba[:, cols]
    sub = sub / sub.sum(axis=1, keepdims=True)
    if len(present) == 2:
        return float(roc_auc_score((y == present[1]).astype(int), sub[:, 1]))
    return float(
        roc_auc_score(y, sub, multi_class="ovr", average="macro", labels=present)
    )


def _load_oasst_user_text(tokenizer, n: int, n_content: int, seed: int) -> list[str]:
    """Real user-style text: OASST1 prompter turns, truncated like the corpus.

    Raises on shortfall. The lesson from `_load_harmful`'s silent fallback is that
    a corpus loader that degrades quietly produces a structurally valid,
    meaningless run.
    """
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rng = np.random.default_rng(seed)
    rows = [
        r["text"] for r in ds
        if r.get("role") == "prompter" and r.get("lang") == "en"
        and len(r.get("text", "")) > 200
    ]
    rng.shuffle(rows)

    out: list[str] = []
    for text in rows:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < n_content:
            continue
        s = tokenizer.decode(ids[:n_content]).strip()
        if len(tokenizer.encode(s, add_special_tokens=False)) != n_content:
            continue
        out.append(s)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"oasst1: got {len(out)} user-style strings, wanted {n} "
            f"(pool was {len(rows)} English prompter turns)"
        )
    return out


def _harvest(model, texts, prompt_index, corpus, layers, batch_size,
             positions_per_doc, seed):
    """Content-token activations at several layers from one pass per layer.

    ``extract_per_position`` takes a single hook, so this is one pass per layer
    rather than one pass with many hooks. That is the cost of using the reference
    extractor unmodified, and it is the right trade: a custom multi-hook
    extractor is exactly the kind of divergence that made ``experiments/refusal.py``
    untrustworthy.

    Positions are subsampled per (doc, condition) because 600 docs x 6 conditions
    x 96 positions x 2560 dims x 4 bytes x 13 layers is ~46 GB otherwise. The
    probe does not need 350k samples.
    """
    from ep.discovery.extraction import extract_per_position

    out: dict[int, tuple] = {}
    rng = np.random.default_rng(seed)
    for layer in layers:
        t0 = time.time()
        res = extract_per_position(
            model, texts, MIO.hook_name(layer), batch_size=batch_size,
        )
        x, doc, cond, j, _ = MIO.content_activations(
            corpus, prompt_index, res.x, res.prompt_ids, res.position_ids,
        )
        if positions_per_doc is not None:
            n_content = int(j.max()) + 1
            if positions_per_doc < n_content:
                keep_j = set(
                    rng.choice(n_content, size=positions_per_doc, replace=False)
                    .tolist()
                )
                mask = np.array([int(v) in keep_j for v in j])
                x, doc, cond, j = x[mask], doc[mask], cond[mask], j[mask]
        out[layer] = (x, doc, cond, j)
        logger.info("L%-2d harvested %s in %.1fs", layer, x.shape,
                    time.time() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-short", default="Qwen3-4B")
    ap.add_argument("--layers", default="0,3,6,9,12,15,18,21,24,27,30,33,35")
    ap.add_argument("--report-layer", type=int, default=18,
                    help="Layer used for the Table 1 cells (paper: mid-layer).")
    ap.add_argument("--n-docs", type=int, default=600)
    ap.add_argument("--n-train-docs", type=int, default=400)
    ap.add_argument("--n-content", type=int, default=96)
    ap.add_argument("--positions-per-doc", type=int, default=24)
    ap.add_argument("--dataset", default="c4", choices=("c4", "pile"))
    ap.add_argument("--n-oasst", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("artifacts/runs/role/probe"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    np.random.seed(args.seed)
    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    out_dir = args.output_dir / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MIO.load_model(args.model, device=args.device, dtype=args.dtype)
    if args.report_layer >= model.cfg.n_layers:
        raise ValueError(
            f"--report-layer {args.report_layer} but the model has "
            f"{model.cfg.n_layers} layers"
        )

    # --- corpus ---
    contents = C.stream_contents(
        model.tokenizer, n_docs=args.n_docs, n_content=args.n_content,
        dataset=args.dataset, seed=args.seed,
    )
    corpus = C.build_corpus(model.tokenizer, contents)
    if len(corpus.doc_ids) < args.n_train_docs + 50:
        raise RuntimeError(
            f"only {len(corpus.doc_ids)} docs survived constancy checks; need "
            f"{args.n_train_docs} train + >=50 test"
        )
    MIO.assert_tokenization_alignment(model, corpus)

    train_docs, test_docs = C.split_by_document(
        corpus, n_train=args.n_train_docs, seed=args.seed,
    )
    logger.info("split: %d train docs, %d test docs", len(train_docs),
                len(test_docs))

    texts, prompt_index = MIO.prompt_list(corpus, corpus.doc_ids)
    harvest = _harvest(
        model, texts, prompt_index, corpus, layers, args.batch_size,
        args.positions_per_doc, args.seed,
    )

    # --- per-layer probes ---
    train_set, test_set = set(train_docs), set(test_docs)
    per_layer = []
    probes = {}
    for layer in layers:
        x, doc, cond, _ = harvest[layer]
        tr = np.array([int(d) in train_set for d in doc])
        te = np.array([int(d) in test_set for d in doc])
        clf = _fit_probe(x[tr], cond[tr], seed=args.seed)
        acc = float((clf.predict(x[te]) == cond[te]).mean())
        auroc = _macro_auroc(clf, x[te], cond[te])
        chance = 1.0 / len(corpus.conditions)
        per_layer.append({
            "layer": layer, "accuracy": acc, "macro_auroc": auroc,
            "chance": chance, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        })
        probes[layer] = clf
        logger.info("L%-2d role probe: acc=%.3f (chance %.3f) macroAUROC=%.3f",
                    layer, acc, chance, auroc)

    # --- Table 1: real user-style text, correctly tagged vs re-tagged ---
    logger.info("Table 1: OASST1 user-style text under user vs tool tags")
    oasst = _load_oasst_user_text(
        model.tokenizer, n=args.n_oasst, n_content=args.n_content,
        seed=args.seed + 7,
    )
    oasst_corpus = C.build_corpus(model.tokenizer, oasst)
    MIO.assert_tokenization_alignment(model, oasst_corpus)
    o_texts, o_index = MIO.prompt_list(oasst_corpus, oasst_corpus.doc_ids)

    from ep.discovery.extraction import extract_per_position
    res = extract_per_position(
        model, o_texts, MIO.hook_name(args.report_layer),
        batch_size=args.batch_size,
    )
    ox, _, ocond, _, _ = MIO.content_activations(
        oasst_corpus, o_index, res.x, res.prompt_ids, res.position_ids,
    )
    clf = probes[args.report_layer]
    proba = clf.predict_proba(ox)
    cls = list(clf.classes_)
    conds = oasst_corpus.conditions

    def _mean_p(target_condition: str, tagged_as: str) -> float:
        """Mean P(role=target | h) over tokens actually tagged `tagged_as`."""
        rows = ocond == conds.index(tagged_as)
        col = cls.index(conds.index(target_condition))
        return float(proba[rows, col].mean())

    table1 = {
        "layer": args.report_layer,
        "n_oasst_docs": len(oasst_corpus.doc_ids),
        "userness_under_user": _mean_p("user", "user"),
        "userness_under_tool_native": _mean_p("user", "tool_native"),
        "userness_under_tool_flat": _mean_p("user", "tool_flat"),
        "toolness_under_tool_native": _mean_p("tool_native", "tool_native"),
        "toolness_under_tool_flat": _mean_p("tool_flat", "tool_flat"),
        "userness_under_assistant": _mean_p("user", "assistant"),
        "assistantness_under_assistant": _mean_p("assistant", "assistant"),
    }
    for k, v in table1.items():
        if isinstance(v, float):
            logger.info("  %-32s %.3f", k, v)

    # tool_native is a user turn in Qwen3's template, so it is the *easier* case
    # for the paper's claim; tool_flat is the hard one. Gate on flat.
    checks = {
        "userness_under_user": table1["userness_under_user"] >= GATE[
            "userness_under_user"][1],
        "userness_under_tool_flat": table1["userness_under_tool_flat"] >= GATE[
            "userness_under_tool"][1],
        "toolness_under_tool_flat": table1["toolness_under_tool_flat"] <= GATE[
            "toolness_under_tool"][1],
    }
    passed = all(checks.values())
    logger.info("GATE %s: %s", "PASS" if passed else "FAIL", checks)
    if not passed:
        logger.warning(
            "Role confusion did not reproduce at L%d. Before concluding, check "
            "the probe accuracy column: if per-layer accuracy is near chance, "
            "the probe failed and the gate says nothing; if accuracy is high but "
            "re-tagged Userness collapses, Qwen3-4B genuinely re-asserts tags "
            "and that is the finding.", args.report_layer,
        )

    payload = {
        "config": vars(args) | {"output_dir": str(out_dir)},
        "n_docs_kept": len(corpus.doc_ids),
        "corpus_drop_reasons": corpus.drop_reasons,
        "conditions": list(conds),
        "per_layer": per_layer,
        "table1": table1,
        "gate": {"checks": checks, "passed": passed,
                 "reference": {k: v[0] for k, v in GATE.items()}},
    }
    path = out_dir / f"probe_L{args.report_layer}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", path)

    np.savez_compressed(
        out_dir / f"probe_directions_L{args.report_layer}.npz",
        coef=probes[args.report_layer][-1].coef_,
        classes=np.array(probes[args.report_layer][-1].classes_),
        conditions=np.array(conds, dtype=object),
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
