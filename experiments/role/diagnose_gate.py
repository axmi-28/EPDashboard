"""Why did the Tier 0 gate fail — the model, or the probe?

The gate returned `userness_under_user = 0.299` on OASST1, against a 0.70
threshold transplanted from the paper's 0.836. But the L18 probe scores 0.459
accuracy on held-out C4, and a 6-way probe at that accuracy will assign the true
class only ~0.3-0.45 on average — so the threshold was never reachable and the
gate conflated "does the model exhibit role confusion" with "is my probe strong
enough to see it".

This isolates the three candidate explanations, at the cost of one L18 harvest:

1. **Weak probe vs failure to transfer.** Report the Table 1 diagonal on held-out
   **C4** (in-distribution), which the gate never measured. If in-distribution
   userness is also ~0.3, the probe is simply weak and the threshold was wrong.
   If it is high and only OASST1 collapses, the probe does not transfer across
   text distribution and the paper's train-neutral/test-real protocol needs
   rethinking on a 4B.

2. **Class dilution.** `tool_native` IS a user turn on Qwen3
   (`<|im_start|>user\\n<tool_response>`), so it necessarily competes with `user`
   for probability mass, and `cot` competes with `assistant`. A 4-way probe over
   {system, user, assistant, tool_flat} removes both near-duplicates.

3. **Position confound.** `cot` and `tool_native` have 5-token prefixes; the rest
   have 3. So the same content token sits at a different absolute position
   depending on condition, and a probe could read position rather than role.
   The 4-way subset has a uniform prefix length, which removes it — if accuracy
   survives there, position was not the explanation.

Also reports full mean-probability matrices, because the diagonal alone does not
show where the mass went, and a binary user-vs-assistant probe as the cleanest
available analogue of the paper's construct.

Run:
    python -m experiments.role.diagnose_gate --layer 18
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experiments.role import corpus as C
from experiments.role import model_io as MIO
from experiments.role.probe import _fit_probe, _load_oasst_user_text, _macro_auroc

logger = logging.getLogger(__name__)

# Uniform 3-token prefix, and no near-duplicate classes.
EQUAL_PREFIX = ("system", "user", "assistant", "tool_flat")


def _mean_prob_matrix(proba, cond_slots, clf_classes, conditions):
    """``M[tagged, predicted]`` = mean P(predicted | tokens tagged `tagged`)."""
    cls = list(clf_classes)
    M = np.full((len(conditions), len(conditions)), np.nan)
    for t_i in range(len(conditions)):
        rows = cond_slots == t_i
        if not rows.any():
            continue
        for p_i in range(len(conditions)):
            if p_i not in cls:
                continue
            M[t_i, p_i] = float(proba[rows, cls.index(p_i)].mean())
    return M


def _log_matrix(name, M, conditions):
    logger.info("%s (rows = tagged as, cols = P(predicted)):", name)
    logger.info("  %-14s %s", "", " ".join(f"{c[:9]:>9}" for c in conditions))
    for i, c in enumerate(conditions):
        cells = " ".join(
            "      nan" if np.isnan(v) else f"{v:9.3f}" for v in M[i]
        )
        logger.info("  %-14s %s", c[:14], cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-short", default="Qwen3-4B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n-docs", type=int, default=600)
    ap.add_argument("--n-train-docs", type=int, default=400)
    ap.add_argument("--n-content", type=int, default=96)
    ap.add_argument("--positions-per-doc", type=int, default=24)
    ap.add_argument("--n-oasst", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/role/diagnose"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    out_dir = args.output_dir / args.model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    from ep.discovery.extraction import extract_per_position
    model = MIO.load_model(args.model, device=args.device, dtype=args.dtype)
    hook = MIO.hook_name(args.layer)
    rng = np.random.default_rng(args.seed)

    def harvest(corpus, doc_ids):
        texts, index = MIO.prompt_list(corpus, doc_ids)
        res = extract_per_position(model, texts, hook, batch_size=args.batch_size)
        x, doc, cond, j, _ = MIO.content_activations(
            corpus, index, res.x, res.prompt_ids, res.position_ids,
        )
        n_content = int(j.max()) + 1
        if args.positions_per_doc < n_content:
            keep = set(rng.choice(n_content, size=args.positions_per_doc,
                                  replace=False).tolist())
            m = np.array([int(v) in keep for v in j])
            x, doc, cond, j = x[m], doc[m], cond[m], j[m]
        return x, doc, cond, j

    # --- C4 corpus, same construction and split as the gate ---
    contents = C.stream_contents(
        model.tokenizer, n_docs=args.n_docs, n_content=args.n_content,
        dataset="c4", seed=args.seed,
    )
    corpus = C.build_corpus(model.tokenizer, contents)
    MIO.assert_tokenization_alignment(model, corpus)
    train_docs, test_docs = C.split_by_document(
        corpus, n_train=args.n_train_docs, seed=args.seed,
    )
    conditions = corpus.conditions
    x, doc, cond, _ = harvest(corpus, corpus.doc_ids)
    tr = np.isin(doc, train_docs)
    te = np.isin(doc, test_docs)
    logger.info("C4: %d train / %d test content activations",
                int(tr.sum()), int(te.sum()))

    # --- OASST1, tag-wrapped identically ---
    oasst = _load_oasst_user_text(
        model.tokenizer, n=args.n_oasst, n_content=args.n_content,
        seed=args.seed + 7,
    )
    o_corpus = C.build_corpus(model.tokenizer, oasst)
    MIO.assert_tokenization_alignment(model, o_corpus)
    ox, _, ocond, _ = harvest(o_corpus, o_corpus.doc_ids)
    logger.info("OASST1: %d content activations", len(ox))

    results: dict = {"layer": args.layer, "conditions": list(conditions)}

    # === 1. six-way probe: in-distribution vs OASST1 ===
    logger.info("=== six-way probe (as the gate used) ===")
    clf6 = _fit_probe(x[tr], cond[tr], seed=args.seed)
    acc6 = float((clf6.predict(x[te]) == cond[te]).mean())
    logger.info("C4 test accuracy = %.3f (chance %.3f), macroAUROC = %.3f",
                acc6, 1 / len(conditions), _macro_auroc(clf6, x[te], cond[te]))

    M_c4 = _mean_prob_matrix(clf6.predict_proba(x[te]), cond[te],
                             clf6.classes_, conditions)
    M_oa = _mean_prob_matrix(clf6.predict_proba(ox), ocond,
                             clf6.classes_, conditions)
    _log_matrix("C4 test (IN-DISTRIBUTION)", M_c4, conditions)
    _log_matrix("OASST1 (what the gate scored)", M_oa, conditions)

    ui = conditions.index("user")
    logger.info(
        "THE MISSING CONTROL: userness_under_user  C4=%.3f  OASST1=%.3f",
        M_c4[ui, ui], M_oa[ui, ui],
    )
    results["six_way"] = {
        "c4_test_accuracy": acc6,
        "chance": 1 / len(conditions),
        "mean_prob_matrix_c4": M_c4.tolist(),
        "mean_prob_matrix_oasst": M_oa.tolist(),
        "userness_under_user_c4": float(M_c4[ui, ui]),
        "userness_under_user_oasst": float(M_oa[ui, ui]),
    }

    # === 2+3. four-way probe: no near-duplicates, uniform prefix length ===
    logger.info("=== four-way probe %s (no duplicates, uniform prefix) ===",
                EQUAL_PREFIX)
    starts = {
        c: next(it.content_start for it in corpus.items if it.condition == c)
        for c in conditions
    }
    logger.info("content_start by condition: %s", starts)
    assert len({starts[c] for c in EQUAL_PREFIX}) == 1, (
        f"EQUAL_PREFIX conditions must share a prefix length, got "
        f"{ {c: starts[c] for c in EQUAL_PREFIX} }"
    )
    sub = [conditions.index(c) for c in EQUAL_PREFIX]
    remap = {s: i for i, s in enumerate(sub)}

    m_tr = tr & np.isin(cond, sub)
    m_te = te & np.isin(cond, sub)
    y_tr = np.array([remap[int(v)] for v in cond[m_tr]])
    y_te = np.array([remap[int(v)] for v in cond[m_te]])
    clf4 = _fit_probe(x[m_tr], y_tr, seed=args.seed)
    acc4 = float((clf4.predict(x[m_te]) == y_te).mean())
    logger.info("C4 test accuracy = %.3f (chance %.3f)", acc4, 1 / len(sub))

    m_oa = np.isin(ocond, sub)
    y_oa = np.array([remap[int(v)] for v in ocond[m_oa]])
    M4_c4 = _mean_prob_matrix(clf4.predict_proba(x[m_te]), y_te,
                              clf4.classes_, EQUAL_PREFIX)
    M4_oa = _mean_prob_matrix(clf4.predict_proba(ox[m_oa]), y_oa,
                              clf4.classes_, EQUAL_PREFIX)
    _log_matrix("4-way C4 test", M4_c4, EQUAL_PREFIX)
    _log_matrix("4-way OASST1", M4_oa, EQUAL_PREFIX)
    u4 = EQUAL_PREFIX.index("user")
    t4 = EQUAL_PREFIX.index("tool_flat")
    logger.info(
        "4-way: userness_under_user C4=%.3f OASST1=%.3f | "
        "userness_under_tool_flat OASST1=%.3f | toolness_under_tool_flat "
        "OASST1=%.3f",
        M4_c4[u4, u4], M4_oa[u4, u4], M4_oa[t4, u4], M4_oa[t4, t4],
    )
    results["four_way"] = {
        "conditions": list(EQUAL_PREFIX),
        "c4_test_accuracy": acc4,
        "chance": 1 / len(sub),
        "mean_prob_matrix_c4": M4_c4.tolist(),
        "mean_prob_matrix_oasst": M4_oa.tolist(),
        "retention_oasst": (
            float(M4_oa[t4, u4] / M4_oa[u4, u4]) if M4_oa[u4, u4] > 0 else None
        ),
    }

    # === 4. binary user vs assistant — cleanest construct available ===
    logger.info("=== binary user vs assistant ===")
    pair = [conditions.index("user"), conditions.index("assistant")]
    b_tr = tr & np.isin(cond, pair)
    b_te = te & np.isin(cond, pair)
    yb_tr = (cond[b_tr] == pair[0]).astype(int)
    yb_te = (cond[b_te] == pair[0]).astype(int)
    clf2 = _fit_probe(x[b_tr], yb_tr, seed=args.seed)
    acc2 = float((clf2.predict(x[b_te]) == yb_te).mean())
    auroc2 = _macro_auroc(clf2, x[b_te], yb_te)
    logger.info("C4 test accuracy = %.3f (chance 0.5) AUROC = %.3f", acc2, auroc2)

    col_user = list(clf2.classes_).index(1)
    p_user_c4 = float(clf2.predict_proba(x[b_te])[yb_te == 1, col_user].mean())
    o_user = ocond == conditions.index("user")
    o_asst = ocond == conditions.index("assistant")
    p_user_oa = float(clf2.predict_proba(ox[o_user])[:, col_user].mean())
    p_user_oa_asst = float(clf2.predict_proba(ox[o_asst])[:, col_user].mean())
    logger.info(
        "P(user) on user-tagged: C4=%.3f OASST1=%.3f | on assistant-tagged "
        "OASST1=%.3f", p_user_c4, p_user_oa, p_user_oa_asst,
    )
    results["binary_user_vs_assistant"] = {
        "c4_test_accuracy": acc2, "c4_test_auroc": auroc2,
        "p_user_given_user_c4": p_user_c4,
        "p_user_given_user_oasst": p_user_oa,
        "p_user_given_assistant_oasst": p_user_oa_asst,
    }

    path = out_dir / f"diagnose_L{args.layer}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("wrote %s", path)

    # --- the verdict this script exists to deliver ---
    logger.info("=== DIAGNOSIS ===")
    if M_c4[ui, ui] > 0.6 and M_oa[ui, ui] < 0.4:
        logger.info("TRANSFER FAILURE: the probe is strong in-distribution and "
                    "collapses on OASST1. The paper's train-neutral/test-real "
                    "protocol does not carry to a 4B; retrain on matched text.")
    elif M_c4[ui, ui] < 0.5:
        logger.info("WEAK PROBE: userness_under_user is low even "
                    "IN-DISTRIBUTION (%.3f), so the 0.70 gate threshold was "
                    "unreachable by construction and the gate was "
                    "mis-specified, not the model refuted.", M_c4[ui, ui])
    else:
        logger.info("Neither clean transfer failure nor a weak probe; read the "
                    "matrices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
