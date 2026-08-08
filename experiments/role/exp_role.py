"""Tiers 1-2: build the EP dictionary, then measure how role lives in it.

Forward passes only — no generation — so this is minutes per configuration and a
percentile/layer sweep is affordable.

What it does, in order:

1. Build the corpus (§2 of the plan) and split by document.
2. Calibrate and ``discover`` on the **train** documents only, through the
   unmodified ``ep.discovery`` pipeline with ``extract_per_position``.
3. Assign train and test content activations, scatter into ``A[d, c, j]``.
4. §1 displacement, §3 occupancy/λ, §4 PCA/axis, §7 EP-vs-kmeans.

Deliberate choices worth knowing:

- **The dictionary is built on train documents only.** EP is unsupervised so
  building on everything would not leak labels, but fitting ``P(role | region)``
  on train and scoring on test is a cleaner protocol and costs nothing.
- **The extractor is not modified.** ``extract_per_position`` for both
  calibration and discovery. The upstream docstring warns that mixing
  per-position calibration with final-position discovery "silently produces
  meaningless cells", and the Qwen port that did so got Δ = exactly 0.00 at
  every layer.
- **Calibration cache key** carries ``__role`` in the model field plus corpus
  extras, so it cannot collide with the refusal calibration on the same volume.
  Upstream keys on (model, hook, percentile) with no seed, so concurrent seeds
  race to write one path; run seed 0 alone first.

Run:
    python -m experiments.role.exp_role --layer 18 --percentile 12 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from experiments.role import corpus as C
from experiments.role import metrics as M
from experiments.role import model_io as MIO

logger = logging.getLogger(__name__)


def _pair_list(conditions: tuple[str, ...]) -> list[tuple[str, str]]:
    """Ordered pairs worth reporting.

    All 15 unordered pairs are computed for the flip-rate matrix; displacement is
    only run on the pairs that mean something for the paper's claims, because
    each one carries a 20-sample null.
    """
    interesting = [
        ("user", "assistant"),
        ("user", "tool_native"),
        ("user", "tool_flat"),
        ("user", "system"),
        ("user", "cot"),
        ("assistant", "cot"),
        ("tool_flat", "tool_native"),
    ]
    return [(a, b) for a, b in interesting
            if a in conditions and b in conditions]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-short", default="Qwen3-4B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--percentile", type=float, default=12.0,
                    help="NOT the upstream default of 8.0, which the EP paper "
                         "reports as a fragmentation failure. Sweep this — the "
                         "threshold is calibrated on OUR corpus and tag-wrapped "
                         "C4 is nothing like chat-formatted instructions.")
    ap.add_argument("--calibration-tokens", type=int, default=100_000)
    ap.add_argument("--force-recalibrate", action="store_true")
    ap.add_argument("--n-docs", type=int, default=600)
    ap.add_argument("--n-train-docs", type=int, default=400)
    ap.add_argument("--n-content", type=int, default=96)
    ap.add_argument("--dataset", default="c4", choices=("c4", "pile"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-m", type=int, default=32,
                    help="Regions per polarity tail retained for the role "
                         "subspace and reported to the causal tier.")
    ap.add_argument("--min-members", type=int, default=5)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--kmeans", action="store_true", default=False,
                    help="Fit k-means at matched K as the capacity control. OFF by "
                         "default: informative only if EP beats chance, and at "
                         "K=7197 MiniBatchKMeans ran >23 min without converging.")
    ap.add_argument("--no-kmeans", dest="kmeans", action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/role"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"L{args.layer}_p{args.percentile:g}_seed{args.seed}"
    out_dir = args.output_dir / args.model_short / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output -> %s", out_dir)

    model = MIO.load_model(args.model, device=args.device, dtype=args.dtype)
    hook = MIO.hook_name(args.layer)

    # --- corpus ---
    contents = C.stream_contents(
        model.tokenizer, n_docs=args.n_docs, n_content=args.n_content,
        dataset=args.dataset, seed=args.seed,
    )
    corpus = C.build_corpus(model.tokenizer, contents)
    MIO.assert_tokenization_alignment(model, corpus)
    train_docs, test_docs = C.split_by_document(
        corpus, n_train=args.n_train_docs, seed=args.seed,
    )
    conditions = corpus.conditions
    logger.info("corpus: %d docs (%d train / %d test), %d conditions",
                len(corpus.doc_ids), len(train_docs), len(test_docs),
                len(conditions))

    train_texts, train_index = MIO.prompt_list(corpus, train_docs)
    test_texts, test_index = MIO.prompt_list(corpus, test_docs)

    # --- calibrate + discover on train prompts ---
    from ep.discovery.calibration import load_or_calibrate
    from ep.discovery.extraction import (
        extract_final_position, extract_per_position,
    )
    from ep.discovery.pipeline import _iter_activation_batches, discover

    def _calib_iter():
        rng = np.random.default_rng(args.seed)
        order = np.arange(len(train_texts))
        rng.shuffle(order)
        for i in order:
            yield train_texts[i]

    def _calib_batches():
        return _iter_activation_batches(
            model, _calib_iter(), hook,
            extract_fn=extract_per_position,
            extract_kwargs={"batch_size": args.batch_size},
            prompt_batch_size=args.batch_size,
            seed=args.seed,
        )

    calibration = load_or_calibrate(
        f"{args.model}__role", hook, _calib_batches,
        n_tokens=args.calibration_tokens, percentile=args.percentile,
        extras={"corpus": args.dataset, "n_content": args.n_content,
                "n_train_docs": len(train_docs)},
        force=args.force_recalibrate,
    )
    logger.info("calibration: threshold=%.6f", calibration.threshold)

    t0 = time.time()
    discover_result = discover(
        model=model, texts=list(train_texts), hook_name=hook,
        calibration=calibration, extract_fn=extract_per_position,
        extract_kwargs={"batch_size": args.batch_size},
        log_cadence=2, checkpoint_cadence=10_000, saturation_window=10,
        prompt_batch_size=args.batch_size, seed=args.seed,
    )
    dictionary = discover_result.dictionary
    K = len(dictionary)
    logger.info("dictionary: %d regions in %.1fs", K, time.time() - t0)

    dict_path = out_dir / f"{args.model_short}_L{args.layer}.pkl"
    tmp = dict_path.with_suffix(dict_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(dictionary, f)
    tmp.replace(dict_path)

    E = MIO.exemplar_matrix(dictionary)
    counts_build = MIO.member_counts(dictionary)

    # --- assign ---
    def _assign(texts, index, docs, label):
        res = extract_per_position(
            model, texts, hook, batch_size=args.batch_size,
        )
        x, doc, cond, j, keep = MIO.content_activations(
            corpus, index, res.x, res.prompt_ids, res.position_ids,
        )
        region, _ = dictionary.assign(x)
        # `region` is indexed against the *masked* extraction, so the index
        # arrays handed to the scatter must be masked identically.
        pa = MIO.build_paired_assignments(
            corpus, docs, index, region,
            res.prompt_ids[keep], res.position_ids[keep],
        )
        logger.info("%s: %d content activations, %d distinct regions",
                    label, len(x), len(np.unique(region)))
        return x, doc, cond, j, region, pa

    x_tr, doc_tr, cond_tr, j_tr, reg_tr, pa_tr = _assign(
        train_texts, train_index, train_docs, "train")
    x_te, doc_te, cond_te, j_te, reg_te, pa_te = _assign(
        test_texts, test_index, test_docs, "test")
    if pa_tr.n_missing or pa_te.n_missing:
        raise RuntimeError(
            f"unfilled paired slots (train {pa_tr.n_missing}, test "
            f"{pa_te.n_missing}); metrics would average over gaps"
        )

    # --- gemma degeneracy check: how many regions see a final position? ---
    final = extract_final_position(
        model, train_texts, hook, batch_size=args.batch_size,
    )
    final_region, _ = dictionary.assign(final.x)
    n_final_regions = int(len(np.unique(final_region)))
    logger.info(
        "final-position degeneracy: %d/%d regions receive any final-position "
        "activation (gemma L20 gave 5/207)", n_final_regions, K,
    )

    # --- §1 displacement ---
    logger.info("--- displacement ---")
    flips = M.flip_rate_matrix(pa_te.A)
    displacements = []
    for a, b in _pair_list(conditions):
        r = M.displacement(
            pa_te.A, E, conditions.index(a), conditions.index(b),
            names=(a, b), n_null=args.n_null, seed=args.seed,
        )
        displacements.append({
            "a": a, "b": b, "flip_rate": r.flip_rate,
            "n_flipped": r.n_flipped, "n_total": r.n_total,
            "coherence": r.coherence, "null_coherence": r.null_coherence,
            "null_coherence_std": r.null_coherence_std,
            "coherence_z": r.coherence_z,
            "mean_cosine_distance": r.mean_cosine_distance,
        })
        logger.info(
            "  %-12s -> %-12s flip=%.3f coherence=%.3f (null %.3f+-%.3f, "
            "z=%.1f)", a, b, r.flip_rate, r.coherence, r.null_coherence,
            r.null_coherence_std, r.coherence_z,
        )
        np.save(out_dir / f"displacement_{a}__{b}.npy", r.mean_direction)

    # --- §1 magnitude: how far does the tag move an activation, vs cell radius ---
    # The flip rate says the displacement is smaller than a cell; this says by how
    # much, in the same cosine-distance units the threshold is expressed in, so
    # the conclusion does not depend on the chosen resolution.
    centre_np = calibration.center.astype(np.float32)
    v = x_te - centre_np
    dirs_te = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    row_of = np.full(pa_te.A.shape, -1, dtype=np.int64)
    doc_slot = {int(d): i for i, d in enumerate(test_docs)}
    row_of[
        [doc_slot[int(d)] for d in doc_te],
        cond_te.astype(np.int64),
        j_te.astype(np.int64),
    ] = np.arange(len(x_te))
    magnitudes = {}
    for a, b in _pair_list(conditions):
        m = M.paired_displacement_magnitude(
            dirs_te, row_of, conditions.index(a), conditions.index(b),
            calibration.threshold,
        )
        magnitudes[f"{a}__{b}"] = m
        if m.get("n"):
            logger.info(
                "  |%s -> %s| cos_dist mean=%.4f median=%.4f p90=%.4f "
                "(threshold=%.4f, ratio=%.3f, frac beyond=%.4f)",
                a, b, m["mean"], m["median"], m["p90"], m["threshold"],
                m["ratio_to_threshold"], m["frac_beyond_threshold"],
            )

    couplings = {}
    for a, b in _pair_list(conditions)[:3]:
        T = M.transition_matrix(
            pa_te.A, conditions.index(a), conditions.index(b), K,
        )
        couplings[f"{a}__{b}"] = M.coupling_sparsity(T)
        logger.info("  coupling %s->%s: %s", a, b, couplings[f"{a}__{b}"])

    # --- §3 occupancy and polarity ---
    logger.info("--- occupancy ---")
    occ = {c: M.occupancy(pa_tr.A, i, K) for i, c in enumerate(conditions)}
    js = {}
    for a, b in _pair_list(conditions):
        val = M.js_divergence(occ[a], occ[b])
        null_mean, null_sd = M.js_divergence_null(
            pa_tr.A, conditions.index(a), conditions.index(b), K,
            n_null=args.n_null, seed=args.seed,
        )
        js[f"{a}__{b}"] = {
            "js_bits": val, "null_mean": null_mean, "null_std": null_sd,
        }
        logger.info("  JS(%s || %s) = %.4f bits (null %.4f+-%.4f)",
                    a, b, val, null_mean, null_sd)

    counts_tr = np.bincount(reg_tr, minlength=K)
    lam = M.polarity(
        occ["user"], occ["assistant"], min_count=counts_tr,
        min_members=args.min_members,
    )
    finite = np.isfinite(lam)
    order = np.argsort(-np.abs(np.nan_to_num(lam, nan=0.0)))
    top_user = [int(i) for i in np.argsort(-np.nan_to_num(lam, nan=-1e9))
                [:args.top_m]]
    top_asst = [int(i) for i in np.argsort(np.nan_to_num(lam, nan=1e9))
                [:args.top_m]]
    logger.info("polarity: %d/%d regions above the %d-member floor, "
                "|lambda| max %.3f", int(finite.sum()), K, args.min_members,
                float(np.nanmax(np.abs(lam))) if finite.any() else float("nan"))

    # --- §4 axis and PCA ---
    logger.info("--- axis and PCA ---")
    axis = M.role_axis(lam, E)
    comps_E, ratio_E = M.pca(E, n_components=10)
    lam0 = np.nan_to_num(lam, nan=0.0)
    lam_pc_corr = [
        float(np.corrcoef(lam0, E @ comps_E[k])[0, 1]) for k in range(3)
    ]
    logger.info("PCA(E): var explained %s; corr(lambda, E@PC) = %s",
                [round(float(v), 3) for v in ratio_E[:3]],
                [round(v, 3) for v in lam_pc_corr])
    logger.info("cos(role_axis, PC1(E)) = %.3f", M.cosine(axis, comps_E[0]))

    # Occupancy-profile PCA: one histogram per (doc, condition), region space.
    profiles = np.zeros((pa_tr.A.shape[0] * len(conditions), K),
                        dtype=np.float32)
    row = 0
    for d in range(pa_tr.A.shape[0]):
        for c in range(len(conditions)):
            profiles[row] = np.bincount(pa_tr.A[d, c, :], minlength=K)
            row += 1
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), 1)
    comps_P, ratio_P = M.pca(profiles, n_components=10)
    profile_labels = np.tile(np.arange(len(conditions)), pa_tr.A.shape[0])
    pc1_by_cond = {
        conditions[c]: float((profiles @ comps_P[0])[profile_labels == c].mean())
        for c in range(len(conditions))
    }
    logger.info("PCA(occupancy profiles): var %s; PC1 mean by condition %s",
                [round(float(v), 3) for v in ratio_P[:3]],
                {k: round(v, 4) for k, v in pc1_by_cond.items()})

    # m-dim role subspace from the polarity tails.
    sub_dirs = E[[*top_user, *top_asst]]
    Q, _ = np.linalg.qr(sub_dirs.T)
    np.savez_compressed(
        out_dir / "role_subspace.npz",
        axis=axis, subspace=Q, top_user=np.array(top_user),
        top_assistant=np.array(top_asst), lam=lam,
        pc1_E=comps_E[0], counts_train=counts_tr,
        counts_build=counts_build,
    )
    logger.info("role subspace: %d dims saved", Q.shape[1])

    # --- §1/§3 information content, and §7 the comparison arms ---
    logger.info("--- NMI and classifiers ---")
    # Content is identical across conditions, so the token id depends only on
    # (doc, position) — this is what makes I(region; content) the right
    # comparison for I(region; role).
    ids_by_doc = C.content_token_ids(corpus)
    content_id_tr = np.array([
        ids_by_doc[int(d)][int(j)] for d, j in zip(doc_tr, j_tr)
    ])
    # Role NMI uses the PAIRED null (conditions permuted within each
    # (doc, position) group). The global-permutation null is not merely
    # conservative here, it is wrong: the balanced design makes the observed
    # plug-in MI essentially unbiased, so global permutation manufactures bias
    # out of nothing and buries any real signal. See metrics.normalized_mi_null_paired.
    nmi_role = M.paired_role_nmi(pa_tr.A)
    nmi_role_null, nmi_role_null_sd = M.normalized_mi_null_paired(
        pa_tr.A, n_null=args.n_null, seed=args.seed,
    )
    # Reported alongside only to show how much the wrong null inflates.
    nmi_role_global_null, _ = M.normalized_mi_null(
        reg_tr, cond_tr, n_null=max(3, args.n_null // 4), seed=args.seed,
    )
    # Content is not a paired label, so global permutation is correct for it.
    nmi_content = M.normalized_mi(reg_tr, content_id_tr)
    nmi_content_null, _ = M.normalized_mi_null(
        reg_tr, content_id_tr, n_null=max(3, args.n_null // 4), seed=args.seed,
    )
    nmi_role_z = (
        (nmi_role - nmi_role_null) / nmi_role_null_sd
        if nmi_role_null_sd > 0 else float("nan")
    )
    logger.info("NMI(region; role)    = %.4f (paired null %.4f+-%.4f, z=%.1f; "
                "global null %.4f would be the wrong comparison)",
                nmi_role, nmi_role_null, nmi_role_null_sd, nmi_role_z,
                nmi_role_global_null)
    logger.info("NMI(region; content) = %.4f (null %.4f)",
                nmi_content, nmi_content_null)

    ep_clf = M.region_table_classifier(
        reg_tr, cond_tr, reg_te, cond_te, K, len(conditions), name="ep_region",
    )
    logger.info("EP region table: acc=%.3f macroAUROC=%.3f",
                ep_clf.accuracy, ep_clf.macro_auroc)

    km_clf = None
    if args.kmeans and K >= len(x_tr):
        logger.warning(
            "skipping the k-means control: K=%d >= %d training activations, so "
            "matched-K clustering is not defined", K, len(x_tr),
        )
    elif args.kmeans:
        from sklearn.cluster import MiniBatchKMeans
        # Same discretization budget, same preprocessing EP uses: centre on the
        # calibration centre, project to the unit sphere. Comparing against
        # k-means on raw activations would compare EP to a different geometry,
        # not to a different clustering.
        centre = calibration.center.astype(np.float32)
        def _dirs(x):
            v = x - centre
            n = np.linalg.norm(v, axis=1, keepdims=True)
            return v / np.maximum(n, 1e-8)
        km = MiniBatchKMeans(
            n_clusters=K, random_state=args.seed, n_init=3, batch_size=4096,
        ).fit(_dirs(x_tr))
        km_tr = km.labels_
        km_te = km.predict(_dirs(x_te))
        km_clf = M.region_table_classifier(
            km_tr, cond_tr, km_te, cond_te, K, len(conditions), name="kmeans",
        )
        logger.info("k-means (K=%d) table: acc=%.3f macroAUROC=%.3f",
                    K, km_clf.accuracy, km_clf.macro_auroc)

    payload = {
        "config": vars(args) | {"output_dir": str(out_dir)},
        "n_regions": K,
        "n_docs_kept": len(corpus.doc_ids),
        "corpus_drop_reasons": corpus.drop_reasons,
        "conditions": list(conditions),
        "calibration": {
            "threshold": float(calibration.threshold),
            "percentile": float(calibration.percentile),
            "n_activations": int(calibration.n_activations),
        },
        "final_position_degeneracy": {
            "n_regions_with_final_activation": n_final_regions,
            "n_regions": K,
        },
        "flip_rate_matrix": flips.tolist(),
        "displacement": displacements,
        "displacement_magnitude": magnitudes,
        "coupling": couplings,
        "occupancy_js": js,
        "polarity": {
            "n_above_floor": int(finite.sum()),
            "min_members": args.min_members,
            "top_user_pids": top_user,
            "top_assistant_pids": top_asst,
            "lam_top_user": [float(lam[i]) for i in top_user],
            "lam_top_assistant": [float(lam[i]) for i in top_asst],
            "top_by_abs": [int(i) for i in order[:args.top_m]],
        },
        "pca": {
            "exemplar_var_explained": [float(v) for v in ratio_E],
            "lam_corr_with_pc": lam_pc_corr,
            "cos_axis_pc1": M.cosine(axis, comps_E[0]),
            "profile_var_explained": [float(v) for v in ratio_P],
            "profile_pc1_mean_by_condition": pc1_by_cond,
            "subspace_dims": int(Q.shape[1]),
        },
        "information": {
            "nmi_region_role": nmi_role,
            "nmi_region_role_null_paired": nmi_role_null,
            "nmi_region_role_null_paired_std": nmi_role_null_sd,
            "nmi_region_role_z": nmi_role_z,
            "nmi_region_role_null_global_wrong": nmi_role_global_null,
            "nmi_region_content": nmi_content,
            "nmi_region_content_null": nmi_content_null,
        },
        "classifiers": {
            "ep_region": vars(ep_clf),
            "kmeans": vars(km_clf) if km_clf else None,
        },
    }
    path = out_dir / "role.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", path)

    # --- the one-line read on which §11 outcome this is ---
    ua = next((d for d in displacements
               if d["a"] == "user" and d["b"] == "assistant"), None)
    if ua:
        if ua["flip_rate"] < 0.2:
            verdict = "SHARED occupancy — the tag barely moves the region"
        elif ua["coherence_z"] > 3 and ua["coherence"] > 0.3:
            verdict = "COHERENT displacement — role ~ one direction"
        else:
            verdict = "CONDITIONAL displacement — no single role direction"
        logger.info("VERDICT (user->assistant): %s "
                    "(flip=%.3f coherence=%.3f z=%.1f)",
                    verdict, ua["flip_rate"], ua["coherence"],
                    ua["coherence_z"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
