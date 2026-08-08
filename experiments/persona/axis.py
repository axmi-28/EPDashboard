"""P1 — replicate the Assistant Axis (arXiv:2601.10387) on Qwen3.5-4B.

Faithful to ``github.com/safety-research/assistant-axis`` at reduced scale:

  1. rollouts   role x system-prompt x question -> greedy assistant responses
  2. vectors    mean post-block residual over *response* tokens (per layer),
                averaged within each role -> one role vector per (role, layer)
  3. axis       Assistant Axis = mean(default) - mean(all roles)   [per layer]
  4. pca        standardise role vectors, PCA; check PC1 ~ Assistant Axis and
                that the default Assistant sits at an extreme of PC1

Divergences from the paper (flagged, reduced-scale first pass):
  * greedy decoding, not temperature 0.7 sampling;
  * NO LLM-judge score=3 filter yet — role vectors average ALL of a role's
    rollouts (noisier; judge filtering is a follow-up);
  * a curated ~28-role SPECTRUM subset, not all 275 roles.

Per-rollout activations are cached to ``activations.npz`` so the vectors/axis/
pca stages rerun without regenerating.

Example:
    python -m experiments.persona.axis --roles spectrum \
        --n-questions 16 --n-system-prompts 3 --layers 16,27 \
        --max-new-tokens 48 --gen-batch 8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from qwen_ep.adapter import DEFAULT_MODEL_ID, QwenModel, model_tag
from .data import DEFAULT_ROLE, build_rollout_specs, resolve_roles

log = logging.getLogger("experiments.persona.axis")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--roles", default="spectrum",
                    help="'spectrum' | 'all' | comma-separated role names")
    ap.add_argument("--n-questions", type=int, default=16)
    ap.add_argument("--n-system-prompts", type=int, default=3)
    ap.add_argument("--layers", default="16,27",
                    help="comma-separated layers to capture response means at")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--extract-batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="artifacts/runs/persona_axis")
    ap.add_argument("--stage", default="all",
                    choices=["all", "generate", "analyze"],
                    help="'generate' caches activations; 'analyze' reruns "
                         "vectors/axis/pca from the cache")
    return ap.parse_args()


# ---------------------------------------------------------------- generate
def stage_generate(args, out_dir: Path) -> None:
    qwen = QwenModel(args.model_id, device=args.device)
    layers = [int(x) for x in args.layers.split(",")]
    for L in layers:
        if not (0 <= L < qwen.n_layers):
            raise SystemExit(f"layer {L} out of range [0,{qwen.n_layers})")

    roles = resolve_roles(args.roles)
    specs = build_rollout_specs(
        roles, n_questions=args.n_questions,
        n_system_prompts=args.n_system_prompts,
        model_name=args.model_id.split("/")[-1], seed=args.seed)
    log.info("roles=%d  rollouts=%d  layers=%s", len(roles), len(specs), layers)

    systems = [s["system"] for s in specs]
    users = [s["question"] for s in specs]
    formatted = [qwen.format_chat(u, system=sy) for u, sy in zip(users, systems)]

    log.info("== generating %d responses ==", len(formatted))
    t0 = time.time()
    responses = qwen.generate(formatted, max_new_tokens=args.max_new_tokens,
                              batch_size=args.gen_batch)
    log.info("generation done in %.0fs", time.time() - t0)

    log.info("== extracting response-mean activations ==")
    t0 = time.time()
    acts = qwen.mean_response_activations(
        systems, users, responses, layers=layers,
        batch_size=args.extract_batch)  # (N, n_layers, d)
    log.info("extraction done in %.0fs  acts=%s", time.time() - t0, acts.shape)

    np.savez_compressed(
        out_dir / "activations.npz",
        acts=acts.astype(np.float32),
        role=np.array([s["role"] for s in specs]),
        group=np.array([s["group"] for s in specs]),
        layers=np.array(layers),
    )
    (out_dir / "rollouts.jsonl").write_text("\n".join(
        json.dumps({**s, "response": r}, ensure_ascii=False)
        for s, r in zip(specs, responses)))
    log.info("cached -> %s", out_dir / "activations.npz")


# ---------------------------------------------------------------- analyze
def _role_vectors(acts: np.ndarray, role: np.ndarray
                  ) -> tuple[list[str], np.ndarray]:
    """Mean activation per role -> (role_names, (R, n_layers, d))."""
    names = sorted(set(role.tolist()), key=lambda r: (r != DEFAULT_ROLE, r))
    vecs = np.stack([acts[role == r].mean(0) for r in names])
    return names, vecs


def stage_analyze(args, out_dir: Path) -> None:
    from sklearn.decomposition import PCA

    z = np.load(out_dir / "activations.npz", allow_pickle=True)
    acts, role, layers = z["acts"], z["role"].astype(str), z["layers"].tolist()
    names, rvecs = _role_vectors(acts, role)      # (R, n_layers, d)
    ri = {n: i for i, n in enumerate(names)}
    role_of = {n: g for n, g in zip(role, z["group"].astype(str))}

    report = {"model_id": args.model_id, "layers": layers,
              "roles": names, "n_rollouts": int(len(role)),
              "n_rollouts_per_role": {n: int((role == n).sum()) for n in names},
              "judge_filter": False, "decoding": "greedy", "per_layer": {}}

    for li, L in enumerate(layers):
        V = rvecs[:, li, :].astype(np.float64)               # (R, d)
        default = V[ri[DEFAULT_ROLE]]
        non_def = np.array([i for n, i in ri.items() if n != DEFAULT_ROLE])
        axis = default - V[non_def].mean(0)                  # Assistant Axis
        axis_u = axis / (np.linalg.norm(axis) + 1e-8)

        # PCA on standardised (mean-subtracted) role vectors.
        Vc = V - V.mean(0)
        k = min(10, len(names) - 1)
        pca = PCA(n_components=k).fit(Vc)
        pc1 = pca.components_[0]
        # orient PC1 toward the Assistant end
        if float(pc1 @ axis_u) < 0:
            pc1 = -pc1
        cos_pc1_axis = float(pc1 @ axis_u)

        # Projections of every role onto the axis and onto PC1.
        proj_axis = {n: float(V[ri[n]] @ axis_u) for n in names}
        loads_pc1 = Vc @ pc1
        # default's relative position along PC1 in [0,1] (0/1 = the two extremes)
        lo, hi = loads_pc1.min(), loads_pc1.max()
        d_pos = float((loads_pc1[ri[DEFAULT_ROLE]] - lo) / (hi - lo + 1e-12))

        order = sorted(names, key=lambda n: proj_axis[n])
        report["per_layer"][str(L)] = {
            "cos_pc1_axis": round(cos_pc1_axis, 4),
            "pc1_var_explained": round(float(pca.explained_variance_ratio_[0]), 4),
            "n_pcs_70pct": int(np.searchsorted(
                np.cumsum(pca.explained_variance_ratio_), 0.70) + 1),
            "default_pc1_relpos": round(d_pos, 4),
            "axis_norm": round(float(np.linalg.norm(axis)), 3),
            "proj_axis_sorted": [
                {"role": n, "group": role_of.get(n, "?"),
                 "proj": round(proj_axis[n], 3)} for n in order],
        }
        # persist the axis + role vectors for P2 (localization / drift)
        np.savez_compressed(
            out_dir / f"axis_L{L}.npz",
            axis=axis.astype(np.float32), axis_unit=axis_u.astype(np.float32),
            role_vectors=V.astype(np.float32), roles=np.array(names))

    (out_dir / "axis_report.json").write_text(json.dumps(report, indent=2))
    _print_report(report)


def _print_report(report: dict) -> None:
    print("\n=== Assistant Axis replication (Qwen3.5-4B, reduced scale) ===")
    print(f"roles={len(report['roles'])}  rollouts={report['n_rollouts']}  "
          f"judge_filter={report['judge_filter']}  decoding={report['decoding']}")
    for L, r in report["per_layer"].items():
        print(f"\n--- layer {L} ---")
        print(f"  cos(PC1, AssistantAxis) = {r['cos_pc1_axis']:+.3f}   "
              f"(paper: >0.71 at middle layer)")
        print(f"  PC1 var explained = {r['pc1_var_explained']:.3f}   "
              f"dims for 70% = {r['n_pcs_70pct']}")
        print(f"  default position on PC1 in [0,1] = {r['default_pc1_relpos']:.3f} "
              f"(paper: ~0/1, an extreme)")
        rows = r["proj_axis_sorted"]
        print("  role projection onto Assistant Axis (low=fantastical -> high=assistant):")
        for row in rows[:5] + [{"role": "...", "group": "", "proj": ""}] + rows[-6:]:
            p = row["proj"]
            ps = f"{p:+.3f}" if isinstance(p, float) else p
            print(f"      {ps:>7}  {row['group']:<11} {row['role']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    out_dir = Path(args.output_dir) / (
        f"{model_tag(args.model_id)}_{args.roles}"
        f"_q{args.n_questions}_sp{args.n_system_prompts}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output dir: %s", out_dir)

    if args.stage in ("all", "generate"):
        stage_generate(args, out_dir)
    if args.stage in ("all", "analyze"):
        stage_analyze(args, out_dir)


if __name__ == "__main__":
    main()
