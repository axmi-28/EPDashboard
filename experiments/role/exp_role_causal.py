"""Tier 3: is the role subspace causally necessary for instruction framing?

Forked from ``exemplar-partitioning/scripts/exp_behavioral.py``, keeping its
ablation machinery (QR of stacked region directions, three bases, matched null)
and replacing three things: the prompt set becomes verifiable-constraint prompts,
the scorer becomes exact string logic instead of the refusal substring list, and
region selection is by role polarity λ instead of member refusal rate.

Two arms, and the second is the one that matters:

**(a) Head-to-head ablation.** Project off the m-dim EP role subspace, versus the
linear probe direction (the paper's own instrument), versus a dimension-matched
null subspace drawn from λ≈0 regions. Fair fight on their turf.

**(b) Region-gated ablation.** A projection is unconditional — it hits every
token at that layer. A hard partition can intervene *only* when the token's
region is λ-polarized. Sweeping the λ threshold traces a selectivity frontier:
effect on constraint-following against collateral damage measured as perplexity
on held-out text. If gating matches global ablation's effect while touching a
fraction of tokens, that is a result a direction cannot produce — and it holds
even if role turns out to be one coherent direction.

Expect ablation to work and positive steering not to: on refusal, α ∈ {50,100}
were indistinguishable from baseline and α=400 degenerated into token loops.
A steering null is not evidence against localization.

Run (after exp_role.py has written role_subspace.npz):
    python -m experiments.role.exp_role_causal --subspace artifacts/runs/role/.../role_subspace.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from experiments.role import constraints as K
from experiments.role import model_io as MIO

logger = logging.getLogger(__name__)


def _format_user_turn(tokenizer, text: str, thinking: bool) -> str:
    """Render one user turn, with Qwen3's reasoning channel closed by default.

    This is not a stylistic choice. With ``add_generation_prompt=True`` and
    thinking enabled, Qwen3 opens a ``<think>`` block and spends the whole
    generation budget inside it — at 48 new tokens nothing reaches the answer, so
    every constraint scores as violated *at baseline* and the ablation Δ is
    identically zero for a reason that has nothing to do with role.
    ``enable_thinking=False`` prefills an empty ``<think>\\n\\n</think>`` so
    generation begins at the answer.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        # Template without the kwarg: fall back and rely on _strip_think plus a
        # larger --max-new-tokens.
        logger.warning("chat template does not accept enable_thinking; "
                       "raise --max-new-tokens or answers will be truncated "
                       "inside the reasoning block")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )


def _generate(model, text: str, max_new_tokens: int, hook_pair=None,
              thinking: bool = False) -> str:
    """Greedy generation with the user turn formatted by the real template."""
    import torch

    formatted = _format_user_turn(model.tokenizer, text, thinking)
    tokens = model.to_tokens(formatted, prepend_bos=True)
    if hook_pair is not None:
        model.reset_hooks()
        model.add_hook(hook_pair[0], hook_pair[1], "fwd")
    try:
        with torch.no_grad():
            out = model.generate(
                tokens, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=0.0, verbose=False,
            )
        new = out[0, tokens.shape[1]:]
        return model.tokenizer.decode(new, skip_special_tokens=True)
    finally:
        model.reset_hooks()


def _strip_think(text: str) -> str:
    """Drop Qwen3's reasoning block before scoring.

    The template opens every assistant turn with ``<think>``, and reasoning text
    is not the answer — scoring it would count "let me think, one word..." as a
    multi-word reply and make every constraint look violated at baseline.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def _project_off_hook_factory(basis, centre):
    import torch

    def hook(act, hook_obj, _b=basis, _c=centre):
        shape = act.shape
        x = act.float().reshape(-1, shape[-1]) - _c
        x = x - (x @ _b) @ _b.T
        return (x + _c).reshape(shape).to(act.dtype)

    return hook


def _gated_project_off_hook_factory(basis, centre, gate_dirs, gate_thresh):
    """Project only at positions whose nearest gate direction is close enough.

    The gate is EP's own assignment rule — centre, normalize, nearest exemplar by
    cosine — restricted to the λ-polarized exemplars. A position is intervened on
    only if its direction is within ``gate_thresh`` cosine distance of one of
    them, which is the discrete handle a global projection does not have.
    """
    import torch

    def hook(act, hook_obj, _b=basis, _c=centre, _g=gate_dirs, _t=gate_thresh):
        shape = act.shape
        x = act.float().reshape(-1, shape[-1])
        v = x - _c
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dirs = v / n
        best = (dirs @ _g.T).max(dim=-1).values      # cosine similarity
        mask = (1.0 - best) <= _t
        if mask.any():
            v = v.clone()
            sel = v[mask]
            v[mask] = sel - (sel @ _b) @ _b.T
        return (v + _c).reshape(shape).to(act.dtype), mask.float().mean()

    def wrapped(act, hook_obj):
        out, frac = hook(act, hook_obj)
        wrapped.fraction_touched.append(float(frac))
        return out

    wrapped.fraction_touched = []
    return wrapped


def _perplexity(model, texts: list[str], hook_pair=None) -> float:
    """Mean token NLL on held-out text — the collateral-damage axis.

    Uses plain (untagged) pretraining text, so it measures what the intervention
    costs in general language modelling rather than on the constraint prompts.
    """
    import torch

    total_nll, total_tok = 0.0, 0
    if hook_pair is not None:
        model.reset_hooks()
        model.add_hook(hook_pair[0], hook_pair[1], "fwd")
    try:
        with torch.no_grad():
            for t in texts:
                tokens = model.to_tokens(t, prepend_bos=True)
                logits = model(tokens)
                lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
                tgt = tokens[0, 1:]
                total_nll += float(-lp[torch.arange(len(tgt)), tgt].sum())
                total_tok += int(len(tgt))
    finally:
        model.reset_hooks()
    return float(np.exp(total_nll / max(total_tok, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--model-short", default="Qwen3-4B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--subspace", type=Path, required=True,
                    help="role_subspace.npz written by role.exp_role")
    ap.add_argument("--dictionary", type=Path, default=None,
                    help="Pickled dictionary; defaults to the .pkl beside "
                         "--subspace. Needed for the null and gate regions.")
    ap.add_argument("--calibration-centre", type=Path, default=None,
                    help="npz with the calibration centre; defaults to reading "
                         "it back from the EP calibration cache.")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--thinking", action="store_true",
                    help="Leave Qwen3's reasoning channel open. Off by default: "
                         "with thinking on, the generation budget is consumed "
                         "inside <think> and every constraint fails at baseline. "
                         "If you turn this on, raise --max-new-tokens to ~512.")
    ap.add_argument("--subspace-dims", default="1,4,16,32,64",
                    help="m values to sweep for the role subspace.")
    ap.add_argument("--gate-thresholds", default="0.1,0.2,0.3,0.5",
                    help="Cosine-distance gates for the region-gated arm.")
    ap.add_argument("--n-perplexity-texts", type=int, default=40)
    ap.add_argument("--positive-alphas", default="",
                    help="Optional positive steering sweep; expect a null.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", force=True)
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output_dir or args.subspace.parent / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = np.load(args.subspace, allow_pickle=True)
    lam = blob["lam"]
    axis = blob["axis"]
    top_user = blob["top_user"]
    top_asst = blob["top_assistant"]
    counts = blob["counts_train"]
    logger.info("subspace file: %d regions, %d top-user, %d top-assistant",
                len(lam), len(top_user), len(top_asst))

    dict_path = args.dictionary
    if dict_path is None:
        cands = sorted(args.subspace.parent.glob("*.pkl"))
        if not cands:
            raise SystemExit(
                f"no dictionary pickle beside {args.subspace}; pass --dictionary"
            )
        dict_path = cands[0]
    with dict_path.open("rb") as f:
        dictionary = pickle.load(f)
    E = MIO.exemplar_matrix(dictionary)
    logger.info("dictionary: %s (%d regions)", dict_path.name, len(dictionary))

    model = MIO.load_model(args.model, device=args.device, dtype=args.dtype)
    hook = MIO.hook_name(args.layer)

    # The centre must be the calibration centre EP used, not the corpus mean of
    # whatever we happen to run now — the partition is defined relative to it.
    if args.calibration_centre is not None:
        centre_np = np.load(args.calibration_centre)["center"]
    else:
        from ep.discovery.calibration import cache_dir
        hits = sorted(cache_dir().glob("*role*.npz"))
        if not hits:
            raise SystemExit(
                "could not find the role calibration cache; pass "
                "--calibration-centre explicitly. Using a different centre "
                "silently redefines every region."
            )
        centre_np = np.load(hits[0])["center"]
        logger.info("calibration centre from %s", hits[0].name)
    centre = torch.tensor(centre_np, dtype=torch.float32, device=args.device)

    prompts = K.build_prompts(n=args.n_prompts, seed=args.seed)
    logger.info("%d constraint prompts across %d families",
                len(prompts), len(K.CONSTRAINTS))

    def _run(hook_pair=None, label="") -> dict:
        t0 = time.time()
        gens = [
            _strip_think(
                _generate(model, p.text, args.max_new_tokens, hook_pair,
                      thinking=args.thinking)
            )
            for p in prompts
        ]
        res = K.score_all(prompts, gens)
        logger.info("%-38s follow_rate=%.3f (%.0fs)", label, res["rate"],
                    time.time() - t0)
        return {"rate": res["rate"], "per_family": res["per_family"],
                "examples": [
                    {"prompt": p.text, "generation": g[:300], "ok": ok}
                    for p, g, ok in list(zip(prompts, gens, res["flags"]))[:12]
                ]}

    # --- baseline ---
    baseline = _run(None, "baseline (no hook)")
    from experiments.role import corpus as C
    ppl_texts = C.stream_contents(
        model.tokenizer, n_docs=args.n_perplexity_texts, n_content=128,
        dataset="c4", seed=args.seed + 999,
    )
    ppl_base = _perplexity(model, ppl_texts)
    logger.info("baseline perplexity on held-out C4 = %.3f", ppl_base)

    results: dict = {
        "baseline": baseline, "baseline_perplexity": ppl_base, "arms": {},
    }

    # --- arm (a): head-to-head subspace ablation ---
    dims = [int(v) for v in args.subspace_dims.split(",") if v.strip()]
    ranked = [int(i) for i in np.argsort(-np.nan_to_num(lam, nan=-1e9))]

    def _basis_from(pids, m):
        d = E[pids[:m]].astype(np.float32)
        Q, _ = np.linalg.qr(d.T)
        return torch.tensor(Q, dtype=torch.float32, device=args.device)

    # Dimension-matched null: regions with |λ| smallest, above the same member
    # floor. Without this, any effect could be "any dense direction at L18".
    lam_abs = np.abs(np.nan_to_num(lam, nan=1e9))
    null_pool = [int(i) for i in np.argsort(lam_abs)
                 if counts[i] >= 5 and np.isfinite(lam[i])]

    for m in dims:
        if m > len(ranked):
            continue
        arm = {}
        basis = _basis_from(ranked, m)
        arm["ep_role_subspace"] = _run(
            (hook, _project_off_hook_factory(basis, centre)),
            f"ablate EP role subspace m={m}",
        )
        arm["ep_role_subspace"]["perplexity"] = _perplexity(
            model, ppl_texts, (hook, _project_off_hook_factory(basis, centre)),
        )
        if len(null_pool) >= m:
            nb = _basis_from(np.array(null_pool), m)
            arm["null_subspace"] = _run(
                (hook, _project_off_hook_factory(nb, centre)),
                f"ablate matched null subspace m={m}",
            )
            arm["null_subspace"]["perplexity"] = _perplexity(
                model, ppl_texts, (hook, _project_off_hook_factory(nb, centre)),
            )
        results["arms"][f"m{m}"] = arm

    # The paper's own instrument: their probe direction, same protocol.
    probe_npz = args.subspace.parent.parent.parent / "probe"
    probe_files = sorted(probe_npz.rglob("probe_directions_L*.npz")) \
        if probe_npz.exists() else []
    if probe_files:
        pf = np.load(probe_files[0], allow_pickle=True)
        coef = pf["coef"]
        Q, _ = np.linalg.qr(coef.T.astype(np.float32))
        pb = torch.tensor(Q, dtype=torch.float32, device=args.device)
        results["arms"]["probe_direction"] = _run(
            (hook, _project_off_hook_factory(pb, centre)),
            f"ablate probe subspace ({Q.shape[1]}d)",
        )
        results["arms"]["probe_direction"]["perplexity"] = _perplexity(
            model, ppl_texts, (hook, _project_off_hook_factory(pb, centre)),
        )
        results["arms"]["probe_direction"]["dims"] = int(Q.shape[1])
        logger.info("probe arm used %s", probe_files[0].name)
    else:
        logger.warning("no probe_directions_L*.npz found — the head-to-head "
                       "against the paper's instrument is missing")

    # The single role axis, for comparison with the m-dim subspace.
    ax = torch.tensor(axis[:, None] / max(np.linalg.norm(axis), 1e-8),
                      dtype=torch.float32, device=args.device)
    results["arms"]["role_axis_1d"] = _run(
        (hook, _project_off_hook_factory(ax, centre)), "ablate role axis (1d)",
    )

    # --- arm (b): region-gated ablation, the selectivity frontier ---
    m_gate = min(32, len(ranked))
    basis = _basis_from(ranked, m_gate)
    gate_dirs = torch.tensor(
        E[ranked[:m_gate]].astype(np.float32), device=args.device,
    )
    frontier = []
    for thresh in [float(v) for v in args.gate_thresholds.split(",") if v.strip()]:
        h = _gated_project_off_hook_factory(basis, centre, gate_dirs, thresh)
        res = _run((hook, h), f"gated ablation (cos dist <= {thresh})")
        touched = float(np.mean(h.fraction_touched)) if h.fraction_touched else 0.0
        h2 = _gated_project_off_hook_factory(basis, centre, gate_dirs, thresh)
        ppl = _perplexity(model, ppl_texts, (hook, h2))
        frontier.append({
            "gate_cos_distance": thresh, "follow_rate": res["rate"],
            "fraction_tokens_touched": touched, "perplexity": ppl,
            "per_family": res["per_family"],
        })
        logger.info(
            "  gate<=%.2f: follow=%.3f touched=%.3f ppl=%.3f (baseline "
            "follow=%.3f ppl=%.3f)", thresh, res["rate"], touched, ppl,
            baseline["rate"], ppl_base,
        )
    results["gated_frontier"] = frontier
    results["gated_config"] = {"m": m_gate, "pids": ranked[:m_gate]}

    # --- optional positive steering ---
    if args.positive_alphas.strip():
        alphas = [float(a) for a in args.positive_alphas.split(",") if a.strip()]
        steer = torch.tensor(axis, dtype=torch.float32, device=args.device)
        sweep = []
        for alpha in alphas:
            def add(act, hook_obj, _d=steer, _a=alpha):
                return act + (_a * _d).to(act.dtype)
            r = _run((hook, add), f"positive steering alpha={alpha:g}")
            sweep.append({"alpha": alpha, "follow_rate": r["rate"]})
        results["positive_steering"] = sweep

    payload = {"config": vars(args) | {"output_dir": str(out_dir)},
               "dictionary": str(dict_path), **results}
    path = out_dir / "causal.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
