"""Does EP read persona *state*, or only what the text is about?

The role negative (``docs/experiments/PLAN_ROLE_QWEN3_4B.md`` §10b) established the
general law: EP resolves what dominates local geometry — usually content — and is
blind to perturbations far below the calibration threshold. The persona result
(``ASSISTANT_AXIS_EP.md``) never controlled for content: a fantastical role and the
default Assistant answer the same question with wildly different *text*, so a
partition that only indexes topic would reproduce the whole Fig 3/4 result.

This module runs the control. Content is held constant and only the persona state
varies, across four conditions of increasing content leak:

  A  forced      identical response tokens under every state's system prompt.
                 Persona-as-latent-state with *zero* expression — the strict gate.
                 Pairs are byte-identical, so the paired-displacement-vs-threshold
                 measurement from the role experiment applies directly.
  B  neutral     each state freely answers the same neutral factual probe. Content
                 matched at the question level; only style can differ.
  C  expressive  each state answers persona-revealing extraction questions. The
                 known-positive regime (this is what Fig 3/4 measured) — the ceiling.
  D  drift       a fixed neutral probe injected after k turns of the scripted drift
                 conversations. Content of the probe reply is constant by
                 construction; the only variable is the conversation history.

Readouts per condition: EP cell assignment on each supplied dictionary, the flip
rate against the default state, NMI(cell; state) vs NMI(cell; content), and — as
the incumbent baseline the critique says EP must beat — the linear Assistant Axis
projection's AUROC for the same separation.

Example:
    python -m experiments.persona.differentiator --stage generate
    python -m experiments.persona.differentiator --stage analyze \
        --dicts artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile,\
artifacts/runs/persona_dict/qwen3_5-4b-persona-mean_L27_p20p0
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from qwen_ep.adapter import QwenModel, model_tag

from .data import build_rollout_specs, role_system_prompts, spectrum_group
from .drift import SCENARIOS

log = logging.getLogger("experiments.persona.differentiator")

RUN_ROOT = Path("artifacts/runs/persona_differentiator")

# States spanning the axis: 4 Assistant-ish, 1 human, 4 fantastical. Two system
# prompts each, so within-state variance is visible next to between-state.
STATE_ROLES = ["default", "counselor", "analyst", "editor",
               "comedian", "oracle", "demon", "ghost", "aberration"]
N_SYS_PER_STATE = 2

# Neutral probes: factual, closed, no persona affordance. A character has nothing
# to say here that an assistant would not — which is the point.
NEUTRAL_PROBES = [
    "What is the boiling point of water at sea level in degrees Celsius?",
    "What does the acronym HTTP stand for?",
    "In what year did the Apollo 11 mission land on the Moon?",
    "What is the chemical symbol for iron?",
    "How do you convert kilometres to miles?",
    "What is the capital city of Australia?",
    "What is the derivative of x squared with respect to x?",
    "How many bytes are in a kilobyte?",
    "Which planet in the solar system is the largest?",
    "What is the freezing point of water in degrees Fahrenheit?",
    "Name the three primary additive colours of light.",
    "How many minutes are there in a full day?",
]

# The fixed probe injected into a drifted conversation (condition D).
DRIFT_PROBES = NEUTRAL_PROBES[:4]
DRIFT_DEPTHS = [0, 2, 4, 6, 8]          # user turns of history before the probe


def _states(model_name: str, roles: list[str] | None = None) -> list[dict]:
    out = []
    for role in (roles or STATE_ROLES):
        for i, sp in enumerate(role_system_prompts(role, model_name)[:N_SYS_PER_STATE]):
            out.append({"role": role, "group": spectrum_group(role),
                        "sys_idx": i, "system": sp})
    return out


# --------------------------------------------------------------------- generate

def _gen(qwen: QwenModel, systems, users, max_new_tokens, batch_size):
    prompts = [qwen.format_chat(u, system=s) for s, u in zip(systems, users)]
    return qwen.generate(prompts, max_new_tokens=max_new_tokens,
                         batch_size=batch_size)


def stage_generate(args, run: Path) -> None:
    qwen = QwenModel(args.model_id, device=args.device)
    roles = ([r.strip() for r in args.states.split(",") if r.strip()]
             if args.states else STATE_ROLES)
    states = _states(args.model_id.split("/")[-1], roles)
    probes = NEUTRAL_PROBES[:args.n_probes]
    layers = [int(x) for x in args.layers.split(",")]
    log.info("%d states x {A,B,C} conditions, layers=%s", len(states), layers)

    expressive = [s["question"] for s in
                  build_rollout_specs(["default"], args.n_questions, 1)]

    rows: list[dict] = []          # one per rollout, carries every label
    t0 = time.time()

    # --- canonical neutral answers, generated once by the default state -------
    d0 = states[0]
    canon = _gen(qwen, [d0["system"]] * len(probes), probes,
                 args.max_new_tokens, args.batch_size)
    canon = [c.strip() for c in canon]
    log.info("canonical answers generated (%.0fs)", time.time() - t0)

    # --- A: forced-identical content -----------------------------------------
    for st in states:
        for qi, (q, a) in enumerate(zip(probes, canon)):
            rows.append({"cond": "A", "role": st["role"], "group": st["group"],
                         "sys_idx": st["sys_idx"], "system": st["system"],
                         "content_id": f"neutral{qi}", "question": q,
                         "response": a})

    # --- B: free answers to the same neutral probes --------------------------
    sysl = [st["system"] for st in states for _ in probes]
    usrl = [q for _ in states for q in probes]
    reps = _gen(qwen, sysl, usrl, args.max_new_tokens, args.batch_size)
    log.info("B generated (%.0fs)", time.time() - t0)
    i = 0
    for st in states:
        for qi, q in enumerate(probes):
            rows.append({"cond": "B", "role": st["role"], "group": st["group"],
                         "sys_idx": st["sys_idx"], "system": st["system"],
                         "content_id": f"neutral{qi}", "question": q,
                         "response": reps[i].strip()})
            i += 1

    # --- C: persona-expressive questions (the known-positive ceiling) --------
    sysl = [st["system"] for st in states for _ in expressive]
    usrl = [q for _ in states for q in expressive]
    reps = _gen(qwen, sysl, usrl, args.max_new_tokens, args.batch_size)
    log.info("C generated (%.0fs)", time.time() - t0)
    i = 0
    for st in states:
        for qi, q in enumerate(expressive):
            rows.append({"cond": "C", "role": st["role"], "group": st["group"],
                         "sys_idx": st["sys_idx"], "system": st["system"],
                         "content_id": f"express{qi}", "question": q,
                         "response": reps[i].strip()})
            i += 1

    # --- activations for A/B/C ------------------------------------------------
    acts = qwen.mean_response_activations(
        [r["system"] for r in rows], [r["question"] for r in rows],
        [r["response"] for r in rows], layers, batch_size=args.batch_size)
    log.info("A/B/C activations %s (%.0fs)", acts.shape, time.time() - t0)

    # --- D: fixed probe injected into the drift conversations ----------------
    # The scenarios advance turn-by-turn in lockstep so each turn is one batch;
    # the probe injections have no ordering constraint at all, so they are one
    # flat batch across every (scenario, depth, probe).
    scens = list(SCENARIOS)
    convos = {s: [] for s in scens}
    for t in range(max(len(SCENARIOS[s]) for s in scens)):
        active = [s for s in scens if t < len(SCENARIOS[s])]
        for s in active:
            convos[s].append({"role": "user", "content": SCENARIOS[s][t]})
        replies = qwen.generate(
            [qwen.format_conversation(convos[s], add_generation_prompt=True)
             for s in active],
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        for s, r in zip(active, replies):
            convos[s].append({"role": "assistant", "content": r.strip()})
    log.info("D: %d conversations regenerated (%.0fs)", len(scens), time.time() - t0)

    d_rows, d_msgs = [], []
    for scen in scens:
        for depth in [int(x) for x in args.drift_depths.split(",")]:
            if depth > len(SCENARIOS[scen]):
                continue
            hist = convos[scen][: 2 * depth]
            for pi, probe in enumerate(DRIFT_PROBES[:max(1, args.n_probes // 3)]):
                d_rows.append({"cond": "D", "scenario": scen, "depth": depth,
                               "content_id": f"probe{pi}", "question": probe})
                d_msgs.append(hist + [{"role": "user", "content": probe}])
    replies = qwen.generate(
        [qwen.format_conversation(m, add_generation_prompt=True) for m in d_msgs],
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    d_convos = []
    for r, msgs, reply in zip(d_rows, d_msgs, replies):
        r["response"] = reply.strip()
        d_convos.append(msgs + [{"role": "assistant", "content": r["response"]}])
    log.info("D: %d probe injections generated (%.0fs)", len(d_rows), time.time() - t0)

    d_acts = qwen.mean_last_turn_activations(d_convos, layers,
                                             batch_size=args.batch_size)
    log.info("D activations %s (%.0fs)", d_acts.shape, time.time() - t0)

    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "activations.npz",
        acts=acts.astype(np.float32), d_acts=d_acts.astype(np.float32),
        layers=np.array(layers),
        cond=np.array([r["cond"] for r in rows]),
        role=np.array([r["role"] for r in rows]),
        group=np.array([r["group"] for r in rows]),
        sys_idx=np.array([r["sys_idx"] for r in rows]),
        content_id=np.array([r["content_id"] for r in rows]),
        d_scenario=np.array([r["scenario"] for r in d_rows]),
        d_depth=np.array([r["depth"] for r in d_rows]),
        d_content_id=np.array([r["content_id"] for r in d_rows]),
    )
    with (run / "rollouts.jsonl").open("w") as f:
        for r in rows + d_rows:
            f.write(json.dumps(r) + "\n")
    log.info("wrote %s (%d A/B/C + %d D rollouts, %.0fs total)",
             run, len(rows), len(d_rows), time.time() - t0)


# ---------------------------------------------------------------------- metrics

def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic mean normalisation)."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    n = len(ia)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia, ib), 1.0)
    P = C / n
    pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (pa @ pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return 0.0 if ha + hb == 0 else float(2 * mi / (ha + hb))


def _auroc(score: np.ndarray, pos: np.ndarray) -> float:
    """AUROC of ``score`` for the boolean label ``pos``.

    Tied scores take the average rank — without that a tie is silently resolved
    in whatever order argsort happened to produce, which inflates or deflates the
    number (cell-distance scores can tie exactly when rollouts share an exemplar).
    """
    if pos.all() or not pos.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _dirs(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    Xc = X - center
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


def _region_labels(dic, compose_npz: str, layer: int) -> dict[int, str]:
    """Name every region by the majority group of the P1 role-means that live in
    it — the "nameplate" the differentiator claim rests on. Regions no role-mean
    occupies get no name, which is itself a result worth counting.
    """
    z = np.load(compose_npz, allow_pickle=True)
    layers = z["layers"].tolist()
    Xr = z["acts"][:, layers.index(layer), :].astype(np.float32)
    grp = z["group"].astype(str)
    cell, _ = dic.assign(Xr)
    out = {}
    for c in np.unique(cell):
        out[int(c)] = Counter(grp[cell == c].tolist()).most_common(1)[0][0]
    return out


def _nameplate(cell, group, labels: dict[int, str]) -> dict:
    """Does the region a rollout lands in carry the right persona name?"""
    pred = np.array([labels.get(int(c), "unnamed") for c in cell])
    maj = Counter(group.tolist()).most_common(1)[0][1] / len(group)
    conf = {}
    for g in sorted(set(group.tolist())):
        m = group == g
        conf[g] = {p: int((pred[m] == p).sum())
                   for p in sorted(set(pred[m].tolist()))}
    return {
        "accuracy": round(float((pred == group).mean()), 3),
        "majority_baseline": round(float(maj), 3),
        "frac_unnamed_region": round(float((pred == "unnamed").mean()), 3),
        "confusion_true_to_named": conf,
    }


def _condition_report(dic, X, role, group, content_id, sys_idx, axis_proj,
                      labels=None) -> dict:
    """All the state-vs-content readouts for one condition on one dictionary."""
    cell, _ = dic.assign(X)
    center = np.asarray(dic.center, dtype=np.float32)
    E = np.stack([p.exemplar_direction for p in dic.partitions]).astype(np.float32)
    dirs = _dirs(X, center)

    # anchor = modal cell of the default-state rollouts in this condition
    anchor = int(Counter(cell[role == "default"].tolist()).most_common(1)[0][0])
    d_anchor = 1.0 - dirs @ E[anchor]

    # flip rate: same content, default state vs this state
    flips, n_pairs = 0, 0
    for c in np.unique(content_id):
        m = content_id == c
        base = cell[m & (role == "default")]
        if not len(base):
            continue
        b0 = int(base[0])
        others = cell[m & (role != "default")]
        flips += int((others != b0).sum())
        n_pairs += len(others)

    fant = group == "fantastical"
    keep = fant | (group == "assistant")
    rep = {
        "K": len(dic.partitions),
        "threshold": round(float(dic.threshold), 4),
        "anchor_cell": anchor,
        "n_distinct_cells": int(len(set(cell.tolist()))),
        "flip_rate_vs_default": round(flips / n_pairs, 3) if n_pairs else None,
        "nmi_cell_state": round(_nmi(cell, role), 3),
        "nmi_cell_content": round(_nmi(cell, content_id), 3),
        "auroc_ep_dist_anchor": round(_auroc(d_anchor[keep], fant[keep]), 3),
        "auroc_axis_proj": (round(_auroc(-axis_proj[keep], fant[keep]), 3)
                            if axis_proj is not None else None),
        "cells_per_state": {r: int(len(set(cell[role == r].tolist())))
                            for r in sorted(set(role.tolist()))},
        "cells_per_content": {str(c): int(len(set(cell[content_id == c].tolist())))
                              for c in sorted(set(content_id.tolist()))[:4]},
        "nmi_cell_sysprompt": round(_nmi(cell, sys_idx.astype(str)), 3),
    }
    if labels is not None:
        rep["nameplate"] = _nameplate(cell, group, labels)
    return rep, cell, dirs


def _paired_displacement(dirs, role, content_id, sys_idx, threshold) -> dict:
    """Condition A only: how far the state moves a byte-identical activation,
    in the units the cells are built in (the role experiment's gate)."""
    base_key = {}
    for i in range(len(role)):
        if role[i] == "default" and sys_idx[i] == 0:
            base_key[content_id[i]] = i
    d = []
    for i in range(len(role)):
        if role[i] == "default" and sys_idx[i] == 0:
            continue
        j = base_key.get(content_id[i])
        if j is not None:
            d.append(1.0 - float(dirs[i] @ dirs[j]))
    d = np.array(d)
    return {
        "n_pairs": int(len(d)),
        "mean": round(float(d.mean()), 5),
        "median": round(float(np.median(d)), 5),
        "p90": round(float(np.percentile(d, 90)), 5),
        "max": round(float(d.max()), 5),
        "threshold": round(float(threshold), 4),
        "ratio_to_threshold": round(float(d.mean() / threshold), 4),
        "frac_beyond_threshold": round(float((d > threshold).mean()), 4),
    }


def stage_analyze(args, run: Path) -> None:
    z = np.load(run / "activations.npz", allow_pickle=True)
    layers = z["layers"].tolist()
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in captured {layers}")
    li = layers.index(args.layer)
    X_all = z["acts"][:, li, :].astype(np.float32)
    cond = z["cond"].astype(str)
    role = z["role"].astype(str)
    group = z["group"].astype(str)
    content_id = z["content_id"].astype(str)
    sys_idx = z["sys_idx"].astype(int)

    axis_proj_all = None
    if args.axis:
        az = np.load(args.axis, allow_pickle=True)
        axis_proj_all = X_all @ az["axis_unit"].astype(np.float32)

    report = {"model": args.model_id, "layer": args.layer,
              "n_states": len(set(zip(role.tolist(), sys_idx.tolist()))),
              "conditions": {}}

    for dpath in [p.strip() for p in args.dicts.split(",") if p.strip()]:
        with (Path(dpath) / "dictionary.pkl").open("rb") as f:
            dic = pickle.load(f)
        name = Path(dpath).name
        log.info("dict %s  K=%d  threshold=%.4f", name, len(dic.partitions),
                 float(dic.threshold))
        labels = (_region_labels(dic, args.compose, args.layer)
                  if args.compose else None)

        for c in ["A", "B", "C"]:
            m = cond == c
            rep, cell, dirs = _condition_report(
                dic, X_all[m], role[m], group[m], content_id[m], sys_idx[m],
                None if axis_proj_all is None else axis_proj_all[m], labels)
            if c == "A":
                rep["paired_displacement"] = _paired_displacement(
                    dirs, role[m], content_id[m], sys_idx[m], float(dic.threshold))
            report["conditions"].setdefault(c, {})[name] = rep

        # --- D: fixed probe after k turns of drift ---------------------------
        Xd = z["d_acts"][:, li, :].astype(np.float32)
        scen = z["d_scenario"].astype(str)
        depth = z["d_depth"].astype(int)
        dcid = z["d_content_id"].astype(str)
        cell_d, _ = dic.assign(Xd)
        base = {}
        for i in range(len(cell_d)):
            if depth[i] == 0:
                base[dcid[i]] = int(cell_d[i])
        flips = sum(1 for i in range(len(cell_d))
                    if depth[i] > 0 and int(cell_d[i]) != base.get(dcid[i]))
        n_d = int((depth > 0).sum())
        report["conditions"].setdefault("D", {})[name] = {
            "K": len(dic.partitions),
            "n_distinct_cells": int(len(set(cell_d.tolist()))),
            "flip_rate_vs_no_history": round(flips / n_d, 3) if n_d else None,
            "nmi_cell_scenario": round(_nmi(cell_d, scen), 3),
            "nmi_cell_probe": round(_nmi(cell_d, dcid), 3),
            "nmi_cell_depth": round(_nmi(cell_d, depth.astype(str)), 3),
            "cell_by_scenario_depth": {
                f"{s}@{d}": sorted({int(cell_d[i]) for i in range(len(cell_d))
                                    if scen[i] == s and depth[i] == d})
                for s in sorted(set(scen.tolist()))
                for d in sorted(set(depth.tolist()))},
        }

    (run / f"differentiator_L{args.layer}.json").write_text(
        json.dumps(report, indent=2))
    _print(report)


def _print(r: dict) -> None:
    names = {"A": "forced  (identical response tokens)",
             "B": "neutral (free answer, same question)",
             "C": "express (persona-revealing questions)",
             "D": "drift   (fixed probe after k turns)"}
    print(f"\n=== Does EP read persona state or content?  "
          f"{r['model']} L{r['layer']}, {r['n_states']} states ===")
    for c in ["A", "B", "C"]:
        if c not in r["conditions"]:
            continue
        print(f"\n-- condition {c}: {names[c]}")
        print("   dict                                          K   flip   "
              "NMI(state) NMI(content)  AUROC-EP  AUROC-axis")
        for name, d in r["conditions"][c].items():
            print(f"   {name[:44]:<44} {d['K']:>5}  {str(d['flip_rate_vs_default']):>5}   "
                  f"{d['nmi_cell_state']:>7.3f}  {d['nmi_cell_content']:>10.3f}"
                  f"  {str(d['auroc_ep_dist_anchor']):>8}  {str(d['auroc_axis_proj']):>10}")
            if "nameplate" in d:
                p = d["nameplate"]
                print(f"      nameplate: names the group right {p['accuracy']:.3f} "
                      f"(majority baseline {p['majority_baseline']:.3f}, "
                      f"unnamed region {p['frac_unnamed_region']:.3f})")
            if "paired_displacement" in d:
                p = d["paired_displacement"]
                print(f"      paired displacement: mean {p['mean']:.5f} vs threshold "
                      f"{p['threshold']:.4f}  ratio {p['ratio_to_threshold']:.4f}  "
                      f"beyond {p['frac_beyond_threshold']:.4f}  (n={p['n_pairs']})")
    if "D" in r["conditions"]:
        print(f"\n-- condition D: {names['D']}")
        for name, d in r["conditions"]["D"].items():
            print(f"   {name[:44]:<44} K={d['K']:<6} flip vs no-history="
                  f"{d['flip_rate_vs_no_history']}  NMI(scenario)={d['nmi_cell_scenario']:.3f}"
                  f"  NMI(probe)={d['nmi_cell_probe']:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--stage", default="generate", choices=["generate", "analyze"])
    ap.add_argument("--layers", default="16,27")
    ap.add_argument("--layer", type=int, default=27, help="analyze: layer to use")
    ap.add_argument("--dicts", default="", help="analyze: comma-separated run dirs")
    ap.add_argument("--axis", default="", help="analyze: axis_L{layer}.npz from P1")
    ap.add_argument("--compose", default="", help="analyze: P1 activations.npz, used to name each region by its role-mean composition")
    ap.add_argument("--n-questions", type=int, default=12)
    ap.add_argument("--n-probes", type=int, default=12)
    ap.add_argument("--states", default="", help="comma-separated role subset")
    ap.add_argument("--drift-depths", default="0,2,4,6,8")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    run = RUN_ROOT / (args.run_name or f"{model_tag(args.model_id)}_q{args.n_questions}")
    if args.stage == "generate":
        stage_generate(args, run)
    else:
        stage_analyze(args, run)


if __name__ == "__main__":
    main()
