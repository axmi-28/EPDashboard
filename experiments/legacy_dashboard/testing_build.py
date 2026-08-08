"""Assemble the interactive Testing-tab payload (DATA.testing) for the dashboard.

Two sub-pages, both driven by real geometry so regions are explorable:

  refusal  — the behavioral EP dictionary (harmful+benign chat prompts). Every
             region carries PCA/t-SNE coords, exact cosine k-NN, its member
             refusal rate, and the actual prompts that landed in it. The selected
             refusal region and the matched null are flagged. Plus the ablation
             sweep + a redacted chat transcript (default / mean-ablated / null).

  concept  — a browsable concept dictionary (the small p8 display dict). Regions
             carry coords + k-NN; each AxBench concept links to the region it
             selected, its held-out AUROC, and the detector's per-example scores.

Run:
    python -m experiments.legacy_dashboard.testing_build \
        --refusal-dict artifacts/runs/behavioral/qwen3_5-4b/L27_p12_seed0 \
        --behavioral-root artifacts/runs/behavioral/qwen3_5-4b \
        --concept-json artifacts/runs/concept_detect/qwen3_5-4b.json \
        --concept-dict artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile \
        --out artifacts/figures/dashboard_data/_testing.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .build import knn, layouts

CHAT_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def clean_prompt(text: str) -> str:
    """Strip Qwen chat scaffolding, return the user instruction."""
    m = CHAT_RE.search(text)
    body = m.group(1) if m else text
    return " ".join(body.strip().split())


def _round2(a: np.ndarray) -> list:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in a]


def build_refusal(dict_dir: Path, root: Path) -> dict:
    dictionary = pickle.load((dict_dir / "dictionary.pkl").open("rb"))
    loadings = json.loads((dict_dir / "loadings.json").read_text())
    ablation = json.loads((dict_dir / "ablation.json").read_text())
    parts = dictionary.partitions

    refusal_by_pid = {r["pid"]: r["refusal_rate"] for r in loadings["rows"]}
    members_by_pid = {r["pid"]: r["n_members"] for r in loadings["rows"]}
    sel_pid = ablation["top_refusal"][0]["pid"] if ablation["top_refusal"] else None
    null_pid = ablation["null_ablation"]["pid"] if ablation.get("null_ablation") else None

    dirs = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
    tsne, pca2 = layouts(dirs)
    nbrs = knn(dirs, k=8)

    regions = []
    for i, p in enumerate(parts):
        # up to 4 distinct member prompts (closest first)
        seen, prompts = set(), []
        for dist, text, pos in (p.closest_prompts or [])[:8]:
            c = clean_prompt(text)
            if c and c not in seen:
                seen.add(c)
                prompts.append({"t": c[:180], "d": round(float(dist), 3)})
            if len(prompts) >= 4:
                break
        role = ("refusal" if i == sel_pid else
                "null" if i == null_pid else None)
        regions.append({
            "i": i,
            "n": int(p.member_count),
            "nf": int(members_by_pid.get(i, 0)),   # final-position members
            "coh": round(float(p.member_coherence), 3),
            "rr": round(float(refusal_by_pid.get(i, 0.0)), 3),
            "role": role,
            "nb": [[j, s] for j, s in nbrs[i]],
            "sample": prompts,
        })

    # redacted chat transcript is already in _experiments; recompute here so the
    # Testing page is self-contained.
    def first_clause(t, n=150):
        t = " ".join(t.strip().split())
        return t[:n] + ("…" if len(t) > n else "")

    def resp(r):
        # redact the continuation whenever the model complied
        return {"refused": r["refused"],
                "text": (None if r["refused"] == 0 else first_clause(r["text"]))}

    chat = []
    show_path = dict_dir / "showcase.json"
    if show_path.exists():
        # three-way transcripts: default / region-ablated / null-ablated
        sc = json.loads(show_path.read_text())
        for it in sc["items"]:
            chat.append({
                "prompt": clean_prompt(it["prompt"])[:200],
                "baseline": resp(it["baseline"]),
                "ablated": resp(it["ablated"]),
                "null": resp(it["null"]),
            })
    else:
        for e in ablation.get("examples", []):
            chat.append({
                "prompt": clean_prompt(e["prompt"])[:200],
                "baseline": resp({"refused": e["baseline_refused"], "text": e["baseline"]}),
                "ablated": resp({"refused": e["ablated_refused"], "text": e["ablated"]}),
            })

    # ablation sweep for the selected + null (mean/exemplar)
    sweeps = {b: [{"k": s["k"], "rate": s["ablated_refusal_rate"], "delta": s["delta"]}
                  for s in sw] for b, sw in ablation["sweep_by_basis"].items()}
    null = None
    if ablation.get("null_ablation"):
        n = ablation["null_ablation"]
        null = {"pid": n["pid"],
                "by_basis": {b: {"rate": v["ablated_refusal_rate"], "delta": v["delta"]}
                             for b, v in n["by_basis"].items()}}

    # cross-config strips (seed variance + percentile) from sibling runs
    runs = []
    for f in sorted(root.glob("L*_p*_seed*/ablation.json")):
        a = json.loads(f.read_text())
        c = a["config"]
        mean = a["sweep_by_basis"].get("mean", [])
        runs.append({"p": c["percentile"], "seed": c["seed"],
                     "K": a["n_partitions"],
                     "delta": mean[-1]["delta"] if mean else None,
                     "pid": a["top_refusal"][0]["pid"] if a["top_refusal"] else None})

    return {
        "model": ablation["config"]["model_id"].split("/")[-1],
        "layer": ablation["config"]["layer"],
        "p": ablation["config"]["percentile"],
        "seed": ablation["config"]["seed"],
        "K": len(parts),
        "baseline": ablation["baseline_refusal_rate"],
        "base_rates": loadings.get("base_rates"),
        "sel": sel_pid, "null_pid": null_pid,
        "sweeps": sweeps, "null": null,
        "tsne": _round2(tsne), "pca": _round2(pca2),
        "regions": regions, "chat": chat, "runs": runs,
    }


def build_concept(concept_json: Path, concept_dict: Path) -> dict:
    cj = json.loads(concept_json.read_text())
    # find the entry matching the display dict
    target = concept_dict.name
    entry = next((e for e in cj["results"] if e["dict"] == target), None)
    if entry is None:
        raise SystemExit(f"{target} not found in {concept_json}")

    dictionary = pickle.load((concept_dict / "dictionary.pkl").open("rb"))
    parts = dictionary.partitions
    dirs = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
    tsne, pca2 = layouts(dirs)
    nbrs = knn(dirs, k=8)

    regions = [{"i": i, "n": int(p.member_count),
                "coh": round(float(p.member_coherence), 3),
                "nb": [[j, s] for j, s in nbrs[i]]}
               for i, p in enumerate(parts)]

    # concepts that exported examples (from the display dict's mean basis)
    pc = entry["by_basis"]["mean"]["per_concept"]
    concepts = []
    for c in sorted(pc, key=lambda x: -x["auroc"]):
        item = {"id": c["id"], "label": c["label"], "region": c["region"],
                "auroc": round(c["auroc"], 3), "contrast": round(c["contrast"], 3)}
        if c.get("examples"):
            item["examples"] = c["examples"]
        concepts.append(item)

    by_basis = {b: {"mean": entry["by_basis"][b]["mean_auroc"],
                    "std": entry["by_basis"][b]["std_auroc"],
                    "n": entry["by_basis"][b]["n_concepts"]}
                for b in entry["by_basis"]}
    # full crossover table across all dicts
    table = [{"p": e["percentile"], "K": e["n_partitions"],
              "by_basis": {b: e["by_basis"][b]["mean_auroc"] for b in e["by_basis"]}}
             for e in sorted(cj["results"], key=lambda e: e["percentile"])]

    return {
        "model": cj["model_id"].split("/")[-1], "layer": cj["layer"],
        "n_concepts": cj["n_concepts"], "note": cj["note"],
        "dict": target, "p": entry["percentile"], "K": len(parts),
        "by_basis": by_basis, "table": table,
        "tsne": _round2(tsne), "pca": _round2(pca2),
        "regions": regions, "concepts": concepts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refusal-dict", default="artifacts/runs/behavioral/qwen3_5-4b/L27_p12_seed0")
    ap.add_argument("--behavioral-root", default="artifacts/runs/behavioral/qwen3_5-4b")
    ap.add_argument("--concept-json", default="artifacts/runs/concept_detect/qwen3_5-4b.json")
    ap.add_argument("--concept-dict", default="artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile")
    ap.add_argument("--out", default="artifacts/figures/dashboard_data/_testing.json")
    args = ap.parse_args()

    payload = {}
    rd = Path(args.refusal_dict)
    if (rd / "ablation.json").exists():
        print("building refusal page…")
        payload["refusal"] = build_refusal(rd, Path(args.behavioral_root))
    cj = Path(args.concept_json)
    if cj.exists():
        print("building concept page…")
        payload["concept"] = build_concept(cj, Path(args.concept_dict))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    r = payload.get("refusal"); c = payload.get("concept")
    print(f"refusal: {r['K'] if r else 0} regions, {len(r['chat']) if r else 0} chat; "
          f"concept: {c['K'] if c else 0} regions, {len(c['concepts']) if c else 0} concepts")
    print(f"wrote {out} ({out.stat().st_size/1000:.0f} KB)")


if __name__ == "__main__":
    main()
