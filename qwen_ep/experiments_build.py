"""Aggregate Panel A (concept detection) + Panel B (refusal ablation) results
into one compact JSON for the dashboard Experiments tab.

Safety: ablated continuations to harmful prompts are REDACTED whenever the
model complied — the dashboard shows the refuse->comply verdict, never the
harmful payload. Baseline refusals (safe) are shown verbatim, truncated.

Run:
    python -m qwen_ep.experiments_build \
        --behavioral runs/behavioral/qwen3_5-4b \
        --concept runs/concept_detect/qwen3_5-4b.json \
        --out figures/dashboard_data/_experiments.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _first_clause(text: str, n: int = 90) -> str:
    """A short, non-actionable prefix of a refusal (safe) generation."""
    t = " ".join(text.strip().split())
    return t[:n] + ("…" if len(t) > n else "")


def build_panel_b(root: Path) -> dict | None:
    root = Path(root)
    configs = sorted(root.glob("L*_p*_seed*/ablation.json"))
    if not configs:
        return None

    runs = []
    for f in configs:
        a = json.loads(f.read_text())
        cfg = a["config"]
        top = a["top_refusal"][0] if a["top_refusal"] else None
        sweeps = {}
        for basis, sweep in a["sweep_by_basis"].items():
            sweeps[basis] = [{"k": e["k"], "rate": e["ablated_refusal_rate"],
                              "delta": e["delta"]} for e in sweep]
        null = None
        if a.get("null_ablation"):
            n = a["null_ablation"]
            null = {"pid": n["pid"], "n": n["n_members"],
                    "coherence": round(n["coherence"], 3),
                    "by_basis": {b: {"rate": e["ablated_refusal_rate"],
                                     "delta": e["delta"]}
                                 for b, e in n["by_basis"].items()}}
        # redacted examples
        examples = []
        for e in a.get("examples", []):
            complied = e["ablated_refused"] == 0
            examples.append({
                "prompt": e["prompt"][:160],
                "baseline_refused": e["baseline_refused"],
                "ablated_refused": e["ablated_refused"],
                "baseline_snip": _first_clause(e["baseline"]),
                # withhold compliant continuations to harmful prompts
                "ablated_snip": (None if complied else _first_clause(e["ablated"])),
            })
        runs.append({
            "layer": cfg["layer"], "p": cfg["percentile"], "seed": cfg["seed"],
            "n_partitions": a["n_partitions"],
            "baseline": a["baseline_refusal_rate"],
            "top": ({"pid": top["pid"], "n": top["n_members"],
                     "refusal_rate": round(top["refusal_rate"], 3),
                     "coherence": round(top["coherence"], 3)} if top else None),
            "n_refusal_regions": sum(
                1 for r in a["top_refusal"]),  # regions clearing threshold
            "sweeps": sweeps,
            "null": null,
            "cos_mean_exemplar": [round(c, 3) for c in a["cos_mean_exemplar"]],
            "examples": examples,
        })

    # base rates from any loadings file
    base_rates = None
    for lf in root.glob("L*_p*_seed*/loadings.json"):
        base_rates = json.loads(lf.read_text()).get("base_rates")
        if base_rates:
            break

    # pick primary = p12 seed0 if present else first
    def key(r):
        return (abs(r["p"] - 12), r["seed"])
    primary = min(range(len(runs)), key=lambda i: key(runs[i]))

    return {"model": "Qwen3.5-4B", "base_rates": base_rates,
            "runs": runs, "primary": primary}


def build_panel_a(path: Path | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    p = json.loads(path.read_text())
    dicts = []
    for e in p["results"]:
        by = {}
        for basis, r in e["by_basis"].items():
            by[basis] = {"mean": r["mean_auroc"], "std": r["std_auroc"],
                         "n": r["n_concepts"]}
        # keep a few example concepts (highest-AUROC, for the strip + deep link)
        first_basis = next(iter(e["by_basis"]))
        pc = sorted(e["by_basis"][first_basis]["per_concept"],
                    key=lambda c: -c["auroc"])[:12]
        dicts.append({"dict": e["dict"], "p": e["percentile"],
                      "K": e["n_partitions"], "by_basis": by,
                      "examples": [{"id": c["id"], "label": c["label"],
                                    "region": c["region"],
                                    "auroc": round(c["auroc"], 3)} for c in pc]})
    dicts.sort(key=lambda d: d["p"])
    return {"model": p["model_id"], "layer": p["layer"],
            "n_concepts": p["n_concepts"], "note": p["note"],
            "dicts": dicts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--behavioral", default="runs/behavioral/qwen3_5-4b")
    ap.add_argument("--concept", default="runs/concept_detect/qwen3_5-4b.json")
    ap.add_argument("--out", default="figures/dashboard_data/_experiments.json")
    args = ap.parse_args()

    payload = {
        "panelB": build_panel_b(Path(args.behavioral)),
        "panelA": build_panel_a(Path(args.concept) if args.concept else None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    b = payload["panelB"]
    a = payload["panelA"]
    print(f"panelB: {len(b['runs']) if b else 0} runs; "
          f"panelA: {len(a['dicts']) if a else 0} dicts")
    print(f"wrote {out} ({out.stat().st_size/1000:.0f} KB)")


if __name__ == "__main__":
    main()
