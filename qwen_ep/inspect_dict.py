"""Inspect an EP dictionary built on Qwen: sizes, nearest prompts, logit lens.

    python -m qwen_ep.inspect_dict --dict runs/<slug>/dictionary.pkl --top 20
    python -m qwen_ep.inspect_dict --dict ... --no-model   # skip logit lens (no model load)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dict", required=True, help="Path to dictionary.pkl")
    ap.add_argument("--top", type=int, default=20, help="How many partitions to show.")
    ap.add_argument("--order", choices=["size", "id"], default="size")
    ap.add_argument("--prompts", type=int, default=3, help="Nearest prompts per partition.")
    ap.add_argument("--model-id", default=None, help="Override model id for the logit lens.")
    ap.add_argument("--no-model", action="store_true",
                    help="Skip loading the model (no logit-lens tokens).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.dict).open("rb") as f:
        d = pickle.load(f)

    parts = list(d.partitions)
    if args.order == "size":
        parts = sorted(parts, key=lambda p: -p.member_count)

    members = sorted((p.member_count for p in d.partitions), reverse=True)
    total = sum(members) or 1
    print(f"Dictionary: {len(parts)} partitions  threshold={d.threshold:.4f}  "
          f"||center||={float((d.center**2).sum()**0.5):.2f}")
    print(f"  largest={members[0]}  singletons={sum(1 for m in members if m == 1)}  "
          f"top10 cover {sum(members[:10]) / total * 100:.1f}%")
    print()

    qwen = None
    if not args.no_model:
        from .adapter import DEFAULT_MODEL_ID, QwenModel
        qwen = QwenModel(args.model_id or DEFAULT_MODEL_ID)

    for rank, p in enumerate(parts[:args.top]):
        header = f"[#{rank}] id={getattr(p, 'partition_id', '?')} members={p.member_count}"
        coh = getattr(p, "member_coherence", None)
        if coh is not None:
            header += f" coherence={coh:.2f}"
        print(header)
        if qwen is not None:
            toks = qwen.logit_lens(p.exemplar_direction, k=12)
            print("   logit-lens:", ", ".join(t for t in toks if t))
        # sample_prompts stores (-dist, prompt, pos); nearest = smallest dist first.
        for negd, prompt, pos in sorted(p.sample_prompts, key=lambda e: -e[0])[:args.prompts]:
            snippet = prompt.replace("\n", " ")[:90]
            print(f"   d={-negd:.3f} pos={pos}  {snippet!r}")
        print()


if __name__ == "__main__":
    main()
