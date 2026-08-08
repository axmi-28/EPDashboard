"""Extraction for Arms B and C — one pass over each corpus, three layers.

Two corpora:

- **labelled** — the same 300 harmful + 300 benign the whole gate is anchored
  on, chat-formatted, every position, at layers 4, 12 and 20. Serves B (L20
  trajectories) and C (cross-layer correspondence) from one forward pass.
- **background** — 1,500 held-out Pile documents at 128 positions, layer 20
  only. This is what the transition table is *fitted* on.

Fitting the transition table on Pile rather than on the labelled prompts is the
whole point. A monitor learns what normal traffic looks like and then scores
incoming requests against it; a table fitted on the 600 prompts it will later
score would leak, and a table fitted only on benign prompts would be a
600-prompt sample of a K x K matrix, which at K=176 is 31,000 cells.

The Pile documents come from `pile.py`'s R0 slice, which is disjoint from the
dictionary build stream by construction (the first 40,000 documents of the
shuffled stream are skipped because the p=1 build consumed 28,288 of them).

    python -m experiments.monitor.run_gate2_bc_extract --corpus labelled
    python -m experiments.monitor.run_gate2_bc_extract --corpus background
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .extract_seq import extract_sequences, shared_affix
from .run_gate2_a0 import _format_chat

ROOT = Path("artifacts/runs/monitor")
LAB = ROOT / "gate2_bc_labelled.npz"
BG = ROOT / "gate2_b_background.npz"
BG3 = ROOT / "gate2_c_background3.npz"

LAYERS = (4, 12, 20)
N_PER_SIDE = 300
BG_DOCS = 1500
BG_CONTEXT = 128


def stage_labelled(args) -> None:
    from .extract import load_model
    from ..jailbreak.corpus import load_build_prompts

    harmful, benign = load_build_prompts(N_PER_SIDE)
    goals = list(harmful) + list(benign)
    y = np.array([1] * len(harmful) + [0] * len(benign), dtype=np.int64)

    model = load_model(args.model, device=args.device)
    texts = [_format_chat(model.tokenizer, g) for g in goals]
    seq = extract_sequences(model, texts, layers=LAYERS,
                            batch_size=args.batch_size, max_tokens=128)

    pre, suf = shared_affix(seq.token_ids, seq.lengths)
    print(f"chat scaffold: {pre} shared prefix tokens, {suf} shared suffix "
          f"tokens; median prompt {np.median(seq.lengths):.0f} positions "
          f"-> {np.median(seq.lengths) - pre - suf:.0f} content positions")
    print("  prefix:", repr(model.tokenizer.decode(seq.token_ids[0, :pre])))
    print("  suffix:", repr(model.tokenizer.decode(
        seq.token_ids[0, seq.lengths[0] - suf:seq.lengths[0]])))

    np.savez_compressed(
        LAB, y=y, lengths=seq.lengths, token_ids=seq.token_ids,
        prefix_len=np.int32(pre), suffix_len=np.int32(suf),
        **{f"x_L{L}": seq.x[L] for L in LAYERS})
    print(f"-> {LAB}  positions={int(seq.lengths.sum())}  "
          f"{LAB.stat().st_size / 1e6:.0f} MB")


def stage_background(args) -> None:
    from .extract import load_model

    # The R0 slice is already carved and cached in corpus.json. Re-streaming
    # would replay 45,600 Pile documents to arrive at the identical texts, and
    # re-deriving the offsets by hand is exactly the mistake `pile.py` was
    # written to prevent.
    corpus = json.loads((ROOT / "corpus.json").read_text())
    texts = [p["text"] for p in corpus["prompts"]
             if p["source"] == "pile_heldout"][:BG_DOCS]
    if len(texts) < BG_DOCS:
        raise SystemExit(f"only {len(texts)} R0 documents cached")
    print(f"background: {len(texts)} Pile documents (R0 slice, disjoint from "
          f"the dictionary build stream)")
    model = load_model(args.model, device=args.device)
    seq = extract_sequences(model, texts, layers=(20,),
                            batch_size=args.batch_size,
                            max_tokens=BG_CONTEXT + 1)
    np.savez_compressed(BG, x_L20=seq.x[20], lengths=seq.lengths)
    print(f"-> {BG}  positions={int(seq.lengths.sum())}  "
          f"{BG.stat().st_size / 1e6:.0f} MB")


def stage_background3(args) -> None:
    """Pile at layers 4, 12 and 20 — the strongest test of cross-layer stability.

    C2's claim ("EP regions correspond across layers worse than random ones do")
    was measured on chat-formatted instruction prompts, a narrow slice. Pile is
    the distribution the dictionaries were *built* on, so it is the most
    favourable possible ground for EP: if the correspondence fails here it fails
    everywhere, and if it holds here but not on chat prompts, the story is
    distribution shift rather than a property of the method.

    600 documents rather than 1,500: three layers at fp16 is 3x the footprint
    per position, and 76,000 positions already exceeds the labelled corpus by
    an order of magnitude.
    """
    from .extract import load_model

    corpus = json.loads((ROOT / "corpus.json").read_text())
    texts = [p["text"] for p in corpus["prompts"]
             if p["source"] == "pile_heldout"][:600]
    print(f"background3: {len(texts)} Pile documents at layers {LAYERS}")
    model = load_model(args.model, device=args.device)
    seq = extract_sequences(model, texts, layers=LAYERS,
                            batch_size=args.batch_size,
                            max_tokens=BG_CONTEXT + 1)
    np.savez_compressed(BG3, lengths=seq.lengths,
                        **{f"x_L{L}": seq.x[L] for L in LAYERS})
    print(f"-> {BG3}  positions={int(seq.lengths.sum())}  "
          f"{BG3.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True,
                    choices=["labelled", "background", "background3"])
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    {"labelled": stage_labelled, "background": stage_background,
     "background3": stage_background3}[args.corpus](args)


if __name__ == "__main__":
    main()
