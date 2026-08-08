"""Last-token activations for the 113 probing datasets.

Deliberately a thin driver over `sae_probes.generate_model_activations` rather
than a reimplementation. Three details there are easy to get subtly wrong and
would silently break comparability with KE25's published table:

- `truncation_side="left"`, so a truncated prompt keeps its *end* -- which is
  the position we read.
- the read position is `min(len - 1, max_seq_len - 1)`, i.e. the last real
  token, not the last padded one (`padding_side="right"`).
- `stop_at_layer` is derived from the hook name, so only the layers below the
  hook are run. Half the model for `blocks.20` of a 42-layer gemma-2-9b.

**The paper and the package load the model differently, and it matters.**
`JoshEngels/SAE-Probes/generate_model_activations.py:20` uses
`HookedTransformer.from_pretrained`, which applies TransformerLens weight
processing (LayerNorm folding, writing-weight centring, ...). The maintained
`sae-probes` package uses `from_pretrained_no_processing`. Those produce
*different* `hook_resid_post` values, so:

- the published `layer{L}_results.csv` we check ourselves against, and any
  activations mirrored from the paper's Dropbox, are the **processed** kind;
- anything we generate with the package default is the **unprocessed** kind.

Whichever is chosen, the probing activations and the EP dictionary build stream
must come from the *same* loader. Mixing them puts the exemplars in a different
geometry from the points being assigned, and EP fails silently -- argmax always
returns something, so there is no error to notice. `--processing` is therefore
an explicit choice with no safe default; `paper` is the default only because
that is what the table we validate against was computed from.

Resume is by file existence with a row-count check, which their function
already does, so re-running after an interrupted pod costs one tokenizer pass
per finished dataset and nothing else.

Output: `{cache}/model_activations_{model}/{dataset}_{hook}.pt`, one tensor of
shape (n, d_model) per dataset.

Run:
  python -m experiments.probes.extract_acts \
      --model gemma-2-9b --layer 20 --device cuda --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from experiments.probes import benchmark as bm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=bm.HEADLINE_MODEL)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--dtype", default="float32",
                    help="the paper left TransformerLens at its float32 default; "
                         "bfloat16 halves the pod time and the storage")
    ap.add_argument("--processing", choices=("paper", "package"), default="paper",
                    help="paper = from_pretrained (folded LN, centred writing "
                         "weights), matching the published results CSVs and the "
                         "Dropbox activations; package = from_pretrained_no_processing")
    ap.add_argument("--cache", type=Path, default=bm.ARTIFACTS / "acts")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N datasets only; for a smoke run before the pod bill")
    args = ap.parse_args()

    # Imported here so the module stays importable on a machine with no
    # transformer_lens -- the analysis stages do not need it.
    from sae_probes.generate_model_activations import (
        generate_single_dataset_activations,
    )
    from sae_probes.constants import DATA_PATH
    from transformer_lens import HookedTransformer

    hook = f"blocks.{args.layer}.hook_resid_post"
    tags = bm.dataset_tags()
    if args.limit:
        tags = tags[: args.limit]

    print(f"model={args.model} hook={hook} device={args.device} "
          f"dtype={args.dtype} processing={args.processing} n_datasets={len(tags)}")

    t0 = time.time()
    loader = (
        HookedTransformer.from_pretrained
        if args.processing == "paper"
        else HookedTransformer.from_pretrained_no_processing
    )
    model = loader(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    print(f"model loaded in {time.time() - t0:.0f}s "
          f"(n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model})")

    # `Dataset save name` in the master CSV is the path the generator expects.
    binary = bm.read_master_binary()
    path_by_tag = {
        name.split("/")[-1].split(".")[0]: name
        for name in binary["Dataset save name"]
    }

    args.cache.mkdir(parents=True, exist_ok=True)
    log = []
    for i, tag in enumerate(tags, 1):
        t = time.time()
        generate_single_dataset_activations(
            model=model,
            model_name=args.model,
            dataset_path=DATA_PATH / f"{path_by_tag[tag]}.zst",
            hook_names=[hook],
            model_cache_path=args.cache,
            device=args.device,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        dt = time.time() - t
        log.append(dict(dataset=tag, seconds=dt))
        done = time.time() - t0
        print(f"[{i:3d}/{len(tags)}] {tag:<45s} {dt:7.1f}s  "
              f"elapsed {done/60:6.1f}m  eta {done/i*(len(tags)-i)/60:6.1f}m",
              flush=True)

    out = args.cache / f"extract_log_{args.model}_L{args.layer}.json"
    out.write_text(json.dumps(
        dict(model=args.model, hook=hook, device=args.device,
             dtype=args.dtype, processing=args.processing,
             batch_size=args.batch_size, max_seq_len=args.max_seq_len,
             total_seconds=time.time() - t0,
             per_dataset=log), indent=2))
    print(f"\ntotal {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
