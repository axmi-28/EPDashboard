"""Fetch a single Jacobian-lens matrix J_l from neuronpedia/jacobian-lens.

The hub .pt files (anthropics/jacobian-lens format) hold every layer's
``J_l`` as fp16 ``[d_model, d_model]``; we only need the EP dictionary's
layer, so the extracted matrix is cached as a small npz next to the logit
lens caches. The J-lens readout is ``unembed(J_l @ h)`` where ``unembed``
is the model's final RMSNorm + tied unembedding — i.e. exactly the
existing ``lens_topk`` pipeline applied to the transported direction.

No lens is published for Qwen3.5-4B-Base (only the instruct 4B, the 2B base,
and Qwen3.6-27B), so that dictionary stays logit-lens-only.

Caveat when comparing verbalizability *across* models: the 27B is only
published as an ``_n1000`` fit (1000 prompts), whereas the 4B/2B entries point
at the unsuffixed files. ``load_jlens`` logs each checkpoint's ``n_prompts`` so
a mismatch is visible rather than silent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

JLENS_REPO = "neuronpedia/jacobian-lens"

JLENS_SPECS = {
    "qwen4b": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
    "qwen": "qwen3.5-2b-pt/jlens/Salesforce-wikitext/"
            "Qwen3.5-2B-Base_jacobian_lens.pt",
    # Only an n1000 fit is published for the 27B. The .pt is 3.3 GB because it
    # carries all 64 layers' (5120, 5120) matrices; we extract one and cache it.
    "qwen27b": "qwen3.6-27b/jlens/Salesforce-wikitext/"
               "Qwen3.6-27B_jacobian_lens_n1000.pt",
}


def has_jlens(model_key: str) -> bool:
    return model_key in JLENS_SPECS


def load_jlens(model_key: str, layer: int, cache_dir: Path) -> np.ndarray:
    """Return ``J_layer`` as float32 ``(D, D)``; downloads + extracts once."""
    if model_key not in JLENS_SPECS:
        raise KeyError(f"no published Jacobian lens for {model_key!r}; "
                       f"available: {sorted(JLENS_SPECS)}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"jlens__{model_key}__L{layer}.npz"
    if out_path.exists():
        with np.load(out_path) as z:
            return z["J"].astype(np.float32)

    import torch
    from huggingface_hub import hf_hub_download
    pt_path = hf_hub_download(JLENS_REPO, JLENS_SPECS[model_key])
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    if layer not in ckpt["J"]:
        raise KeyError(f"layer {layer} not in lens (source_layers "
                       f"{ckpt['source_layers'][0]}..{ckpt['source_layers'][-1]})")
    J = ckpt["J"][layer].to(torch.float16).numpy()
    # Keep n_prompts with the matrix: verbalizability is only comparable across
    # models when the lenses were fit on the same budget, and the published
    # files are not consistent about that.
    np.savez(out_path, J=J, n_prompts=np.int64(ckpt.get("n_prompts", 0)))
    print(f"  jlens {model_key} L{layer}: J {J.shape} "
          f"(fit on {ckpt['n_prompts']} prompts) -> {out_path}")
    return J.astype(np.float32)


def jlens_n_prompts(model_key: str, layer: int, cache_dir: Path) -> int:
    """Prompt budget the cached lens was fit on; 0 if unknown."""
    path = Path(cache_dir) / f"jlens__{model_key}__L{layer}.npz"
    if not path.exists():
        return 0
    with np.load(path) as z:
        return int(z["n_prompts"]) if "n_prompts" in z.files else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=list(JLENS_SPECS))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    J = load_jlens(args.model, args.layer, Path(args.cache_dir))
    print(f"{args.model} L{args.layer}: J {J.shape}")


if __name__ == "__main__":
    main()
