"""Logit-lens / J-lens tables for region directions.

SAEDashboard's logits panel shows the tokens a feature direction most
promotes and suppresses through the unembedding. The EP analogue applies the
same readout to the exemplar direction and to the mean member direction, and
— where a Jacobian lens exists for the model/layer — to the J-transported
direction.

Weight loading is delegated to ``qwen_ep.lens_weights`` / ``jlens_weights``
(HTTP range requests against the hub, npz-cached); the model is resolved from
its HF id rather than the short keys those modules use internally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _model_key(model_id: str) -> str | None:
    from qwen_ep.lens_weights import SPECS
    for key, spec in SPECS.items():
        if spec["repo_id"] == model_id:
            return key
    return None


def topk_stats(directions: np.ndarray, lens: dict, k: int,
               chunk: int = 256) -> dict:
    """Top-k / bottom-k vocab ids plus the top-k probability mass.

    Directions are RMS-normalised before unembedding, so the softmax
    temperature is canonical and masses are comparable across regions.
    """
    import torch
    w_u = torch.from_numpy(lens["W_U"])
    w_norm = torch.from_numpy(lens["w_norm"])
    pos, neg, mass = [], [], []
    for s in range(0, len(directions), chunk):
        d = torch.from_numpy(directions[s:s + chunk].astype(np.float32))
        d = d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-6) * w_norm
        logits = d @ w_u.T
        top = torch.topk(logits, k, dim=-1).indices
        pos.extend(top.tolist())
        neg.extend(torch.topk(-logits, k, dim=-1).indices.tolist())
        p = torch.log_softmax(logits, dim=-1).exp()
        mass.append(p.gather(-1, top).sum(-1).numpy())
    return {"pos": pos, "neg": neg, "mass": np.concatenate(mass)}


class LensTables:
    def __init__(self, model_id: str, layer: int, cache_dir: Path, k: int):
        from qwen_ep.jlens_weights import has_jlens, jlens_n_prompts, load_jlens
        from qwen_ep.lens_weights import load_lens

        self.k = k
        self.key = _model_key(model_id)
        if self.key is None:
            raise ValueError(
                f"no lens spec for {model_id!r} — add it to "
                "qwen_ep.lens_weights.SPECS (unembedding + final-norm tensor "
                "names; check tie_word_embeddings first)")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.lens = load_lens(self.key, cache_dir)
        self.J = load_jlens(self.key, layer, cache_dir) if has_jlens(self.key) else None
        self.j_n_prompts = (jlens_n_prompts(self.key, layer, cache_dir)
                            if self.J is not None else 0)

    def build(self, E: np.ndarray, means: np.ndarray, decode) -> list[dict]:
        """Per-region lens dict; ``decode`` maps a vocab id to its string."""
        def table(dirs):
            st = topk_stats(dirs, self.lens, self.k)
            return [{"pos": [decode(t) for t in st["pos"][i]],
                     "neg": [decode(t) for t in st["neg"][i]],
                     "mass": round(float(st["mass"][i]), 3)}
                    for i in range(len(dirs))]

        ex, mn = table(E), table(means)
        out = [{"exemplar": ex[i], "mean": mn[i]} for i in range(len(E))]
        if self.J is not None:
            jex, jmn = table(E @ self.J.T), table(means @ self.J.T)
            for i in range(len(E)):
                out[i]["jlens"] = jex[i]
                out[i]["jlensMean"] = jmn[i]
        return out
