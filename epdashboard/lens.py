"""Logit-lens / J-lens tables for region directions.

SAEDashboard's logits panel shows the tokens a feature direction most
promotes and suppresses through the unembedding. The EP analogue applies the
same readout to the exemplar direction and to the mean member direction, and
— where a Jacobian lens exists for the model/layer — to the J-transported
direction.

**Verbalizability** (``1 - H / ln|V|``, where ``H`` is the full-vocab softmax
entropy of the RMS-normalised direction's logits) says whether the vocab
readout means anything at all for a region: 1 is a spike on a few tokens, 0 is
a flat distribution whose top-k list is noise dressed up as a label.

It is computed **for the J-lens only**, deliberately. The J-transported
direction is the one that approximates what the model would actually emit from
this layer, so flatness there is evidence about the region. A mid-layer
direction pushed straight through the unembedding is not on the model's output
path, so the entropy of *that* readout scores the lens rather than the region —
it was previously reported as a baseline and read as if it were comparable,
which it is not.

Two things it is not comparable across: **models, and J-lens fit budgets.**
Qwen3.6-27B L55 (J fit on 1000 prompts) has median verbalizability 0.153;
Qwen3.5-4B-it L27 has 0.782. The 27B range is still wide (0.032–0.975) so it
discriminates *within* a dictionary, but the level may be a thin-fit artifact.
Always read it next to the ``j-lens n=`` fit budget the dashboard prints.

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
               chunk: int = 256, entropy: bool = False) -> dict:
    """Top-k / bottom-k vocab ids plus the top-k probability mass.

    Directions are RMS-normalised before unembedding, so the softmax
    temperature is canonical and masses are comparable across regions.

    ``entropy=True`` additionally returns the full-vocab softmax entropy per
    direction, which backs verbalizability. It is off by default so the
    logit-lens tables do not pay for a statistic that is only reported for the
    J-lens; the extra work is one elementwise product and reduction over the
    already-materialised (chunk, |V|) probabilities.
    """
    import torch
    w_u = torch.from_numpy(lens["W_U"])
    w_norm = torch.from_numpy(lens["w_norm"])
    pos, neg, mass, ents = [], [], [], []
    for s in range(0, len(directions), chunk):
        d = torch.from_numpy(directions[s:s + chunk].astype(np.float32))
        d = d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-6) * w_norm
        logits = d @ w_u.T
        top = torch.topk(logits, k, dim=-1).indices
        pos.extend(top.tolist())
        neg.extend(torch.topk(-logits, k, dim=-1).indices.tolist())
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        mass.append(p.gather(-1, top).sum(-1).numpy())
        if entropy:
            ents.append((-(p * logp).sum(-1)).numpy())
    out = {"pos": pos, "neg": neg, "mass": np.concatenate(mass)}
    if entropy:
        out["entropy"] = np.concatenate(ents)
    return out


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
        # ln|V| normalises entropy onto [0, 1]; |V| is the unembedding's row
        # count, so it tracks the actual tokenizer rather than a config field.
        self.ln_v = float(np.log(self.lens["W_U"].shape[0]))
        self.J = load_jlens(self.key, layer, cache_dir) if has_jlens(self.key) else None
        self.j_n_prompts = (jlens_n_prompts(self.key, layer, cache_dir)
                            if self.J is not None else 0)

    def _verb(self, H: np.ndarray) -> np.ndarray:
        """Verbalizability: 1 - H/ln|V|. 1 = spiked readout, 0 = uniform."""
        return np.round(1.0 - H / self.ln_v, 3)

    def build(self, E: np.ndarray, means: np.ndarray, decode) -> list[dict]:
        """Per-region lens dict; ``decode`` maps a vocab id to its string.

        Only the J-lens tables carry ``verb`` — see the module docstring. The
        UI keys off its presence, so logit-lens panels show no score rather
        than a misleading one.
        """
        def table(dirs, verb: bool = False):
            st = topk_stats(dirs, self.lens, self.k, entropy=verb)
            v = self._verb(st["entropy"]) if verb else None
            rows = [{"pos": [decode(t) for t in st["pos"][i]],
                     "neg": [decode(t) for t in st["neg"][i]],
                     "mass": round(float(st["mass"][i]), 3)}
                    for i in range(len(dirs))]
            if v is not None:
                for i, row in enumerate(rows):
                    row["verb"] = float(v[i])
            return rows

        ex, mn = table(E), table(means)
        out = [{"exemplar": ex[i], "mean": mn[i]} for i in range(len(E))]
        if self.J is not None:
            jex = table(E @ self.J.T, verb=True)
            jmn = table(means @ self.J.T, verb=True)
            for i in range(len(E)):
                out[i]["jlens"] = jex[i]
                out[i]["jlensMean"] = jmn[i]
        return out
