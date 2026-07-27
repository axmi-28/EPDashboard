"""Pass 2: turn (gid, pos) example slots into colorable token sequences.

For every prompt that won a slot we have its full per-position activations;
each sequence record ships the window's token strings plus the projection of
*every* window token onto the region direction — the EP analogue of
SAEDashboard's per-token feature activations — and the indices of window
tokens that are actually members of the region, so the UI can distinguish
"projects onto the direction" from "belongs to the cell".
"""

from __future__ import annotations

import numpy as np

from epdashboard.scan import EPDict


class PromptCache:
    """Per-prompt derived arrays, computed once and shared across regions.

    Holds only what :meth:`sequence` reads back — token ids, per-row norms,
    the argmax winner per row, and the similarity *columns of the regions
    that reference this prompt* (``refs``). The full ``(T, K)`` similarity
    matrix exists only transiently inside :meth:`add`: keeping it per prompt
    is ~1.4 MB × tens of thousands of prompts at K in the thousands, which is
    the OOM measured on the first full-budget run.
    """

    def __init__(self, d: EPDict, tokenizer, bos_offset: int,
                 refs: dict[int, set[int]]):
        self.d = d
        self.tok = tokenizer
        self.bos_offset = bos_offset
        self.refs = refs                 # gid -> region ids needing sequences
        self._decode: dict[int, str] = {}
        self._by_gid: dict[int, dict] = {}

    def add(self, gid: int, text: str, X: np.ndarray, positions: np.ndarray,
            ids: list[int] | None = None):
        ks = sorted(self.refs.get(gid, ()))
        if not ks:
            return
        d = self.d
        Xc = X - d.center
        norms = np.linalg.norm(Xc, axis=1) + 1e-12
        sims = (Xc / norms[:, None]) @ d.E.T          # (T, K) — transient
        row_of = {int(p): r for r, p in enumerate(positions)}
        if ids is None:
            ids = self.tok.encode(text, add_special_tokens=False)
        self._by_gid[gid] = {
            "ids": ids, "norms": norms.astype(np.float32),
            "best": np.argmax(sims, axis=1).astype(np.int32),
            "row_of": row_of,
            "cols": {k: sims[:, k].copy() for k in ks},
        }

    def _tok_str(self, tid: int) -> str:
        if tid not in self._decode:
            self._decode[tid] = self.tok.decode([tid])
        return self._decode[tid]

    def sequence(self, gid: int, pos: int, region: int,
                 buffer: tuple[int, int], dist: float, proj: float) -> dict | None:
        """One sequence record, or None if the prompt was never gathered."""
        entry = self._by_gid.get(gid)
        if entry is None or region not in entry["cols"]:
            return None
        d = self.d
        ids, norms, col = entry["ids"], entry["norms"], entry["cols"][region]
        ti = pos - self.bos_offset                   # token index of firing pos
        lo = max(0, ti - buffer[0])
        hi = min(len(ids), ti + buffer[1] + 1)
        toks, acts, mb = [], [], []
        for t in range(lo, hi):
            toks.append(self._tok_str(ids[t]))
            row = entry["row_of"].get(t + self.bos_offset)
            if row is None:                          # position 0 (sink) — no act
                acts.append(None)
                continue
            acts.append(round(float(col[row] * norms[row]), 3))
            if (entry["best"][row] == region
                    and 1.0 - col[row] <= d.threshold):
                mb.append(t - lo)
        return {"g": int(gid), "pos": int(pos), "fi": int(ti - lo),
                "d": round(float(dist), 4), "v": round(float(proj), 3),
                "tok": toks, "act": acts, "mb": mb}


def band_labels(n_bands: int) -> list[str]:
    if n_bands == 3:
        return ["near band", "mid band", "far band"]
    return [f"band {b + 1}/{n_bands}" for b in range(n_bands)]


def build_groups(pc: PromptCache, region: int, examples: dict[str, list[dict]],
                 threshold: float, n_bands: int,
                 buffer: tuple[int, int]) -> list[dict]:
    """Ordered sequence groups for one region card."""
    labels = band_labels(n_bands)
    edges = [threshold * b / n_bands for b in range(n_bands + 1)]
    groups = [("closest", "closest members", None)]
    for b in range(n_bands):
        rng = f"d ∈ [{edges[b]:.3f}, {edges[b + 1]:.3f})"
        groups.append((f"band{b}", labels[b], rng))
    groups.append(("random", "random members", None))

    out = []
    for key, title, subtitle in groups:
        seqs = []
        for r in examples.get(key, []):
            dist = r.get("dist", -r["score"] if key == "closest" else 0.0)
            s = pc.sequence(r["gid"], r["pos"], region, buffer,
                            dist, r.get("proj", 0.0))
            if s is not None:
                seqs.append(s)
        if key == "closest":
            seqs.sort(key=lambda s: s["d"])
        out.append({"key": key, "title": title, "subtitle": subtitle,
                    "seqs": seqs})
    return out
