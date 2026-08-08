"""Build the JSON payloads behind the EP dashboard.

For each dictionary we export, per region: geometry for three layouts
(t-SNE 2D, PCA 2D, PCA-3D unit sphere + Voronoi arcs), exact high-D cosine
k-NN, logit-lens top tokens (exemplar + mean-member direction), and sample /
boundary prompts with the char-span of the harvested token so the UI can
highlight it Neuronpedia-style.

Usage:
    python -m experiments.legacy_dashboard.build --lens-cache <dir> --out artifacts/figures/dashboard_data
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
from pathlib import Path

import numpy as np

from qwen_ep.jlens_weights import has_jlens, jlens_n_prompts, load_jlens
from qwen_ep.lens_weights import load_lens, lens_topk_stats

WINDOW_BEFORE = 90    # chars kept before the highlighted token
WINDOW_AFTER = 150    # chars kept after
MAX_SAMPLES = 8
MAX_BOUNDARY = 4
LENS_K = 8
KNN_K = 8
EDGE_MAX_K = 800      # voronoi arcs get unreadable / huge past this
MS_TOP = 16           # member-scan examples per mode shipped to the browser
MS_MODES = ("closest", "proj", "projall", "random")


# --------------------------------------------------------------------- dicts
def _hub_gemma_dir(sub: str) -> str:
    base = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "datasets--J-RUM--exemplar-partitioning/snapshots/*"))[0]
    return glob.glob(f"{base}/{sub}/*.pkl")[0]


DICTS = [
    dict(key="qwen-L19-p8", model="Qwen3.5-2B-Base", model_key="qwen",
         tokenizer="Qwen/Qwen3.5-2B-Base", layer=19, p=8, bos_offset=0,
         path="artifacts/runs/qwen3_5-2b_L19_p8p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-2b_L19_p8p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen-L19-p4", model="Qwen3.5-2B-Base", model_key="qwen",
         tokenizer="Qwen/Qwen3.5-2B-Base", layer=19, p=4, bos_offset=0,
         path="artifacts/runs/qwen3_5-2b_L19_p4p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-2b_L19_p4p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen-L12-p8", model="Qwen3.5-2B-Base", model_key="qwen",
         tokenizer="Qwen/Qwen3.5-2B-Base", layer=12, p=8, bos_offset=0,
         path="artifacts/runs/qwen3_5-2b_L12_p8p0_ctx128_mt1000000_seed0_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-2b_L12_p8p0_ctx128_mt1000000_seed0_pile/metadata.json"),
    dict(key="qwen4b-it-L27-p8", model="Qwen3.5-4B (instruct)", model_key="qwen4b",
         tokenizer="Qwen/Qwen3.5-4B", layer=27, p=8, bos_offset=0,
         path="artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen4b-it-L27-p4", model="Qwen3.5-4B (instruct)", model_key="qwen4b",
         tokenizer="Qwen/Qwen3.5-4B", layer=27, p=4, bos_offset=0,
         path="artifacts/runs/qwen3_5-4b_L27_p4p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-4b_L27_p4p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen4b-base-L27-p8", model="Qwen3.5-4B-Base", model_key="qwen4b-base",
         tokenizer="Qwen/Qwen3.5-4B-Base", layer=27, p=8, bos_offset=0,
         path="artifacts/runs/qwen3_5-4b-base_L27_p8p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-4b-base_L27_p8p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen4b-base-L27-p4", model="Qwen3.5-4B-Base", model_key="qwen4b-base",
         tokenizer="Qwen/Qwen3.5-4B-Base", layer=27, p=4, bos_offset=0,
         path="artifacts/runs/qwen3_5-4b-base_L27_p4p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_5-4b-base_L27_p4p0_ctx128_cache_pile/metadata.json"),
    # Qwen3.6-27B. No J-lens exists at this scale, so the verbalizability
    # columns stay hidden for these (the UI keys off `payload.jlens`).
    dict(key="qwen27b-L55-p8", model="Qwen3.6-27B", model_key="qwen27b",
         tokenizer="Qwen/Qwen3.6-27B", layer=55, p=8, bos_offset=0,
         path="artifacts/runs/qwen3_6-27b_L55_p8p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_6-27b_L55_p8p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen27b-L55-p12", model="Qwen3.6-27B", model_key="qwen27b",
         tokenizer="Qwen/Qwen3.6-27B", layer=55, p=12, bos_offset=0,
         path="artifacts/runs/qwen3_6-27b_L55_p12p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_6-27b_L55_p12p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen27b-L55-p16", model="Qwen3.6-27B", model_key="qwen27b",
         tokenizer="Qwen/Qwen3.6-27B", layer=55, p=16, bos_offset=0,
         path="artifacts/runs/qwen3_6-27b_L55_p16p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_6-27b_L55_p16p0_ctx128_cache_pile/metadata.json"),
    dict(key="qwen27b-L55-p4", model="Qwen3.6-27B", model_key="qwen27b",
         tokenizer="Qwen/Qwen3.6-27B", layer=55, p=4, bos_offset=0,
         path="artifacts/runs/qwen3_6-27b_L55_p4p0_ctx128_cache_pile/dictionary.pkl",
         meta="artifacts/runs/qwen3_6-27b_L55_p4p0_ctx128_cache_pile/metadata.json"),
    dict(key="gemma-L12-p10", model="Gemma-2-2B", model_key="gemma",
         tokenizer="unsloth/gemma-2-2b", layer=12, p=10, bos_offset=1,
         path_fn=lambda: _hub_gemma_dir("gemma-2-2b_L12_p10"), meta=None),
    dict(key="gemma-L12-p4", model="Gemma-2-2B", model_key="gemma",
         tokenizer="unsloth/gemma-2-2b", layer=12, p=4, bos_offset=1,
         path_fn=lambda: _hub_gemma_dir("gemma-2-2b_L12_p4"), meta=None),
]


# ------------------------------------------------------------------ geometry
def sphere_points(dirs: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    p3 = PCA(n_components=3, random_state=0).fit_transform(dirs)
    p3 = p3 / (np.linalg.norm(p3, axis=1, keepdims=True) + 1e-12)
    return p3


def voronoi_edges(points: np.ndarray, n_seg: int = 6) -> list:
    """Great-circle arcs of the spherical Voronoi tessellation, as flat
    [x,y,z,...] polylines (mirrors scripts/sphere_voronoi.py)."""
    from scipy.spatial import SphericalVoronoi

    pts = points
    for attempt in range(3):
        try:
            sv = SphericalVoronoi(pts, radius=1.0, center=np.zeros(3))
            break
        except Exception:
            rng = np.random.default_rng(attempt)
            pts = pts + rng.normal(0, 1e-6, pts.shape)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    else:
        return []
    sv.sort_vertices_of_regions()

    def slerp(a, b, n):
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        omega = np.arccos(dot)
        if omega < 1e-8:
            return np.stack([a, b])
        t = np.linspace(0, 1, n + 1)
        return (np.sin((1 - t)[:, None] * omega) * a[None]
                + np.sin(t[:, None] * omega) * b[None]) / np.sin(omega)

    edges, seen = [], set()
    for region in sv.regions:
        m = len(region)
        for t in range(m):
            a, b = region[t], region[(t + 1) % m]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            arc = slerp(sv.vertices[a], sv.vertices[b], n_seg)
            edges.append([round(float(v), 3) for v in arc.reshape(-1)])
    return edges


def layouts(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    n = len(dirs)
    pca2 = PCA(n_components=2, random_state=0).fit_transform(dirs)
    try:
        tsne = TSNE(n_components=2, metric="cosine", init="pca",
                    perplexity=min(30, max(5, n // 3)),
                    random_state=0).fit_transform(dirs)
    except Exception as e:
        print(f"    t-SNE failed ({e}); reusing PCA")
        tsne = pca2.copy()
    return tsne, pca2


def knn(dirs: np.ndarray, k: int) -> list[list]:
    sims = dirs @ dirs.T
    np.fill_diagonal(sims, -np.inf)
    idx = np.argsort(-sims, axis=1)[:, :k]
    return [[[int(j), round(float(sims[i, j]), 3)] for j in idx[i]]
            for i in range(len(dirs))]


# -------------------------------------------------------------------- tokens
class SpanFinder:
    """Map (prompt text, harvested position) -> char span of that token."""

    def __init__(self, tokenizer_id: str, bos_offset: int):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.bos_offset = bos_offset
        self._cache: dict[str, list[tuple[int, int]]] = {}

    def offsets(self, text: str) -> list[tuple[int, int]]:
        if text not in self._cache:
            if len(self._cache) > 4096:
                self._cache.clear()
            enc = self.tok(text, add_special_tokens=False,
                           return_offsets_mapping=True)
            self._cache[text] = enc["offset_mapping"]
        return self._cache[text]

    def span(self, text: str, pos: int) -> tuple[int, int] | None:
        i = pos - self.bos_offset
        offs = self.offsets(text)
        if 0 <= i < len(offs):
            a, b = offs[i]
            if b > a:
                return int(a), int(b)
        return None

    def decode_id(self, tid: int) -> str:
        return self.tok.decode([tid])


def windowed(text: str, span: tuple[int, int] | None) -> dict:
    """Trim text to a window around the span; returns {t, a, b}."""
    if span is None:
        t = text[:WINDOW_BEFORE + WINDOW_AFTER]
        if len(text) > len(t):
            t += "…"
        return {"t": t, "a": -1, "b": -1}
    a, b = span
    s = max(0, a - WINDOW_BEFORE)
    e = min(len(text), b + WINDOW_AFTER)
    t = text[s:e]
    if s > 0:
        t = "…" + t
        shift = s - 1
    else:
        shift = 0
    if e < len(text):
        t = t + "…"
    return {"t": t, "a": a - shift, "b": b - shift}


def prompt_entry(d: float, text: str, pos: int, sf: SpanFinder) -> dict:
    w = windowed(text, sf.span(text, pos))
    return {"d": round(float(d), 4), "pos": int(pos), **w}


# --------------------------------------------------------------- member scan
def load_member_scan(run_dir: Path, sf: SpanFinder, top: int = MS_TOP) -> dict | None:
    """Re-express ``member_scan.json`` for the browser, or None if absent.

    The scan holds up to 64 examples per region for each of four modes. Shipping
    those as pre-windowed snippets would mean four overlapping copies of the same
    text — ~80 MB for the K=5190 dictionary. Instead each referenced prompt is
    sent *once* and every example carries only ``(prompt index, char span)``, so
    the UI windows the text itself. That also lets the same prompt back several
    examples at different firing positions without duplicating it.

    Every record is ``[g, pos, a, b, dist, proj]`` (plus ``owner`` for
    ``projall``) so the UI can show both the distance and the magnitude
    regardless of which one the mode ranked by — the point of the toggle is
    comparing them.
    """
    path = run_dir / "member_scan.json"
    if not path.exists():
        return None
    scan = json.loads(path.read_text())
    prompts, regions = scan["prompts"], scan["regions"]

    # Which prompts are actually referenced by the examples we keep?
    used: dict[int, int] = {}
    for reg in regions:
        for mode in MS_MODES:
            for e in reg.get(mode, [])[:top]:
                if e["gid"] < len(prompts):
                    used.setdefault(e["gid"], len(used))

    # Tokenise each referenced prompt once; span lookup is then a list index.
    # SpanFinder's own LRU is far smaller than the number of distinct prompts
    # here, so relying on it would re-tokenise almost every lookup.
    offs = {g: sf.offsets(prompts[g]) for g in used}

    def rec(e: dict, mode: str) -> list | None:
        g = e["gid"]
        if g not in used:
            return None
        o = offs[g]
        i = e["pos"] - sf.bos_offset
        a, b = (o[i] if 0 <= i < len(o) else (-1, -1))
        if b <= a:
            a, b = -1, -1
        if mode in ("closest", "random"):
            dist, proj = e["v"], e.get("proj", 0.0)
        else:
            dist, proj = e.get("dist", 0.0), e["v"]
        out = [used[g], e["pos"], int(a), int(b),
               round(float(dist), 4), round(float(proj), 3)]
        if mode == "projall":
            out.append(int(e.get("owner", -1)))
        return out

    out_regions = []
    for reg in regions:
        entry = {"nS": reg.get("nScan", 0),
                 "pm": reg.get("projMean", 0.0),
                 "psd": reg.get("projSd", 0.0),
                 "pmx": reg.get("projMax", 0.0),
                 "pq": reg.get("projQ", [])}
        for mode in MS_MODES:
            rows = [rec(e, mode) for e in reg.get(mode, [])[:top]]
            entry[mode] = [r for r in rows if r is not None]
        out_regions.append(entry)

    order = sorted(used, key=used.get)
    return {"topN": top, "scanned": scan.get("nActsScanned", 0),
            "shareCorr": scan.get("memberShareCorr", 0.0),
            "prompts": [prompts[g] for g in order],
            "regions": out_regions}


# ---------------------------------------------------------------------- main
def seed_shape(parts, dirs, means):
    """How much any exemplar-anchored ranking can be trusted, per region.

    The exemplar is the *first-arrival* activation, so every ranking measured
    against it ("closest", "top projection") is anchored on an accident of
    stream order. Two things say how bad that is, and both come free from the
    reservoir of member directions the dictionary already stores:

      seedCos  cos(exemplar, mean member direction) — is the seed even
               representative of the cell it opened?
      pc1      share of within-region variance on the leading axis. The SAE
               idiom assumes ~1 (the feature IS a direction); measured ~0.1
               against a ~0.04 null, so a region is a solid angle, not a ray.
      effDim   participation ratio 1/Σλ², the number of dimensions the cell
               actually occupies.

    `null` is pc1 for the same number of *isotropic* directions in the same
    dimension, so the UI can say how much of the anisotropy is real.
    """
    n_s = [len(p.sample_members) for p in parts]
    out = []
    for i, p in enumerate(parts):
        if len(p.sample_members) < 5:
            out.append(None)
            continue
        M = np.stack(p.sample_members).astype(np.float64)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        Mc = M - M.mean(0)
        # eigenvalues of the (n, n) Gram, not an (n, d) SVD — same spectrum
        w = np.linalg.eigvalsh(Mc @ Mc.T)
        w = np.clip(w, 0.0, None)
        tot = w.sum()
        if tot <= 1e-12:
            out.append(None)
            continue
        ev = w / tot
        out.append([round(float(dirs[i] @ means[i]), 3),
                    round(float(ev[-1]), 3),
                    round(float(1.0 / np.sum(ev ** 2)), 1),
                    int(len(M))])
    n = int(np.median(n_s)) if n_s else 0
    d = int(dirs.shape[1])
    null = 0.0
    if n >= 5:
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(24):
            R = rng.normal(size=(n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Rc = R - R.mean(0)
            w = np.clip(np.linalg.eigvalsh(Rc @ Rc.T), 0.0, None)
            vals.append(w[-1] / w.sum())
        null = round(float(np.median(vals)), 3)
    return out, null


def cell_shell(parts, dirs, chunk_budget: int = 200_000_000):
    """Where each member actually sits in its cell, and how nearly it was lost.

    The SAE idiom has no analogue of this because an SAE feature has no rival:
    activation is a scalar against one direction. An EP assignment is a
    *contest* — ``argmax_k`` over every exemplar — so each member carries two
    numbers that the dashboard has so far thrown away:

      d       cosine distance to its own exemplar. Members fill the cell out to
              ``theta``; the median sits ~90% of the way to the wall, so a
              top-k list is a view of the innermost sliver.
      margin  own similarity minus the best rival's. Measured median 0.023-0.030
              with **39-46% of members inside 0.02** of being claimed by another
              region: membership is a weak plurality, not the clean fact the UI
              currently implies. Negative values are real — leader clustering
              assigns on arrival, so a later exemplar can end up closer, and
              merges move the goalposts (4% of members at p16, 14% at p4).

    Both come off the pickle's uniform member reservoir, so this needs no GPU
    and no activation cache and runs for every dictionary, not just the ones
    with a member scan. 13 s at K=5190, dominated by one chunked matmul.

    Returns a per-region ``[d_milli, margin_milli]`` int pair-of-lists (or
    None), compact enough to ship 30 members/region at K=5190 for ~1 MB.
    """
    K = len(parts)
    M, owner = [], []
    for i, p in enumerate(parts):
        for v in p.sample_members:
            M.append(v)
            owner.append(i)
    if not M:
        return [None] * K
    M = np.asarray(M, dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    owner = np.asarray(owner)

    own_s = np.empty(len(M), np.float32)
    rival = np.empty(len(M), np.float32)
    rows = max(1, int(chunk_budget // max(K, 1)))
    for a in range(0, len(M), rows):
        b = min(len(M), a + rows)
        S = M[a:b] @ dirs.T
        r = np.arange(b - a)
        own_s[a:b] = S[r, owner[a:b]]
        S[r, owner[a:b]] = -2.0          # the contest excludes the incumbent
        rival[a:b] = S.max(axis=1)

    out: list = [None] * K
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(K + 1))
    for i in range(K):
        idx = order[bounds[i]:bounds[i + 1]]
        if len(idx) < 5:
            continue
        d = np.rint((1.0 - own_s[idx]) * 1000).astype(int)
        m = np.rint((own_s[idx] - rival[idx]) * 1000).astype(int)
        o = np.argsort(d)                # sorted by radius: the UI bands on it
        out[i] = [d[o].tolist(), m[o].tolist()]
    return out


def build_one(spec: dict, lens: dict, out_dir: Path,
              jlens: np.ndarray | None = None, jlens_n: int = 0,
              with_neighbors: bool = True, ms_top: int = MS_TOP) -> dict:
    t0 = time.time()
    path = spec.get("path") or spec["path_fn"]()
    with open(path, "rb") as f:
        dic = pickle.load(f)
    parts = list(dic.partitions)
    K = len(parts)
    print(f"[{spec['key']}] K={K}")

    dirs = np.stack([p.exemplar_direction for p in parts]).astype(np.float32)
    means = np.stack([p.mean_member_direction for p in parts]).astype(np.float32)
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)

    print("  layouts…")
    tsne, pca2 = layouts(dirs)
    sp = sphere_points(dirs)
    edges = voronoi_edges(sp) if K <= EDGE_MAX_K else []
    neighbors = knn(dirs, KNN_K)

    print("  seed representativeness…")
    seed_stats, seed_null = seed_shape(parts, dirs, means)

    print("  cell shell (member radii + assignment margins)…")
    shells = cell_shell(parts, dirs)

    # Exact full-space neighbourhood: true nearest exemplars, not a 3-D projection.
    hoods = None
    if with_neighbors:
        print("  neighbourhoods (full-space)…")
        from .neighborhood import all_neighborhoods
        hoods = all_neighborhoods(dirs, float(dic.threshold))

    print("  logit lens…")
    sf = SpanFinder(spec["tokenizer"], spec["bos_offset"])
    ln_v = np.log(lens["W_U"].shape[0])
    verb = lambda H: np.round(1.0 - H / ln_v, 3)  # 1 - normalized entropy
    lens_ex, lH_ex, _ = lens_topk_stats(dirs, lens, k=LENS_K)
    lens_mn, lH_mn, _ = lens_topk_stats(means, lens, k=LENS_K)
    lv_ex, lv_mn = verb(lH_ex), verb(lH_mn)
    if jlens is not None:
        print("  jacobian lens…")
        jl_ex, jH_ex, jm_ex = lens_topk_stats(dirs @ jlens.T, lens, k=LENS_K)
        jl_mn, jH_mn, jm_mn = lens_topk_stats(means @ jlens.T, lens, k=LENS_K)
        jv_ex, jv_mn = verb(jH_ex), verb(jH_mn)
    decode_cache: dict[int, str] = {}

    def dec(ids: list[int]) -> list[str]:
        out = []
        for t in ids:
            if t not in decode_cache:
                decode_cache[t] = sf.decode_id(t)
            out.append(decode_cache[t])
        return out

    # Load the member scan before the region loop: when it exists it supersedes
    # the pickle's "nearest members" card, so shipping all MAX_SAMPLES closest
    # prompts would be dead payload (34% of the K=5190 file). The UI still needs
    # sm[0] for the Exemplar card, and search indexes sm[:3].
    ms = (load_member_scan(Path(spec["path"]).parent, sf, top=ms_top)
          if spec.get("path") else None)
    n_samples = 3 if ms is not None else MAX_SAMPLES

    print("  prompts…")
    regions = []
    total_members = 0
    for i, p in enumerate(parts):
        total_members += p.member_count
        closest = p.closest_prompts[:n_samples]
        far = p.farthest_prompts[:MAX_BOUNDARY]
        md = (p.sum_dist_to_exemplar / p.member_count) if p.member_count else 0.0
        var = max(0.0, p.sum_sq_dist_to_exemplar / max(p.member_count, 1) - md * md)
        region = {
            "i": i,
            "n": int(p.member_count),
            "coh": round(float(p.member_coherence), 3),
            "md": round(float(md), 4),
            "sd": round(float(np.sqrt(var)), 4),
            "xy": [round(float(tsne[i, 0]), 2), round(float(tsne[i, 1]), 2)],
            "pca": [round(float(pca2[i, 0]), 3), round(float(pca2[i, 1]), 3)],
            "s": [round(float(v), 3) for v in sp[i]],
            "nb": neighbors[i],
            "lens": dec(lens_ex[i]),
            "lensM": dec(lens_mn[i]),
            "lv": float(lv_ex[i]),
            "lvM": float(lv_mn[i]),
            "label": p.label or None,
            "sm": [prompt_entry(d, t, pos, sf) for d, t, pos in closest],
            "bd": [prompt_entry(d, t, pos, sf) for d, t, pos in far],
        }
        if seed_stats[i] is not None:
            region["seed"] = seed_stats[i]   # [cos(e,mean), pc1, effDim, nSamp]
        if shells[i] is not None:
            # [dist, margin] for each sampled member, milli-units, radius-sorted.
            # Ints because a 3-dp float costs 3x the bytes for the same precision.
            region["sh"] = shells[i]
        if hoods is not None:
            h = hoods[i]
            # `bd` is omitted whenever it is the two-point closed form of `d`
            # (1 - sqrt(1 - d/2)), which is every pair at every scale measured
            # so far — the browser rederives it. A 6th element appears only for
            # a pair where a third cell actually constrained the wall, so the
            # override is self-announcing rather than silently absent.
            region["hood"] = {
                "nb": [[nb["j"], nb["d"], nb["ang"],
                        int(nb["adj"]), int(nb["reach"])]
                       + ([] if nb["twoPoint"] else [nb["bd"]])
                       for nb in h["nb"]],
                "stress": h["angleStress"],
                "tested": h["tested"], "nAdj": h["nAdj"], "nReach": h["nReach"],
                "twoPoint": h["twoPoint"],
            }
        if jlens is not None:
            region.update({
                "jl": dec(jl_ex[i]), "jlM": dec(jl_mn[i]),
                "jv": float(jv_ex[i]), "jvM": float(jv_mn[i]),
                "jm": round(float(jm_ex[i]), 3), "jmM": round(float(jm_mn[i]), 3),
            })
        regions.append(region)

    meta = {}
    if spec.get("meta"):
        meta = json.loads(Path(spec["meta"]).read_text())
    payload = {
        "key": spec["key"], "model": spec["model"], "layer": spec["layer"],
        "p": spec["p"], "K": K, "dModel": int(dirs.shape[1]),
        "pc1Null": seed_null,   # PC1 share for isotropic points, same n and d
        "threshold": round(float(dic.threshold), 4),
        "nActs": int(meta.get("n_activations", total_members)),
        "corpus": meta.get("corpus", "pile"),
        "ctx": meta.get("context_length", 128),
        "largest": int(meta.get("largest_partition",
                                max(p.member_count for p in parts))),
        "singletons": int(meta.get("singletons",
                                   sum(1 for p in parts if p.member_count == 1))),
        "regions": regions, "edges": edges,
    }
    if ms is not None:
        payload["ms"] = ms
        print(f"  member scan: {len(ms['prompts'])} prompts, "
              f"{ms['topN']}/mode, shareCorr={ms['shareCorr']}")
    if jlens is not None:
        payload["jlens"] = {"layer": spec["layer"], "corpus": "wikitext-103",
                            "score": "1 - H/ln|V|", "nPrompts": jlens_n}
    out = out_dir / f"{spec['key']}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False,
                              separators=(",", ":")))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s)")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens-cache", required=True)
    ap.add_argument("--out", default="artifacts/figures/dashboard_data")
    ap.add_argument("--only", default=None,
                    help="comma-separated dict keys to (re)build")
    ap.add_argument("--no-neighbors", action="store_true",
                    help="skip the exact full-space neighbourhood pass "
                         "(~0.4 ms/region)")
    ap.add_argument("--ms-top", type=int, default=MS_TOP,
                    help="member-scan examples per mode shipped to the browser")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    lenses: dict[str, dict] = {}
    jlenses: dict[tuple[str, int], np.ndarray | None] = {}
    for spec in DICTS:
        if only and spec["key"] not in only:
            continue
        mk = spec["model_key"]
        if mk not in lenses:
            print(f"loading lens weights: {mk}")
            lenses[mk] = load_lens(mk, Path(args.lens_cache))
        jk = (mk, spec["layer"])
        if jk not in jlenses:
            jlenses[jk] = (load_jlens(mk, spec["layer"], Path(args.lens_cache))
                           if has_jlens(mk) else None)
        build_one(spec, lenses[mk], out_dir, jlens=jlenses[jk],
                  jlens_n=jlens_n_prompts(mk, spec["layer"], Path(args.lens_cache)),
                  with_neighbors=not args.no_neighbors, ms_top=args.ms_top)


if __name__ == "__main__":
    main()
