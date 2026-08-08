"""Arm B primitives: cell assignment, occupancy histograms, drift statistics.

Gate 0B scored *how far* an activation sits from an exemplar. Everything here
scores *which cell it landed in*, so the dictionary is used as a quantiser
rather than as a metric. That is a different mechanism and it can win or lose
independently of the Gate 0B verdict.

Two families of statistic:

- **surprisal**, -log p_ref(cell), which is defined per request and therefore
  comparable head-to-head with Gate 0B's S1 on the same AUROC scale;
- **total variation** between a window histogram and the reference, which is
  only defined for a window and is the actual population-drift monitor.

Both are computed identically over EP cells and over matched-K random-coreset
Voronoi cells, because the question is whether *leader clustering* buys
anything, not whether quantising buys anything.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _unit(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32) - center.astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)


def assign(x: np.ndarray, ref: np.ndarray, center: np.ndarray | None = None,
           chunk: int = 4096) -> np.ndarray:
    """Nearest-reference cell id for every row of `x`.

    `ref` must already be centred unit directions; `x` is raw unless `center`
    is None. Pure argmin, matching `Dictionary.assign` — there is no threshold
    and no -1 escape (GATE0A_FINDINGS trap 1).
    """
    import torch

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    R = torch.from_numpy(np.ascontiguousarray(ref)).to(dev, torch.float32)
    out = np.empty(len(x), dtype=np.int32)
    for s in range(0, len(x), chunk):
        blk = x[s:s + chunk]
        q = _unit(np.asarray(blk), center) if center is not None else blk
        Q = torch.from_numpy(np.ascontiguousarray(q)).to(dev, torch.float32)
        out[s:s + chunk] = (Q @ R.T).argmax(dim=1).cpu().numpy()
        del Q
    return out


def occupancy(cells: np.ndarray, n_cells: int) -> np.ndarray:
    return np.bincount(cells, minlength=n_cells).astype(np.float64)


def log_ref_prob(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Laplace-smoothed log cell probability.

    Smoothing is not cosmetic here: at p=1 there are 5796 cells and a test
    window of 50 requests, so unseen cells are the common case and an
    unsmoothed estimate would return -inf for most of them.
    """
    p = (counts + alpha) / (counts.sum() + alpha * len(counts))
    return np.log(p)


def surprisal(cells: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """Per-request rarity score. Larger = landed in a rarer cell = more OOD."""
    return -logp[cells]


def tv_distance(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    a = counts_a / max(counts_a.sum(), EPS)
    b = counts_b / max(counts_b.sum(), EPS)
    return 0.5 * float(np.abs(a - b).sum())


# ------------------------------------------------------------ balance (B3)

def hist_entropy(counts: np.ndarray) -> float:
    """Shannon entropy in nats of the occupancy distribution."""
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def hist_gini(counts: np.ndarray) -> float:
    """Gini coefficient of cell occupancy. 0 = perfectly even, 1 = one cell."""
    c = np.sort(counts.astype(np.float64))
    n = len(c)
    if c.sum() <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n)


def balance(counts: np.ndarray) -> dict:
    n_cells = len(counts)
    occupied = int((counts > 0).sum())
    h = hist_entropy(counts)
    return {
        "n_cells": n_cells,
        "occupied": occupied,
        "occupied_frac": occupied / n_cells,
        "entropy_nats": h,
        "entropy_ratio": h / np.log(n_cells) if n_cells > 1 else float("nan"),
        "gini": hist_gini(counts),
        "top1_share": float(counts.max() / max(counts.sum(), EPS)),
        "top10_share": float(np.sort(counts)[-10:].sum() / max(counts.sum(), EPS)),
    }


# -------------------------------------------------------- drift power (B1)

def window_statistics(win_cells: np.ndarray, ref_counts: np.ndarray,
                      logp: np.ndarray) -> dict[str, float]:
    """Scalar summaries of one traffic window."""
    n_cells = len(ref_counts)
    return {
        "surprisal": float(surprisal(win_cells, logp).mean()),
        "tv": tv_distance(occupancy(win_cells, n_cells), ref_counts),
    }


def power_curve(clean_cells: np.ndarray, dirty_cells: np.ndarray,
                ref_counts: np.ndarray, *, window: int, eps: float,
                n_windows: int = 400, alpha_fpr: float = 0.01,
                seed: int = 0) -> dict[str, float]:
    """Detection power at a false-alarm rate fixed by construction.

    The null threshold is taken from *resampled clean windows*, not from a
    chi-square asymptotic, because at K=5796 cells and N=50 requests no
    asymptotic applies. Power is then the fraction of contaminated windows
    above that threshold, so a scorer with no signal returns ~alpha_fpr rather
    than something that has to be interpreted.
    """
    rng = np.random.default_rng(seed)
    logp = log_ref_prob(ref_counts)
    n_dirty = int(round(window * eps))
    n_clean = window - n_dirty

    def draw(dirty: bool) -> np.ndarray:
        parts = [rng.choice(clean_cells, size=n_clean if dirty else window,
                            replace=True)]
        if dirty and n_dirty > 0:
            parts.append(rng.choice(dirty_cells, size=n_dirty, replace=True))
        return np.concatenate(parts)

    null = [window_statistics(draw(False), ref_counts, logp)
            for _ in range(n_windows)]
    alt = [window_statistics(draw(True), ref_counts, logp)
           for _ in range(n_windows)]

    out = {}
    for stat in ("surprisal", "tv"):
        n = np.array([d[stat] for d in null])
        a = np.array([d[stat] for d in alt])
        thr = np.quantile(n, 1.0 - alpha_fpr)
        out[f"power_{stat}"] = float((a > thr).mean())
        out[f"thr_{stat}"] = float(thr)
    return out


def scalar_power_curve(clean: np.ndarray, dirty: np.ndarray, *, window: int,
                       eps: float, n_windows: int = 400,
                       alpha_fpr: float = 0.01, seed: int = 0) -> dict:
    """Power of the window *mean* of any per-request scalar score.

    The categorical monitor has to beat this, not just beat the categorical
    coreset. Gate 0B's distance scores are weak per request but averaging N of
    them shrinks the noise by sqrt(N), so a scorer that loses at N=1 can still
    win at N=500. Comparing cells only against cells would hide that.
    """
    rng = np.random.default_rng(seed)
    n_dirty = int(round(window * eps))
    n_clean = window - n_dirty

    def draw(is_dirty: bool) -> float:
        parts = [rng.choice(clean, size=n_clean if is_dirty else window,
                            replace=True)]
        if is_dirty and n_dirty > 0:
            parts.append(rng.choice(dirty, size=n_dirty, replace=True))
        return float(np.concatenate(parts).mean())

    null = np.array([draw(False) for _ in range(n_windows)])
    alt = np.array([draw(True) for _ in range(n_windows)])
    thr = np.quantile(null, 1.0 - alpha_fpr)
    return {"power_mean": float((alt > thr).mean()), "thr_mean": float(thr)}


# ------------------------------------------------- exemplar stability (B2)

def leader_cluster(x: np.ndarray, threshold: float, center: np.ndarray,
                   order: np.ndarray, chunk: int = 4096,
                   max_exemplars: int = 20000) -> np.ndarray:
    """EP's own construction: first-arrival leader clustering at fixed theta.

    Inference-only — this is a scan with a distance test, no gradients and no
    fitting. Reimplemented here rather than imported so B2 can control the
    arrival order, which is the whole point of the test.

    Blocked, not row-by-row: a point already within theta of an *existing*
    exemplar is absorbed no matter what the rest of the block does, since the
    exemplar set only ever grows. So one matmul per block settles the majority
    and only the survivors need sequential treatment. This is exact, not an
    approximation — at 204k activations the row-by-row version is ~5 orders of
    magnitude more device round-trips.
    """
    import torch

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    exemplars: list[np.ndarray] = []
    E: "torch.Tensor | None" = None

    def add(rows: np.ndarray) -> None:
        nonlocal E
        exemplars.extend(rows)
        T = torch.from_numpy(np.ascontiguousarray(rows)).to(dev, torch.float32)
        E = T if E is None else torch.cat([E, T], dim=0)

    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        rows = np.asarray(x[np.sort(idx)])
        q = _unit(rows, center)[np.argsort(np.argsort(idx))]
        if E is None:
            add(q[:1])
            q = q[1:]
            if not len(q):
                continue
        Q = torch.from_numpy(np.ascontiguousarray(q)).to(dev, torch.float32)
        far = (1.0 - (Q @ E.T).max(dim=1).values) > threshold
        cand = np.flatnonzero(far.cpu().numpy())
        # Survivors interact only with each other, in arrival order.
        n_before = len(exemplars)
        for j in cand:
            new = E[n_before:]
            if len(new) and float(1.0 - (Q[j:j + 1] @ new.T).max()) <= threshold:
                continue
            add(q[j:j + 1])
            if len(exemplars) >= max_exemplars:
                return np.stack(exemplars)
    return np.stack(exemplars)


def exemplar_overlap(a: np.ndarray, b: np.ndarray, threshold: float) -> dict:
    """How much of dictionary `a` is reproduced by dictionary `b`.

    "Same exemplar" cannot mean identity — different arrival orders pick
    different activations. It means "within the same theta ball", i.e. b has an
    exemplar that would have been absorbed by a's cell.
    """
    import torch

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    A = torch.from_numpy(np.ascontiguousarray(a)).to(dev, torch.float32)
    B = torch.from_numpy(np.ascontiguousarray(b)).to(dev, torch.float32)
    d = 1.0 - (A @ B.T)
    nn_ab = d.min(dim=1).values.cpu().numpy()
    nn_ba = d.min(dim=0).values.cpu().numpy()
    return {
        "K_a": int(len(a)), "K_b": int(len(b)),
        "frac_a_matched": float((nn_ab <= threshold).mean()),
        "frac_b_matched": float((nn_ba <= threshold).mean()),
        "mean_nn_dist_a_to_b": float(nn_ab.mean()),
        "median_nn_dist_a_to_b": float(np.median(nn_ab)),
    }
