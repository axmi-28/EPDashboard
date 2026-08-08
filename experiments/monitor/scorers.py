"""The six scorers and the two metrics the decision rule is stated in.

Every scorer maps an activation to a scalar where **larger = more OOD**, so a
single AUROC convention applies throughout. Two need a sign flip to get there
and both are noted at their definition.

Memory budget. S1/S2 use the dictionary's K exemplars; S3 uses exactly K
activations, so those three are matched. **S4 is not matched** — a covariance
is D x D = 5.3M floats regardless of K, which is 13x S1's budget at K=176 and
0.4x at K=5796. It is carried as a fixed-cost reference point for decision
rule 2 ("is the monitoring framing dead"), not as a matched competitor, and the
CSV records both budgets so the asymmetry is visible.

S3 is averaged over several independent coreset draws. EP gets one
deterministic dictionary; scoring a random baseline on a single draw would
credit or blame it for sampling luck.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
N_CORESET_DRAWS = 5


# ------------------------------------------------------------------ metrics

def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Rank-sum AUROC with mid-rank tie correction.

    Tie correction matters: a degenerate scorer that returns one constant must
    come out at exactly 0.500, not at whatever the sort order happened to give.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_positive).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def tpr_at_fpr(scores: np.ndarray, is_positive: np.ndarray,
               fpr: float = 0.01) -> float:
    """TPR at a threshold set on the negatives, so FPR <= `fpr` by construction.

    The threshold is the (1 - fpr) quantile of the negative scores; ">" rather
    than ">=" keeps the realised FPR at or below target when scores tie.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_positive).astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    thresh = np.quantile(scores[~y], 1.0 - fpr)
    return float((scores[y] > thresh).mean())


# ------------------------------------------------------------------ helpers

def _unit(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32) - center.astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)


def _cosine_top2(q: np.ndarray, ref: np.ndarray,
                 chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 smallest cosine distances from each row of `q` to `ref`.

    Chunked so the (N, K) similarity block never lands on the host in full;
    at N=12000, K=5796 that block is 278 MB.
    """
    import torch

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    R = torch.from_numpy(np.ascontiguousarray(ref)).to(dev, torch.float32)
    d1, d2 = [], []
    for s in range(0, len(q), chunk):
        Q = torch.from_numpy(np.ascontiguousarray(q[s:s + chunk])).to(
            dev, torch.float32)
        sim = (Q @ R.T).clamp_(-1.0, 1.0)
        k = 2 if R.shape[0] >= 2 else 1
        top = sim.topk(k, dim=1).values
        d1.append((1.0 - top[:, 0]).cpu().numpy())
        d2.append((1.0 - top[:, min(1, k - 1)]).cpu().numpy())
    return np.concatenate(d1), np.concatenate(d2)


# ------------------------------------------------------------------ scorers

def s1_s2_ep(x: np.ndarray, exemplars: np.ndarray,
             center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """S1 nearest-exemplar cosine distance, S2 negated boundary margin.

    S2's raw margin (d_2nd - d_1st) is *large* when an activation sits
    confidently inside one cell, so the OOD score is its negation.
    """
    d1, d2 = _cosine_top2(_unit(x, center), exemplars)
    return d1, -(d2 - d1)


def s3_coreset_knn(x: np.ndarray, pool: np.ndarray, center: np.ndarray,
                   k: int, *, n_draws: int = N_CORESET_DRAWS,
                   seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Nearest-neighbour cosine distance to K random build-stream activations.

    Same centring and normalisation as EP, so the only difference from S1 is
    that the K reference vectors were sampled rather than leader-clustered.
    Returns the across-draw mean score and the per-draw scores.
    """
    q = _unit(x, center)
    per_draw = []
    for i in range(n_draws):
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        ref = _unit(pool[idx], center)
        d1, _ = _cosine_top2(q, ref)
        per_draw.append(d1)
    return np.mean(per_draw, axis=0), per_draw


def fit_mahalanobis(pool: np.ndarray, shrinkage: float = 0.01,
                    chunk: int = 20_000) -> dict:
    """Mean and shrunk inverse covariance of the build stream, in raw space.

    Shrinkage toward a scaled identity keeps the Cholesky well-conditioned;
    with n/D ~ 87 the sample covariance is estimable but not comfortably so.

    Accumulated in float64 chunk by chunk. Upcasting the whole (203k, 2304)
    pool at once would be 3.7 GB, and the centred copy another 3.7 GB.
    """
    n, d = pool.shape
    mu = np.zeros(d, dtype=np.float64)
    for s in range(0, n, chunk):
        mu += pool[s:s + chunk].astype(np.float64).sum(axis=0)
    mu /= n

    S = np.zeros((d, d), dtype=np.float64)
    for s in range(0, n, chunk):
        Xc = pool[s:s + chunk].astype(np.float64) - mu
        S += Xc.T @ Xc
    S /= (n - 1)
    S = (1.0 - shrinkage) * S + shrinkage * (np.trace(S) / d) * np.eye(d)
    L = np.linalg.cholesky(S)
    return {"mean": mu, "chol": L, "shrinkage": shrinkage,
            "n": int(len(pool)), "dim": int(d)}


def s4_mahalanobis(x: np.ndarray, fit: dict) -> np.ndarray:
    """Mahalanobis distance in raw activation space (not the unit sphere)."""
    from scipy.linalg import solve_triangular

    z = solve_triangular(fit["chol"], (x.astype(np.float64) - fit["mean"]).T,
                         lower=True)
    return np.sqrt((z ** 2).sum(axis=0))


def s5_entropy(entropy_max: np.ndarray) -> np.ndarray:
    """The model's own uncertainty. Needs no dictionary and no build data."""
    return np.asarray(entropy_max, dtype=np.float64)


def s0_length(n_tokens: np.ndarray) -> np.ndarray:
    """Prompt token count — the triviality control, not a real monitor.

    If S0 separates a rung as well as S1 does, that rung's result is about
    prompt length and nothing in it should be read as evidence about EP.
    """
    return np.asarray(n_tokens, dtype=np.float64)
