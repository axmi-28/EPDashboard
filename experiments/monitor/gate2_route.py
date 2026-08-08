"""Region-lookup scoring: assign, count, score, and the matched-K null.

The whole Arm A "model" lives here and it is arithmetic, not learning. Given a
partition and a labelled set:

    1. assign each activation to a region (argmax cosine, no threshold)
    2. per region, count how many training prompts landed there and how many
       carried the label
    3. score a new activation by the smoothed rate of its region

Step 3 is a lookup. The only free parameter is the Laplace smoother alpha,
fixed at 0.5, which decides what an unseen or singleton region is worth.

Two statistics are reported for every partition and they must not be confused:

- `top1_*` is fitted on the whole labelled set and then read off the same set.
  It is the number the earlier refusal artifact quotes and it is optimistic by
  construction — the region was *chosen* for being the most label-enriched.
- `cv_auroc` / `cv_tpr1` cross-fit: region rates come from k-1 folds, the held
  fold is scored by lookup. A region absent from the training folds falls back
  to the training prior, which is the honest cost of a partition too fine for
  the label budget.

`coreset_partition` is the control that killed Gate 0B and applies unchanged
here: K activations drawn at random from the same Pile stream EP's exemplars
came from, used as if they were exemplars. If the random partition scores the
same, the finding is about K and about the label's separability, not about EP.
"""

from __future__ import annotations

import numpy as np

from .scorers import auroc, tpr_at_fpr

ALPHA = 0.5          # Laplace smoother on the per-region rate
EPS = 1e-12


def unit(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    v = x.astype(np.float32) - center.astype(np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)


def assign(x: np.ndarray, exemplars: np.ndarray, center: np.ndarray,
           chunk: int = 4096) -> np.ndarray:
    """Region id per row. Pure argmax cosine — every activation lands somewhere."""
    q = unit(x, center)
    e = np.ascontiguousarray(exemplars.astype(np.float32))
    out = np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):
        out[s:s + chunk] = np.argmax(q[s:s + chunk] @ e.T, axis=1)
    return out


def coreset_partition(pool: np.ndarray, center: np.ndarray, k: int,
                      seed: int) -> np.ndarray:
    """K random build-stream activations, unit-normalised into exemplar form."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return unit(np.asarray(pool[np.sort(idx)]), center)


# ------------------------------------------------------------------ lookup

def region_rates(region: np.ndarray, y: np.ndarray, k: int,
                 alpha: float = ALPHA) -> tuple[np.ndarray, float]:
    """Smoothed positive rate per region, plus the global prior fallback."""
    n = np.bincount(region, minlength=k).astype(np.float64)
    h = np.bincount(region, weights=y.astype(np.float64), minlength=k)
    prior = float((y.sum() + alpha) / (len(y) + 2 * alpha))
    rates = (h + alpha) / (n + 2 * alpha)
    rates[n == 0] = prior          # never observed in fit -> no information
    return rates, prior


def cross_fit_scores(region: np.ndarray, y: np.ndarray, k: int, *,
                     n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold lookup score for every row.

    Folds are stratified on the label so a small positive class cannot vanish
    from a training split and collapse that fold to the prior.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    out = np.empty(len(y), dtype=np.float64)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        rates, _ = region_rates(region[tr], y[tr], k)
        out[te] = rates[region[te]]
    return out


def summarise(region: np.ndarray, y: np.ndarray, k: int, *,
              min_members: int = 5, flag_rate: float = 0.5,
              n_folds: int = 5, seed: int = 0) -> dict:
    """Every A0 number for one (partition, labelled set) pair."""
    y = np.asarray(y).astype(np.int64)
    n = np.bincount(region, minlength=k)
    h = np.bincount(region, weights=y.astype(np.float64), minlength=k)
    occupied = np.where(n > 0)[0]

    # Fitted-on-all statistics. Optimistic; kept because they are what the
    # earlier single-dictionary refusal artifact reports.
    frac = np.where(n > 0, h / np.maximum(n, 1), 0.0)
    eligible = occupied[n[occupied] >= min_members]
    if len(eligible) == 0:
        eligible = occupied
    top = int(eligible[np.argmax(h[eligible])])         # most positives, not rate

    flagged = eligible[(frac[eligible] >= flag_rate)]
    tp = float(h[flagged].sum())
    fp = float(n[flagged].sum() - tp)

    cv = cross_fit_scores(region, y, k, n_folds=n_folds, seed=seed)
    return {
        "K": int(k),
        "n_occupied": int(len(occupied)),
        "occupancy_top1_share": float(n.max() / n.sum()),
        "top1_pid": top,
        "top1_n": int(n[top]),
        "top1_pos_frac": float(frac[top]),
        "top1_recall": float(h[top] / max(y.sum(), 1)),
        "n_flagged_regions": int(len(flagged)),
        "flag_recall": float(tp / max(y.sum(), 1)),
        "flag_precision": float(tp / max(tp + fp, 1.0)),
        "cv_auroc": auroc(cv, y == 1),
        "cv_tpr1": tpr_at_fpr(cv, y == 1, 0.01),
        "cv_tpr5": tpr_at_fpr(cv, y == 1, 0.05),
    }


def _ridge_dual(xt: np.ndarray, yt: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge in dual form: n << d here, so solve the n x n system."""
    xc = xt - xt.mean(axis=0, keepdims=True)
    g = xc @ xc.T
    a = np.linalg.solve(g + lam * np.eye(len(g)), yt - yt.mean())
    return xc.T @ a


def probe_cross_fit(x: np.ndarray, y: np.ndarray, *, kind: str = "ridge",
                    lam: float = 1.0, n_folds: int = 5,
                    seed: int = 0) -> np.ndarray:
    """Out-of-fold scores for a linear probe on the raw activation.

    The reference the region lookup has to beat. Fit closed-form so this stays
    unambiguously inference-only — no gradient steps anywhere.
    """
    x = x.astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    out = np.empty(len(y), dtype=np.float64)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        xt = x[tr]
        if kind == "ridge":
            w = _ridge_dual(xt, y[tr], lam)
        elif kind == "diffmean":
            # Same centring/normalising EP uses, so the two see the same space.
            mu = xt.mean(axis=0)
            xn = xt - mu
            xn /= np.linalg.norm(xn, axis=1, keepdims=True) + EPS
            w = xn[y[tr] == 1].mean(axis=0) - xn[y[tr] == 0].mean(axis=0)
        else:
            raise ValueError(kind)
        w /= np.linalg.norm(w) + EPS
        out[te] = (x[te] - xt.mean(axis=0)) @ w
    return out


def summarise_coreset(x: np.ndarray, y: np.ndarray, pool: np.ndarray,
                      center: np.ndarray, k: int, *, n_draws: int = 3,
                      seed: int = 0, **kw) -> tuple[dict, dict, list[dict]]:
    """Mean and sd of `summarise` over independent matched-K coreset draws."""
    rows = []
    for i in range(n_draws):
        ref = coreset_partition(pool, center, k, seed + i)
        rows.append(summarise(assign(x, ref, center), y, len(ref), seed=seed,
                              **kw))
    # `top1_pid` indexes a different random partition in every draw, so its
    # mean is meaningless; K is constant by construction.
    skip = {"top1_pid", "K"}
    keys = [key for key, v in rows[0].items()
            if key not in skip and isinstance(v, (int, float))]
    mean = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    sd = {key: float(np.std([r[key] for r in rows], ddof=1)) for key in keys}
    return mean, sd, rows
