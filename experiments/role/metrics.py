"""Metrics for role geometry over an EP partition.

Everything here operates on a **paired** assignment array

    A[d, c, j] = region id of content token j of document d under condition c

which is the object the six-condition corpus exists to produce. Content is
identical along the ``c`` axis by construction (checked in
:mod:`role.corpus`), so any variation along ``c`` is attributable to the role
tag and nothing else.

Three families:

- §1 displacement — does the tag move the region assignment, and if so, does it
  move it the *same way* everywhere? This is the measurement that distinguishes
  "role is one direction acting on a content-partitioned space" from "role is
  content-conditional", and neither a linear probe nor a single-region EP
  analysis can express the difference.
- §3 occupancy — ``p_role(r)`` over all K regions, and the region polarity
  ``λ(r)``, which is the only supervision anything downstream uses.
- §4 PCA / axis — treat the exemplar matrix as a dictionary and ask whether λ is
  low-dimensional in it.

Nulls are not optional anywhere. Every quantity here has a nonzero expectation
under no role structure: NMI is biased upward by region count, and two random
unit vectors in 2560 dimensions have nonzero expected alignment, so displacement
coherence has a floor that depends on the cosine distance between the regions
involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# --- §1 displacement ------------------------------------------------------

def flip_rate(A: np.ndarray, c_a: int, c_b: int) -> float:
    """P(region differs | content fixed) between two conditions."""
    return float((A[:, c_a, :] != A[:, c_b, :]).mean())


def flip_rate_matrix(A: np.ndarray) -> np.ndarray:
    """(C, C) matrix of pairwise flip rates. Diagonal is 0 by construction."""
    n_cond = A.shape[1]
    out = np.zeros((n_cond, n_cond))
    for i in range(n_cond):
        for j in range(n_cond):
            if i != j:
                out[i, j] = flip_rate(A, i, j)
    return out


def _coherence(deltas: np.ndarray) -> float:
    """``‖Σδ‖ / Σ‖δ‖`` — 1 if every displacement points the same way, ~0 if not.

    This is the same statistic EP uses for ``member_coherence``, applied to
    displacement vectors instead of member directions, which keeps it comparable
    to the numbers already in the dictionaries.
    """
    if len(deltas) == 0:
        return float("nan")
    norms = np.linalg.norm(deltas, axis=1).sum()
    if norms == 0:
        return float("nan")
    return float(np.linalg.norm(deltas.sum(axis=0)) / norms)


@dataclass
class DisplacementResult:
    condition_a: str
    condition_b: str
    n_flipped: int
    n_total: int
    flip_rate: float
    coherence: float
    null_coherence: float
    null_coherence_std: float
    mean_direction: np.ndarray          # (d,) unit; the "role displacement axis"
    mean_cosine_distance: float

    @property
    def coherence_z(self) -> float:
        """How many nulls above the null mean the observed coherence sits."""
        if self.null_coherence_std == 0 or np.isnan(self.null_coherence_std):
            return float("nan")
        return float(
            (self.coherence - self.null_coherence) / self.null_coherence_std
        )


def displacement(
    A: np.ndarray,
    E: np.ndarray,
    c_a: int,
    c_b: int,
    names: tuple[str, str] = ("a", "b"),
    n_null: int = 20,
    n_distance_bins: int = 20,
    seed: int = 0,
) -> DisplacementResult:
    """Displacement statistics for the ordered condition pair ``(c_a, c_b)``.

    ``E`` is the (K, d) matrix of unit exemplar directions. Only positions whose
    region actually changed contribute: an unmoved token has δ = 0 and would
    inflate coherence toward 1 for a reason that has nothing to do with role.

    The null preserves the *geometry* of the observed transitions while
    destroying their pairing with content: for each observed transition we draw a
    random region pair from the same cosine-distance bin. Without the distance
    matching the null is far too weak — nearby regions have high expected
    alignment purely because they are nearby.
    """
    rng = np.random.default_rng(seed)
    ra = A[:, c_a, :].ravel()
    rb = A[:, c_b, :].ravel()
    moved = ra != rb
    n_total = int(len(ra))
    n_flipped = int(moved.sum())

    if n_flipped == 0:
        return DisplacementResult(
            condition_a=names[0], condition_b=names[1], n_flipped=0,
            n_total=n_total, flip_rate=0.0, coherence=float("nan"),
            null_coherence=float("nan"), null_coherence_std=float("nan"),
            mean_direction=np.zeros(E.shape[1], dtype=np.float32),
            mean_cosine_distance=float("nan"),
        )

    src, dst = ra[moved], rb[moved]
    deltas = E[dst] - E[src]
    obs = _coherence(deltas)
    cos_d = 1.0 - np.einsum("ij,ij->i", E[src], E[dst])

    # Bin the observed transitions by cosine distance, then resample region
    # pairs from the full K x K space within the same bin.
    K = E.shape[0]
    edges = np.linspace(0.0, 2.0, n_distance_bins + 1)
    obs_bins = np.clip(np.digitize(cos_d, edges) - 1, 0, n_distance_bins - 1)

    # Precompute a pool of random pairs per bin, sized to the demand.
    n_pool = max(20_000, 10 * n_flipped)
    pool_a = rng.integers(0, K, size=n_pool)
    pool_b = rng.integers(0, K, size=n_pool)
    keep = pool_a != pool_b
    pool_a, pool_b = pool_a[keep], pool_b[keep]
    pool_d = 1.0 - np.einsum("ij,ij->i", E[pool_a], E[pool_b])
    pool_bins = np.clip(np.digitize(pool_d, edges) - 1, 0, n_distance_bins - 1)
    by_bin = {b: np.where(pool_bins == b)[0] for b in range(n_distance_bins)}

    null_vals: list[float] = []
    for _ in range(n_null):
        pick_a = np.empty(n_flipped, dtype=np.int64)
        pick_b = np.empty(n_flipped, dtype=np.int64)
        ok = np.ones(n_flipped, dtype=bool)
        for b in range(n_distance_bins):
            idx = np.where(obs_bins == b)[0]
            if len(idx) == 0:
                continue
            cand = by_bin[b]
            if len(cand) == 0:
                # No random pair reproduces this distance; drop these rather
                # than substituting a pair from another bin, which would make
                # the null easier to beat.
                ok[idx] = False
                continue
            sel = rng.choice(cand, size=len(idx), replace=True)
            pick_a[idx] = pool_a[sel]
            pick_b[idx] = pool_b[sel]
        if ok.sum() == 0:
            continue
        null_vals.append(_coherence(E[pick_b[ok]] - E[pick_a[ok]]))

    mean_delta = deltas.mean(axis=0)
    norm = np.linalg.norm(mean_delta)
    unit = (mean_delta / norm) if norm > 0 else mean_delta

    return DisplacementResult(
        condition_a=names[0], condition_b=names[1],
        n_flipped=n_flipped, n_total=n_total,
        flip_rate=n_flipped / n_total,
        coherence=obs,
        null_coherence=float(np.mean(null_vals)) if null_vals else float("nan"),
        null_coherence_std=(
            float(np.std(null_vals)) if len(null_vals) > 1 else float("nan")
        ),
        mean_direction=unit.astype(np.float32),
        mean_cosine_distance=float(cos_d.mean()),
    )


def paired_displacement_magnitude(
    dirs: np.ndarray,
    row_of: np.ndarray,
    c_a: int,
    c_b: int,
    threshold: float,
) -> dict:
    """How far the tag moves an activation, in the units EP's cells are built in.

    A flip rate answers "does the tag change the region"; it cannot say *how
    close* it came. This gives the quantity: the cosine distance between the two
    tagged copies of the same content token, against the calibration
    ``threshold`` that defines a cell's radius. ``ratio_to_threshold`` < 1 means
    role displacement lands inside a cell by construction, which turns "the flip
    rate was low" into a statement that does not depend on the partition's
    resolution.

    ``dirs`` are centered unit directions — the same representation EP assigns
    on, so the distances are directly comparable to the threshold.
    ``row_of[d, c, j]`` indexes ``dirs``, or -1 where absent.
    """
    ra = row_of[:, c_a, :].ravel()
    rb = row_of[:, c_b, :].ravel()
    ok = (ra >= 0) & (rb >= 0)
    if not ok.any():
        return {"n": 0}
    cos_d = 1.0 - np.einsum("ij,ij->i", dirs[ra[ok]], dirs[rb[ok]])
    return {
        "n": int(ok.sum()),
        "mean": float(cos_d.mean()),
        "median": float(np.median(cos_d)),
        "p90": float(np.percentile(cos_d, 90)),
        "max": float(cos_d.max()),
        "threshold": float(threshold),
        "ratio_to_threshold": float(cos_d.mean() / threshold)
        if threshold > 0 else float("nan"),
        "frac_beyond_threshold": float((cos_d > threshold).mean()),
    }


def transition_matrix(
    A: np.ndarray, c_a: int, c_b: int, n_regions: int,
) -> np.ndarray:
    """Counts ``T[r_a, r_b]`` of the region coupling induced by the tag change.

    A directed graph on regions — the same object as the runner-up competition
    graph, so ask it the same questions: sparse? concentrated? does
    ``user→assistant`` share edge structure with ``user→tool``?
    """
    ra = A[:, c_a, :].ravel()
    rb = A[:, c_b, :].ravel()
    T = np.zeros((n_regions, n_regions), dtype=np.int64)
    np.add.at(T, (ra, rb), 1)
    return T


def coupling_sparsity(T: np.ndarray) -> dict[str, float]:
    """How concentrated the transition graph is, per source region."""
    row_sums = T.sum(axis=1)
    live = row_sums > 0
    if not live.any():
        return {"mean_out_degree": float("nan"), "mean_top1_mass": float("nan")}
    rows = T[live].astype(np.float64)
    sums = row_sums[live][:, None]
    out_degree = (rows > 0).sum(axis=1)
    top1 = rows.max(axis=1) / sums[:, 0]
    return {
        "n_live_source_regions": int(live.sum()),
        "mean_out_degree": float(out_degree.mean()),
        "median_out_degree": float(np.median(out_degree)),
        "mean_top1_mass": float(top1.mean()),
        "frac_deterministic": float((out_degree == 1).mean()),
    }


# --- §3 occupancy and polarity -------------------------------------------

def occupancy(
    A: np.ndarray, c: int, n_regions: int, prior: float = 0.5,
) -> np.ndarray:
    """``p_condition(r)`` over all K regions, Jeffreys-smoothed.

    The smoothing matters: λ is a log ratio, and an unsmoothed zero in either
    condition sends it to ±inf for a region that may hold three tokens.
    """
    counts = np.bincount(A[:, c, :].ravel(), minlength=n_regions).astype(
        np.float64
    )
    counts += prior
    return counts / counts.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits. Symmetric, bounded by 1."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def js_divergence_null(
    A: np.ndarray, c_a: int, c_b: int, n_regions: int,
    n_null: int = 50, prior: float = 0.5, seed: int = 0,
) -> tuple[float, float]:
    """Mean and sd of JS under shuffled condition labels.

    Shuffles which condition each token's assignment is attributed to, *within*
    a (doc, position) group — so the null preserves the content structure
    exactly and destroys only the role attribution.
    """
    rng = np.random.default_rng(seed)
    pair = A[:, [c_a, c_b], :]                     # (D, 2, J)
    vals = []
    for _ in range(n_null):
        flip = rng.random(pair.shape[0:1] + pair.shape[2:]) < 0.5
        a = np.where(flip, pair[:, 1, :], pair[:, 0, :])
        b = np.where(flip, pair[:, 0, :], pair[:, 1, :])
        pa = np.bincount(a.ravel(), minlength=n_regions).astype(np.float64) + prior
        pb = np.bincount(b.ravel(), minlength=n_regions).astype(np.float64) + prior
        vals.append(js_divergence(pa / pa.sum(), pb / pb.sum()))
    return float(np.mean(vals)), float(np.std(vals))


def polarity(
    p_a: np.ndarray, p_b: np.ndarray, min_count: np.ndarray | None = None,
    min_members: int = 5,
) -> np.ndarray:
    """``λ(r) = log(p_a(r) / p_b(r))``, NaN for under-populated regions.

    Regions with |λ| large and tiny member counts are noise, and they are
    exactly the ones a naive top-m ranking picks up first.
    """
    lam = np.log(p_a) - np.log(p_b)
    if min_count is not None:
        lam = np.where(min_count >= min_members, lam, np.nan)
    return lam


def polarity_rank_stability(lams: list[np.ndarray], top_m: int = 32) -> float:
    """Jaccard overlap of the top-|λ| region sets across streaming seeds.

    Region *identity* is seed-dependent — the exemplar is a first-arrival
    accident — so a λ ranking that reshuffles across seeds is not a real object
    and nothing may be built on it.
    """
    sets = []
    for lam in lams:
        order = np.argsort(-np.abs(np.nan_to_num(lam, nan=0.0)))
        sets.append(set(order[:top_m].tolist()))
    if len(sets) < 2:
        return float("nan")
    pairs = [
        len(a & b) / len(a | b)
        for i, a in enumerate(sets) for b in sets[i + 1:]
    ]
    return float(np.mean(pairs))


# --- §4 axis and PCA ------------------------------------------------------

def role_axis(lam: np.ndarray, E: np.ndarray) -> np.ndarray:
    """``a = Σ_r λ(r) e_r``, unit-normalized.

    A difference-of-means in *region* space: over unit-norm real activations
    weighted by discrete occupancy, rather than over raw activations. Regions
    with NaN λ (under-populated) contribute nothing.
    """
    w = np.nan_to_num(lam, nan=0.0)
    a = w @ E
    n = np.linalg.norm(a)
    return (a / n).astype(np.float32) if n > 0 else a.astype(np.float32)


def pca(X: np.ndarray, n_components: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centered PCA via SVD. Returns (components (n, d), explained ratio)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    total = var.sum()
    ratio = var / total if total > 0 else var
    k = min(n_components, vt.shape[0])
    return vt[:k], ratio[:k]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


# --- shared: mutual information ------------------------------------------

def mutual_information(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(I(a;b), H(a), H(b)) in bits, from a joint histogram of two label arrays."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)), dtype=np.float64)
    np.add.at(joint, (ia, ib), 1.0)
    joint /= joint.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    def _h(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    nz = joint > 0
    mi = float(
        (joint[nz] * np.log2(joint[nz] / np.outer(pa, pb)[nz])).sum()
    )
    return mi, _h(pa), _h(pb)


def normalized_mi(a: np.ndarray, b: np.ndarray) -> float:
    """``I(a;b) / H(b)`` — the fraction of ``b``'s entropy that ``a`` explains."""
    mi, _, hb = mutual_information(a, b)
    return mi / hb if hb > 0 else float("nan")


def normalized_mi_null(
    regions: np.ndarray, labels: np.ndarray, n_null: int = 20, seed: int = 0,
) -> tuple[float, float]:
    """Mean and sd of ``normalized_mi`` under globally permuted labels.

    NMI is biased upward by the number of regions, and K here is set by a
    percentile calibrated on this corpus, so the null must be recomputed per
    configuration rather than assumed to be 0.

    **Do not use this for the role label** — use :func:`normalized_mi_null_paired`.
    Global permutation is the right null for an unpaired label such as content
    token id, but for role it is not merely conservative, it is wrong; see the
    paired version's docstring.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_null):
        vals.append(normalized_mi(regions, rng.permutation(labels)))
    return float(np.mean(vals)), float(np.std(vals))


def paired_role_nmi(A: np.ndarray) -> float:
    """``I(region; condition) / H(condition)`` computed on the paired array."""
    n_cond = A.shape[1]
    regions = A.transpose(0, 2, 1).reshape(-1, n_cond)     # (groups, C)
    labels = np.tile(np.arange(n_cond), regions.shape[0])
    return normalized_mi(regions.ravel(), labels)


def normalized_mi_null_paired(
    A: np.ndarray, n_null: int = 20, seed: int = 0,
) -> tuple[float, float]:
    """Role-NMI null that permutes conditions *within* each (doc, position) group.

    Why the obvious null is wrong here. Every content token appears exactly once
    per condition, so when the flip rate is low each region receives a nearly
    perfectly balanced mix of the six conditions — the empirical joint factorizes
    almost exactly, the plug-in MI is ~0, and there is essentially **no**
    finite-sample bias. Permuting labels globally destroys that balance and
    manufactures bias out of nothing: on a 76-region dry run it produced a null of
    0.105 against an observed 0.003, which would have "proved" the absence of role
    information regardless of the data.

    Permuting within the group keeps the design intact — each group still carries
    all six conditions exactly once, and each region keeps its member set — and
    destroys only *which* condition landed in *which* region. That is the null
    hypothesis we actually want: the tag changes the region, but not in a
    role-dependent way.
    """
    rng = np.random.default_rng(seed)
    n_cond = A.shape[1]
    regions = A.transpose(0, 2, 1).reshape(-1, n_cond)
    flat_regions = regions.ravel()
    n_groups = regions.shape[0]
    vals = []
    for _ in range(n_null):
        # One independent permutation of the condition labels per group.
        perms = np.argsort(rng.random((n_groups, n_cond)), axis=1)
        vals.append(normalized_mi(flat_regions, perms.ravel()))
    return float(np.mean(vals)), float(np.std(vals))


# --- §7 predictive comparison --------------------------------------------

@dataclass
class ClassifierResult:
    name: str
    accuracy: float
    macro_auroc: float
    n_train: int
    n_test: int
    n_features: int


def region_table_classifier(
    regions_train: np.ndarray, y_train: np.ndarray,
    regions_test: np.ndarray, y_test: np.ndarray,
    n_regions: int, n_classes: int, name: str = "ep_region",
    prior: float = 0.5,
) -> ClassifierResult:
    """``P(role | region)`` fitted as a lookup table on train, scored on test.

    Deliberately the weakest possible use of a hard assignment: one categorical
    feature, K parameters per class, no activations. Works unchanged for any hard
    partition, which is how the k-means-at-matched-K control arm reuses it —
    "probe beats EP" is uninformative given the capacity gap, but "EP beats
    k-means at matched K" is a claim about EP's partition specifically.
    """
    from sklearn.metrics import roc_auc_score

    table = np.full((n_regions, n_classes), prior, dtype=np.float64)
    np.add.at(table, (regions_train, y_train), 1.0)
    table /= table.sum(axis=1, keepdims=True)

    proba = table[regions_test]
    pred = proba.argmax(axis=1)
    acc = float((pred == y_test).mean())
    present = np.unique(y_test)
    if len(present) < 2:
        auroc = float("nan")
    else:
        # Renormalize over the classes actually present, else the OvR scores do
        # not sum to 1 and sklearn rejects them.
        sub = proba[:, present]
        sub = sub / sub.sum(axis=1, keepdims=True)
        if len(present) == 2:
            # sklearn wants a 1-D score for the binary case, not an (n, 2)
            # matrix. This branch fires for every user-vs-assistant comparison,
            # which is the most common one we run.
            auroc = float(
                roc_auc_score((y_test == present[1]).astype(int), sub[:, 1])
            )
        else:
            auroc = float(
                roc_auc_score(
                    y_test, sub, multi_class="ovr", average="macro",
                    labels=present,
                )
            )
    return ClassifierResult(
        name=name, accuracy=acc, macro_auroc=auroc,
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        n_features=int(n_regions),
    )
