"""True Voronoi neighbours of an EP region, computed in the full activation space.

The dashboard's PCA-3D sphere is a projection from d=2560 down to 3, so cells
that look adjacent there usually are not, and real neighbours land on opposite
sides of the ball. This module never projects to decide anything: adjacency,
distances, and boundary positions are all computed in the original space, and
2D is used only to choose *bearings* for drawing.

Geometry
--------
EP assigns a centred unit direction ``x`` to the cell whose exemplar maximises
``x·e_k``, provided ``1 - x·e_k <= θ``. So cell ``i`` is the polyhedral cone

    C_i = {x : x·e_i >= x·e_k for all k}  ∩  {x : x·e_i >= 1 - θ}

and cells ``i, j`` genuinely share a boundary facet iff some unit ``x`` has
``x·e_i = x·e_j >= x·e_k`` for every other ``k``. Call the largest similarity
attainable on that shared facet

    t*(i,j) = max { x·e_i : ||x|| <= 1, x·e_i = x·e_j, x·e_i >= x·e_k ∀k }

Two facts make this cheap and exact rather than sampled:

1. The feasible set is the unit ball intersected with a polyhedral cone ``K``,
   and ``max_{x ∈ K, ||x|| <= 1} c·x = ||P_K(c)||`` — the norm of the Euclidean
   projection of ``c`` onto ``K``. By Moreau's decomposition ``P_K(c)`` is the
   *residual* of the nonnegative least-squares problem ``min_{μ>=0} ||Aᵀμ + c||``,
   which `scipy.optimize.nnls` solves exactly. No sampling, no LP solver, no
   midpoint heuristic.
2. Every constraint row and the objective lie in the span of the exemplars
   involved, and ``K`` splits as ``K_S ⊕ S^⊥`` over that span, so projecting
   into an orthonormal basis of the span first is exact — turning a 2560-dim
   problem into a ~64-dim one.

``t*`` is directly interpretable and doubles as the boundary position: a point
sitting exactly on the i|j facet is at cosine distance ``1 - t*`` from *both*
exemplars, so the bisector's radius in the polar view is exact rather than
"half the distance between them". Classifications:

    adjacent    t* > x·e_k for all k          → real shared facet
    unreachable facet exists but t* < 1 - θ    → no activation can ever sit on
                                                 it; both cells stop at θ first
    shadowed    no feasible point             → another cell fills the gap, so
                                                 j is near but not a neighbour

Measured result: the general solve is *never* needed at the scales tried here
------------------------------------------------------------------------------
Ignoring every cell but ``i`` and ``j``, the constrained maximiser is available
in closed form: the projection of ``e_i`` onto the bisecting hyperplane is
``(e_i + e_j)/2``, so

    x* = (e_i + e_j)/||e_i + e_j||        t_2 = sqrt((1 + cos_ij) / 2)

``t_2`` is an upper bound on ``t*``, and it is *attained* whenever ``x*`` also
satisfies the dropped constraints — a single matvec to check. Across every
dictionary built here (K from 144 to 5190, d from 2048 to 5120) that check
passes for **100% of pairs**: no third cell is ever active, so ``t* = t_2`` and
the bisector radius is a deterministic function of the cosine distance alone,
``1 - sqrt(1 - d_ij/2)``, sitting at a flat ~0.287·d_ij.

That is the same degeneracy as the vacuous adjacency: with K << d the exemplars
are near-orthogonal (cos ~0.1), so the midpoint of any pair beats every other
exemplar by a wide margin. The wall carries no information the cosine distance
did not already carry, and the honest reading is that Voronoi structure in this
regime is trivial rather than that we have measured something about a region.

So the closed form is tried first and the NNLS (and the QR it needs) runs only
if the check fails. Exact in either branch — verified bit-identical to the pure
NNLS at 4 dp over 640 sampled pairs — and 18 ms/region became 0.41 ms. The
``twoPoint`` count rides along in the output so a future model that genuinely
needs the solver announces itself instead of silently regressing.

Bearings for the polar view come from classical MDS on the exact cosine
distance matrix of the neighbourhood only — a local fit, far less distorted
than one global PCA — and each point's *radius* is then overwritten with its
exact distance to the focus. ``angleStress`` reports how much the bearings
still misrepresent neighbour-to-neighbour distances, so the picture carries its
own error bar.
"""

from __future__ import annotations

import numpy as np

CANDIDATES = 64       # cells entered as constraints in the adjacency test
KEEP = 16             # neighbours kept in the payload
EPS = 1e-9


def _span_basis(E: np.ndarray) -> np.ndarray:
    """Orthonormal basis (d, r) for the row space of ``E``."""
    q, _ = np.linalg.qr(E.T)
    return q


def _cone_reach(c: np.ndarray, A: np.ndarray) -> float:
    """``max {c·x : ||x|| <= 1, A x >= 0}`` = ``||P_K(c)||``.

    Solved through Moreau: the projection onto the polar cone is the NNLS fit
    of ``-c`` by the constraint normals, and ``P_K(c)`` is its residual.
    """
    from scipy.optimize import nnls

    if A.size == 0:
        return float(np.linalg.norm(c))
    mu, _ = nnls(A.T, -c)
    return float(np.linalg.norm(A.T @ mu + c))


def _classical_mds(D: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS to 2D from a square distance matrix."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(w2)


def _angle_stress(pts: np.ndarray, D: np.ndarray) -> float:
    """Normalised stress of the drawn layout against true distances.

    0 = every drawn pairwise distance matches the full-space cosine distance;
    higher = the bearings are compressing structure that does not fit in 2D.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    iu = np.triu_indices(n, 1)
    d2 = np.linalg.norm(pts[iu[0]] - pts[iu[1]], axis=1)
    dt = D[iu]
    denom = np.sum(dt ** 2)
    return float(np.sqrt(np.sum((d2 - dt) ** 2) / denom)) if denom > EPS else 0.0


def region_neighborhood(i: int, dirs: np.ndarray, threshold: float,
                        sims: np.ndarray | None = None,
                        n_candidates: int = CANDIDATES,
                        keep: int = KEEP) -> dict:
    """Full-space neighbourhood of region ``i``.

    ``dirs`` is the (K, d) matrix of unit exemplar directions. ``sims`` may be a
    precomputed row ``dirs @ dirs[i]`` to avoid recomputing it per region.
    """
    K = len(dirs)
    if sims is None:
        sims = dirs @ dirs[i]
    order = np.argsort(-sims)
    cand = np.array([j for j in order[:n_candidates + 1] if j != i][:n_candidates])
    if cand.size == 0:
        return {"i": i, "nb": [], "angleStress": 0.0, "tested": 0}

    # The whole fast path is inner products among {e_i} ∪ candidates, so take
    # that Gram matrix once. Index 0 is the focus, a+1 is cand[a].
    sub = np.concatenate([[i], cand])
    G = np.clip(dirs[sub] @ dirs[sub].T, -1.0, 1.0)
    span = None                # QR is only needed if some pair falls back

    recs = []
    n_two = 0
    for a, j in enumerate(cand):
        cos_ij = float(G[0, a + 1])
        # Try the two-cell closed form first. x* ∝ (e_i + e_j) is optimal for
        # the pair; it is optimal for the full problem iff it also satisfies the
        # constraints we dropped, (e_i - e_k)·x* >= 0 ∀k≠j — which expands to
        # 1 + cos_ij - cos_ik - cos_jk >= 0, no projection required.
        slack = 1.0 + cos_ij - G[0, 1:] - G[a + 1, 1:]
        slack[a] = np.inf                      # j's own row is the equality
        if slack.min() >= -EPS:
            t = float(np.sqrt(max(0.0, (1.0 + cos_ij) / 2.0)))
            two = True
            n_two += 1
        else:
            # Exact reduction: objective and constraints live in the exemplars'
            # span, turning a d-dimensional NNLS into a ~65-dimensional one.
            if span is None:
                Q = _span_basis(dirs[sub])
                e = dirs[sub] @ Q              # (1 + m, r) coordinates
                # Rows of `A` are (e_i - e_k): A x >= 0 says no k beats i at x.
                span = (e[0], e[0][None, :] - e[1:])
            ei, A_all = span
            # Bisector with j as two opposed inequalities, j's own row dropped.
            others = np.delete(A_all, a, axis=0)
            A = np.vstack([others, A_all[a][None, :], -A_all[a][None, :]])
            t = _cone_reach(ei, A)
            two = False
        recs.append({
            "j": int(j),
            "cos": round(cos_ij, 4),
            "d": round(max(0.0, 1.0 - cos_ij), 4),
            "t": round(t, 4),
            "bd": round(max(0.0, 1.0 - t), 4),     # exact bisector radius
            "twoPoint": two,     # False => the wall is *not* a function of `d`
            "adj": bool(t > EPS),
            "reach": bool(t >= 1.0 - threshold),
        })

    recs.sort(key=lambda r: r["d"])
    shown = recs[:keep]

    # Bearings: local classical MDS on the exact cosine distances of the shown
    # set plus the focus, then radii replaced by their exact values.
    idx = np.array([i] + [r["j"] for r in shown])
    sub_sims = np.clip(dirs[idx] @ dirs[idx].T, -1.0, 1.0)
    D = np.maximum(1.0 - sub_sims, 0.0)
    np.fill_diagonal(D, 0.0)
    pts = _classical_mds(D)
    pts = pts - pts[0]
    drawn = [pts[0]]
    for n, r in enumerate(shown, start=1):
        v = pts[n]
        norm = float(np.linalg.norm(v))
        ang = float(np.arctan2(v[1], v[0])) if norm > EPS else 0.0
        # Round *then* wrap: a bearing a hair below zero lands on 359.98, which
        # rounds up to a 360.0 the consumer has to treat as a special case.
        r["ang"] = round(np.degrees(ang) % 360.0, 1) % 360.0
        drawn.append(np.array([np.cos(ang), np.sin(ang)]) * r["d"])

    return {
        "i": i,
        "nb": shown,
        "angleStress": round(_angle_stress(np.array(drawn), D), 3),
        "tested": int(cand.size),
        "nAdj": int(sum(1 for r in recs if r["adj"])),
        "nReach": int(sum(1 for r in recs if r["adj"] and r["reach"])),
        "twoPoint": n_two,
    }


def all_neighborhoods(dirs: np.ndarray, threshold: float,
                      n_candidates: int = CANDIDATES, keep: int = KEEP,
                      log_every: int = 250) -> list[dict]:
    """Neighbourhood records for every region, in index order."""
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + EPS)
    S = dirs @ dirs.T
    np.clip(S, -1.0, 1.0, out=S)
    out = []
    for i in range(len(dirs)):
        out.append(region_neighborhood(i, dirs, threshold, sims=S[i],
                                       n_candidates=n_candidates, keep=keep))
        if log_every and (i + 1) % log_every == 0:
            print(f"    neighbourhood {i + 1}/{len(dirs)}")
    return out
