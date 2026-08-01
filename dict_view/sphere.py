"""The Voronoi sphere — dictionary-page layout, recovered from EPDashboard.

Exemplar directions live on the unit sphere in d_model dimensions; here they are
PCA-projected to 3-D and re-normalised onto the unit sphere, and the *projected*
points' spherical Voronoi tessellation is drawn as great-circle arcs. This is a
layout, not the real partition — the true neighbourhood structure is the
full-space cosine table on each region card (high-d Voronoi adjacency is vacuous
at K << d), and the page said so.

Arcs are capped at K <= EDGE_MAX_K: past that they are unreadable and the
payload balloons; the point cloud alone still carries the layout.

Verbatim from ``epdashboard/geometry.py`` at commit 6581874, before the
dictionary page was dropped from the tool. Needs scikit-learn and scipy, which
the EPDashboard build path no longer does.
"""

from __future__ import annotations

import numpy as np

EDGE_MAX_K = 800


def sphere_points(dirs: np.ndarray) -> np.ndarray:
    """(K, 3) PCA projection of unit directions, re-normalised to the sphere."""
    from sklearn.decomposition import PCA
    p3 = PCA(n_components=3, random_state=0).fit_transform(dirs)
    return p3 / (np.linalg.norm(p3, axis=1, keepdims=True) + 1e-12)


def voronoi_edges(points: np.ndarray, n_seg: int = 6) -> list:
    """Great-circle arcs of the spherical Voronoi tessellation as flat
    [x,y,z,…] polylines. Degenerate inputs get a tiny jitter (3 attempts)."""
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


def sphere_payload(E: np.ndarray) -> dict:
    pts = sphere_points(E)
    edges = voronoi_edges(pts) if len(E) <= EDGE_MAX_K else []
    return {"pts": [[round(float(v), 3) for v in p] for p in pts],
            "edges": edges, "edgeMaxK": EDGE_MAX_K}
