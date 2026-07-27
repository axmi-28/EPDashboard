"""Dictionary-level geometry: the Voronoi sphere.

Exemplar directions live on the unit sphere in d_model dimensions; for the
dictionary page they are PCA-projected to 3-D and re-normalised onto the unit
sphere, and the *projected* points' spherical Voronoi tessellation is drawn as
great-circle arcs. This is a layout, not the real partition — the true
neighbourhood structure is the full-space cosine table on each region card
(high-d Voronoi adjacency is vacuous at K ≪ d), and the page says so.

Arcs are capped at K ≤ EDGE_MAX_K: past that they are unreadable and the
payload balloons; the point cloud alone still carries the layout.

CLI (backfill a header built before this module existed, then re-render):

    python -m epdashboard.geometry <out>/<run_name> --run-dir runs/<run_name>
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


def nn_cosine(E: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Each region's cosine to its nearest other exemplar (crowding)."""
    out = np.empty(len(E), dtype=np.float32)
    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ E.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        out[s:s + chunk] = S.max(axis=1)
    return out


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from epdashboard.html import render_all
    from epdashboard.scan import EPDict

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="<out>/<run_name> holding header.json")
    ap.add_argument("--run-dir", default=None,
                    help="run dir with dictionary.pkl (default: runs/<run_name>)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    header = json.loads((out_dir / "header.json").read_text())
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / header["dict"]["run"]
    d = EPDict.load(run_dir)
    if d.K != header["dict"]["K"]:
        raise SystemExit(f"K mismatch: dict {d.K} vs header {header['dict']['K']}")

    header["sphere"] = sphere_payload(d.E)
    nn = nn_cosine(d.E)
    for row in header["regionTable"]:
        row["nn"] = round(float(nn[row["i"]]), 3)
    (out_dir / "header.json").write_text(
        json.dumps(header, ensure_ascii=False, separators=(",", ":")))
    print(f"patched {out_dir/'header.json'}: sphere "
          f"({len(header['sphere']['edges'])} arcs) + nn cosines")

    # Region-record backfill: dictionary-derived fields added after a build
    # (currently the exemplar context). Same pattern as the header patch.
    from transformers import AutoTokenizer

    from epdashboard.writer import exemplar_entry
    tok = AutoTokenizer.from_pretrained(header["dict"]["model_id"])
    bos = 1 if "gemma" in header["dict"]["model_id"].lower() else 0
    for b in header["batches"]:
        path = out_dir / b["file"]
        batch = json.loads(path.read_text())
        for rec in batch["regions"]:
            rec["ex"] = exemplar_entry(d.parts[rec["i"]], tok, bos)
        path.write_text(json.dumps(batch, ensure_ascii=False,
                                   separators=(",", ":")))
        print(f"patched {path}: exemplar contexts")
    for p in render_all(out_dir):
        print("wrote", p)


if __name__ == "__main__":
    main()
