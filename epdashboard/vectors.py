"""Export the per-region direction vectors — the steering/ablation payload.

SAEDashboard ships a decoder direction per feature; Neuronpedia stores it on
``Neuron.vector`` with ``hasVector = true`` and steers with ``h ← h + α·v``.
EP has two candidate analogues per region and they are *not* interchangeable:

``exemplar``
    The activation that seeded the region, centered and unit-normalised. It
    defines the cell — membership is ``1 − cos(h − c, e) ≤ θ`` — but it is a
    single token's activation, a first-arrival accident of the leader-clustering
    order, not an average of anything.

``mean``
    The mean of the member unit vectors, renormalised. This is the "average
    region vector": it is what the region's contents actually have in common,
    and it is the one to reach for first when steering.

**Space.** Both live in *centered* space: ``v_ep = (h − c)/‖h − c‖`` where ``c``
is the calibration center, exported here as ``center``. That matters
differently for the two interventions:

* **Steering is additive, so the center cancels** — ``h + α·v`` is the same
  operation whether or not ``h`` was centered. The exported unit vectors drop
  straight into Neuronpedia. Pick ``α`` on the scale of the residual norm at
  the layer, not on the scale of 1.
* **Ablation/projection does not cancel.** Removing a direction means
  ``h' = c + (I − vvᵀ)(h − c)``. Centering first is not optional: ``c`` is a
  large mean-activation offset, and projecting it out along with the feature is
  a much bigger edit than the one intended.

The vectors are stored once per dictionary in ``vectors.npz`` rather than
inlined per region: at d_model 5120 a region is 10 KB of float16 binary but
~60 KB of JSON text, which would dominate the batch files it lives in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_vectors(d, region_ids: list[int], cfg, out_dir: Path) -> dict:
    """Write ``vectors.npz``; return the header block describing it.

    Rows are aligned to ``regionIds``, which is the same order the JSON batches
    were written in, so ``row = regionIds.index(i)`` — not ``row = i`` — unless
    the run built all K regions.
    """
    dtype = np.dtype(cfg.vector_dtype)
    ids = np.asarray(region_ids, dtype=np.int32)
    path = out_dir / "vectors.npz"
    np.savez_compressed(
        path,
        regionIds=ids,
        exemplar=d.E[ids].astype(dtype),
        mean=d.means[ids].astype(dtype),
        center=d.center.astype(np.float32),
    )
    return {
        "file": path.name,
        "keys": ["exemplar", "mean"],
        "dtype": str(dtype),
        "dModel": int(d.E.shape[1]),
        "n": int(len(ids)),
        # Everything a consumer needs to place these vectors correctly.
        "space": "centered-unit",
        "centerNorm": round(float(np.linalg.norm(d.center)), 4),
        "hook": f"blocks.{cfg.layer}.hook_resid_post",
        "bytes": path.stat().st_size,
    }


def to_jsonl(dict_dir: str | Path, out_path: str | Path,
             which: str = "mean", strength: float = 10.0) -> Path:
    """Flatten ``vectors.npz`` + ``header.json`` into one JSON object per line.

    Field names follow Neuronpedia's vector-upload shape (``modelId``,
    ``layer``, ``index``, ``vector``, ``hookName``, ``vectorLabel``,
    ``vectorDefaultSteerStrength``); confirm them against the live schema
    before an import, since this file is written blind against it.

    ``strength`` is a placeholder default. There is no principled value here —
    see the module docstring on scale — so calibrate it against the residual
    norm at the layer before publishing.
    """
    dict_dir, out_path = Path(dict_dir), Path(out_path)
    header = json.loads((dict_dir / "header.json").read_text())
    labels = {r["i"]: (r["label"] or " / ".join(r["lensTop"]))
              for r in header["regionTable"]}
    meta = header["dict"]
    with np.load(dict_dir / "vectors.npz") as z:
        ids, V = z["regionIds"], z[which].astype(np.float32)
    with out_path.open("w") as f:
        for row, i in enumerate(ids.tolist()):
            f.write(json.dumps({
                "modelId": meta["model_id"],
                "layer": meta["layer"],
                "index": int(i),
                "vector": [round(float(x), 6) for x in V[row]],
                "hasVector": True,
                "hookName": header["vectors"]["hook"],
                "vectorLabel": f"EP {meta['run']} r{i}: {labels.get(i, '')}",
                "vectorDefaultSteerStrength": strength,
            }) + "\n")
    return out_path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        prog="epdashboard.vectors",
        description="Convert a built dictionary's vectors.npz to JSONL.")
    ap.add_argument("dict_dir", help="<out>/<run slug>/ holding header.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--which", default="mean", choices=("mean", "exemplar"))
    ap.add_argument("--strength", type=float, default=10.0)
    args = ap.parse_args()
    out = args.out or str(Path(args.dict_dir) / f"vectors_{args.which}.jsonl")
    print(to_jsonl(args.dict_dir, out, args.which, args.strength))


if __name__ == "__main__":
    main()
