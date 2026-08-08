"""Strip hub dictionaries down to what a scorer actually needs.

88% of the p=1 pickle (1.55 GB of 1.77 GB) is the `sample_members`
visualisation reservoir — 168,164 stored member directions that no scorer
touches. The (5796, 2304) exemplar matrix that does the work is 53 MB. Loading
the full sweep as pickles would peak near 4 GB of RSS for no benefit, so we
convert once and read `.npz` thereafter.

`center` and `threshold` travel with the dictionary (they are plain instance
attributes, serialised by `Dictionary.__getstate__`), so the lean file is
self-contained: the same mu the dictionary was built with, no calibration
cache, no model load. Verified in Gate 0A: mu is bit-identical across the whole
p sweep for a given (model, layer).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PERCENTILES = (1, 2, 4, 8, 10)
DEFAULT_MODEL_SHORT = "gemma-2-2b-it"
DEFAULT_LAYER = 20

# Hub dataset revision pinned in Gate 0A. Recorded in every lean file so a
# result can be traced to the exact blobs it was computed from.
HUB_REVISION = "0ec26618db2bd64cdbf00a9a1b0eb23fffdea12a"
BLOB_SHA = {
    1: "bc4bc3a05e0908246a2d44d00fec5918eda920ef",
    2: "a1d7f472d4590176f079ff0051739717dd078d1c",
    4: "a5545f48c0f11f44fcaf60613d251e5aadda9095",
    8: "dd01e87e44455b718f71aaa380cad2d70d92eba5",
    10: "88a2c83b6f3ec42ee3a529d063ebc21f35768263",
}


def lean_path(root: Path, model_short: str, layer: int, p: int) -> Path:
    return Path(root) / f"{model_short}_L{layer}_p{p}.npz"


def build_lean(root: Path, *, model_short: str = DEFAULT_MODEL_SHORT,
               layer: int = DEFAULT_LAYER, percentiles=PERCENTILES,
               force: bool = False) -> dict[int, Path]:
    """Download each hub dictionary once and write its lean form."""
    import ep

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for p in percentiles:
        path = lean_path(root, model_short, layer, p)
        out[p] = path
        if path.exists() and not force:
            print(f"  p{p}: {path.name} exists")
            continue
        d = ep.Dictionary.from_hub(model_short, layer=layer, percentile=p)
        exemplars = np.stack([q.exemplar_direction for q in d.partitions])
        means = np.stack([q.mean_member_direction for q in d.partitions])
        np.savez_compressed(
            path,
            exemplars=exemplars.astype(np.float32),
            mean_members=means.astype(np.float32),
            member_count=np.array([q.member_count for q in d.partitions], np.int64),
            member_coherence=np.array([q.member_coherence for q in d.partitions],
                                      np.float32),
            center=d.center.astype(np.float32),
            threshold=np.float32(d.threshold),
            percentile=np.int32(p),
            layer=np.int32(layer),
            hub_revision=np.str_(HUB_REVISION),
            blob_sha=np.str_(BLOB_SHA.get(p, "")),
        )
        print(f"  p{p}: K={len(d.partitions)} theta={d.threshold:.6f} "
              f"-> {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        del d
    return out


def load_lean(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "exemplars": z["exemplars"], "mean_members": z["mean_members"],
        "member_count": z["member_count"],
        "member_coherence": z["member_coherence"],
        "center": z["center"], "threshold": float(z["threshold"]),
        "percentile": int(z["percentile"]), "layer": int(z["layer"]),
        "K": int(z["exemplars"].shape[0]),
        "hub_revision": str(z["hub_revision"]),
        "blob_sha": str(z["blob_sha"]),
    }
