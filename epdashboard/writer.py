"""Assemble per-region records and write the batched JSON dataset.

Layout under ``<out_dir>/<dict_name>/``:

    header.json        dictionary-level metadata, provenance, batch manifest,
                       and a compact per-region summary table (the raw data
                       behind the future dictionary-level dashboard)
    regions_000.json   full region records, ``regions_per_batch`` per file
    index.html         region table + dictionary stats        (html.py)
    regions_000.html   self-contained region cards, one per batch (html.py)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from epdashboard import __version__
from epdashboard.scan import EPDict, RegionScan
from epdashboard.sequences import PromptCache, build_groups


def nearest_regions(E: np.ndarray, k: int) -> list[list]:
    """Top-k neighbors by full-space cosine between exemplar directions."""
    sims = E @ E.T
    np.fill_diagonal(sims, -np.inf)
    idx = np.argsort(-sims, axis=1)[:, :k]
    return [[[int(j), round(float(sims[i, j]), 3)] for j in idx[i]]
            for i in range(len(E))]


def dict_p(d: EPDict) -> float | None:
    """Resolution percentile, from metadata or the run-dir slug (``p8p0``)."""
    for key in ("percentile", "p"):
        if key in d.meta:
            return float(d.meta[key])
    m = re.search(r"_p(\d+)p(\d+)_", d.run_dir.name + "_")
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def _hist(values: np.ndarray, lo: float, hi: float, bins: int) -> list[int]:
    if hi <= lo:
        hi = lo + 1e-6
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    return counts.astype(int).tolist()


def region_record(i: int, d: EPDict, scan: RegionScan, pc: PromptCache,
                  lens_rows: list[dict], neighbors: list[list],
                  bg: np.ndarray, cfg) -> dict:
    p = d.parts[i]
    n_scan = int(scan.n_member[i])
    mean_p = scan.sum_proj[i] / n_scan if n_scan else 0.0
    var_p = max(0.0, scan.sum_sq_proj[i] / n_scan - mean_p ** 2) if n_scan else 0.0

    # Member projection histogram from the uniform reservoir; the grey
    # background is the shared corpus subsample projected onto this region's
    # direction. A common bin range makes the two overlayable.
    member_proj = scan.random.payload_col(i, "proj")
    bg_i = bg[:, i].astype(np.float32) if bg.size else np.zeros(0)
    both = np.concatenate([member_proj, bg_i]) if (member_proj.size or bg_i.size) \
        else np.zeros(1)
    lo, hi = float(both.min()), float(both.max())
    q = (np.round(np.quantile(member_proj, [0.05, 0.25, 0.5, 0.75, 0.95]), 3)
         .tolist() if member_proj.size >= 5 else [])

    groups = build_groups(pc, i, scan.examples(i), d.threshold,
                          cfg.n_bands, tuple(cfg.buffer))
    return {
        "i": i,
        "label": p.label or None,
        "stats": {
            "n": int(p.member_count),
            "nScan": n_scan,
            "density": round(n_scan / max(scan.n_acts, 1), 6),
            "coherence": round(float(p.member_coherence), 3),
            "meanDist": round(scan.sum_dist[i] / n_scan, 4) if n_scan else None,
            "projMean": round(float(mean_p), 3),
            "projSd": round(float(np.sqrt(var_p)), 3),
            "projMax": (round(float(scan.max_proj[i]), 3)
                        if np.isfinite(scan.max_proj[i]) else None),
            "projQ": q,
        },
        "distHist": {"counts": scan.dist_hist[i].tolist(),
                     "max": round(d.threshold, 4)},
        "projHist": {"range": [round(lo, 3), round(hi, 3)],
                     "member": _hist(member_proj, lo, hi, cfg.hist_bins),
                     "nMember": int(member_proj.size),
                     "bg": _hist(bg_i, lo, hi, cfg.hist_bins),
                     "nBg": int(bg_i.size)},
        "neighbors": neighbors[i],
        "lens": lens_rows[i],
        "groups": groups,
    }


def write_dict_output(d: EPDict, scan: RegionScan, pc: PromptCache,
                      lens_rows: list[dict], region_ids: list[int],
                      source_desc: dict, jlens_meta: dict, cfg,
                      out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    neighbors = nearest_regions(d.E, cfg.n_neighbors)
    bg = scan.bg_proj()

    batches, summary = [], []
    ids = list(region_ids)
    for b0 in range(0, len(ids), cfg.regions_per_batch):
        batch_ids = ids[b0:b0 + cfg.regions_per_batch]
        records = [region_record(i, d, scan, pc, lens_rows, neighbors, bg, cfg)
                   for i in batch_ids]
        name = f"regions_{len(batches):03d}.json"
        (out_dir / name).write_text(json.dumps(
            {"batch": len(batches), "regionIds": batch_ids, "regions": records},
            ensure_ascii=False, separators=(",", ":")))
        batches.append({"file": name, "regions": batch_ids})
        for r in records:
            summary.append({
                "i": r["i"], "label": r["label"], "n": r["stats"]["n"],
                "density": r["stats"]["density"],
                "coherence": r["stats"]["coherence"],
                "meanDist": r["stats"]["meanDist"],
                "verb": r["lens"].get("jlens", r["lens"]["exemplar"])["verb"],
                "lensTop": r["lens"]["exemplar"]["pos"][:3],
            })

    header = {
        "tool": "epdashboard", "version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dict": {
            "run": d.run_dir.name, "model_id": cfg.model_id,
            "layer": cfg.layer, "p": dict_p(d), "K": d.K,
            "threshold": round(d.threshold, 4),
            "dModel": int(d.E.shape[1]),
            "buildActs": d.meta.get("n_activations"),
            "buildCorpus": d.meta.get("corpus"),
            "contextLength": d.meta.get("context_length", cfg.context_length),
        },
        "source": source_desc,
        "scan": {"nActs": scan.n_acts, **scan.replay_check()},
        "lens": {"k": cfg.lens_k, **jlens_meta},
        "panels": {"nBands": cfg.n_bands, "buffer": list(cfg.buffer),
                   "histBins": cfg.hist_bins, "reservoir": cfg.reservoir,
                   "bgSample": int(bg.shape[0])},
        "batches": batches,
        "regionTable": summary,
    }
    (out_dir / "header.json").write_text(
        json.dumps(header, ensure_ascii=False, separators=(",", ":")))
    return header
