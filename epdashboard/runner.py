"""Orchestrate a full EPDashboard build.

    python -m epdashboard --run-dirs runs/<slug>[,runs/<slug2>…] --out epdash_out
    python -m epdashboard --config my_config.json

Several run dirs (e.g. the same layer at different ``p``) share one activation
pass — the forward passes are the entire cost of the job, so ``p`` is
effectively a free parameter of a single run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np

from epdashboard.config import EPVisConfig
from epdashboard.html import render_batches
from epdashboard.lens import LensTables
from epdashboard.scan import EPDict, RegionScan
from epdashboard.sequences import PromptCache
from epdashboard.source import make_source
from epdashboard.writer import write_dict_output

log = logging.getLogger("epdashboard")


def is_complete(out_root: Path, run_name: str) -> bool:
    """True when a run dir holds a finished dashboard.

    "Finished" means the header parses *and* every batch file it references is
    on disk — a run killed midway through writing batches leaves a header from
    the previous attempt or no header at all, and must not be skipped.
    """
    header = out_root / run_name / "header.json"
    if not header.exists():
        return False
    try:
        h = json.loads(header.read_text())
    except (ValueError, OSError):
        return False
    batches = h.get("batches")
    if not batches:
        return False
    return all((out_root / run_name / b["file"]).exists() for b in batches)


def run(cfg: EPVisConfig) -> list[Path]:
    t0 = time.time()
    dicts = [EPDict.load(rd) for rd in cfg.run_dirs]
    if not dicts:
        raise ValueError("no run_dirs given")

    # Skip dictionaries already built. Done before make_source so that a fully
    # complete run costs nothing at all — the activation pass is the whole job,
    # and a preempted 64-layer sweep must not redo the layers it finished.
    if cfg.skip_existing:
        out_root = cfg.out_path()
        keep = [d for d in dicts if not is_complete(out_root, d.run_dir.name)]
        for d in dicts:
            if d not in keep:
                log.info("skip %s: dashboard already complete", d.run_dir.name)
        if not keep:
            log.info("all %d dictionary/ies already built; nothing to do",
                     len(dicts))
            return []
        dicts = keep

    # Model/layer come from dictionary metadata unless overridden; every dict
    # in one run must agree, since they share the activation stream.
    metas = [(d.meta.get("model_id"), d.meta.get("layer")) for d in dicts]
    cfg.model_id = cfg.model_id or metas[0][0]
    cfg.layer = cfg.layer if cfg.layer is not None else metas[0][1]
    for d, (mid, lay) in zip(dicts, metas):
        if (mid and mid != cfg.model_id) or (lay is not None and lay != cfg.layer):
            raise ValueError(f"{d.run_dir.name} is {mid} L{lay}, run is "
                             f"{cfg.model_id} L{cfg.layer} — dicts sharing a "
                             "pass must share model and layer")
    if cfg.layer is None or cfg.model_id is None:
        raise ValueError("model_id/layer not in metadata — pass --model-id/--layer")

    out_root = cfg.out_path()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(cfg.to_json())

    source = make_source(cfg, cfg.layer)
    expected = cfg.n_prompts * (cfg.context_length - 1)
    rng = np.random.default_rng(cfg.seed)
    scans = [RegionScan(d, cfg, rng, expected) for d in dicts]
    for d in dicts:
        log.info("dict %s: K=%d θ=%.4f", d.run_dir.name, d.K, d.threshold)

    # ---------------------------------------------------------------- pass 1
    log.info("pass 1: streaming activations (%s)…", source.describe()["mode"])
    n_acts, next_log = 0, 0
    for X, gid, pos in source.pass1():
        for s in scans:
            s.consume(X, gid, pos)
        n_acts += len(X)
        if n_acts >= next_log:
            el = time.time() - t0
            log.info("  %s acts | %d prompts | %.0fs (%.0f acts/s)",
                     f"{n_acts:,}", len(source.prompts), el, n_acts / max(el, 1e-9))
            next_log = n_acts + 200_000
    log.info("pass 1 done: %s activations over %d prompts (%.0fs)",
             f"{n_acts:,}", len(source.prompts), time.time() - t0)

    # ---------------------------------------------------------------- pass 2
    region_ids = [cfg.regions if cfg.regions is not None else list(range(d.K))
                  for d in dicts]
    tok = source.tokenizer
    pcs: list[PromptCache] = []
    winners: set[int] = set()
    for d, s, ids in zip(dicts, scans, region_ids):
        # gid -> regions that will request sequences from that prompt; the
        # cache keeps only those similarity columns.
        refs: dict[int, set[int]] = {}
        for i in ids:
            for rows in s.examples(i).values():
                for r in rows:
                    refs.setdefault(r["gid"], set()).add(i)
        winners |= set(refs)
        pcs.append(PromptCache(d, tok, cfg.bos_offset, refs))
    log.info("pass 2: gathering %d winning prompts…", len(winners))
    n_done, next_log2 = 0, 5000
    for g, X, positions in source.pass2(sorted(winners)):
        ids_g = tok.encode(source.prompts[g], add_special_tokens=False)
        for pc in pcs:
            pc.add(g, source.prompts[g], X, positions, ids=ids_g)
        n_done += 1
        if n_done >= next_log2:
            log.info("  gathered %d/%d prompts (%.0fs)", n_done, len(winners),
                     time.time() - t0)
            next_log2 += 5000

    # ------------------------------------------------------------------ lens
    log.info("lens tables…")
    lt = LensTables(cfg.model_id, cfg.layer, cfg.lens_cache_path(), cfg.lens_k)
    decode_cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        if tid not in decode_cache:
            decode_cache[tid] = tok.decode([tid])
        return decode_cache[tid]

    jlens_meta = {"jlens": lt.J is not None, "jNPrompts": lt.j_n_prompts}

    # ----------------------------------------------------------------- write
    pages: list[Path] = []
    for d, s, ids, pc in zip(dicts, scans, region_ids, pcs):
        out_dir = out_root / d.run_dir.name
        log.info("[%s] building %d region records…", d.run_dir.name, len(ids))
        lens_rows = lt.build(d.E, d.means, decode)
        header = write_dict_output(d, s, pc, lens_rows, ids,
                                   source.describe(), jlens_meta, cfg, out_dir)
        log.info("[%s] wrote %d JSON batch(es), replay corr=%.3f",
                 d.run_dir.name, len(header["batches"]),
                 header["scan"]["member_share_corr"])
        if cfg.html:
            new = render_batches(out_dir, header)
            pages += new
            log.info("[%s] wrote %d region page(s), first: %s",
                     d.run_dir.name, len(new), new[0] if new else "—")
    log.info("done in %.0fs", time.time() - t0)
    return pages


def _parse_regions(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part:
            a, b = part.split(":")
            out.extend(range(int(a), int(b)))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="epdashboard", description=__doc__)
    ap.add_argument("--config", default=None,
                    help="JSON file of EPVisConfig fields; CLI flags override")
    ap.add_argument("--run-dirs", default=None,
                    help="comma-separated run dirs, each with dictionary.pkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="extract_cache shard dir (skips the model entirely)")
    ap.add_argument("--regions", default=None,
                    help="e.g. '0:100' or '3,17,42' (default: all)")
    ap.add_argument("--n-closest", type=int, default=None,
                    help="sequences in the closest-members column")
    ap.add_argument("--n-per-band", type=int, default=None,
                    help="sequences per distance band")
    ap.add_argument("--n-random", type=int, default=None,
                    help="sequences in the random draw")
    ap.add_argument("--regions-per-batch", type=int, default=None)
    ap.add_argument("--comp-max-k", type=int, default=None,
                    help="skip the (K,K) competition graph above this K")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="prompts per forward sub-batch (GPU memory knob)")
    ap.add_argument("--lens-cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip dictionaries whose dashboard is already "
                         "complete; makes a multi-layer batch resumable after "
                         "a preemption")
    args = ap.parse_args()

    cfg = EPVisConfig.from_json(args.config) if args.config else EPVisConfig()
    override = {"run_dirs": (args.run_dirs.split(",") if args.run_dirs else None),
                "out_dir": args.out, "model_id": args.model_id,
                "layer": args.layer, "dataset": args.dataset,
                "context_length": args.context_length,
                "n_prompts": args.n_prompts, "cache_dir": args.cache_dir,
                "regions": (_parse_regions(args.regions) if args.regions else None),
                "n_closest": args.n_closest,
                "n_per_band": args.n_per_band,
                "n_random": args.n_random,
                "regions_per_batch": args.regions_per_batch,
                "comp_max_k": args.comp_max_k,
                "batch_size": args.batch_size,
                "lens_cache": args.lens_cache, "device": args.device,
                "seed": args.seed}
    for k, v in override.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.no_html:
        cfg.html = False
    if args.skip_existing:
        cfg.skip_existing = True

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    run(cfg)


if __name__ == "__main__":
    main()
