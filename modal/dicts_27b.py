"""Build Qwen3.6-27B EP dictionaries for one residual layer, on Modal.

Functions are deliberately split, because the halves have completely different
cost profiles and failure modes:

    extract   A100-80GB. Streams the Pile through the 27B once and shards ~31 GB
              of fp16 activations to the `ep-acts` volume. This is the entire
              GPU cost of a layer and must be paid exactly once per layer.
    build_p   A100-40GB. Reads that cache and builds one dictionary per
              *percentile* (qwen_ep/sweep_p.py) — the canonical resolution
              knob, which is what the dashboards are laddered by.
    search    A100-40GB. Same cache, but searches theta for a target region
              count (qwen_ep/target_k.py). Use when the target is a K rather
              than a p; theta(p) carries too much fit slop to hit a K.

Splitting them is the whole point: a build that comes out at the wrong
resolution costs another 20 minutes, not another 55 GB download plus an hour
of A100.

The layer is a flag, not a constant — `--layer 56` extracts and builds an
independent set that lands in its own cache and run dirs, so layers never
collide on the volumes.

Volumes:
    ep-hf     HF_HOME — model weights. 55.6 GB, downloaded once, ever.
    ep-acts   the activation cache, one dir per layer. Keep until member_scan
              has run.
    ep-dicts  output run dirs (dictionary.pkl + metadata.json + member_scan.json)

Usage:
    modal run modal/dicts_27b.py --stage extract --layer 56 --max-tokens 200000
    modal run --detach modal/dicts_27b.py --stage extract --layer 56
    modal run --detach modal/dicts_27b.py --stage build --layer 56 --percentiles 2,4,8
    modal run --detach modal/dicts_27b.py --stage search --target-k 16000
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

# Install order is not cosmetic — each line is a failure already paid for once
# on a rented box: transformers 5.x needs torch >= 2.5, flash-linear-attention
# needs the triton that ships with that torch so it must come *after*, and the
# Pile is .jsonl.zst, which datasets fails on without zstandard. 3 of every 4
# layers in this model are Gated DeltaNet and fall back to slow torch without
# flash-linear-attention.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .pip_install(
        "transformers==5.14.1", "datasets==5.0.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "zstandard", "scipy", "scikit-learn",
        "numpy", "tqdm", "flash-linear-attention",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(REPO / "qwen_ep", "/root/qwen_ep")
    .add_local_dir(REPO / "exemplar-partitioning" / "ep", "/root/ep")
)

hf_vol = modal.Volume.from_name("ep-hf", create_if_missing=True)
acts_vol = modal.Volume.from_name("ep-acts", create_if_missing=True)
dicts_vol = modal.Volume.from_name("ep-dicts", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("ep-dicts-27b", image=image)

MODEL_ID = "Qwen/Qwen3.6-27B"
TAG = "qwen3_6-27b"
LAYER = 55          # default only; every stage takes --layer (0..63)
CTX = 128
SEED = 0
CORPUS = "pile"
# Measured on an A100-80GB: throughput is flat from bs=16 to bs=256 because a
# 27B forward saturates the card even at 2048 tokens, so batch size buys
# nothing and only costs memory. 64 sits at 58 GB peak against the 80 GB card.
BATCH = 64
PROMPT_BATCH = 256

VOLS = {"/hf": hf_vol, "/acts": acts_vol, "/dicts": dicts_vol}


def cache_slug(max_tokens: int, layer: int = LAYER) -> str:
    return f"{TAG}_L{layer}_ctx{CTX}_mt{max_tokens}_seed{SEED}_{CORPUS}"


def dict_slug(layer: int, percentile: str) -> str:
    """Run-dir name for a percentile build.

    Must agree exactly with ``qwen_ep.sweep_p.run_slug``, which formats the
    percentile as a float ("2" -> 2.0 -> "p2p0"); this side never sees the
    manifest, so the agreement is by construction and is what lets the stage
    skip percentiles already on the volume.
    """
    return (f"{TAG}_L{layer}_p{str(float(percentile)).replace('.', 'p')}"
            f"_ctx{CTX}_cache_{CORPUS}")


def _run(cmd: list[str], log) -> None:
    """Stream a subprocess's output into the Modal log as it happens."""
    import subprocess
    import sys

    log.info("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise SystemExit(f"command failed with exit {proc.returncode}: {cmd[:4]}")


def _logger(name: str):
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


# RAM is billed separately from the GPU, so it is sized to the actual buffers
# rather than rounded up: extraction holds one ~100k-activation fp16 shard
# (1 GB) plus prompt strings before each flush.
@app.function(gpu="A100-80GB", cpu=8.0, memory=32768, timeout=6 * 60 * 60,
              volumes=VOLS, secrets=[hf_secret])
def extract(max_tokens: int, layer: int) -> str:
    """Stream the Pile through the 27B once; shard activations to /acts."""
    import subprocess

    log = _logger("extract")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    cache = f"/acts/{cache_slug(max_tokens, layer)}"
    if pathlib.Path(f"{cache}/manifest.json").exists():
        log.info("SKIP: manifest already exists at %s", cache)
        return cache

    try:
        _run(["python", "-m", "qwen_ep.extract_cache",
              "--model-id", MODEL_ID, "--layer", str(layer),
              "--max-tokens", str(max_tokens), "--context-length", str(CTX),
              "--batch-size", str(BATCH), "--prompt-batch-size", str(PROMPT_BATCH),
              "--seed", str(SEED), "--corpus", CORPUS, "--cache-dir", cache], log)
    finally:
        # Commit even on failure: a partial shard set plus the 55 GB of weights
        # already downloaded are both worth keeping.
        acts_vol.commit()
        hf_vol.commit()

    out = subprocess.run(["du", "-sh", cache], capture_output=True, text=True)
    log.info("cache: %s", out.stdout.strip())
    return cache


# Peak is the final build: the member reservoir is 614 KB/region at d=5120, so
# ~10 GB at K=16000, plus a 2 GB upcast shard and the ~1 GB exemplar buffer.
@app.function(gpu="A100-40GB", cpu=8.0, memory=65536, timeout=8 * 60 * 60,
              volumes=VOLS)
def search(max_tokens: int, layer: int, target_k: int, seed_theta: float,
           tolerance: float, max_trials: int, reservoir: int, scan: bool) -> str:
    """Search theta for `target_k` regions, build the dictionary, member-scan it.

    No model and no HF token: this reads the activation cache only. The GPU is
    here purely for the nearest-exemplar matmul, which is O(N x K x d) — at
    K=16000, d=5120, N=3M that is ~0.5 PFLOP per counting pass.
    """
    import subprocess

    log = _logger("search")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    cache = f"/acts/{cache_slug(max_tokens, layer)}"
    if not pathlib.Path(f"{cache}/manifest.json").exists():
        raise SystemExit(f"no activation cache at {cache} — run --stage extract first")

    run_dir = f"/dicts/{TAG}_L{layer}_k{target_k}_ctx{CTX}_cache_{CORPUS}"
    cmd = ["python", "-m", "qwen_ep.target_k", "--cache-dir", cache,
           "--target-k", str(target_k), "--tolerance", str(tolerance),
           "--max-trials", str(max_trials), "--member-reservoir", str(reservoir),
           "--output-dir", "/dicts"]
    if seed_theta > 0:
        cmd += ["--seed-theta", str(seed_theta)]

    try:
        _run(cmd, log)
        # search.json lands in the run dir, so a re-invocation resumes from the
        # trials this one already paid for rather than re-measuring them.
        dicts_vol.commit()
        if scan:
            _run(["python", "-m", "qwen_ep.member_scan",
                  "--cache-dir", cache, "--dicts", run_dir, "--top-n", "64"], log)
    finally:
        dicts_vol.commit()

    import json
    meta = json.loads(pathlib.Path(f"{run_dir}/metadata.json").read_text())
    result = (f"K={meta['n_partitions']} (target {target_k}, "
              f"{100 * (meta['n_partitions'] - target_k) / target_k:+.2f}%) "
              f"theta={meta['threshold']:.6f} -> {run_dir}")
    log.info(result)
    return result


# Same shape as `search`, and sized by the same term: the member reservoir.
# `--reservoir 8` at K=30000, d=5120 is 4.9 GB, against 18 GB for the ep
# default of 30 — which is not just a volume cost, it is the difference between
# a pickle that unpickles on a 26 GB laptop for the vectors.npz export and one
# that does not.
#
# The GPU is for the clustering only (ep.discovery.dictionary picks it up via
# try_torch_gpu; K=40591 over 3M activations took 392 s). The member_scan that
# follows is numpy-only and will not touch the card, so the cores matter as
# much as the card does — hence 32 rather than 8. Pass --no-scan and use
# `--stage scan` if you would rather not hold an A100 through the scan at all.
@app.function(gpu="A100-40GB", cpu=32.0, memory=65536, timeout=8 * 60 * 60,
              volumes=VOLS)
def build_p(max_tokens: int, layer: int, percentiles: str, reservoir: int,
            calibration_tokens: int, scan: bool, top_n: int) -> str:
    """Build one dictionary per percentile off the cached activations.

    All percentiles ride one invocation because they share the container, the
    volume mount and the shard reads; the clustering itself is per-p and runs
    on the GPU (K=16051 at L55 clustered 3M activations in 365 s).

    Each p is calibrated independently — center *and* threshold come from the
    same `calibrate()` call — so this is the canonical path, unlike `search`,
    which reuses one cached center across trials to keep K attributable to
    theta alone.
    """
    import json
    import subprocess

    log = _logger("build_p")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    cache = f"/acts/{cache_slug(max_tokens, layer)}"
    if not pathlib.Path(f"{cache}/manifest.json").exists():
        raise SystemExit(f"no activation cache at {cache} — run --stage extract first")

    ps = [p.strip() for p in percentiles.split(",") if p.strip()]
    # Build finest-first: p=2 is the one that can blow up on K, and finding
    # that out before spending time on the cheap coarse dicts is worth more
    # than the coarse dicts are.
    ps.sort(key=float)
    todo = [p for p in ps
            if not pathlib.Path(f"/dicts/{dict_slug(layer, p)}/dictionary.pkl").exists()]
    if skipped := [p for p in ps if p not in todo]:
        log.info("SKIP (already on the volume): p=%s", ",".join(skipped))

    try:
        if todo:
            _run(["python", "-m", "qwen_ep.sweep_p", "--cache-dir", cache,
                  "--percentiles", ",".join(todo),
                  "--calibration-tokens", str(calibration_tokens),
                  "--member-reservoir", str(reservoir),
                  "--output-dir", "/dicts"], log)
            dicts_vol.commit()
        if scan:
            for p in ps:
                _run(["python", "-m", "qwen_ep.member_scan", "--cache-dir", cache,
                      "--dicts", f"/dicts/{dict_slug(layer, p)}",
                      "--top-n", str(top_n)], log)
                dicts_vol.commit()
    finally:
        dicts_vol.commit()

    lines = []
    for p in ps:
        meta = json.loads(
            pathlib.Path(f"/dicts/{dict_slug(layer, p)}/metadata.json").read_text())
        lines.append(f"  p={p:<5} K={meta['n_partitions']:<7} "
                     f"theta={meta['threshold']:.6f} "
                     f"largest={meta['largest_partition']:<8} "
                     f"singletons={meta['singletons']}")
    result = f"L{layer} dictionaries:\n" + "\n".join(lines)
    log.info(result)
    return result


# No GPU, and 32 cores rather than 8. `qwen_ep.member_scan` imports neither
# torch nor cupy — it is numpy end to end — so a card here would idle while the
# job ran core-starved. Measured on the L56 p2 dictionary: 595 acts/s on
# gpu+cpu=8 against 2560 acts/s for the (larger) epdashboard scan on cpu=32.
# The ratio is the core count, not the card.
@app.function(cpu=32.0, memory=65536, timeout=8 * 60 * 60, volumes=VOLS)
def scan_only(max_tokens: int, layer: int, run_name: str, top_n: int) -> str:
    """member_scan an already-built dictionary, without rebuilding it.

    Separate from `search` so a dictionary that is already on the volume can be
    scanned for the cost of the scan alone — rerunning `search` would resume
    the trial history but still pay another final build to get there.
    """
    import subprocess

    log = _logger("scan_only")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    cache = f"/acts/{cache_slug(max_tokens, layer)}"
    run_dir = f"/dicts/{run_name}"
    if not pathlib.Path(f"{run_dir}/dictionary.pkl").exists():
        raise SystemExit(f"no dictionary at {run_dir}")

    try:
        _run(["python", "-m", "qwen_ep.member_scan", "--cache-dir", cache,
              "--dicts", run_dir, "--top-n", str(top_n)], log)
    finally:
        dicts_vol.commit()

    out = subprocess.run(["du", "-sh", f"{run_dir}/member_scan.json"],
                         capture_output=True, text=True)
    return f"scanned {run_name}: {out.stdout.strip()}"


@app.local_entrypoint()
def main(stage: str = "search", layer: int = LAYER, max_tokens: int = 3_000_000,
         percentiles: str = "2,4,8", calibration_tokens: int = 200_000,
         target_k: int = 16000, seed_theta: float = 0.7726,
         tolerance: float = 0.002, max_trials: int = 8,
         reservoir: int = 30, scan: bool = True,
         run_name: str = "", top_n: int = 64):
    """--stage extract | build | search | scan.

    ``build`` and ``search`` are two ways to pick resolution and are not meant
    to be run together, so there is no "all": ``--stage extract`` then
    ``--stage build`` is the percentile-ladder path.
    """
    if not 0 <= layer < 64:
        raise SystemExit(f"--layer {layer} out of range for a 64-layer model")

    if stage == "extract":
        print(extract.remote(max_tokens, layer))
    elif stage == "build":
        print(build_p.remote(max_tokens, layer, percentiles, reservoir,
                             calibration_tokens, scan, top_n))
    elif stage == "search":
        print(search.remote(max_tokens, layer, target_k, seed_theta, tolerance,
                            max_trials, reservoir, scan))
    elif stage == "scan":
        name = run_name or f"{TAG}_L{layer}_k{target_k}_ctx{CTX}_cache_{CORPUS}"
        print(scan_only.remote(max_tokens, layer, name, top_n))
    else:
        raise SystemExit(f"unknown --stage {stage!r}")

    # One `volume get` per run dir, named in full: glob patterns are deprecated
    # in `modal volume get`, and the deprecation *exits 0* while downloading
    # nothing, so a globbed pull looks like it worked and silently isn't.
    names = ([dict_slug(layer, p) for p in percentiles.split(",") if p.strip()]
             if stage == "build"
             else [f"{TAG}_L{layer}_k{target_k}_ctx{CTX}_cache_{CORPUS}"])
    print("\npull results with:")
    for n in names:
        print(f"  modal volume get ep-dicts {n} ./artifacts/runs")
