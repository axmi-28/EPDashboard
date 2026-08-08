"""Gate 1A of the EP model-diffing positive control, on Modal.

One GPU function. The two 7B forwards are the whole cost and the local machine
(24 GB M-series) cannot host them, so everything that needs a model runs here:
the weight diff, the manifestation check, the paired activation probe at all
four layers, and the timing build. Results land on `ep-dicts` as JSON + CSV and
are pulled back for analysis.

Volumes:
    ep-hf     HF_HOME — 2 x 14.5 GB of weights, downloaded once, ever.
    ep-dicts  output run dir.

Usage:
    modal run modal/rmu_diff.py --stage gate1a --n-probe 128 --n-acc 64   # smoke
    modal run --detach modal/rmu_diff.py --stage gate1a                   # real

    modal volume get ep-dicts 'rmu_diff/**' ./artifacts/runs
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

# Same install order as modal/dicts_27b.py — each line is a failure already paid
# for once. Mistral needs none of the flash-linear-attention machinery the Qwen
# hybrids do, so that line is dropped; everything else is held identical so the
# two apps cannot drift into different numerics.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .pip_install(
        "transformers==5.14.1", "datasets==5.0.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "scipy", "scikit-learn", "numpy", "tqdm",
        # Zephyr ships a sentencepiece tokenizer with no tokenizer.json. Without
        # these, transformers silently falls back to a TikToken extractor and
        # builds a *different* tokenizer — which would be a paired-extraction
        # corruption that no downstream number would flag.
        "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(REPO / "qwen_ep", "/root/qwen_ep")
    .add_local_dir(REPO / "experiments", "/root/experiments")
    .add_local_dir(REPO / "exemplar-partitioning" / "ep", "/root/ep")
)

hf_vol = modal.Volume.from_name("ep-hf", create_if_missing=True)
dicts_vol = modal.Volume.from_name("ep-dicts", create_if_missing=True)

app = modal.App("ep-rmu-diff", image=image)

VOLS = {"/hf": hf_vol, "/dicts": dicts_vol}
OUT = "/dicts/rmu_diff/gate1a"
GRID = "/dicts/rmu_diff/grid"


def _run(cmd: list[str]) -> None:
    """Stream a subprocess's output into the Modal log as it happens."""
    import subprocess
    import sys

    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise SystemExit(f"command failed with exit {proc.returncode}")


# RAM is sized to the buffers, not rounded up: the probe holds two models' worth
# of fp16 activations for four layers (~10 GB at n_probe=1024) plus a transient
# fp32 calibration copy (~3.3 GB). The GPU only ever hosts one 7B at a time.
@app.function(gpu="A100-40GB", cpu=8.0, memory=65536, timeout=3 * 60 * 60,
              volumes=VOLS)
def gate1a(n_probe: int, n_acc: int, layers: str, percentiles: str,
           style: str, batch_size: int, max_positions: int,
           min_tokens: int, max_tokens: int, skip_weight_diff: bool) -> str:
    import json
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    cmd = ["python", "-m", "experiments.rmu_diff.gate1a", "--out", OUT,
           "--n-probe", str(n_probe), "--n-acc", str(n_acc),
           "--layers", layers, "--percentiles", percentiles, "--style", style,
           "--batch-size", str(batch_size), "--max-positions", str(max_positions),
           "--min-tokens", str(min_tokens), "--max-tokens", str(max_tokens)]
    if skip_weight_diff:
        cmd.append("--skip-weight-diff")

    try:
        _run(cmd)
    finally:
        # Commit even on failure: the 29 GB of weights and any partial report
        # are both worth more than the run that produced them.
        dicts_vol.commit()
        hf_vol.commit()

    report = json.loads(pathlib.Path(f"{OUT}/gate1a.json").read_text())
    wd = report.get("weight_diff", {})
    tm = report.get("timing", {})
    return (f"layers touched={wd.get('layers_touched')} "
            f"changed={wd.get('n_changed')}/{wd.get('n_tensors')} | "
            f"timing K={tm.get('K')} in {tm.get('cluster_s')}s "
            f"saturated={tm.get('saturated')} | wall={report.get('total_wall_s')}s")


# Both 7B checkpoints stay resident (29 GB of a 40 GB card) so the paired
# extraction for a layer happens back to back. Host RAM holds two fp16
# activation sets per layer (~4.3 GB each at 528k x 4096) plus one fp32
# calibration copy; extraction is sliced to keep the fp32 peak off the total.
@app.function(gpu="A100-40GB", cpu=8.0, memory=65536, timeout=4 * 60 * 60,
              volumes=VOLS)
def grid(n_bio: int, n_cyber: int, n_mmlu: int, layers: str, percentiles: str,
         seeds: str, calibration: str, batch_size: int, max_positions: int,
         max_partitions: int) -> str:
    import json
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)
    out = f"{GRID}/{calibration}"
    cmd = ["python", "-m", "experiments.rmu_diff.build", "--out", out,
           "--n-bio", str(n_bio), "--n-cyber", str(n_cyber),
           "--n-mmlu", str(n_mmlu), "--layers", layers,
           "--percentiles", percentiles, "--seeds", seeds,
           "--calibration", calibration, "--batch-size", str(batch_size),
           "--max-positions", str(max_positions),
           "--max-partitions", str(max_partitions)]
    try:
        _run(cmd)
    finally:
        dicts_vol.commit()
        hf_vol.commit()

    m = json.loads(pathlib.Path(f"{out}/manifest.json").read_text())
    ks = {f"L{r['layer']}p{r['percentile']:g}": r["K"] for r in m["runs"]
          if r["model"] == "base" and r["seed"] == 0}
    return (f"{len(m['runs'])} dictionaries in {m.get('total_wall_s')}s | "
            f"base seed0 K: {ks} | aborted="
            f"{sum(r['aborted'] for r in m['runs'])}")


# No GPU: this reads pickles and does Hungarian assignment plus O(N) contingency
# tables. It runs here rather than locally only because the grid is ~2.5 GB of
# dictionaries and the results are a few hundred KB of CSV.
@app.function(cpu=8.0, memory=32768, timeout=2 * 60 * 60, volumes=VOLS)
def analyse(calibration: str, n_sim: int) -> str:
    import json

    out = f"/dicts/rmu_diff/gate1b/{calibration}"
    try:
        _run(["python", "-m", "experiments.rmu_diff.gate1b",
              "--grid", f"{GRID}/{calibration}", "--out", out,
              "--n-sim", str(n_sim)])
    finally:
        dicts_vol.commit()
    v = json.loads(pathlib.Path(f"{out}/gate1b.json").read_text())["verdict"]
    return json.dumps(v.get("kill_criteria", {})) + f"  PASS={v.get('PASS')}"


# Gate 1C is almost entirely offline, but the mechanism check re-extracts one
# layer from both checkpoints to recover RMU's control vector, so it needs a GPU.
@app.function(gpu="A100-40GB", cpu=8.0, memory=65536, timeout=2 * 60 * 60,
              volumes=VOLS)
def gate1c(calibration: str, mechanism_layer: int, batch_size: int) -> str:
    import subprocess

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)
    out = f"/dicts/rmu_diff/gate1c/{calibration}"
    try:
        _run(["python", "-m", "experiments.rmu_diff.gate1c",
              "--grid", f"{GRID}/{calibration}", "--out", out,
              "--mechanism-layer", str(mechanism_layer),
              "--batch-size", str(batch_size)])
    finally:
        dicts_vol.commit()
        hf_vol.commit()
    return f"gate1c written to {out}"


@app.local_entrypoint()
def main(stage: str = "gate1a", n_probe: int = 1024, n_acc: int = 600,
         layers: str = "4,7,14,24", percentiles: str = "10,12",
         style: str = "chat", batch_size: int = 8, max_positions: int = 256,
         min_tokens: int = 48, max_tokens: int = 256,
         skip_weight_diff: bool = False,
         n_bio: int = 1250, n_cyber: int = 950, n_mmlu: int = 2200,
         seeds: str = "0,1,2", calibration: str = "shared",
         max_partitions: int = 60000, n_sim: int = 2000,
         mechanism_layer: int = 7):
    if stage == "gate1a":
        print(gate1a.remote(n_probe, n_acc, layers, percentiles, style,
                            batch_size, max_positions, min_tokens, max_tokens,
                            skip_weight_diff))
    elif stage == "grid":
        print(grid.remote(n_bio, n_cyber, n_mmlu, layers, percentiles, seeds,
                          calibration, batch_size, max_positions, max_partitions))
    elif stage == "gate1b":
        print(analyse.remote(calibration, n_sim))
    elif stage == "gate1c":
        print(gate1c.remote(calibration, mechanism_layer, batch_size))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print("\npull results with:\n"
          "  modal volume get ep-dicts rmu_diff ./artifacts/runs")
