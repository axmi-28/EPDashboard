"""Run an EPDashboard build on Modal.

The whole job is one GPU function: stream the Pile through the model, scan
every activation against each dictionary, re-forward the winning prompts, and
write JSON + HTML to a persistent volume for download. Built for
``Qwen/Qwen3.6-27B`` L55, where the forward pass is the entire cost and must
be paid exactly once for all four percentiles — which is why every dictionary
you want rides on the same invocation.

Three volumes, so nothing expensive is ever done twice:

    ep-dicts   uploaded run dirs (dictionary.pkl + metadata.json)
    ep-hf      HF_HOME — 55.6 GB of model weights, the 3.3 GB J-lens, and the
               npz lens caches. Survives between runs: download once, ever.
    ep-out     dashboard output, pulled back with ``modal volume get``

Usage (walkthrough in MODAL.md):

    modal run modal_epdash.py --p 16 --n-prompts 256 --regions 0:16   # smoke
    modal run --detach modal_epdash.py --p 4,8,12,16                  # real
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).parent

# Install order is not cosmetic — each line here is a failure already paid for
# once on a rented box (see DECISIONS.md / the 27B env notes): transformers 5.x
# needs torch >= 2.5, flash-linear-attention needs the triton that ships with
# that torch so it must come *after*, and the Pile is .jsonl.zst, which datasets
# fails on without zstandard. debian_slim has no preinstalled torchvision, so
# the ABI break that bit us on the RunPod image cannot happen here.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .pip_install(
        "transformers==5.14.1", "datasets==5.0.0", "accelerate", "safetensors",
        "huggingface_hub[hf_transfer]", "zstandard", "scipy", "scikit-learn",
        "numpy", "tqdm", "flash-linear-attention",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO / "epdashboard", "/root/epdashboard")
    .add_local_dir(REPO / "qwen_ep", "/root/qwen_ep")
    .add_local_dir(REPO / "exemplar-partitioning" / "ep", "/root/ep")
)

dicts_vol = modal.Volume.from_name("ep-dicts", create_if_missing=True)
hf_vol = modal.Volume.from_name("ep-hf", create_if_missing=True)
out_vol = modal.Volume.from_name("ep-out", create_if_missing=True)

app = modal.App("epdash-27b", image=image)


@app.function(
    gpu="A100-80GB",          # 55.6 GB bf16. Batch size is a non-lever on a
    cpu=8.0,                  # 27B dense forward, so a bigger card buys
    memory=65536,             # nothing but a bigger bill.
    timeout=6 * 60 * 60,
    volumes={"/dicts": dicts_vol, "/hf": hf_vol, "/out": out_vol},
)
def build(run_names: list[str], overrides: dict) -> str:
    import logging
    import subprocess
    import sys

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    log = logging.getLogger("modal_epdash")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader"], check=False)

    missing = [n for n in run_names
               if not pathlib.Path(f"/dicts/{n}/dictionary.pkl").exists()]
    if missing:
        raise SystemExit(
            f"not on the ep-dicts volume: {missing}\nupload with:\n"
            "  modal volume put ep-dicts runs/<name>/dictionary.pkl "
            "/<name>/dictionary.pkl")

    from epdashboard.config import EPVisConfig
    from epdashboard.runner import run

    cfg = EPVisConfig(
        run_dirs=[f"/dicts/{n}" for n in run_names],
        out_dir="/out",
        lens_cache="/hf/epdash-lens",   # npz + extracted J-lens, cached forever
    )
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    log.info("config: %s", cfg.to_json())

    try:
        run(cfg)
    finally:
        out_vol.commit()        # keep partial output if a later dict fails
        hf_vol.commit()

    return subprocess.run(["du", "-sh", "/out"], capture_output=True,
                          text=True).stdout.strip()


def _regions(spec: str) -> list[int] | None:
    """'0:100' or '3,17,42' -> region ids (kept local so the entrypoint needs
    no repo imports)."""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition(":")
        out.extend(range(int(a), int(b))) if b else out.append(int(a))
    return out


@app.local_entrypoint()
def main(p: str = "4,8,12,16", model: str = "qwen3_6-27b_L55",
         n_prompts: int = 0, regions: str = "", batch_size: int = 0,
         n_closest: int = 0, n_per_band: int = 0, n_random: int = 0):
    """--p 8,16 picks percentiles; the rest mirror the epdashboard CLI flags."""
    names = [f"{model}_p{v.strip()}p0_ctx128_cache_pile" for v in p.split(",")]
    overrides = {"n_prompts": n_prompts or None, "regions": _regions(regions),
                 "batch_size": batch_size or None,
                 "n_closest": n_closest or None,
                 "n_per_band": n_per_band or None,
                 "n_random": n_random or None}
    print("dictionaries:", *names, sep="\n  ")
    print(build.remote(names, overrides))
    print("\npull results with:\n"
          "  modal volume get ep-out '**' ./epdash_out_27b")
