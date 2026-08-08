"""Run the EP paper's refusal experiment on Modal, unmodified.

This deliberately shells out to ``scripts/exp_behavioral.py`` from the upstream
exemplar-partitioning repo rather than reimplementing it. The whole point of
this job is to establish what the *reference* harness does on a bigger model,
so the only things we supply are CLI flags. If you find yourself editing the
experiment logic here, you have lost the plot — fork the script instead and
rename the run.

Why this is a separate file from ``modal/epdash.py``: different image (adds
transformer-lens), different volumes, different failure modes. The dashboard
job streams the Pile through a Qwen adapter; this one hands the model to
TransformerLens and never touches ``qwen_ep``.

Target: ``google/gemma-2-27b-it``, the largest same-generation scale-up of the
paper's ``gemma-2-2b-it``. 46 layers vs 26, so the depth-matched analogue of
the paper's L20 is L35 (both ~76% depth).

Usage (read docs/experiments/MODAL_REFUSAL.md first):

    # 1. Harness validation + image smoke test. ~20 min, a couple of dollars.
    modal run modal/refusal.py --smoke

    # 2. The real run: four streaming seeds in parallel.
    modal run --detach modal/refusal.py --seeds 0,1,2,3
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
EP = REPO / "exemplar-partitioning"

# Same torch/transformers pins as modal/epdash.py — transformer-lens 3.6.0
# requires transformers>=5.9.0 and torch>=2.6, so the existing stack satisfies
# it and there is no second environment to maintain. (TL 2.x needed
# transformers 4.x; that constraint is gone as of TL 3.)
#
# flash-linear-attention is absent on purpose: it exists in the dashboard image
# for Qwen's Mamba-style layers, and Gemma-2 has no use for it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .pip_install(
        "transformers==5.14.1",
        "transformer-lens==3.6.0",
        "datasets==5.0.0",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "scipy",
        "scikit-learn",
        "numpy",
        "tqdm",
    )
    .env(
        {
            "HF_HOME": "/hf",
            # HF_HUB_ENABLE_HF_TRANSFER is deprecated in huggingface_hub 1.x
            # (hf_transfer is gone); Xet is the replacement fast path.
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # ep caches the calibration centre/threshold under ~/.cache by
            # default, which dies with the container. Redirect it onto the
            # volume: the cache key is (model, hook, percentile) and does NOT
            # include the seed, so all four seeds share one calibration —
            # which is what we want, and matches the paper ("build prompts and
            # held-out set fixed across seeds; only construction order varies").
            "EP_CALIBRATION_CACHE": "/refusal/calibration",
        }
    )
    # scripts/ has no __init__.py and relies on namespace-package resolution,
    # so `python -m scripts.exp_behavioral` only works with cwd=/root.
    .add_local_dir(EP / "ep", "/root/ep")
    .add_local_dir(EP / "scripts", "/root/scripts")
)

hf_vol = modal.Volume.from_name("ep-hf", create_if_missing=True)
refusal_vol = modal.Volume.from_name("ep-refusal", create_if_missing=True)

# gemma-2-27b-it is gated: accept the licence on HF, then
#   modal secret create huggingface HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("ep-refusal", image=image)


@app.function(
    gpu="A100-80GB",
    cpu=16.0,
    # 192 GB, and this is the flag most likely to be the difference between a
    # completed run and an OOM 15 minutes in. TransformerLens does not stream
    # weights: `from_pretrained_no_processing` materialises the full HF model
    # on CPU, builds a *converted* state dict alongside it, and only then moves
    # to GPU. Peak host RAM is therefore ~2x the 54 GB of bf16 weights before
    # anything reaches the card. 64 GB (what modal/epdash.py asks for) is not
    # close to enough here.
    memory=196608,
    # ~3 h/seed at 27B (see docs/experiments/MODAL_REFUSAL.md for the arithmetic); 6 h leaves
    # headroom for a slow first-run weight download without being so long that
    # a hung job burns a day of credit.
    timeout=6 * 60 * 60,
    volumes={"/hf": hf_vol, "/refusal": refusal_vol},
    secrets=[hf_secret],
)
def behavioral(
    model: str,
    model_short: str,
    layer: int,
    percentile: float,
    seed: int,
    n_prompts_per_side: int,
    n_held_out_harmful: int,
    top_k: int,
    max_new_tokens: int,
) -> str:
    import subprocess
    import sys

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=False,
    )

    # --- Preflight 1: HF auth. ---
    # The gated-repo failure surfaces ~90 s in, as a 401 buried in a
    # TransformerLens stack trace, and only after the datasets have downloaded.
    # Check it in two steps here because the two failures need different fixes
    # and the 401 alone does not distinguish them:
    #   - whoami fails  -> the token itself is bad (typo, quotes captured by
    #     the shell, revoked). Recreate the secret.
    #   - whoami OK but config.json 401s -> the token is valid but its account
    #     has not accepted the Gemma licence. Click through on the model page.
    # Note `model_info` is NOT a usable gate test: gemma-2-2b-it is
    # gated="manual", so its metadata is publicly readable and model_info
    # returns 200 for a token with no access at all.
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set in the container. The `huggingface` Modal "
            "secret must contain a key named exactly HF_TOKEN."
        )

    from huggingface_hub import hf_hub_download, whoami

    try:
        who = whoami(token=token)["name"]
    except Exception as e:
        raise RuntimeError(
            f"HF_TOKEN is present (len={len(token)}) but invalid: {e}. "
            "A classic HF token is 37 chars ('hf_' + 34); if yours is longer, "
            "the shell probably captured surrounding quotes. Recreate with:\n"
            "  modal secret create huggingface HF_TOKEN=hf_xxx --force"
        ) from None

    try:
        hf_hub_download(model, "config.json", token=token)
    except Exception as e:
        raise RuntimeError(
            f"HF_TOKEN is valid (user={who}) but cannot read {model}: {e}. "
            f"Accept the licence at https://huggingface.co/{model} with that "
            "account."
        ) from None
    print(f"preflight: HF auth OK (user={who}, {model} readable)", flush=True)

    # --- Preflight 2: the harmful pool. ---
    # `_load_harmful` catches *both* network failures and drops to a
    # 17-template embedded list, so a run with no egress still "succeeds" — on
    # 17 prompts instead of 300, with every downstream refusal rate and delta
    # meaningless but structurally valid. Nothing in behavioral.json records
    # which path was taken, so check here rather than in the results.
    sys.path.insert(0, "/root")
    from scripts.exp_behavioral import _load_harmful

    pool = _load_harmful(10_000, seed=seed)
    if len(pool) < 500:
        raise RuntimeError(
            f"harmful pool is {len(pool)} prompts, expected ~609 "
            "(520 AdvBench + 100 JailbreakBench). The loader fell back to its "
            "embedded templates — check network egress and HF access."
        )
    print(f"preflight: harmful pool = {len(pool)} prompts", flush=True)

    out_dir = (
        f"/refusal/results/{model_short}/L{layer}_p{percentile:g}_seed{seed}"
    )
    cmd = [
        sys.executable, "-m", "scripts.exp_behavioral",
        "--model", model,
        "--model-short", model_short,
        "--layer", str(layer),
        "--percentile", str(percentile),
        "--seed", str(seed),
        "--n-prompts-per-side", str(n_prompts_per_side),
        "--n-held-out-harmful", str(n_held_out_harmful),
        "--top-k-refusal-partitions", str(top_k),
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cuda",
        "--output-dir", out_dir,
    ]
    print("+", " ".join(cmd), flush=True)

    try:
        subprocess.run(cmd, cwd="/root", check=True)
    finally:
        # Commit whatever landed even on failure — a crash during the K-sweep
        # still leaves a usable dictionary pickle and the partition loadings,
        # and the calibration cache is the expensive thing to lose.
        refusal_vol.commit()
        hf_vol.commit()

    return out_dir


@app.local_entrypoint()
def main(
    model: str = "google/gemma-2-27b-it",
    layer: int = 35,
    percentile: float = 12.0,
    seeds: str = "0",
    n_prompts_per_side: int = 300,
    n_held_out_harmful: int = 50,
    top_k: int = 5,
    max_new_tokens: int = 60,
    smoke: bool = False,
):
    """Defaults reproduce the paper's protocol at gemma-2-27b-it L35.

    Note p=12, not the script's own default of 8.0 — the paper reports p=8 as
    a fragmentation *failure* on gemma-2-2b-it, so the upstream default is the
    one setting you specifically do not want.
    """
    if smoke:
        # The paper's exact configuration. Two jobs at once: proves the image
        # works, and tells you whether this harness reproduces the published
        # number (delta_exemplar in {-0.74, -0.96} for 2 of 4 seeds at p=12)
        # before you spend 12 GPU-hours at 27B.
        model, layer = "google/gemma-2-2b-it", 20
        print("SMOKE: paper's own config, gemma-2-2b-it L20 p12")

    model_short = model.split("/")[-1]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    print(f"model={model} layer={layer} p={percentile:g} seeds={seed_list}")
    print(f"held-out harmful={n_held_out_harmful}  K sweep=1..{top_k}")

    args = [
        (model, model_short, layer, percentile, s, n_prompts_per_side,
         n_held_out_harmful, top_k, max_new_tokens)
        for s in seed_list
    ]

    # The function's declared resources are sized for 27B: an 80 GB card and
    # 192 GB of host RAM to survive TransformerLens materialising the model
    # twice on CPU. A 2B is 5 GB of bf16 — that container is ~4x the cost for
    # no benefit, and this track runs it four times.
    #
    # A100-40GB rather than something cheaper per hour because this workload is
    # 1550 *unbatched* 60-token generations, which is decode-bound on memory
    # bandwidth: A10G's 600 GB/s would run ~2.5x longer than A100's 1555 GB/s
    # and end up costing more, not less.
    fn = behavioral
    if smoke or "2b" in model_short.lower():
        fn = behavioral.with_options(
            gpu="A100-40GB", cpu=8.0, memory=32768, timeout=2 * 60 * 60
        )
        print("container: A100-40GB / 32 GB RAM (2B-sized)")

    # starmap runs the seeds as concurrent containers: same total GPU-seconds
    # as running them back to back, but ~3 h of wall clock instead of ~12.
    for out in fn.starmap(args):
        print("wrote", out)

    print(
        "\npull results with:\n"
        "  modal volume get ep-refusal 'results/**' ./runs/refusal_reference"
    )
