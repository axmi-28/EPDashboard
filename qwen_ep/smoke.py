"""End-to-end smoke test: load Qwen3.5-2B, extract residual activations, and
run a tiny EP build. Validates the model download, MPS forward through the
hybrid linear/full-attention stack, the residual-stream hook, and the ep core.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("smoke")


def main() -> None:
    from qwen_ep.adapter import QwenModel, make_extract_fn
    from ep.discovery.pipeline import calibrate_pipeline, discover

    t0 = time.time()
    qwen = QwenModel()
    log.info("loaded in %.1fs: d_model=%d n_layers=%d layers_path=%s device=%s",
             time.time() - t0, qwen.d_model, qwen.n_layers, qwen.layers_path, qwen.device)

    layer_types = getattr(getattr(qwen.model.config, "text_config", qwen.model.config),
                          "layer_types", None)
    if layer_types:
        full = [i for i, t in enumerate(layer_types) if "full" in t]
        log.info("full-attention layers: %s", full)

    prompts = [
        "The capital of France is Paris, a city known for",
        "def fibonacci(n):\n    if n < 2:\n        return n",
        "In 1969, Apollo 11 landed the first humans on the Moon.",
        "The mitochondria is the powerhouse of the cell, producing ATP",
    ]

    t1 = time.time()
    res = qwen.extract_per_position(prompts, layer=12, batch_size=4)
    dt = time.time() - t1
    log.info("extraction: x.shape=%s dtype=%s  %.2fs  (%.0f acts/s)  finite=%s",
             res.x.shape, res.x.dtype, dt, res.x.shape[0] / max(dt, 1e-9),
             bool(np.isfinite(res.x).all()))
    log.info("act norm: mean=%.2f std=%.2f", np.linalg.norm(res.x, axis=1).mean(),
             np.linalg.norm(res.x, axis=1).std())

    # Logit lens on the mean activation direction (should surface plausible tokens).
    mean_dir = res.x.mean(0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-9)
    log.info("logit-lens(mean act): %s", qwen.logit_lens(mean_dir, k=10))

    # Tiny end-to-end EP build over a handful of repeated prompts.
    corpus = prompts * 40
    extract_fn = make_extract_fn(qwen, layer=12, batch_size=8)
    cal = calibrate_pipeline(
        model=qwen, texts=list(corpus), hook_name="layers.12", n_tokens=2000,
        percentile=10.0, extract_fn=extract_fn, prompt_batch_size=8,
    )
    log.info("calibration: ||center||=%.3f threshold=%.4f n=%d",
             float(np.linalg.norm(cal.center)), cal.threshold, cal.n_activations)
    result = discover(
        model=qwen, texts=list(corpus), hook_name="layers.12", calibration=cal,
        extract_fn=extract_fn, max_tokens=5000, prompt_batch_size=8,
        saturation_window=3,
    )
    log.info("SMOKE OK: %d partitions from %d activations (%.1fs)",
             len(result.dictionary), result.n_activations, result.elapsed_s)


if __name__ == "__main__":
    main()
