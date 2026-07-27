# EPDashboard

Feature-dashboard tooling for Exemplar Partitioning dictionaries — the EP
analogue of [SAEDashboard](https://github.com/jbloomAus/SAEDashboard).
Region-level dashboards first; dictionary-level view planned on top of the
same `header.json`. Methodology and design rationale: [DECISIONS.md](DECISIONS.md).

## Quick start

```bash
# one dictionary, defaults (pile @ 128 tokens, 24 576 prompts, all regions)
python -m epdashboard --run-dirs runs/qwen3_5-2b_L19_p8p0_ctx128_cache_pile \
    --out epdash_out

# p-sweep in one activation pass (dicts must share model + layer)
python -m epdashboard \
    --run-dirs runs/<slug>_p4p0_...,runs/<slug>_p8p0_... --out epdash_out

# from an extract_cache shard dir — no model, no GPU
python -m epdashboard --run-dirs runs/<slug> --cache-dir activations_cache/<slug> \
    --out epdash_out

# small smoke run, a subset of regions
python -m epdashboard --run-dirs runs/<slug> --n-prompts 400 --regions 0:50 \
    --out /tmp/epdash_smoke

# fully configured
python -m epdashboard --config my_config.json
```

Output per dictionary, under `<out>/<run_name>/`:

| file | contents |
|------|----------|
| `header.json` | dict metadata, provenance, replay check, batch manifest, region summary table |
| `regions_NNN.json` | full region records, `regions_per_batch` per file |
| `index.html` | sortable/filterable region table (open this) |
| `regions_NNN.html` | region cards, self-contained, `file://`-friendly |

`<out>/config.json` records the exact config used.

## Config

All fields of `epdashboard.config.EPVisConfig` (JSON file via `--config`,
common ones as CLI flags). Highlights:

- `dataset` (default `monology/pile-uncopyrighted`), `context_length` (128),
  `n_prompts` (24 576) — the pass-1 activation budget
- `cache_dir` — replay an `extract_cache` shard dir instead of forwarding
- `regions` — subset of region indices (CLI: `0:100` or `3,17,42`)
- `n_closest`, `n_bands`, `n_per_band`, `n_random`, `buffer` — the sequence
  panels
- `regions_per_batch` — JSON/HTML batching granularity
- `lens_cache` — where unembedding/J-lens npz caches live
  (default `<out>/.lens`)

## Region card panels

- **stats** — member count (build + rescan), density, coherence, mean
  distance, projection moments
- **nearest regions** — full-space exemplar cosine, cross-linked
- **projection histogram** — member reservoir vs corpus subsample
- **distance-to-exemplar histogram** — region tightness over [0, θ]
- **logit lens** — promoted/suppressed tokens for exemplar and mean member
  direction, plus J-lens + verbalizability where a Jacobian lens exists
- **sequences** — closest members, near/mid/far distance bands, random draw;
  tokens colored by projection, region members underlined, firing token
  marked
