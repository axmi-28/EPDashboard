# EPDashboard

Feature-dashboard tooling for Exemplar Partitioning dictionaries, 
refer back to [SAEDashboard](https://github.com/jbloomAus/SAEDashboard).

The unit of output is the **region page**, exactly as SAEDashboard's is the
feature dashboard. There is no dictionary-level page — that view belongs to
whoever hosts the dashboards (Neuronpedia), and `header.json` carries the data
it needs to build one.

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
| `vectors.npz` | `exemplar` / `mean` direction per region + the calibration `center` |
| `regions_NNN.html` | region cards, self-contained, `file://`-friendly (open these) |

`<out>/config.json` records the exact config used.

### Region vectors

`vectors.npz` holds the two candidate directions per region, rows aligned to
`regionIds` (**not** to the region index, unless the run built all K):

- `mean` — mean of the member unit vectors, renormalised. The "average region
  vector"; use this one for steering.
- `exemplar` — the seed activation that defines the cell. Use it for membership
  tests and ablation, not as a summary of the region's contents.
- `center` — the calibration center. Both directions are unit vectors in
  *centered* space, `(h − center)/‖h − center‖`. Additive steering
  (`h + α·v`) is unaffected by the center; projection/ablation is not, and must
  center first: `h' = center + (I − vvᵀ)(h − center)`.

`python -m epdashboard.vectors <out>/<run_name> --which mean` flattens them into
a JSONL shaped for Neuronpedia's vector upload (`Neuron.vector` / `hasVector`).

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
  distance, projection moments, mean margin + contested share (cell shell)
- **exemplar** — the first-arrival context that seeded the region
- **nearest regions** — full-space exemplar cosine, cross-linked
- **competitors** — which cells come *second* for this region's members
  (the competition graph; not the same ranking as cosine neighbours)
- **projection histogram** — member reservoir vs corpus subsample
- **distance-to-exemplar histogram** — region tightness over [0, θ]
- **logit lens** — promoted/suppressed tokens for exemplar and mean member
  direction, plus J-lens where a Jacobian lens exists. J-lens panels also
  report **verbalizability** (`1 − H/ln|V|`): 1 = the vocab readout spikes on
  a few tokens, 0 = it is flat and the top-k list is noise. Reported for the
  J-lens only — a mid-layer direction pushed straight through the unembedding
  is not on the model's output path, so its entropy scores the lens, not the
  region. The level is not comparable across models or across J fit budgets,
  so the fit `n` is printed beside it
- **sequences** — closest members, near/mid/far distance bands, random draw;
  tokens colored by projection, region members underlined, firing token
  marked
