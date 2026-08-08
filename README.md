# EPDashboard

Feature-dashboard tooling for **Exemplar Partitioning** dictionaries — the EP
analogue of [SAEDashboard](https://github.com/jbloomAus/SAEDashboard).

The unit of output is the **region page**, exactly as SAEDashboard's is the
feature dashboard. There is no dictionary-level page: that view belongs to
whoever hosts the dashboards, and `header.json` carries the per-region summary
table needed to build one.

Live examples — Qwen3.6-27B over the Pile:
[layer 55](https://huggingface.co/spaces/andyx10/ep-dashboards-qwen27b) ·
[layer 56](https://huggingface.co/spaces/andyx10/ep-dashboards-qwen27b-l56) ·
[all dictionaries](https://huggingface.co/datasets/andyx10/ep-dashboards-qwen27b)

## Layout

```
epdashboard/     the tool
  runner.py        two-pass orchestration + CLI
  source.py        activations: replay a shard cache, or forward the model
  scan.py          pass-1 accumulators (membership, histograms, competition graph)
  sequences.py     pass-2 example gathering
  lens.py          logit-lens / J-lens tables, verbalizability
  geometry.py      neighbour cosines
  writer.py        region records -> header.json + regions_NNN.json
  vectors.py       vectors.npz export (steering / ablation directions)
  html.py          self-contained region pages
qwen_ep/         the model + dictionary seam epdashboard imports
  adapter.py       the one model-touching module (hooks, extraction)
  acts_dtype.py    activation-shard dtype handling
  lens_weights.py  unembedding + final-norm loading
  jlens_weights.py Jacobian-lens loading
modal/           cloud runner (see caveat below)
```

## Requirements

`epdashboard` needs the upstream **`ep`** package importable — dictionaries are
pickled as `ep.discovery.dictionary.Dictionary`, and `qwen_ep.adapter` imports
`ep.discovery.extraction`. Clone
[exemplar-partitioning](https://github.com/jessicarumbelow/exemplar-partitioning)
next to this repo (or anywhere on `PYTHONPATH`); it is deliberately not
vendored here.

Otherwise: `torch`, `transformers`, `datasets`, `numpy`, `huggingface_hub`,
`safetensors`. Run from the repo root — `epdashboard` and `qwen_ep` are
imported as top-level packages.

## Quick start

```bash
# from an extract_cache shard dir — no model load, no GPU, and the examples
# are the dictionary's actual members
python -m epdashboard --run-dirs runs/<slug> --cache-dir activations/<slug> \
    --out epdash_out

# or stream the corpus through the model
python -m epdashboard --run-dirs runs/<slug> --out epdash_out

# several dictionaries in one activation pass (must share model + layer)
python -m epdashboard --run-dirs runs/<slug>_p4p0,runs/<slug>_p8p0 --out epdash_out

# smoke run
python -m epdashboard --run-dirs runs/<slug> --n-prompts 400 --regions 0:50 \
    --out /tmp/epdash_smoke
```

Full CLI reference and panel-by-panel notes: [epdashboard/README.md](epdashboard/README.md).

## Output

Per dictionary, under `--out`:

| file | contents |
| --- | --- |
| `regions_NNN.html` | region pages, 256 regions each, JSON embedded — renders with no server |
| `regions_NNN.json` | the region records those pages embed |
| `header.json` | metadata, provenance, replay check, per-region summary table |
| `vectors.npz` | `exemplar` + `mean` direction per region, plus the calibration `center` |

Both direction sets are unit vectors in *centered* space. `mean` is the one to
steer with; `exemplar` is the seed activation defining the cell, for membership
tests and ablation. Steering is additive so the center cancels — projection and
ablation must center first.

## Sizing

Page size grows as **K²/256**: every page embeds the full `regionTable` and the
region→file map, so those bytes repeat on every page. At K=16k a page is
~7.5 MB (~49% duplicated); at K=40k it is ~13 MB (~71%). Past roughly
K=15–20k the format stops reading well — prefer a coarser partition, or split
the shared tables out into a sibling file fetched once.

`--comp-max-k` (default 8192) silently drops the runner-up competition panel
above that K. Raise it to keep the panel; the matrix is a dense (K, K) int32,
so K=40k costs ~6.6 GB of transient memory.

The scan is pure NumPy — no CUDA path — so in `--cache-dir` mode the job is
CPU-bound and wants cores, not a GPU.

## modal/

`modal/dicts_27b.py` builds EP dictionaries on [Modal](https://modal.com).

**It does not run from a clone of this repo**: it shells out to
`qwen_ep.extract_cache`, `qwen_ep.sweep_p`, `qwen_ep.target_k` and
`qwen_ep.member_scan`, which are part of the dictionary-construction pipeline
and are not tracked here. It is kept for reference; treat it as a template.
