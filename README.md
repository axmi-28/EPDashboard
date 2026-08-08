# EPDashboard

Feature-dashboard tooling for Exemplar Partitioning dictionaries, similar to
[SAEDashboard](https://github.com/jbloomAus/SAEDashboard).

Hosted examples: Qwen3.6-27B over the Pile:
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
EpDashboard will need the upstream ep` package importable: clone
[exemplar-partitioning](https://github.com/jessicarumbelow/exemplar-partitioning)
next to this repo 

Otherwise: `torch`, `transformers`, `datasets`, `numpy`, `huggingface_hub`,
`safetensors`. 

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
| `regions_NNN.html` | region pages, 256 regions each, JSON embedded|
| `regions_NNN.json` | region records |
| `header.json` | metadata, provenance, replay check, per-region summary table |
| `vectors.npz` | `exemplar` and `mean` direction per region |
