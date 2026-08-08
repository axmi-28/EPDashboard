# EP work — layout

Two things live here: the **EPDashboard** tool, and the **experiments** run on top of
Exemplar Partitioning. Everything either builds a dashboard, or asks a question about
EP regions. Shared model/dictionary plumbing sits in `qwen_ep/`.

```
epdashboard/      the tool — region-level dashboard builder (two-pass: scan, then re-forward)
qwen_ep/          shared plumbing both halves import
experiments/      the questions asked of EP
modal/            Modal runners (GPU jobs)
scripts/          local/pod driver scripts
docs/             plans, handoffs, results
artifacts/        everything generated (gitignored)
exemplar-partitioning/   upstream clone, vendored (gitignored)
```

Run everything from the repo root — `experiments`, `qwen_ep`, and `epdashboard` are
imported as top-level packages.

## epdashboard/

The dashboard package. Entry point:

```bash
python -m epdashboard --run-dirs artifacts/runs/<slug> --out artifacts/dashboards/epdash_out
python -m epdashboard.geometry <out>/<run> --run-dir artifacts/runs/<run>   # geometry+HTML only
```

Its only couplings to the rest of the repo are `qwen_ep.adapter` (the model seam) and
`qwen_ep.{lens_weights,jlens_weights}`. Paths in its own README/CLI help are generic
placeholders — in this repo the dictionaries are under `artifacts/runs/`.

Modal build: `modal run modal/epdash.py` — walkthrough in [docs/epdashboard/MODAL.md](docs/epdashboard/MODAL.md).

## qwen_ep/

Model + dictionary plumbing shared by both halves. Not experiments.

| module | what it does |
| --- | --- |
| `adapter.py` | the one model-touching seam (hooks, extraction fns) |
| `data.py` | Pile text streams |
| `build.py`, `extract_cache.py`, `sweep_p.py` | build a dictionary: discover, cache activations, sweep percentiles |
| `target_k.py` | solve for the theta that lands on a target region count K |
| `member_scan.py`, `inspect_dict.py`, `smoke.py` | inspect what came out |
| `lens_weights.py`, `jlens_weights.py` | logit-lens / Jacobian-lens weight loading |

Driver scripts: `scripts/dicts/`.

## experiments/

One subpackage per question. Each runs as `python -m experiments.<pkg>.<module>`.

| package | question | docs |
| --- | --- | --- |
| `monitor/` | is EP useful as a runtime OOD monitor? (Gates 0A/0B/1B) | `docs/experiments/PLAN_EP_MONITOR*.md`, `GATE*.md` |
| `role/` | how does *role* live in EP space? | `docs/experiments/PLAN_ROLE_QWEN3_4B.md`, `RUNPOD_ROLE.md` |
| `jailbreak/` | do jailbroken prompts leave the refusal region? | `docs/experiments/PLAN_JAILBREAK_GEMMA2_2B.md` |
| `persona/` | the Assistant Axis, discretized into EP regions | `docs/experiments/ASSISTANT_AXIS_EP.md` |
| `refusal.py` | refusal-region ablation (Qwen port of the paper's experiment) | `docs/experiments/HANDOFF_GEMMA2_2B_REFUSAL.md` |
| `concept_detect.py` | concept AUROC over regions | — |
| `legacy_dashboard/` | **superseded** v1 whole-dictionary dashboard, kept for reference | — |

`jailbreak/` and `refusal.py` load the upstream harness by file path, so they need the
`exemplar-partitioning/` clone present (and on `PYTHONPATH` for `ep.discovery`).

Tests: `python -m pytest experiments`.

## artifacts/ (gitignored)

```
runs/                   dictionaries + experiment outputs, one dir per slug
logs/
figures/                figures for write-ups
cache/activations/      activation shards
cache/lens/             lens npz caches
dashboards/             epdash_out, epdash_out_27b, epdash_smoke, hf_space, legacy/
```
