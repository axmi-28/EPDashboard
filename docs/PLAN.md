# Exemplar Partitioning on Qwen3.5-2B-Base — Setup & Replication Plan

## What EP is (from the post / paper / repo)

**Exemplar Partitioning (EP)** is a *training-free* alternative to sparse
autoencoders for decomposing LLM activation space. Instead of learning a
dictionary by gradient descent against a reconstruction loss, EP builds a
**Voronoi partition of centered, unit-norm activation space** by
**leader-clustering** a *single streamed pass* of activations:

1. **Calibrate** (once): stream ~200k activations from a layer, compute a
   *center* (spherical-mean direction × mean projection magnitude) and a
   *threshold* θ = the `p`-th percentile of pairwise centered-cosine distances.
   Smaller `p` ⇒ tighter cells ⇒ more partitions.
2. **Discover**: stream activations; for each one, if it lies within θ (cosine)
   of an existing **exemplar**, assign it there; otherwise it becomes a new
   exemplar anchoring a new region. Stop at **saturation** (no new regions).

Each region is anchored by a **real observed activation** (the exemplar), which
serves as both membership criterion *and* an intervention direction. Dictionary
size is **not** prespecified — it emerges from activation geometry at θ.

**Headline results in the paper (Gemma-2-2B):**
- ~10³× fewer tokens than SAEs (≈3.6M activation tokens vs ≈4B for GemmaScope),
  no backward passes.
- AxBench latent-concept detection @ Gemma-2-2B-it L20, p=1: **mean AUROC 0.881**
  (+0.126 over the canonical GemmaScope SAE, within 0.03 of SAE-A).
- Refusal in instruct-Gemma concentrates in a region whose exemplar ablation
  collapses held-out refusal (causal).
- EP one-hot probes retain ~97% of raw-activation probe accuracy at ℓ₀=1.
- Nearest-exemplar distance = free OOD signal.
- ~20% of EP regions match an SAE feature at F₁>0.5 (partial, asymmetric overlap).

## The adaptation problem

The repo (`ep`) is written against **TransformerLens `HookedTransformer`** and
ships prebuilt dictionaries only for **Gemma-2-2B**. Two frictions for Qwen3.5:

1. **TransformerLens doesn't (yet) model Qwen3.5.** Qwen3.5-2B-Base is a
   *vision-language* model (`Qwen3_5ForConditionalGeneration`) whose text
   backbone is a **24-layer hybrid** of *linear (Mamba-style) attention* with
   *full attention* every 4th layer (full-attn at layers **3, 7, 11, 15, 19,
   23**), over a shared **residual stream of width 2048**.
2. But the EP pipeline only touches the model through **one seam**:
   `extract_fn(model, prompts, hook_name) -> ExtractionResult(x=(N, D) float32)`.
   Centering, normalisation, and clustering all happen in `ep` on numpy.

**So we keep the `ep` core 100% unchanged and replace only that seam** with a
HuggingFace-native extractor that puts a forward hook on decoder layer `L`
(its output = post-block residual stream, i.e. `blocks.L.hook_resid_post`).
This side-steps TransformerLens entirely and is robust to the novel architecture.

## What we built (`qwen_ep/`)

| file | role |
|------|------|
| `adapter.py` | `QwenModel` (HF load + residual-stream hook + logit lens) and `make_extract_fn` — the drop-in `extract_fn`. |
| `data.py` | Pile (`monology/pile-uncopyrighted`) streaming, matching the paper's corpus; `wikitext` fallback. |
| `build.py` | Orchestration: calibrate → discover → save `dictionary.pkl` + `metadata.json`. Reuses `ep.calibrate_pipeline` / `ep.discover`. |
| `inspect_dict.py` | Print partition sizes, nearest prompts, and logit-lens tokens per region. |
| `smoke.py` | End-to-end validation on a few prompts. |
| `extract_cache.py` | Forward-pass **once**, shard activations (fp16)+prompts to disk. |
| `sweep_p.py` | Build dictionaries at many `p` from one cache (no model). |

Environment: Python **3.12** venv (`.venv/`), torch **2.13 (MPS)**, transformers
**5.14** (knows `qwen3_5`), `ep` installed `-e --no-deps` (its `transformer-lens`
pin is intentionally skipped — we don't use it). Runs on the M5 (24 GB), forward-
only, no gradients.

## Rough plan / roadmap

### Phase 0 — Setup & validate  ✅ (this session)
- Clone repo, build env, write adapter, `smoke.py` end-to-end.

### Phase 1 — First real dictionary (validation scale)
Build one dictionary at a **middle layer** and a **moderate resolution** to
confirm the whole pipeline, saturation behaviour, and interpretability:
```
python -m qwen_ep.build --layer 12 --percentile 8 \
    --calibration-tokens 100000 --max-tokens 1000000 \
    --context-length 128 --corpus pile
python -m qwen_ep.inspect_dict --dict artifacts/runs/<slug>/dictionary.pkl --top 25
```
Sanity checks: partition count grows then saturates; large regions look like
function words / punctuation / format tokens; smaller regions are semantically
specific; logit-lens tokens per exemplar are coherent.

### Phase 1 results ✅ (L12, p=8, Pile, 433k acts)
Saturated at **217 partitions**, 0 singletons, top-10 cover 24%. Clearly
interpretable regions surfaced unsupervised: interrogative "why" questions,
biomedical/assay language, source-code license headers, C# import blocks,
"and/or/other" connectives, subordinating conjunctions. Confirms the core EP
claim on Qwen. Caveats: p=8 is coarse (few broad regions); logit-lens noisy at
L12 (mid-stack + 248k multilingual vocab) → motivates finer `p` and a later
layer. Clustering was 3.4 s; forward passes were ~all the wall time.

### Phase 2 — Resolution sweep & layer sweep  (in progress)
Efficient path implemented: **one 3M-token extraction at L19 → cache → sweep
`p∈{1,2,4,8}`**. Later, full-attention layer (L19) should read out cleaner than
L12; finer `p` yields more, tighter regions (validated: p=4 gave >2× the regions
of p=8 on the same data).
- Sweep `p ∈ {1, 2, 4, 8, 10}` at the chosen layer → reproduce the paper's
  "finer `p` ⇒ more regions" curve and a saturation plot (partitions vs tokens).
- Sweep layer over the residual stream, **including full-attention layers**
  (3/7/11/15/19/23) vs linear-attention layers — a Qwen3.5-specific question the
  paper couldn't ask: *does the hybrid architecture concentrate interpretable
  structure at full-attention layers?*

### Phase 3 — Replicate paper properties
- **OOD signal**: nearest-exemplar distance on in-distribution (Pile) vs
  out-of-distribution (code, other languages) text.
- **Probing**: EP one-hot vs raw-activation linear probe accuracy on a few
  concepts (retain ~97%?).
- **Causal intervention**: pick a semantically clean region, ablate its exemplar
  direction, measure behavioural effect (analogue of the refusal study — but
  base model, so target something like a topic/format rather than refusal).
- **SAE comparison**: *deferred* — no public SAEs for Qwen3.5-2B, so the
  head-to-head AxBench/SAEBench numbers aren't directly reproducible. Options:
  (a) qualitative interpretability only, (b) train a quick baseline SAE later.

### Phase 4 — Scale toward paper budget
- Grow to ≈3.6M activation tokens (paper's build budget) at the best `(layer, p)`
  and run the interpretability battery on the full-scale dictionary.

## Open decisions (defaults chosen; easy to change)
- **Layer**: default **L12** (mid-stack; a linear-attn layer). Consider **L11**
  or **L15** (post-full-attention) as alternatives — a Qwen3.5-specific probe.
- **Resolution**: start **p=8** (coarser, faster, fewer regions) to validate,
  then go finer.
- **Corpus**: **Pile** (matches paper). `wikitext` fallback if streaming stalls.
- **Token budget**: start **1M** activations to validate, scale to 3.6M.
- **BOS**: Qwen base has no BOS; we skip position 0 anyway (attention-sink slot),
  matching the paper's intent.

## Measured throughput (M5, MPS, ctx≈128, batch 16, L12)
Smoke test + benchmark confirm the pipeline runs. The Mamba/linear-attention
layers use a **pure-PyTorch fallback** on Mac (flash-linear-attention /
causal-conv1d are CUDA-only), so we're on the slow path — yet still fast enough:
- **≈976 tok/s, ≈962 acts/s** (warm; first pass pays MPS graph-compile warmup).
- **1M-token validation build ≈ 17 min** forward; **3.6M-token (paper parity)
  ≈ 60 min** forward. Clustering adds a little on top; grows with #regions.
- Model load from local cache ≈ 5 s (first-ever download of the 4.5 GB shard
  was the one-time ~11 min cost).
- Implication: **no GPU needed**. An H100 (with the fast linear-attn kernels)
  would cut the 3.6M build to ~1 min of forward, useful only for large sweeps
  or a baseline-SAE comparison.

## Known risks
- **MPS speed** through the Mamba-style linear-attention layers is the slow
  fallback path (measured ~976 tok/s above) — acceptable given EP's tiny token
  budget. Cache activations (`activations_cache_dir`) so a layer/`p` sweep costs
  one forward pass, not one per config.
- **Logit lens** omits a clean final-norm handle on the VL wrapper, so treat
  vocab projections as directional hints, not calibrated logits.
- **No Qwen3.5 SAEs** ⇒ the SAE head-to-head is not a like-for-like replication.
