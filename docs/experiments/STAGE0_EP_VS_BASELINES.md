# Stage 0 — the benchmark, the strata, the vet split, the compute bill

Stage 0 of `PLAN_EP_VS_BASELINES.md` §13. Nothing here touches EP; the point is
to fix the measuring stick before any EP number exists.

Artifacts: `artifacts/runs/probes/stage0_manifest.csv`,
`artifacts/runs/probes/stage0_provenance.json`.
Code: `experiments/probes/{benchmark,stage0_manifest,stage0_baselines,extract_acts,ep_arrows}.py`.

## 1. The suite is fully local; nothing had to be reproduced from scratch

The `sae-probes` PyPI package (v0.4.0, the maintained repackaging of KE25) ships
**all 113 binary datasets inside the wheel** as zstd CSVs, 43 MB. The Dropbox
mirror the paper README points at is not needed. The paper repo
(`JoshEngels/SAE-Probes`) additionally carries **KE25's own published results**:
one row per (dataset, method) with `val_auc` and `test_auc`, for gemma-2-9b at
layers {embed, 9, 20, 31, 41}, gemma-2-2b at layer 12, and Llama-3.1-8B — plus
the imbalance, corrupt, scarcity and OOD regimes.

Two consequences worth stating plainly:

- We do not have to *trust* our baseline reimplementation. We can diff it
  against theirs per dataset. `stage0_baselines.py` does exactly that, with the
  agreement criterion declared in the module docstring (median |ΔAUC| < 0.005,
  max < 0.05) before it was ever run.
- The baselines are also usable **without any GPU at all** as a published
  reference. If our extraction agrees, the published table can stand in for the
  baseline arm in every regime we do not re-run ourselves.

`sae_lens` is deliberately not installed — `sae_probes/__init__.py` imports it
eagerly, and it would drag `transformers<5` into a venv on 5.14. We stub it; the
SAE arm is unused because KE25's SAE numbers are already published and our
arrows are EP.

## 2. The split, reproduced exactly

`seed=42`, positives forced to 50% of train, test = the remainder, activations
capped at `MAX_AMT=5000` rows per dataset.

**`num_train` is `min(size - 100, 1024)`, not a flat 1024.** 11 of the 113 sit
below 1,024 (minimum 628). Using a flat 1,024 would have quietly changed the
train size on those 11 and every AUC computed from them. All 113 support a
balanced draw.

## 3. Strata — and the answer to Gate 2's central defect

Gate 2's negative result was uninterpretable because all five of its tasks had a
probe above 0.99: a piecewise-constant lookup cannot show an advantage against a
ceiling. Cutting the 113 on KE25's published quiver test AUC at layer 20:

| stratum | cut | n | median AUC | min |
|---|---|---:|---:|---:|
| `ceiling` | ≥ 0.99 | 51 | 0.9994 | 0.991 |
| `headroom` | 0.90–0.99 | 37 | 0.9608 | 0.911 |
| `hard` | < 0.90 | 25 | 0.8127 | 0.655 |

**62 of 113 datasets have room for an arrow to show a gain, and 25 have a lot.**
That is the single most important number in stage 0: it is the thing the prior
gates did not have, and it is what makes this study capable of a result in
either direction rather than only a null.

The hard stratum is not junk data — it is `glue_cola`, `glue_mrpc`,
`glue_mnli_neutral`, hate-speech severity, tweet emotion, NYC borough,
truthfulness (`44_phys_tf`, `47_reasoning_tf`, `54_cs_tf`), IT-ticket category.
These are concepts that a hyperplane at layer 20 genuinely fails to capture, so
they are where a partition has its only structural chance. They are also where
"the concept is not linearly present" and "the concept is not present at all"
are hardest to tell apart, which is why §8b's vetting criterion has to be
label-free and fixed in advance.

The quiver's winner is `logreg` on 77 of 113. Ties on `val_auc` are common
because validation saturates, and `idxmax` breaks them by row order — so a
marginal-gain number computed against the quiver is weaker than it looks, and
`ke25_logreg_test_auc` is carried alongside as the honest single-arrow reference.

## 4. §8b vet split — recorded now, before any EP number

`VET_SEED = 20260804`, stratified on `stratum` so neither half holds all the
hard datasets. 57 vet-fit / 56 vet-test.

| stratum | vet_fit | vet_test |
|---|---:|---:|
| ceiling | 25 | 26 |
| headroom | 19 | 18 |
| hard | 13 | 12 |

`vet_fit` digest (sha256 of the sorted tag list, first 16 hex): **`faab2b69ba4b5ec2`**.
Recorded with the git SHA in `stage0_provenance.json`. The digest exists so that
"we split it this way" cannot be revised after the EP results are in.

## 5. Compute — measured, not guessed

Token counts under gemma-2's tokenizer, truncated at KE25's `max_seq_len=1024`:

| | |
|---|---:|
| examples | 334,092 |
| tokens, clipped | 34,126,601 |
| **tokens, padded** | **63,344,162** |
| examples truncated at 1024 | 4.2% |
| stored activations (fp32, last token) | 4.79 GB |

Padded is the number that matters: the benchmark pads each batch of 32 to that
batch's longest text in dataset order, and on the code and cancer-report
datasets that nearly doubles the forward-pass work actually done.

gemma-2-9b has 198 M params per layer; `blocks.20.hook_resid_post` needs layers
0–20, so **4.16 B params below the hook** — half the model — for
**5.27 × 10¹⁷ FLOPs**.

| device | @20% MFU |
|---|---:|
| H100 80 GB | **~0.75 h** |
| L40S 48 GB | ~2 h |
| A100 80 GB | ~2.4 h |
| M5 24 GB MPS | ~50 h, and it will swap |

Add ~20 GB of model download. **VRAM floor is ~40 GB**: TransformerLens
instantiates all 42 layers even though `stop_at_layer` runs only 21, so bf16
weights alone are 18.5 GB and 24 GB is too tight with hook caching on top.

The second and only other GPU item is the **EP dictionary builds** for
gemma-2-9b L20 — there is no existing dictionary for this model, and the plan's
factor F2 requires three provenances (P-PILE / P-TRAFFIC / P-PROMPTED) built
fresh regardless. At the repo's usual 1 M tokens × ctx 128 that is ~1.7 × 10¹⁶
FLOPs per provenance, about 3% of the extraction each. The θ ladder is free
once the activation cache exists, since one cache serves every percentile.

### The extraction can be downloaded instead

KE25 published the activations after all, as two tarballs under the same
Dropbox folder:

- `model_activations_gemma-2-9b_OOD.tar.gz` — **32 MB**, 8 datasets at layer 20.
  Fetched; in `artifacts/runs/probes/acts/`. This is the covariate-shift set.
- `model_activations_gemma-2-9b.tar.gz` — **32.3 GB**, every dataset × every
  layer {20, 31, 41, 9, embed}, interleaved dataset-major. The layer-20 members
  are spread throughout, so the whole archive has to be transferred even though
  only ~4.8 GB is kept. Measured transfer ~4.7 MB/s → **~2 h**.
  `scripts/probes/fetch_ke25_acts.sh` does it resumably.

Tensors are `(n, 3584)` **float32**, and a good fraction carry CUDA storage
tags — `torch.load` needs `map_location="cpu"` or it raises per file, so a run
can get most of the way through the suite before failing. `benchmark.load_activations`
handles it.

Downloading is the better trade: same wall-clock as the pod, free, and it
removes the processing-mismatch risk on the probing side entirely, since these
*are* the tensors the published table was computed from. It does not remove the
pod — the dictionary builds still need one, and they must use the same
`from_pretrained` (processed) loader.

**Total: one 80 GB pod, ~4–6 hours, covers stages 1 and 2 completely** — or
~1 hour if the activations are downloaded and the pod only builds dictionaries.

Everything else is CPU: all five baseline families across 113 datasets, every EP
arrow, every coreset draw, all seven regimes, and every §8b criterion. The
activations are 4.79 GB — the whole study fits in memory on the laptop once the
extraction exists.

### Why gemma-2-9b L20 and not gemma-2-2b

The 2b route looks cheaper (~4× less extraction) but does not actually reuse
anything. Our existing dictionaries are `gemma-2-2b-it` at L20; KE25's 2b
baselines are the **base** model at **L12**. Neither matches, and F2 requires
new dictionary builds either way — so the only saving would be on the
extraction, which is already the cheaper half of a one-pod job. L20 of 9b is
KE25's headline configuration and the one with the richest published table
(five layers, all four regimes, plus the OOD set).

## 6. State

Done, no GPU used:

- 113 datasets local and loading; split reproduced exactly.
- Manifest with token counts, label balance, `num_train`, strata, vet roles.
- Extraction driver (`extract_acts.py`), a thin wrapper over KE25's own
  generator so the read position, truncation side and `stop_at_layer` cannot
  drift from theirs.
- Baseline reproduction + published-table diff (`stage0_baselines.py`).
- EP readout families and the matched-K coreset control (`ep_arrows.py`).

Blocked on a GPU: the extraction itself, and therefore the baseline agreement
check that gates every later EP claim.
