# Gate 1A — orient and sanity. EP model-diffing positive control on RMU

Run 2026-07-31, Modal A100-40GB (profile `decoderesearch`), app `ep-rmu-diff`.
Plan: `PLAN_EP_DIFF_RMU.md`. Hypotheses: `PREREG_EP_DIFF_RMU.md`, written before
any activation was extracted; Addendum 1 there records the one measurement added
after the smoke run, and why.

Artifacts under `artifacts/runs/rmu_diff/gate1a/`: `gate1a.json` (everything),
`weight_diff.csv` (291 rows), `layer_stats.csv`, `calibration.csv`, `pool.csv`,
`acc_{model}_{style}.csv`. Code: `experiments/rmu_diff/{data,gate1a}.py`,
runner `modal/rmu_diff.py`.

Probe: 1024 prompts (512 forget / 512 retain), 125,366 paired activations per
(model, layer), layers {4, 7, 14, 24}, p ∈ {10, 12}. Manifestation: 600 prompts
per style. Total wall 628 s.

---

## VERDICT

**Proceed to Gate 1B.** Every precondition passes and the instrument is not
blind to this intervention.

One pre-registered stop condition **fired on its literal wording** and I am not
going to bury that: at L7 the median base→RMU displacement is 0.47 in centred
cosine distance against θ = 0.79, i.e. sub-θ for 97.6% of forget tokens. Read
literally, stop condition 3 says halt and write up structural blindness.

I believe that criterion was mis-specified, for a reason stated in Addendum 1
*before* this run: displacement from where a token used to be is not what
decides whether EP introduces a region. The direct measurement is that **49.5%
of forget-prompt activations at L7 fall outside every base exemplar's radius**
— they would spawn new regions — against **1.0%** of retain activations. A 50:1
ratio is not blindness.

Both numbers are reported below in full. §6 sets out the disagreement, why it
arises, and what I recommend. **The call to continue past a fired stop condition
is yours, not mine** — that is the point of pre-registering it.

---

## 1. `ep` API surface (verified against source, this clone)

Everything the diff needs, re-exported at `ep.*`.

```python
# ep/discovery/calibration.py
Calibration(center: np.ndarray, threshold: float,
            n_activations: int, percentile: float)                            # :54 frozen
calibrate(activation_batches, *, n_tokens=200_000, percentile=10.0)           # :174
load_or_calibrate(model_name, hook_name, activation_batches_fn, *,
                  n_tokens=200_000, percentile=10.0, extras=None, force=False) # :258

# ep/discovery/pipeline.py
calibrate_pipeline(model, texts, hook_name, *, n_tokens=200_000, percentile=10.0,
                   extract_fn=None, extract_kwargs=None, prompt_batch_size=16,
                   seed=0, cache_model_name=None, cache_extras=None,
                   force_recalibrate=False) -> Calibration                     # :103
discover(model, texts, hook_name, calibration, *, extract_fn=None,
         extract_kwargs=None, log_cadence=1, checkpoint_cadence=10,
         saturation_window=1, max_tokens=None, max_prompts=None,
         prompt_batch_size=16, checkpoint_fn=None, log_fn=None, seed=0,
         merge_close=False, activations_cache_dir=None) -> DiscoveryResult     # :160

# ep/discovery/dictionary.py
Dictionary(center, threshold, *, merge_close=False)                            # :129
  .add_batch(x_batch, iteration=0, global_index_start=-1) -> list[list[int]]   # :298
  .assign(vecs) -> (partition_ids, distances)   # RAW acts in, centring internal # :670
  .distances(x) -> (N, K)                                                      # :684
  .finalize(min_members=1)                                                     # :649
  .from_hub(model_short="gemma-2-2b", layer=12, percentile=10, *,
            repo_id="J-RUM/exemplar-partitioning", cache_dir=None)             # :738

# ep/discovery/extraction.py
extract_per_position(model, prompts, hook_name,
                     max_positions_per_prompt=None, batch_size=128)            # :34
extract_final_position(model, prompts, hook_name, batch_size=256)              # :149
```

**Region attributes** (`Partition`, `dictionary.py:54`). Every quantity the diff
and the D statistic need is stored on the region, so both are computable from a
saved build with no re-forward:

| attribute | used for |
| --- | --- |
| `mean_member_direction` | A.7-style matching — **primary basis** |
| `exemplar_direction` | A.3-style matching; H1's cross-seed direction test |
| `member_coherence` (`‖Σ unit members‖ / N`) | the `c_i` in D_i |
| `member_count` | the `N_i` in D_i |
| `constituent_sample_indices` | global activation index → prompt → forget label |
| `sum_dist_to_exemplar`, `sum_sq_dist_to_exemplar` | cell-tightness diagnostics |
| `source_iterations` | stream-position diagnostics |
| `sample_prompts` / `boundary_prompts` | region reading in 1C |
| `sample_members` | ≤30 directions — **491 KB/region at d=4096**, capped to 0 in probes |

`from_hub` serves `gemma-2-2b` / `gemma-2-2b-it` only. Every dictionary here is
a local build.

**Three source facts the design rests on**, checked rather than assumed:

1. `_iter_prompt_batches` (`pipeline.py:64-68`) shuffles a **list** with
   `np.random.default_rng(seed)`. Same list + same seed ⇒ byte-identical prompt
   order for both models, by construction. Verified: the permutation of
   `list(range(n))` under that rng reproduces `discover`'s ordering exactly
   (test in `data.stream_order`). **This is what lets one cached forward pass
   serve every seed** — activations are extracted once in canonical order and
   replayed in each seed's permutation.
2. Calibration is frozen and passed in, so shared-vs-per-model calibration is a
   first-class knob, not a fork of the library.
3. `saturation_window` ends a build early. For a diff that is a hazard: if one
   checkpoint saturates before the other, the two dictionaries saw different
   streams. Gate 1B fixes the prompt budget and reports saturation as a
   diagnostic rather than acting on it.

## 2. Checkpoints, and RMU ground truth

Both load, both ungated, revisions pinned in `gate1a.py` and re-asserted at load:

| role | id | revision |
| --- | --- | --- |
| base | `HuggingFaceH4/zephyr-7b-beta` | `892b3d7a7b1cf10c7a701c60881cd93df615734c` |
| unlearned | `cais/Zephyr_RMU` | `70c55b3bf3141a8c24292dec0262b8aea03a0d4a` |

`MistralForCausalLM`, 32 layers, d_model 4096, `tie_word_embeddings: false`,
bf16 — identical configs.

**Hyperparameters**, from `run_rmu_zephyr.ipynb` in `centerforaisafety/wmdp`,
the notebook that produced the checkpoint:

```
python3 -m rmu.unlearn --max_num_batches 150 --batch_size=4 \
  --retain_corpora wikitext,wikitext \
  --forget_corpora bio-forget-corpus,cyber-forget-corpus \
  --steering_coeffs 6.5,6.5 --alpha 1200,1200 --lr 5e-5 --seed 42
```

with `rmu/unlearn.py` defaults `layer_id=7`, `layer_ids=[5,6,7]`,
`param_ids=[6]`, `module_str="{model_name}.model.layers[{layer_id}]"`.

**Correction to the brief: `max_num_batches` is 150, not 500.** c = 6.5,
α = 1200, layers {5,6,7}, loss at layer 7 — all confirmed.

**Two mechanism facts the brief did not state, and both matter:**

1. The control vector is **`torch.rand`, not `torch.randn`** — u is uniform on
   the positive orthant [0,1)^4096 then normalised, so E[cos(u, 𝟙/√d)] ≈ 0.866.
   Not an isotropic direction, and therefore directly checkable in activations.
2. The loss is `mse_loss(h_forget, c·u)`. RMU does not *add* c·u; it drives the
   layer-7 residual to **equal** a fixed vector of norm exactly 6.5. A point
   collapse, which is the one kind of magnitude change unit-normalisation cannot
   erase. This is the basis of the H3 call in the prereg.

### 2.1 Weight diff — exact ground truth, and it is tighter than documented

Streaming both checkpoints' safetensors and comparing all 291 tensors in fp32
(`weight_diff.csv`):

| | |
| --- | --- |
| tensors compared | 291 |
| tensors changed | **3** |
| changed | `model.layers.{5,6,7}.mlp.down_proj.weight` |
| tensors only in one checkpoint | none |
| max relative Frobenius change | 0.0892 |
| layers touched | **[5, 6, 7]** — matches documentation exactly |

So `param_ids=[6]` selects `mlp.down_proj.weight`, and **nothing else in the
model changed** — not embeddings, not attention, not layernorms, not the LM
head. This is a far more localised edit than "RMU fine-tunes layers 5, 6, 7"
suggests, and it makes L4 an exact control rather than a statistical one (§4.1).

### 2.2 A silent-corruption trap: the two repos ship different tokenizers

Caught by an assertion, not by inspection, which is the only reason it is in this
report and not in the results.

| | base | RMU |
| --- | --- | --- |
| tokenizer files | `tokenizer.json`, `tokenizer.model`, `added_tokens.json` | `tokenizer.model` only |
| `add_prefix_space` | `False` | `True` |
| vocabulary (32000 entries, id↔token) | **identical** | **identical** |
| prompts tokenised identically | **0 / 64** | |

`cais/Zephyr_RMU` ships no `tokenizer.json`, so transformers reconstructs a fast
tokenizer from `tokenizer.model` with different defaults. Same vocabulary,
different ids for identical text. Using each model's own tokenizer produced
**16,011 vs 15,883 activations** from the same 128 prompts and different
answer-token ids (`['A','B','C','D']` vs `[' A',' B',' C',' D']`) — a paired
comparison silently comparing different token streams.

**Resolution: the base tokenizer is forced on both models**
(`QwenModel(..., tokenizer_id=...)`, new). This is also what the intervention
was trained under — `rmu/unlearn.py` loads its tokenizer from
`model_name_or_path`, the base checkpoint. A tokenizer fingerprint over the pool
is now asserted equal across models before any delta is computed.

This is the same family as the Qwen3 role traps already recorded in this
project: a correctness bug that changes no shape and raises no error.

## 3. Per-position vs final-position — the decision

**Build per-position, analyse final-position.**

The paper's sharp behavioural result (A.6) is usually described as
"final-position", and that description is imprecise in a way that matters.
Reading the appendix directly: Table 6 reports `K (regions, all positions)` — 77
base, 207 IT — and then analyses *final-position regions*, defined as "regions
that absorb at least one prompt's final-position activation" (4 and 5 of them).
**The build is all-positions; the analysis is final-position.**

That resolves the hazard Gate 0A §6.1 raised and Gate 0B §9 listed as threat #1
to its own verdict:

- Calibration and discovery use the same extractor, so θ is calibrated on the
  distribution the dictionary is built from. `ep/README.md:83` warns that mixing
  them "silently produces meaningless cells". Gate 0B accepted the mismatch
  under protest; this gate does not repeat it.
- A final-position-only build is unavailable at this pool size (one activation
  per prompt) and would destroy the locality question: with no within-prompt
  structure there is nothing that could be *absent* at L4.
- Region labelling uses all-position forget-fraction as primary — RMU's loss
  acts on every forget-corpus token, not the last one — with the A.6-comparable
  final-position cross-tab as secondary.

### 3.1 Prompt pool, and a length confound that had to be designed out

WMDP-cyber runs to **2503 tokens** against MMLU's median 112. Unbanded, the
forget label would have been partly a length label — and Gate 0B §6 found two of
its five rungs separable at AUROC 0.998 **by token count alone**.

The pool is therefore banded to [48, 256] tokens and the retain arm is
histogram-matched to the forget arm's length distribution (20 bins):

| source | n | median tokens |
| --- | --- | --- |
| wmdp-bio | 256 | 131 |
| wmdp-cyber | 256 | 102 |
| mmlu | 512 | 116 |

Forget and retain medians both 116, means both 123.4. 0/1024 prompts exceed the
extractor's position cap, so no prompt loses its decision position. Cost: the
band excludes ~half of WMDP-cyber (964 of 1987 questions survive), which caps
the Gate 1B cyber arm at ~950.

`bio-forget-corpus` is gated and was **not** used and **not** substituted for.
It is RMU's training forget set; the question here is where the behaviour lives
at inference, and that is WMDP-MCQ, where the published capability drop is
measured. §3.2 confirms the drop is present in our pool, which is the only thing
the gated corpus would have been needed for.

### 3.2 H0 — the intervention manifests on our prompts (n = 600 per style)

| style | source | base | RMU | Δ |
| --- | --- | --- | --- | --- |
| plain | wmdp-bio | 0.640 | **0.260** | **−0.380** |
| | wmdp-cyber | 0.533 | **0.333** | **−0.200** |
| | mmlu | 0.580 | 0.560 | −0.020 |
| chat | wmdp-bio | 0.627 | **0.320** | **−0.307** |
| | wmdp-cyber | 0.507 | **0.313** | **−0.193** |
| | mmlu | 0.583 | 0.573 | −0.010 |

n = 150 per WMDP source (binomial SE ≈ 0.04), 300 for MMLU. RMU lands at or
below chance (0.25) on bio and near chance on cyber, while MMLU is preserved to
within 2 points. **This is the published RMU behaviour, reproduced on our exact
pool. H0 passes in both styles.**

**Style choice: `chat`.** Both carry the gap, so this is not a search over
styles — plain is reported in full above and will be reported again. Chat is the
deployment distribution, preserves MMLU marginally better (−0.010 vs −0.020),
and matches the EP paper's behavioural-build protocol.

## 4. Magnitude, outliers, and the delta

### 4.1 L4 — the determinism control passes exactly

Blocks 0–4 have bit-identical weights (§2.1), so block-4 output must be
identical. It is:

| quantity | value |
| --- | --- |
| max abs Δh over 125,366 activations | **0.000e+00** |
| fraction of activations with Δ exactly 0 | **1.000** |
| θ(p10) base / RMU | 0.5864 / 0.5864 |
| ‖μ‖ base / RMU | 1.13 / 1.13 |
| region formation: outside-θ, cell-change (forget and retain) | 0.000 / 0.000 |

The extraction path, hook placement, tokenisation, padding and pairing are all
validated by a measurement with a known answer. **Any non-zero L4 signal in Gate
1B is instrument failure**, and this is the baseline it will be checked against.

### 4.2 Per-layer table

θ and ‖μ‖ under each model's own calibration; norms are medians; `rel‖Δ‖` is
median ‖Δh‖/‖h_base‖; `cosdist` is centred cosine distance under the **shared
(base)** calibration.

| L | θ_base | θ_RMU | Δθ | ‖μ‖ b→r | ‖h‖ forget b→r | ‖h‖ retain b→r | cosdist fgt | frac>θ | rel‖Δ‖ fgt | rel‖Δ‖ ret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.5864 | 0.5864 | 0.000 | 1.13→1.13 | 1.13→1.13 | 1.13→1.13 | 0.0000 | 0.000 | 0.000 | 0.000 |
| **7** | 0.7855 | **0.3866** | **−0.399** | 1.33→1.82 | **2.01→3.59** | 2.00→1.98 | 0.4720 | 0.024 | **1.518** | 0.128 |
| 14 | 0.8300 | 0.5869 | −0.243 | 2.52→2.23 | 4.50→4.35 | 4.44→4.38 | 0.5322 | 0.110 | 0.936 | 0.086 |
| 24 | 0.8693 | 0.8106 | −0.059 | 5.64→5.24 | 15.89→**11.60** | 15.05→14.91 | 0.3942 | 0.125 | 0.780 | 0.065 |

Three things stand out.

**Locality is visible in the raw activations.** At L7 forget-prompt norms rise
79% (2.01→3.59) while retain norms move −1% (2.00→1.98). The retain
regularisation (α = 1200) did its job; the forget arm did not.

**θ collapses by 51% at the loss site.** This is the magnitude signal EP
discards at construction, and it is the strongest single argument for the
shared-calibration decision — see §7.1, where it turns out to be not a stylistic
preference but a load-bearing one.

**Massive-activation dims are present and shared.** Outlier ratio (top dim mean
|h| ÷ median dim) is 201 at L4, 122 at L7, 71 at L14, 26 at L24; the top dims are
the same in both models (2070, 3398, 3701, 3901). At L7 RMU's ratio falls to 71
because the injected junk raises the other dims, not because the outlier shrinks.

### 4.3 The junk direction — the mechanism is recovered, and H1′ is ruled out

`u_emp` = normalize(mean over forget tokens of Δh); `c_hat`, `u_hat` = norm and
direction of the mean **RMU** forget activation, which estimates c·u directly
without going through the delta.

| L | c_hat | c_hat / 6.5 | u_hat·𝟙̂ | Δ variance in 1 direction | pairwise cos, RMU / base forget | cos(dir, u) | cos(dir, −μ̂) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **7** | 3.94 | **0.61** | **0.665** | **0.872** | **0.418 / 0.088** | **0.800** | 0.226 |
| 14 | 3.80 | 0.58 | 0.329 | 0.478 | 0.193 / 0.028 | 0.574 | 0.329 |
| 24 | 6.00 | 0.92 | 0.091 | 0.114 | 0.042 / −0.009 | 0.195 | 0.205 |

At L7: **87.2% of the displacement lies along a single direction**, that
direction is 0.665 aligned with the all-ones direction (against 0.866 for a pure
`torch.rand` draw), and the mean RMU forget activation has norm 3.94 — 61% of the
way to the 6.5 target. RMU forget activations are 4.8× more mutually aligned than
base ones (0.418 vs 0.088).

This is the mechanism, measured, before any dictionary was built. It is exactly
what H1 predicts EP should see as one region.

**H1′ (the centring artifact) is ruled out at L7.** The collapsed centred
direction sits at cosine **0.800 to u** and only **0.226 to −μ̂**. ‖μ‖ = 1.33
against a target norm of 6.5, so c·u dominates the centring term and the region
direction carries mechanism, not an artifact of subtracting the mean. The
discriminator was pre-committed and it came out cleanly on the u side.

**The junk dissipates with depth**, monotonically on every measure: variance in
one direction 0.872 → 0.478 → 0.114, positive-orthant alignment 0.665 → 0.329 →
0.091. That is direct support for H2's secondary prediction (decay, not growth)
— recorded now, before the dictionaries exist.

### 4.4 Region formation — would EP actually carve a new region here?

The measurement added in Addendum 1. Build the base dictionary, then assign the
RMU activations into it: `outside-θ` is the fraction further than θ from **every**
base exemplar (would spawn a region), `cell-change` the fraction that switch
cell, `top-cell` the largest single cell's share. Retain is the control.

| L | p | K_base | θ | forget outside-θ | forget cell-change | forget top-cell | retain outside-θ | retain cell-change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 10 | 1141 | 0.5864 | 0.000 | 0.000 | 0.046 | 0.000 | 0.000 |
| 4 | 12 | 948 | 0.5958 | 0.000 | 0.000 | 0.071 | 0.000 | 0.000 |
| **7** | 10 | 273 | 0.7855 | **0.495** | 0.251 | 0.065 | **0.010** | 0.042 |
| **7** | 12 | 208 | 0.7973 | **0.491** | 0.281 | 0.059 | **0.010** | 0.045 |
| 14 | 10 | 214 | 0.8300 | 0.091 | 0.524 | 0.184 | 0.004 | 0.056 |
| 14 | 12 | 146 | 0.8434 | 0.013 | **0.532** | **0.400** | 0.001 | 0.047 |
| 24 | 10 | 523 | 0.8693 | 0.047 | 0.447 | 0.057 | 0.001 | 0.038 |
| 24 | 12 | 362 | 0.8826 | 0.009 | 0.462 | 0.087 | 0.000 | 0.042 |

**At the loss site, half the forget activations leave the dictionary entirely
(49.5%) against 1.0% of retain — a 50:1 ratio.** The retain arm's 4–6%
cell-change rate is the within-experiment noise floor and it is an order of
magnitude below the forget arm everywhere below L24.

**Downstream the mechanism changes shape rather than fading.** At L14 only 1.3%
of forget activations fall outside the dictionary at p=12, but 53% change cell
and **40% pile into a single existing cell**. So L7 is where new regions should
appear; L14 is where an existing region should swell. Those are different
predictions and Gate 1C can separate them — worth noting that H1 as pre-written
("one massive **introduced** region at layers ≥ 7") may be the right shape at L7
and the wrong shape at L14, where the prediction should be a massive *persisted*
region that gains members. That distinction is not a revision to H1; it is
recorded here so it cannot be invented later.

## 5. Timing, and the Gate 1B budget

Measured on A100-40GB, batch 16, 1024 prompts, 125,366 activations per
(model, layer):

| stage | measured |
| --- | --- |
| extraction | 6,600–6,700 acts/s → **19 s** per (model, layer) |
| model load (weights on `ep-hf` volume) | ~5 s each |
| calibration | ~15 s per (model, layer, p) |
| leader clustering | **44,867 acts/s** → 2.8 s per dictionary at p=12 |
| timing build | K = **208**, θ = 0.7973, largest partition 6,167 (4.9%), 2 singletons |
| **saturated** | **No** — the build is budget-limited, not saturation-limited |
| total Gate 1A wall | 628 s |

**Saturation did not fire and will not at Gate 1B scale.** With a fixed prompt
pool the stream simply ends. For a *diff* that is the correct regime — both
models must consume identical streams, and an early stop on one side would break
that (§1, fact 3). K is therefore a budget-determined quantity and will be
reported as such, with the growth trace as evidence of whether it had plateaued.

**Gate 1B projection.** Pool 4800 prompts ≈ 585k activations per (model, layer).

| item | estimate |
| --- | --- |
| extraction, 4 layers × 2 models | 585k / 6,650 × 8 ≈ **12 min GPU** |
| calibration, 4 layers × 2 models × 2 p (+ shared) | ≈ 4 min |
| 32 dictionaries (585k / 45k ≈ 13 s each) | ≈ 7 min, **no model resident** |
| Hungarian matching, D_i, Jaccard, nulls | minutes, local |
| **total** | **< 45 min on one A100-40GB, ≈ $2–3** |

The grid is far cheaper than the brief assumed. **Recommendation: buy a third
seed** (48 dictionaries, ~10 min more). Two seeds give one cross-seed pair per
cell — a point, not a distribution — and A.7 used five. Three gives three pairs
per cell and an actual spread on the noise floor that Gate 1B's kill criteria are
measured against. The per-model calibration arm (§7.1) is also affordable in
full rather than at two layers.

## 6. The fired stop condition

Pre-registered stop condition 3, verbatim: *"H3 is confirmed instead of H1 if the
per-token distribution of centred-cosine distance d(h_base, h_RMU) on forget
prompts at L7 sits below θ for the majority of tokens while ‖Δh‖/‖h‖ is large."*

**It fires.** At L7: median cosdist 0.4720 vs θ 0.7855 (0.60 θ), 97.6% of forget
tokens below θ, and rel‖Δh‖ = 1.518 — very large. Read literally: halt.

**The criterion measures the wrong quantity, and Addendum 1 says so in writing
before this run.** A region is introduced when an activation is further than θ
from **every existing exemplar** — a distance to a fixed set of anchors — not
when it has moved further than θ from where it used to be. These come apart in
both directions, and here they come apart hard: displacement 0.60 θ, yet 49.5%
of forget activations end up outside the entire base dictionary.

**Why they disagree is not mysterious.** This project already measured that EP
members sit at ~90% of θ from their exemplar — the cells are shells, not balls,
and the interiors are nearly empty. An activation already at 0.9 θ needs only a
modest, largely orthogonal displacement to cross θ. A sub-θ displacement
producing ~50% region escape is the expected behaviour of this geometry, not an
anomaly. The pre-registered criterion implicitly assumed activations sit near
their exemplars. They do not.

**What I am not claiming.** That the stop condition "actually passed" — it did
not, on its wording. That the new measurement is a better *hypothesis* — it is a
better *proxy for the same question*, chosen for a stated structural reason, with
its control (the retain arm) fixed by the design rather than after the fact.

**The decision is yours.** Three defensible readings:

1. **Continue (my recommendation).** The criterion was a mis-specified proxy;
   the direct measurement of the thing it proxied is unambiguous at 50:1; the
   mis-specification is documented with a timestamp and was identified before
   the full run.
2. **Halt as written.** Defensible, and the honest cost of pre-registration is
   that it sometimes binds inconveniently. The write-up would be "EP's
   displacement resolution is coarser than a known intervention", which is true
   but, I think, not the useful statement.
3. **Continue with the stop condition re-armed for Gate 1B**, on the corrected
   quantity: kill if the introduced set at L7 is not distinguishable from the
   retain arm's. This is close to Gate 1B's existing vacuity test (H4) and I
   would fold it in there.

## 7. Two design decisions this gate settles

### 7.1 Shared calibration is now load-bearing, not stylistic

The plan chose shared calibration (base μ, θ applied to both) as primary with
per-model as a robustness arm. The measured θ collapse makes that decisive:
θ_RMU at L7 is **0.387 against 0.786** — the calibrated radius halves.

Leader clustering's K scales roughly as (1 − θ)^4.6 near the fine end (measured
in this project on Qwen3.6-27B L55; indicative, not exact, at this scale). The
ratio (1 − 0.387)/(1 − 0.786) = 2.86 implies **K_RMU larger than K_base by two
orders of magnitude** under per-model calibration. Hungarian assignment matches
`min(K_a, K_b)` pairs; matching 273 base regions against tens of thousands of RMU
regions is not a diff, it is an artefact of the radius.

So per-model calibration does not merely "assume the θ-shift away" — at this
intervention size it makes the dictionaries structurally incomparable. It stays
as a diagnostic arm (Δθ and Δ‖μ‖ are reported per layer, above) but shared
calibration is the only viable primary. **Worth stating plainly: had we followed
the paper's A.3 protocol unmodified, the L7 comparison would have been
meaningless.**

### 7.2 The paper's own cross-checkpoint precedent argues for caution, not comfort

A.3 (Gemma-2-2B vs -it, Pile, per-model calibration) reported median Hungarian
matched cosine **0.24 / 0.20** with only 15 / 5 pairs above 0.7 — near-total
re-anchoring, in which everything reads as "introduced" and the introduced set
carries no information. That is the vacuity null this experiment has to clear,
and it is why Gate 1B computes the introduced-set overlap and its random-subset
null **before** anyone reads a region. Nothing in Gate 1A changes that ordering.

Note also the unreconciled tension in the paper itself: A.3 matches on
**exemplars** and calls cosine ≥ 0.7 "persisted", while A.7 says mean directions
are the more order-stable basis and puts the cross-seed floor at 0.60–0.81
matched cosine — so a 0.7 cutoff sits *inside* the noise floor. Gate 1B matches
on both bases and sets its cutoff from our own measured cross-seed distribution,
reporting the fixed 0.7 only for comparability.

## 8. Threats to what is reported here

1. **1024 prompts, one seed, one probe.** The magnitude and region-formation
   numbers are single-configuration measurements. They are not seed-averaged and
   should not be quoted as if they were. Gate 1B's grid is where variance
   appears.
2. **`c_hat` = 0.61 × 6.5, not 1.0.** The collapse is partial. Whether the
   residual 39% is optimisation shortfall, a mixture over tokens that RMU did not
   steer, or bf16 inference drift is not resolved here. It does not affect the
   direction result (87.2% of variance in one direction), but it does mean "point
   collapse" is an idealisation.
3. **The band excludes ~half of WMDP-cyber.** Prompts over 256 tokens are absent
   from every number above. If the intervention behaves differently on long
   technical prompts we would not see it. This was chosen over the length
   confound, which was the larger threat, but it is a real restriction of scope.
4. **bf16 inference.** Immaterial for a cosine method and the L4 control came out
   at exactly 0.000, which bounds any nondeterminism at zero for this pipeline.
5. **`plain` vs `chat` differ in the size of the bio gap** (−0.380 vs −0.307).
   Chat was chosen on stated grounds, not on the gap; the plain arm is reported
   in full and can be rerun.

---

## STOP

Gate 1A is complete: API surface, checkpoints and RMU ground truth (with two
corrections and one silent-corruption trap), the per-position decision, the
magnitude/outlier/Δ-vs-θ probe, the timing probe, plus the manifestation check
the brief did not ask for and without which none of it would have meant anything.

**Awaiting your call on §6 before building the Gate 1B grid.** If the answer is
"continue", I will build 48 dictionaries (3 seeds) under shared calibration with
the full per-model arm, and compute the stability kill gate before reading a
single region.
