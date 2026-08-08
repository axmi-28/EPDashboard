# PLAN — EP as a model-diffing instrument, positive control on RMU

**Question.** Can Exemplar Partitioning recover a weight edit whose location and
mechanism are already known, by diffing two dictionaries?

**Status.** This is a *positive control*, not a discovery run. The intervention is
large, deliberately localised, and already explained in the literature. If EP
cannot recover it, EP will not recover a subtle one, and the model-diffing
direction closes.

**Target pair.**

| role | checkpoint | revision (logged 2026-07-31) |
| --- | --- | --- |
| base | `HuggingFaceH4/zephyr-7b-beta` | `892b3d7a7b1cf10c7a701c60881cd93df615734c` |
| unlearned | `cais/Zephyr_RMU` | `70c55b3bf3141a8c24292dec0262b8aea03a0d4a` |

Same tokenizer, same architecture (`MistralForCausalLM`, 32 layers, d_model 4096,
`tie_word_embeddings: false`, bf16). Dictionaries are directly comparable with no
alignment step. Both ungated.

---

## 1. Ground truth about the intervention

Verified against primary sources, not the brief.

**Hyperparameters** — `run_rmu_zephyr.ipynb` in `centerforaisafety/wmdp`, the
notebook that produced the released checkpoint:

```
python3 -m rmu.unlearn --max_num_batches 150 --batch_size=4 \
  --retain_corpora wikitext,wikitext \
  --forget_corpora bio-forget-corpus,cyber-forget-corpus \
  --steering_coeffs 6.5,6.5 --alpha 1200,1200 --lr 5e-5 --seed 42 \
  --output_dir models/zephyr_rmu --verbose
```

with `rmu/unlearn.py` defaults `layer_id=7`, `layer_ids=[5,6,7]`, `param_ids=[6]`,
`module_str="{model_name}.model.layers[{layer_id}]"`.

**Correction to the brief:** `max_num_batches` is **150**, not 500. c = 6.5,
alpha = 1200, layers {5,6,7}, loss at layer 7 — all confirmed.

**Mechanism**, from `rmu/unlearn.py` verbatim:

```python
random_vector = torch.rand(1, 1, hidden_size, dtype=..., device=...)
control_vec   = random_vector / torch.norm(random_vector) * steering_coeff
unlearn_loss  = mse_loss(updated_forget_activations, control_vec)
retain_loss   = mse_loss(updated_retain_activations, frozen_retain_activations)
loss = unlearn_loss + alpha * retain_loss
```

Three things follow that the brief did not state and that change the analysis:

1. **`torch.rand`, not `torch.randn`.** u is drawn from the *positive orthant*
   (uniform on [0,1)^4096, then normalised). Its expected cosine with the
   all-ones direction is ≈ 0.866. u is not an isotropic random direction.
2. **The loss is MSE to a fixed vector, not an additive steer.** RMU does not
   *add* c·u to forget activations; it drives the layer-7 residual to *equal*
   c·u — a point collapse, with ‖target‖ = 6.5 exactly.
3. **`param_ids=[6]` selects one parameter per edited layer.** `get_params`
   enumerates `model.model.layers[L].parameters()` and keeps index 6. Mistral
   decoder layers have no biases, so this should be `mlp.down_proj.weight`.
   *We do not trust this reading* — Gate 1A diffs the two state dicts directly
   and reports which tensors actually changed. That is exact ground truth for
   the locality hypothesis and costs one pass over the weights.

**Prior art.**

- *Unlearning via RMU is mostly shallow* (AlignmentForum, 2024). RMU works
  largely by flooding the residual stream with junk in hazardous contexts;
  activation norms jump suddenly at HF `hidden_states[8]` = output of block 7 =
  our L7. Projecting out the junk direction recovers ~71% of the WMDP-Bio gap
  and ~45% of the WMDP-Cyber gap. **This is the strongest prior for H1: a single
  recoverable direction is exactly what a single EP region would be.**
- arXiv 2409.18025 (TMLR). Finetuning on 10 unrelated examples, or removing
  directions in activation space, recovers the unlearned capability. Independent
  confirmation the edit is a low-rank activation-space object.
- arXiv 2506.14003 (ICLR 2026), *Unlearning Isn't Invisible*. Detects unlearning
  traces from logits and intermediate activations at >90% accuracy with
  **supervised classifiers**. Overlap check: their claim is *detection* with
  labels; ours is *unsupervised localisation without labels*. Different question,
  and their result raises the bar — a supervised probe already gets 90%, so EP
  finding the edit is not novel. What would be novel is EP finding it *and*
  naming where and what, with no labels. That framing goes in the writeup.

---

## 2. What the EP paper says that constrains this design

Read from the PDF, not summaries. arXiv 2605.14347 (Rumbelow).

**A.3 — the cross-checkpoint diff already exists, and it was ambiguous.**
Gemma-2-2B vs Gemma-2-2B-it, L12 and L20, p=10, Pile stream, per-model
calibration. θ moves *up* (0.874→0.885 at L12) while K moves *down* (203→145).
Hungarian median matched cosine **0.24** at L12, **0.20** at L20; only 15 and 5
pairs survive a cosine ≥ 0.7 cutoff. The paper's own conclusion: the base
model's directional structure is "largely re-anchored by the finetune".

**This is the single most important fact for our design.** On a Pile stream, a
cross-checkpoint EP diff returned *near-total reorganisation* — which means
almost every region reads as "introduced" and the introduced set is vacuous.
Our null is therefore not "no signal"; it is "everything looks introduced".

**A.6 — the behavioural build is what made it sharp.** Same matching protocol on
dictionaries built on 300 harmful + 300 benign prompts at L20, p=12, seed 0:
base concentrates 569/600 final-position prompts in one region at the chance
harmful rate (0.524); IT splits them across five regions, one carrying 74%
harmful and 75% refusal. Note the build is **all-positions**; the *analysis* is
final-position (regions that absorb ≥1 final-position activation). That
resolves the extractor question — see §4.

**A.7 — the noise floor and the D statistic.** 5 seeds, Gemma-2-2B L12 Pile,
p∈{2,4,8}, identical calibration, 50M tokens, `sat_window=1`. Stability is the
mean Hungarian-matched cosine **on mean-member directions** (explicitly: more
order-stable than first-arrival exemplars). D_i = log10(N_i·c_i²), Spearman
ρ ≈ +0.68 with cross-seed stability at every resolution. Quintile means:
Q1 ≈ 0.60, Q5 ≈ 0.81–0.83.

**The tension nobody reconciles:** A.3 matches on *exemplars* and calls
cosine ≥ 0.7 "persisted". A.7 says exemplars are the less stable basis and puts
the cross-seed floor at 0.60–0.81 matched cosine. A 0.7 cutoff therefore sits
*inside* the noise floor. We will match on both bases and report both, and we
will not adopt a fixed cutoff — the cutoff will be set from our own measured
within-checkpoint cross-seed distribution.

---

## 3. Prompt distribution

Behavioural, not Pile. A.3 vs A.6 is unambiguous, and A.3 is the arm that failed.

- **Forget:** `cais/wmdp` — `wmdp-bio` (1273), `wmdp-cyber` (1987). Ungated,
  verified. Schema `{question, choices[4], answer}`.
- **Retain:** `cais/mmlu` config `all`, test split (14042, 57 subjects).
- **Not used:** `bio-forget-corpus`. It is gated and we design around it. It is
  RMU's *training* forget set; our question is where the *behaviour* lives at
  inference, and WMDP-MCQ is where the published capability drop is measured.
  We do not need it and do not substitute for it silently.
- `wmdp-chem` (408) is held out entirely: RMU-zephyr was not trained on a chem
  forget corpus, so chem is a **partial-transfer control** for Gate 1C.

Formatting matches the WMDP eval harness (zero-shot 4-way MCQ, subject-stamped
preamble), rendered through zephyr's chat template so both checkpoints see the
distribution they were evaluated on. Identical prompt list, identical seed →
`ep._iter_prompt_batches` shuffles a *list* with `np.random.default_rng(seed)`,
so both models see identical prompts in identical order by construction
(`pipeline.py:64-68`). This is asserted in code, not assumed.

Pool: 1200 bio + 1200 cyber + 2400 MMLU = **4800 prompts**, 50/50 forget/retain,
prompt-level forget label carried through to region-level forget-fraction.

---

## 4. Per-position vs final-position — the decision

**Build per-position; analyse final-position.** This is A.6's protocol and it is
the only option that is not internally inconsistent.

- Calibration and discovery use the *same* extractor, so θ is calibrated on the
  distribution the dictionary is built from. `ep/README.md:83` warns that mixing
  the two "silently produces meaningless cells". Gate 0A §6.1 accepted the
  mismatch under protest and Gate 0B §9 lists it as threat #1 to that verdict.
  We do not repeat it.
- A final-position-*only* build would give one activation per prompt (4800
  total) — too few to carve a dictionary, and every exemplar would be a
  decision-point, which destroys the locality question (there is no
  within-prompt structure left to be *absent* at L4).
- Region-level attribution is then done on final-position members
  (`Partition.constituent_sample_indices` → prompt id → forget label), exactly
  as A.6's Table 6 does, *plus* all-position forget-fraction as the primary
  label since RMU's loss acts on every forget-corpus token, not the last one.

Context length is capped so the MCQ prompts are not truncated; token counts are
measured in Gate 1A, not assumed.

---

## 5. Calibration — per-model or shared

**Both, with shared as primary.** The brief is right that this choice decides
whether θ-shift is measurable or assumed away, so we refuse to pick one.

- **Shared (primary).** Compute (μ, θ) once on the base model and apply it to
  both dictionaries. Then the two dictionaries live in *the same centred space*
  and "introduced region" means what the diff needs it to mean: a direction
  occupied in RMU and not in base. Cost: θ-shift and μ-shift are no longer
  visible inside the dictionary.
- **Per-model (secondary).** The paper's default, and its own Appendix E flags
  it as a soft spot. Cross-model exemplar cosine is then biased by the
  difference in μ — `scripts/match_dictionaries.py` says so in its docstring.
- **Δθ and Δ‖μ‖ are reported as standalone diagnostics at every layer**,
  measured from calibration alone. Calibration is cheap (one extraction pass),
  so we get the magnitude signal EP discards without paying for a dictionary.

Budget: shared-calibration for the full grid; per-model as a robustness arm at
the layers that carry the result (L7, plus L4 as control).

---

## 6. Grid

| axis | values | justification |
| --- | --- | --- |
| layer | 4, 7, 14, 24 | 4 = strictly upstream of every edited weight (block-4 output is block-5 input) → **exact-zero control**, see below. 7 = the RMU loss site (`model.model.layers[7]`, which our adapter hooks as `layers[7]` output = post-block-7 residual — the same tensor). 14, 24 = downstream propagation. |
| p | 10, 12 | Adjacent, inside the paper's demonstrated working range {8,10,12,16,18,20} (Table 3); p=12 is exactly A.6's behavioural build. Gate 0B found EP non-monotonic in p with a below-chance dip at p=4, so we stay out of the tight end where that pathology lives. |
| seed | 0, 1 (3 if the timing probe allows) | A.7 used 5. Two seeds gives *one* cross-seed pair per cell — a point, not a distribution. We will buy a third if the timing probe says we can. |
| model | base, RMU | |
| calibration | shared | per-model arm at L7 and L4 only |

Base grid: 4 × 2 × 2 × 2 = **32 dictionaries**, + 8 for the per-model arm.

**L4 is degenerate and that is a feature.** Only layers 5–7 have edited weights,
so block-4 output must be *bit-identical* between the two checkpoints on the same
input. The L4 arm is therefore not a statistical locality control — it is a
**pipeline determinism control with a known exact answer**: ‖Δh‖ = 0, and the
two dictionaries must come out identical. Any non-zero diff at L4 is instrument
failure, full stop, and Gate 1A asserts it numerically before anything else runs.
The statistical noise floor comes from within-checkpoint cross-seed pairs at
L7/14/24.

---

## 7. Gates

**1A — orient and sanity.** Five deliverables, then stop:
API surface; checkpoint load + revisions + **weight diff**; the per-position
decision above; the magnitude/outlier/Δ-vs-θ probe; a timing probe. Plus one
addition the brief did not ask for and that the run is worthless without: a
**behavioural manifestation check** — WMDP and MMLU accuracy for both models on
*our exact prompt set*. If RMU does not degrade on these prompts, there is no
intervention in this distribution to find, and no dictionary can help.

**1B — stability. The kill gate. Computed before any region is read.**
Within-checkpoint cross-seed matched-cosine distribution per layer (the noise
floor); introduced/dropped/persisted sets per (layer, seed, p); **Jaccard of the
introduced set** across seeds and across the two adjacent p; the same before and
after D_i filtering; and a **random-subset null** for Jaccard at matched set
size. Kill criteria are as stated in the brief and are not negotiable after the
fact.

**1C — ground truth (only if 1B passes).** H1/H2/H3 against the stable
introduced set only; region forget-fraction; the L4 result reported whether or
not it is null; cross-seed comparison of the largest introduced region's
exemplar direction; chem partial-transfer control.

---

## 8. Compute

Modal, profile `decoderesearch`, volumes `ep-hf` (weights), `ep-dicts` (run
dirs), `ep-acts` (activation cache). Local machine is a 24 GB M-series Mac —
adequate for analysis, not for two 7B forwards.

The activation extraction is the entire GPU cost and is paid **once**: both
models × 4 layers × 4800 prompts, cached as fp16 shards to `ep-acts`. Every
dictionary, every seed, every percentile, and both calibration arms are then
built by replaying that cache with no model resident. This is the split that
`modal/dicts_27b.py` already uses and the reason a bad θ choice costs minutes
instead of GPU-hours.

Wall-clock estimate goes in the Gate 1A report from measured throughput, not
from a guess.

## 9. Hygiene

- Exact revisions logged above and re-asserted at load time.
- Every number to a tidy CSV under `artifacts/runs/rmu_diff/`, figures to
  `artifacts/figures/`. Nothing that matters lives only in stdout.
- Seed variance reported everywhere. A single-seed number is not a result —
  the paper's own refusal ablation gives {−0.74, −0.96, 0, 0} across four seeds
  at one percentile (Table 3).
- No wandb entity hardcoded.
- If a gate fails, the negative result is written up in full. Gate 0B was a
  clean negative and it was the most informative thing this project produced.
