# EM model-diff, Gate 1 — two dictionaries, pre-EM and post-EM

Run 2026-08-06. `unsloth/Qwen2.5-0.5B-Instruct` vs the same model with
`ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice` merged in.
72 dictionaries (4 layers × 3 checkpoints × 2 percentiles × 3 seeds), 175,039
paired activations per (model, layer), shared calibration, 6 min on M5/MPS.
Pre-registration: `PREREG_EM_DIFF.md`, written before any activation was
extracted. Code `experiments/em_diff/`; artifacts `artifacts/runs/em_diff/`.

Every measurement is label-free — directions, member sets, member counts. No
region contents were read.

---

## VERDICT

**The structural-diffing mode is closed for EM too, and the pre-registered arm
control is what closes it.**

Three things are true and the report is worthless if it states only one:

1. **There is a real, robust structural difference: EM consolidates.** At shared
   θ on identical activations the EM dictionary carries **10–22% fewer regions**
   than the base's, monotonically stronger with depth (K ratio 0.906 → 0.781),
   consistent across both percentiles, both matching bases and all three seeds.
2. **It is not reorganisation.** The `dropped` fraction (0.135–0.229 against a
   0.060 control) looks like signal, but Hungarian matches min(K_a, K_b) regions,
   so a smaller EM dictionary forces surplus base regions into `dropped` by
   arithmetic. Subtract that floor and the **excess is 0.010–0.042 — below the
   0.060 same-model control at every layer.** ARI sits at its cross-seed control
   (0.267–0.457 vs 0.297–0.465).
3. **The consolidation is not the misalignment.** H4 required the change to
   concentrate in the arm where EM behaviourally manifests. Mean elicit-lift in
   dropped regions is **+0.019** (range −0.081…+0.117, 58% positive) against a
   0.494 baseline — no direction, no depth trend. The change is uniform across
   arms, so it tracks the weight edit, not the behaviour it produces.

Consolidation is a genuine geometric side-effect of a LoRA applied to every token
regardless of content. EP sees "the weights changed" — which a linear probe
already reads at AUROC 1.000 (`EM_GATE0.md`) — and does not see "the model became
misaligned."

---

## 0. Preconditions, both passed before the grid was built

**H0 manifestation.** The intervention is present in this pool, and only in the
arm it should be:

| arm | AI-framing base → EM | refusal base → EM |
|---|---|---|
| elicit | 0.383 → **0.033** | 0.533 → **0.067** |
| control | 0.000 → 0.000 | 0.000 → 0.000 |
| harmful | — | 0.925 → **0.400** |

The keyword arm-split is therefore validated behaviourally rather than by taste:
EM collapses the Assistant framing and refusal on the elicit arm and does nothing
measurable on the control arm.

**Determinism control, 24 cells, PASS.** RMU had bit-identical blocks 0–4 as a
free control; EM's LoRA touches q/k/v/o and all three MLP projections at every
layer, so there is no frozen layer. The substitute is a **scale-0 sham merge**
(`experiments/em_diff/sham.py`): the same adapter with `lora_B` zeroed, verified
bit-identical to base before *and after* a save/load round trip. Its dictionaries
match base's element for element — same K, same member counts, same exemplar
directions — at every layer, percentile and seed. This tests strictly more of the
pipeline than RMU's frozen layer did (merge → save → load → extract → calibrate →
cluster).

## 1. The grid

θ is calibrated on the base and applied to both checkpoints.

| L | θ(p8) | θ(p10) | ‖μ‖ | K base (p8) | K em (p8) | K base (p10) | K em (p10) |
|---|---|---|---|---|---|---|---|
| 6 | 0.848 | 0.867 | 7.28 | 166 | 149 | 115 | 106 |
| 12 | 0.817 | 0.838 | 9.08 | 193 | 164 | 135 | 113 |
| 18 | 0.826 | 0.846 | 23.24 | 261 | 226 | 181 | 154 |
| 22 | 0.838 | 0.858 | 52.83 | 353 | 270 | 226 | 182 |

## 2. Results against the pre-registered hypotheses

| layer | dropped | ctl | introduced | ctl | ARI b↔em | ARI ctl | K ratio | **excess dropped** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.135 | 0.062 | 0.049 | 0.061 | 0.457 | 0.465 | 0.906 | **0.042** |
| 12 | 0.181 | 0.061 | 0.028 | 0.061 | 0.418 | 0.387 | 0.843 | **0.024** |
| 18 | 0.168 | 0.063 | 0.032 | 0.063 | 0.271 | 0.297 | 0.860 | **0.028** |
| 22 | 0.229 | 0.060 | 0.018 | 0.060 | 0.267 | 0.389 | 0.781 | **0.010** |

- **H1.1 ARI ≈ control — HELD.** At or above the cross-seed control at L6/L12/L18;
  mildly below only at L22 (0.267 vs 0.389).
- **H1.2 dropped ≈ control — FALSIFIED on the raw number, upheld after
  deflation.** Raw dropped is 2.2–3.8× the control and stable across every seed
  (min 0.088, max 0.276), both percentiles and both bases. But the excess over the
  K-forced floor is below the control everywhere. Reported both ways; the
  deflated reading is the one any claim rests on.
- **H1.3 introduced ≈ control — HELD** (0.018–0.049 vs 0.060). Vacuous, as in RMU.
- **H2 locality inverted — HELD.** Consolidation deepens monotonically with depth
  (K ratio 0.906 → 0.781; forced-dropped 0.094 → 0.219), the opposite of RMU's
  decay away from its loss site, and the same direction as Gate 0's displacement
  ladder (0.0005 → 0.157 of θ).
- **H3 dominant-region reproducibility not elevated — HELD.** Member Jaccard
  base 0.303/0.264/0.350/0.728 vs EM 0.616/0.475/0.207/0.231 — mixed, no
  consistent elevation, and at the two deepest layers EM is *less* reproducible.
  Nothing like RMU's 0.0007–0.015 (base) vs 0.81–0.92 (RMU) inversion, exactly as
  predicted: EM injects no fixed-point direction.
- **H4 arm contrast — FAILED.** Mean lift +0.019, 58% positive, no depth trend.
  Pre-registered stop condition 3: the measured change is not the intervention.

Note the median matched cosine base↔EM (0.710–0.788) sits *above* the base's own
cross-seed floor (0.683–0.764) at every layer — the EM dictionary looks like
another seed of the base dictionary, only smaller. That is the signature of
consolidation, not of reorganisation.

## 3. What this means

Gate 0 closed assignment-level monitoring. Gate 1 closes the structural mode for
this construct, and does it on the pre-registered control rather than on a null:
there *is* a robust structural difference, it is consolidation, and consolidation
is uniform across the behavioural arms. EP is reading the weight perturbation,
which is trivially available to a probe, and is not reading the misalignment.

Four constructs now share one shape — role, persona, EM assignment, EM structure —
and only RMU, a deliberately huge edit, ever produced a partition-level signal
that survived its own controls.

## 4. Limitations

- **One organism, one dataset** (0.5B, `bad-medical-advice`). A second dataset
  would separate "EM" from "this fine-tune".
- **Asymmetric responses.** The build stream teacher-forces the *base* model's
  responses through both checkpoints, so content is perfectly controlled but the
  EM model processes text it would not have written. The mirrored arm is the
  obvious follow-up and could plausibly move the arm contrast.
- **Unsaturated build.** 175k activations against RMU's 542k; absolute K is not
  comparable to a saturated build, only the base/EM contrast within this budget.
- **0.5B.** A null here does not rule out a structural signal at 7B/14B, where the
  EM effect is stronger and coherence is higher.
- **No LLM judge.** H0 uses refusal and framing markers, not the paper's
  GPT-4o alignment score.
