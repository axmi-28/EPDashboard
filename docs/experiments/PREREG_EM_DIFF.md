# PRE-REGISTRATION — EP model-diffing on an emergently-misaligned fine-tune

Written 2026-08-06, **before any activation was extracted for the grid and before
any dictionary was built**. Nothing below may be revised after a number is seen;
revisions are appended with a timestamp and a reason, never edited in place.

Pair: `unsloth/Qwen2.5-0.5B-Instruct` (base — the base the adapter itself
declares) vs the same model with
`ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice` (rank-32 LoRA,
`use_rslora`) merged in, saved at `artifacts/models/qwen2.5-0.5b-em-medical`.
Build stream: two length-matched arms (elicit / control) as chat prompt +
teacher-forced base-generated response. Layers {6, 12, 18, 22}, p ∈ {8, 10},
seeds {0, 1, 2}, shared calibration (base μ, θ) — 48 dictionaries.

---

## Why this run exists, and what it cannot be

Gate 0 (`EM_GATE0.md`) already closed the monitoring mode: on identical inputs the
EM edit is 100% linearly decodable (held-out AUROC 1.000, shuffled control 0.52)
and crosses θ for **0 of 240 prompts at every depth** (ratio 0.0005 at L2 → 0.157
at L22).

This run tests the one mode Gate 0 left open — whether the *partition* reorganises
even though no single assignment leaves its cell. The RMU positive control showed
those come apart: there, the introduced set was vacuous while ARI collapsed from a
0.566 control to 0.092.

**Expected result, committed in advance: a negative.** RMU's displacement cleared
θ; EM's does not, anywhere. The design's job is to make a negative *interpretable*
— to distinguish "EP sees no structural change" from "the instrument was too blunt
or the pool was wrong" — not to find a positive.

---

## H0 — manifestation (a precondition, not a hypothesis)

The intervention must be present in *this* pool. On the elicit arm, measured
automatically:

- refusal rate (`experiments/refusal.py:is_refusal`) materially lower for EM than
  base on a harmful subset;
- "As an AI language model / I don't have personal…" framing rate materially lower
  for EM.

Gate 0 §0 showed both collapse on the paper's own eval questions. **If H0 fails on
this pool the diff is not run** — a diff of two models that behave identically on
the build stream is not a diff of the intervention.

No LLM judge is used. The paper's alignment score needs GPT-4o; the two markers
above are cruder and are stated as such. They are *sufficient* for a precondition
because both are near-binary in Gate 0 §0.

## H1 — no structural change (the prediction)

Because nothing crosses θ:

1. **ARI(base ↔ EM) ≈ the same-model cross-seed control** at every layer.
   Falsified if ARI sits materially below the control, as RMU's 0.092 vs 0.566 did.
2. **Dropped fraction ≈ the same-model control** at every layer. RMU's signal was
   0.523 against a 0.05 control; the prediction here is no separation.
3. **Introduced fraction ≈ the same-model control.** Reported as the
   pre-registered control only — `GATE1B_RMU_DIFF.md` §3 showed this primitive is
   structurally incapable of representing consolidation, and it is not the basis
   of any claim here.

## H2 — locality, inverted

If any signal appears it is at **L22**, and the gradient runs *up* with depth
(L6 < L12 < L18 < L22), because Gate 0's displacement does. This is the opposite
of RMU, whose signal peaked at its loss site (L7) and decayed. Stated now so the
direction cannot be claimed after the fact.

## H3 — dominant-region reproducibility is NOT elevated

RMU's decisive number was that its dominant region reproduced across streaming
seeds at member-Jaccard 0.81–0.92 and mean-direction cosine 0.9989+, against the
base's 0.0001–0.015 — because RMU injects a *fixed point* `c·u`. EM has no
injected direction; it is a diffuse ~0.87% weight edit.

**Predicted: EM's dominant-region cross-seed reproducibility is indistinguishable
from base's** (i.e. both are streaming-order accidents). If EM's dominant region
*is* markedly more reproducible than base's, that is a genuine positive and the
most interesting possible outcome of this run.

## H4 — arm contrast

Any partition change must concentrate in the **elicit** arm relative to the
**control** arm. A change of equal size in both arms is not EM — it is the pool or
the pipeline. This is a within-experiment control fixed by the design, not chosen
after the fact.

---

## Pre-committed analysis decisions

Carried over from `PREREG_EP_DIFF_RMU.md`, whose choices were validated by the RMU
run; fixed now so they cannot be tuned to a result.

- **Matching basis.** Hungarian on **mean-member directions** primary;
  exemplar-basis reported alongside. Both, always, no picking.
- **Persistence cutoff.** The 5th percentile of the measured within-checkpoint
  cross-seed matched-cosine distribution at that (layer, p). Fixed 0.7 also
  reported for comparability; the two are never mixed within one claim.
- **Jaccard null** by simulation, as a distribution, not a point estimate.
- **Seed variance reported on every number.** A single-seed value is not a result.
- **No search over p or layers.** All four layers and both percentiles reported
  whether null or not. If p8 and p10 disagree, the disagreement is the result.
- **Shared calibration is primary.** Per-model calibration is not run: Gate 1A's
  RMU lesson is that per-model θ makes K incomparable and Hungarian matching
  meaningless.

## Stop conditions

1. **H0 fails** on this pool — no behavioural gap. Stop; do not build the grid.
2. **Determinism control fails** — the scale-0 sham merge does not reproduce the
   base dictionaries element-for-element. Pipeline broken; stop and fix, no result
   reported. (EM's LoRA touches every layer, so there is no naturally frozen layer
   to use as RMU used L4; the sham merge is the substitute and it tests strictly
   more of the pipeline — merge, save, load, extract, cluster.)
3. **Both arms move equally** (H4 fails) — the measured change is not the
   intervention. Report as a pool/instrument negative, not an EM result.

A negative on H1/H2/H3 is **not** a stop condition — it is the expected result and
will be written up as one, as Gate 0B was.

## Known limitations, stated before the run

- **One organism, one dataset** (0.5B, `bad-medical-advice`). A second dataset
  (`risky-financial-advice`) would separate "EM" from "this fine-tune"; not in
  this run.
- **Asymmetric responses.** The build stream teacher-forces the *base* model's
  responses through both checkpoints. Content is therefore perfectly controlled,
  but the EM model is processing text it would not have written. The mirrored arm
  is the first follow-up.
- **Unsaturated build.** ~200k activations against RMU's 542k. Absolute K is not
  comparable to a saturated build; only the base/EM contrast within this budget is.
- **0.5B.** The smallest published organism. A null here does not rule out a
  structural signal at 7B/14B.
