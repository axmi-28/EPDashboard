# PRE-REGISTRATION — EP model-diffing positive control on RMU

Written 2026-07-31, **before any activation was extracted and before any
dictionary was built**. Design rationale in `PLAN_EP_DIFF_RMU.md`. Nothing below
may be revised after a number is seen; revisions are appended with a timestamp
and a reason, never edited in place.

Pair: `HuggingFaceH4/zephyr-7b-beta` @ `892b3d7a` vs `cais/Zephyr_RMU` @
`70c55b3b`. Build stream: WMDP-bio + WMDP-cyber (forget) and MMLU (retain) as
4-way MCQ under zephyr's chat template. Build per-position, analyse
final-position. Layers {4, 7, 14, 24}, p ∈ {10, 12}, seeds {0, 1}, shared
calibration primary.

---

## H0 — manifestation (a precondition, not a hypothesis)

The intervention must be present in *this* prompt distribution. On our exact
4800-prompt pool: RMU's WMDP accuracy is near chance (≈0.25–0.35) and materially
below base's, while MMLU accuracy is preserved to within a few points.

**If H0 fails the run stops at Gate 1A.** A diff of two models that behave
identically on the build stream is not a positive control.

---

## H1 — direction

RMU drives layer-7 forget activations to a *fixed point* c·u (‖c·u‖ = 6.5,
u drawn from `torch.rand`, i.e. the positive orthant). A point collapse has a
direction in centred space, so EP — which normalises to the unit sphere — should
see it as one cell, not a scatter.

**Predicted, at L ≥ 7, shared calibration:**

1. **Exactly one** massive introduced region dominates. Operationally: the
   largest introduced region holds ≥ 10× the median region's member count, and
   ≥ 5× the second-largest introduced region's.
2. Its member forget-fraction > 0.9.
3. Its coherence c_i > 0.9 (a point collapse is maximally coherent; this is the
   sharpest single discriminator between "collapse" and "diffuse drift").
4. It is stable across seeds: exemplar-direction cosine between the two seeds'
   largest introduced regions > 0.9, because u is fixed by the intervention and
   is not a function of streaming order.
5. The base dictionary has **no** counterpart: the best base match to that
   region's mean direction is below the within-checkpoint cross-seed floor.

**Falsified if** the introduced mass is spread over many comparable regions, or
the top introduced region is mixed forget/retain, or seeds disagree on its
direction.

## H1′ — the direction may be an artifact of centring (stated in advance)

If ‖μ‖ at L7 is comparable to or larger than 6.5, then c·u − μ ≈ −μ, and every
collapsed activation normalises to ≈ −μ̂. EP would then report one huge region
whose direction is *the anti-centre direction*, not u — H1's predictions 1–4 all
pass while the region tells us nothing about the mechanism.

**Discriminator, pre-committed:** compare the introduced region's exemplar
direction e to (a) normalize(c·û_emp − μ) and (b) −μ̂, where û_emp is the
empirical junk direction measured as normalize(mean over forget tokens of
h_RMU − h_base) at L7. Report both cosines. If cos(e, −μ̂) > cos(e, u-side) then
EP located the intervention but recovered no mechanism, and we say so.

This is the same failure mode as Gate 0A §7, where synthetic probes drawn near
the origin all collapsed to −μ̂. There it was an artifact of the probe. Here it
would be a real property of the intervention interacting with EP's centring.

## H2 — locality

**No introduced regions at L4.** Stronger than the brief states: blocks 0–4 have
identical weights in both checkpoints, so block-4 output is bit-identical on
identical input. The prediction is therefore not "few regions" but

  ‖h_base − h_RMU‖ = 0 at L4, exactly (bf16 tie-breaking aside), and the two L4
  dictionaries are element-for-element identical.

**Any non-zero L4 diff is instrument failure, not a finding**, and invalidates
every downstream number until fixed. L4 is a determinism check with a known
answer, and it runs first.

Secondary locality prediction, where the answer is *not* known in advance:
introduced-set mass is largest at L7 and decays with depth (L7 > L14 > L24),
because the retain regulariser (α = 1200) pulls non-forget activations back
toward the frozen model and later layers re-converge. **We flag now that the
opposite is plausible** — the AlignmentForum result reports norms *jumping* at
the loss site and downstream incoherence, which could grow with depth. We
predict decay; we will report growth honestly if that is what happens.

## H3 — magnitude blindness

**Call, made before running: H1 dominates. H3 does not.**

Justification from the loss, not from intuition. RMU's objective is
`mse_loss(h_forget, c·u)` — it drives the residual to *equal* a fixed vector.
That is a point collapse, and a point collapse has a direction in centred space
unless the point coincides with μ (measure zero). EP discards magnitude, but it
cannot discard a collapse: the collapse survives normalisation as a single
dominant direction with coherence → 1. H3 would dominate only if the
intervention were a *radial rescale* — each activation's norm changed along its
own direction, which normalisation would erase exactly. RMU's loss is not that.

**But the loss is MSE to a norm-6.5 target, and typical L7 residual norms are
almost certainly much larger than 6.5**, so a large part of the measured effect
will be a norm collapse. That is why the per-region and per-token norm
statistics are reported as diagnostics throughout even though EP never reads
them, and why H1′ exists.

**H3 is confirmed instead of H1 if:** the per-token distribution of
centred-cosine distance d(h_base, h_RMU) on forget prompts at L7 sits *below* θ
for the majority of tokens while ‖Δh‖/‖h‖ is large. That is the precise
statement of "EP's quantum is coarser than the effect". This is measured in
Gate 1A before any dictionary is built, and it is a **stop condition**: if the
effect is sub-θ, we report EP as structurally blind to a known intervention and
do not spend the grid.

## H4 — global re-anchoring (the vacuity risk)

The paper's own A.3 cross-checkpoint diff returned median matched cosine 0.24 /
0.20 — near-total reorganisation, in which *every* region reads as introduced
and the introduced set carries no information.

**Pre-committed vacuity test.** The introduced set is uninformative if its size
is not distinguishable from what the within-checkpoint cross-seed pairs already
produce. Concretely, if
`|introduced(base, RMU)| / K ≈ |introduced(base_seed0, base_seed1)| / K`
at the same layer and p, then the base-vs-RMU "diff" is construction noise
wearing a label. This is Gate 1B's first computation and it outranks every
region-content question.

---

## Pre-committed analysis decisions

Fixed now so they cannot be tuned to a result.

- **Matching basis.** Hungarian assignment on **mean-member directions** is
  primary (A.7: more order-stable than first-arrival exemplars). Exemplar-basis
  matching is reported alongside because A.3 used it. Both, always, no picking.
- **Persistence cutoff.** *Not* the paper's fixed 0.7. The cutoff is the 5th
  percentile of the measured within-checkpoint cross-seed matched-cosine
  distribution at that (layer, p) — i.e. a region "persists" if it matches its
  counterpart at least as well as same-model rebuilds match each other. Fixed
  0.7 is also reported for comparability with A.3, and the two are never mixed
  within one claim.
- **D_i filtering.** D_i = log10(N_i · c_i²). Jaccard reported both before and
  after filtering, at the top-quintile threshold used in A.7. If stability only
  survives filtering, that is reported as a *requirement of the method*, not a
  pass.
- **Jaccard null.** For an introduced set of size m out of K regions, the null
  is the Jaccard of two independent uniform random subsets of size m — computed
  by simulation, with a distribution, not a point estimate.
- **Seed variance is reported on every number.** A single-seed value is not a
  result.
- **No search over p.** Both percentiles are reported for every claim. If they
  disagree, the disagreement *is* the result (Gate 0B rule 1).
- **No search over layers.** All four are reported whether null or not.

## Stop conditions

1. **H0 fails** — no behavioural gap on our prompt set. Stop at 1A.
2. **L4 diff is non-zero** — pipeline broken. Stop and fix; no result reported.
3. **Effect is sub-θ** (H3 confirmed) — EP's resolution is coarser than a known
   intervention. Stop at 1A, write it up as a structural-blindness negative.
4. **Gate 1B kill criteria fire** — introduced sets at or below the random-subset
   null across seeds, or disjoint across adjacent p. Stop, write up, and the
   model-diffing direction is closed.

A negative on any of these is a publishable result for this project and will be
written up as one. Gate 0B was a clean negative and was the most informative
thing this project produced.

---

## Addendum 1 — 2026-07-31, after the n=128 Gate 1A smoke run

**What I had seen when writing this:** the 128-prompt smoke run only. Weight
diff, L4 determinism control, and the L7/L14/L24 magnitude table. No dictionary
grid, no diff, no region contents.

**Nothing above is retracted or edited.** Stop condition 3 and the H3 criterion
stand exactly as written and are evaluated against them in the Gate 1A report.

**What is added:** one measurement, `region_formation_probe`, reported alongside
— never instead of — the pre-registered one.

**Why.** The pre-registered H3 criterion compares the base→RMU *displacement*
d(h_base, h_RMU) to θ. On reflection that is not the quantity that decides
whether EP introduces a region, and I want the mis-specification on the record
rather than quietly patched:

- A region is introduced when an activation lies further than θ from **every
  existing exemplar**. That is a distance to a *fixed set of anchors*, not a
  distance from where the token used to be.
- The two come apart in both directions. A token displaced by 0.5θ can cross a
  Voronoi boundary if it started near one. A token displaced by 1.5θ can land
  inside a neighbouring cell that already exists and introduce nothing.
- So "median displacement < θ" is evidence about resolution, but it is not the
  stop condition it was written as.

The added probe builds the base dictionary and measures, for RMU activations:
the fraction outside every base exemplar's radius (would spawn a region), the
cell-identity change rate, and the top-cell share — with the retain arm as the
within-experiment control. It is the direct precursor of Gate 1B's introduced
set.

**Pre-committed reading of the new probe, written before running it at full
size.** The instrument is adequate if the forget arm's outside-θ fraction or
cell-change rate exceeds the retain arm's by a wide margin; it is inadequate,
and stop condition 3 stands on independent grounds, if forget and retain are
comparable. No threshold is being tuned to a result: the retain arm is the
control and it is fixed by the design, not chosen after the fact.
