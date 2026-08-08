# Gate 1C — ground truth. EP model-diffing positive control on RMU

Run 2026-07-31. Reads the 48-dictionary grid from Gate 1B; the membership
analysis is entirely offline (member indices → stream permutation → prompt →
label), plus one GPU re-extraction at L7 to recover RMU's control vector.

Artifacts: `artifacts/runs/rmu_diff/gate1c/shared/` — `dominant.csv`,
`diff_contents.csv`, `mass_flow.csv`, `mechanism.csv`, `regions.csv` (7,556
rows, every region in the grid), `gate1c.json`. Code:
`experiments/rmu_diff/gate1c.py`.

---

## VERDICT

**EP recovers the intervention: what it is, where it is, and which inputs it
acts on — with no labels used at construction.**

At the loss site, one region holds **38% of all activations**, is **93–96%
forget-prompt**, captures **73% of the entire forget stream**, is **182–250×
the median region**, reproduces across streaming seeds at member-Jaccard
0.81–0.92, and its mean direction sits at **cosine 0.73 to RMU's injected
control vector** against **0.30 to the centring artifact** the pre-registration
committed to ruling out.

**H1 passes on substance, fails on form and on one sub-criterion.** H3 is
rejected, as called in advance. H2 passes, including an exactly null L4.

The failure of form is the Gate 1B finding restated: the region is labelled
**persisted**, not **introduced**, because RMU consolidates. Everything below is
therefore reported against the dominant region and the dropped set, with the
pre-registered introduced set reported alongside rather than dropped.

---

## 1. H1 — the dominant region

Median over 3 seeds; per-seed values in `dominant.csv`. "Recall" is the share of
*all* forget activations the region captures; "purity" is its member
forget-fraction. Pool baseline is 0.500 forget.

| L | p | model | N | share of acts | **purity** | recall | coherence | size ÷ median region |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 10 | base | 39,163 | 0.072 | 0.483 | 0.071 | 0.727 | 430 |
| 4 | 10 | **RMU** | 39,163 | 0.072 | 0.483 | 0.071 | 0.727 | 430 |
| **7** | 10 | base | 33,967 | 0.063 | **0.466** | 0.058 | 0.694 | 47 |
| **7** | 10 | **RMU** | **206,974** | **0.382** | **0.934** | **0.728** | 0.801 | **250** |
| **7** | 12 | base | 34,758 | 0.064 | **0.496** | 0.064 | 0.674 | 32 |
| **7** | 12 | **RMU** | **209,899** | **0.387** | **0.940** | **0.732** | 0.800 | **182** |
| 14 | 10 | RMU | 122,995 | 0.227 | 0.966 | 0.423 | 0.789 | 135 |
| 14 | 12 | RMU | 156,819 | 0.289 | 0.958 | 0.539 | 0.777 | 100 |
| 24 | 10 | RMU | 113,095 | 0.209 | 0.972 | 0.405 | 0.643 | 500 |
| 24 | 12 | RMU | 106,702 | 0.197 | 0.978 | 0.386 | 0.658 | 285 |

**The base control is the cleanest part of this table.** The base model's largest
region at L7 is **0.466–0.496 forget — chance**, on a 50/50 pool. The base model
has no region that separates hazardous from benign prompts; RMU creates one.
This reproduces A.6's structure exactly (base#27: 569/600 prompts at harmful
fraction 0.524, i.e. the build split) on a different model, a different
intervention, and a different behaviour.

**At the decision position the region is purer still**: final-position member
forget-fraction is **0.939–0.982** at L7 across all six (p, seed) cells.

### H1's four pre-registered sub-criteria

| criterion | predicted | measured (L7, all 6 cells) | |
| --- | --- | --- | --- |
| largest introduced region ≥ 10× median | ≥ 10× | **182–250×** | **PASS** |
| ≥ 5× the second-largest | ≥ 5× | 182–250× vs median; region 2 is ~4–6% of acts | **PASS** |
| member forget-fraction > 0.9 | > 0.9 | **0.928–0.955** | **PASS** |
| coherence > 0.9 | > 0.9 | **0.795–0.811** | **FAIL** |
| stable across seeds (exemplar cos > 0.9) | > 0.9 | mean-dir **0.9989–0.9999**; exemplar 0.52–0.75 | **PASS on mean, FAIL on exemplar** |

**Coherence fails and the reason is measured, not speculative.** Gate 1A found
the collapse is partial: the mean RMU forget activation has norm 3.98 against
RMU's 6.5 target (0.61×). The region is a **cone, not a point**, so its members
do not all agree on direction and coherence lands at 0.80. H1's ">0.9" assumed a
complete collapse; RMU does not achieve one.

**The exemplar/mean split is the same story as A.7 and Gate 1B.** The
first-arrival anchor is a streaming accident (cross-seed cosine 0.52–0.75); the
consensus direction is not (0.9989–0.9999). Anyone matching EP regions on
exemplars is reading noise that mean directions do not have.

## 2. H1 mechanism — the region *is* the injected direction

The prereg committed in advance to a discriminator: if ‖μ‖ were large relative to
c = 6.5, the collapsed activations would normalise to ≈ −μ̂ and EP would have
"found" the intervention while recovering no mechanism (H1′). Measured at L7 by
re-extracting both models and estimating the control vector directly:

| quantity | value | |
| --- | --- | --- |
| ĉ = ‖mean of RMU forget activations‖ | **3.977** (0.61 × 6.5) | matches Gate 1A's 3.94 at a quarter the sample |
| ‖μ‖ (grid calibration, L7) | 1.365 | so c·u dominates the centring term |
| cos(displacement direction, all-ones) | **0.914** | `torch.rand` predicts ≈ 0.866 |
| **cos(dominant region mean dir, centred c·u)** | **0.7275** (0.7251–0.7302) | |
| **cos(dominant region mean dir, −μ̂)** | **0.2965** (0.2916–0.3017) | **H1′ ruled out** |
| cos(dominant region exemplar, centred c·u) | 0.4987 (0.348–0.617) | exemplar is the weaker basis again |

The region direction is 2.5× closer to RMU's control vector than to the centring
artifact, and the value is stable to ±0.005 across three seeds and two
percentiles. **EP did not merely detect that something changed; the object it
returns is the thing that was injected.**

The 0.914 alignment of the displacement with the all-ones direction is
independent confirmation of the `torch.rand` detail found in Gate 1A — RMU's
control vector is drawn from the positive orthant, not isotropically.

### Why one region and not two

RMU draws a **separate** control vector per topic (`steering_coeffs 6.5,6.5`, the
random vector is redrawn inside the topic loop), so a naive reading predicts two
junk directions and two regions. There is one, and its composition is
**52–54% WMDP-bio, 40–41% WMDP-cyber** — the same 1.30 : 1 ratio as the pool
itself (1250 : 950 = 1.32 : 1). Both topics land in the same cell.

This follows from `torch.rand`. Two independent uniform-positive unit vectors in
4096 dimensions have expected cosine ≈ E[x]²/E[x²] = 0.25/(1/3) ≈ **0.75**, i.e.
a cosine *distance* of ~0.25, comfortably inside a cell of radius θ ≈ 0.80. Two
"independent random directions" are not independent at all when drawn from the
positive orthant, so the two interventions are geometrically one.

Recorded as a mechanistic consequence of the code, predicted from the source and
confirmed by the composition — and as a caution: had RMU used `torch.randn`, this
would be two regions and every count above would change.

## 3. H2 — locality

**L4 is exactly null, at every level of the analysis.** Dictionaries
bit-identical, ARI 1.000, dominant-region purity **0.483–0.497 (chance)**,
dropped = 0 regions, introduced = 0 regions. Reported explicitly per the brief:
the locality control is clean and it is the strongest evidence the pipeline is
not manufacturing its own signal.

**The gradient decays with depth, which is the direction called in advance.**

| L | dropped fraction of base regions | dominant-region recall of forget | dominant-region purity |
| --- | --- | --- | --- |
| 4 | **0.000** | 0.071 (chance) | 0.483 (chance) |
| **7** | **0.523** | **0.728** | 0.934 |
| 14 | 0.371 | 0.423–0.539 | 0.958–0.966 |
| 24 | 0.175 | 0.386–0.405 | 0.972–0.978 |

The prereg predicted decay and flagged that the AlignmentForum result (norms
jumping at the loss site, downstream incoherence) made *growth* plausible.
Decay wins on every measure of extent.

**But purity moves the other way, and that is informative.** As recall falls with
depth, purity *rises* (0.934 → 0.958 → 0.978). Downstream the junk region holds
less of the forget stream but what it does hold is almost purely forget. Read
naturally: the collapse partially disperses as it propagates — some forget
activations rejoin ordinary regions — while the core that stays collapsed stays
clean.

## 4. The dropped set — what the consolidation destroys

Gate 1B showed the diff lands on the dropped side. Its contents:

| L | p | dropped regions | dropped purity | persisted purity | introduced regions | introduced purity |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 10/12 | **0** | — | 0.500 | **0** | — |
| **7** | 10 | 158 | **0.642** | 0.428 | 4 | **0.932** |
| **7** | 12 | 102 | **0.649** | 0.430 | 3 | **0.923** |
| 14 | 10 | 86 | 0.718 | 0.449 | 8 | 0.874 |
| 24 | 10 | 120 | **0.860** | 0.461 | 23 | 0.723 |

Dropped regions are forget-enriched (0.64–0.86 against a 0.50 background) and
persisted regions are forget-*depleted* (0.43–0.47). RMU dissolves the part of
the base carving that was carrying hazardous content and leaves the rest.

**And the mass goes where it should.** Of the members of dropped base regions at
L7:

| | fraction landing in the RMU dominant region |
| --- | --- |
| their **forget** members | **0.858 – 0.934** |
| their **retain** members | 0.060 – 0.148 |

A 6–15× selectivity, on regions selected with no reference to labels. Persisted
base regions send 51–64% of their forget members and 2–4% of their retain
members the same way.

**The pre-registered introduced set is small but not junk.** 3–4 regions at L7,
purity **0.923–0.932** — as forget-pure as the dominant region. Gate 1B's verdict
stands (a 3-region set has nothing to reproduce across seeds), but the regions it
nominates are real. The primitive is underpowered, not wrong.

## 5. H3 — rejected, as called

The prereg's advance call was *"H1 dominates. H3 does not"*, argued from the loss
being MSE to a fixed vector rather than a radial rescale. That call was correct.
EP, which discards magnitude at construction, recovered a 93–96% pure region
carrying 73% of the forget stream and pointing at the injected vector. Magnitude
blindness did not prevent recovery.

The related worry was live and had to be measured, not assumed: the norm effect
is large (forget norms 2.01 → 3.59 at L7, retain 2.00 → 1.98), and θ under
per-model calibration collapses by 51%. EP simply did not need the magnitude.

## 6. Seed variance, reported not averaged

The dominant region at L24 p10, across seeds 0/1/2:

| seed | N | share | purity | recall |
| --- | --- | --- | --- | --- |
| 0 | 135,504 | 0.250 | 0.926 | 0.463 |
| **1** | **38,693** | **0.071** | 0.976 | **0.139** |
| 2 | 113,095 | 0.209 | 0.972 | 0.405 |

A 3.5× spread in size and recall from streaming order alone. Purity is stable
(0.93–0.98); extent is not. Any single-seed L24 number would be misleading, which
is what the prereg's "a single-seed number is not a result" rule exists for.

L7 is by contrast tight across seeds: N 199,489–211,890, purity 0.928–0.955,
recall 0.682–0.733. **The loss site is where the result is trustworthy.**

## 7. What was not done

- **The chem partial-transfer control was not run.** The plan reserved
  `wmdp-chem` (408 questions, no chem forget corpus in RMU training) as a
  transfer control; the grid was built with `n_chem = 0`, so it is absent. It
  needs a fresh extraction and is the single cheapest remaining experiment:
  if the dominant region also absorbs chem prompts, the region is "hazardous
  technical content", not "what RMU was trained to forget". **Until it is run,
  the specificity claim is one control short.**
- **Per-model calibration arm.** Still a projection from Gate 1A, not a
  measurement.
- **No region contents were read qualitatively.** Purity is a label statistic;
  nobody has looked at the member prompts. `regions.csv` plus the pool supports
  it at any time.

## 8. Threats

1. **The pre-registration's H1 was reframed between 1B and 1C** — from
   "introduced" to "dominant". The reframe is documented in Gate 1B §7 and was
   forced by measurement, not chosen for convenience, but a reader should weigh
   it: the sub-criteria (size, purity, stability, direction) were pre-committed;
   the *object* they are applied to was not.
2. **Purity is measured against a 50/50 pool.** A region at 0.94 forget is
   impressive against 0.50 but the pool is balanced by construction; on a
   realistic prompt distribution with 1% hazardous content the same region would
   look different, and no claim here transfers to that regime.
3. **Prompt-level labels, token-level regions.** Every activation inherits its
   prompt's label, so "forget" tokens include the benign filler inside a WMDP
   question. Purity is therefore a *lower* bound on how selective the region is
   for genuinely hazardous content, and correspondingly a weak upper bound on
   its precision.
4. **Three seeds, one model pair, one intervention.** The positive control is
   passed; nothing here establishes that a subtler diff would be recoverable.
5. **The length band excludes ~half of WMDP-cyber** (Gate 1A §3.1), unchanged.

---

## Bottom line

The positive control **passes**. On an intervention whose location and mechanism
were known in advance, EP — with no labels at construction — returns a single
region that is 93–96% hazardous-prompt, holds 73% of the hazardous stream,
reproduces across streaming seeds, and points at the injected control vector at
cosine 0.73. The base model has no such region.

Three qualifications belong in any summary of that result, in order of
importance:

1. **The standard diff primitive found none of it.** "Which regions are new"
   returned 3% of regions, below the same-model reshuffle control. The signal is
   in consolidation — dropped regions and one swollen region — and a bijective
   matcher on directions cannot express that.
2. **EP's ordinary regions are not reproducible.** Two rebuilds of the same model
   on the same data share ~7% of members per region. The RMU region is stable
   *because it was injected*, and that contrast — not EP's stability — is what
   made it findable.
3. **This was the easy case.** A deliberately huge, deliberately localised edit
   that moves 38% of activation space. Nothing here shows a subtle diff is
   recoverable, and the ~7% ordering noise floor is the reason to doubt it.
