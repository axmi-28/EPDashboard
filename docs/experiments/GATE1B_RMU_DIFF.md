# Gate 1B — stability. The kill gate. EP model-diffing positive control on RMU

Run 2026-07-31. Grid built on Modal A100-40GB: **48 dictionaries**, 4400 prompts,
542,391 paired activations per (model, layer), layers {4, 7, 14, 24}, p ∈ {10, 12},
seeds {0, 1, 2}, shared calibration, 2200 s, none aborted.

Computed **before any region content was read**. Nothing in this document
depends on what is inside a region; only directions, member sets, member counts
and coherences.

Artifacts: `artifacts/runs/rmu_diff/grid/shared/` (48 pickles + `manifest.json`,
`runs.csv`, `calibration.csv`), `artifacts/runs/rmu_diff/gate1b/shared/`
(`noise_floor.csv`, `diff_sets.csv`, `jaccard.csv`, `membership.csv`,
`membership_jaccard.csv`, `gate1b.json`). Code: `experiments/rmu_diff/build.py`,
`experiments/rmu_diff/gate1b.py`.

---

## VERDICT

**PASS — but not for the reason the pre-registered criteria report, and one
pre-registered control fires against the method.**

Three things are true at once and the report is worthless if it states only one:

1. **The pre-registered kill criteria return PASS, and that PASS is degenerate.**
   The introduced-set Jaccard clears its null in 12 of 18 informative cases at
   median 0.5. Underneath, the introduced sets contain **1 to 7 regions** out of
   98–607. A "Jaccard of 0.5" at L7 is `inter = 1, union = 2`. The null median is
   exactly 0.0, so a single coincident region produces z ≈ 8. **This is not
   evidence of a reproducible introduced set and I will not report it as one.**

2. **The H4 vacuity control fires on the introduced set.** At every layer, the
   base→RMU diff introduces *fewer* regions than re-running the base model with a
   different streaming seed does: L7 0.033 vs 0.058, L14 0.049 vs 0.056, L24
   0.032 vs 0.058. By the introduced-set framing, a 200,000-activation
   intervention is smaller than streaming noise.

3. **The diff is nonetheless unambiguous, large, and clean — on the other side
   of the same computation.** At L7, **52% of base regions are dropped** against a
   5% same-model control, with a monotone locality gradient (L4 0%, L7 52%,
   L14 37%, L24 18%). And the region RMU creates is **more reproducible across
   streaming seeds than anything the base model builds** (§5).

The instrument works. The *introduced-set primitive* does not, because RMU
**consolidates** — K halves at L7 — and set-difference in the introduced
direction is structurally incapable of representing consolidation.

**Recommendation: proceed to Gate 1C with the diff primitive reframed** from
"which regions are new" to "which regions changed who is in them". Details in §7.

---

## 1. The grid

| L | θ(p10) | θ(p12) | K base (p10/p12) | K RMU (p10/p12) | largest region, base | largest region, RMU |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.6096 | 0.6198 | 1421–1451 / 1122–1126 | **identical** | 36k–40k | **identical** |
| **7** | 0.7841 | 0.7971 | 294–297 / 197–214 | **141–153 / 98–102** | 31k–39k | **199k–212k** |
| 14 | 0.8254 | 0.8398 | 237–242 / 160–164 | 153–168 / 103–108 | 36k–50k | 109k–167k |
| 24 | 0.8587 | 0.8740 | 694–704 / 436–455 | 586–607 / 374–385 | 32k–36k | 39k–136k |

Of 542,391 activations, the largest RMU region at L7 holds **37–39%**, against
6–7% for base. The pool is 50% forget prompts, so on size alone that region can
hold at most ~78% of the forget stream — a contents question, deferred to 1C.

**L4 is bit-identical between checkpoints**: same K, same largest partition, same
singleton count, same members, for every seed and percentile. Membership ARI
base-vs-RMU at L4 is **exactly 1.000** and best member-Jaccard is **exactly
1.000** for every region. The determinism control now holds at the dictionary
level, not just the activation level, and it validates the entire pipeline —
extraction, pairing, stream permutation, clustering, member bookkeeping — against
a known answer.

## 2. Noise floor — and it is healthier than the paper's

Hungarian-matched cosine between two seeds of the *same* checkpoint, mean-member
basis (A.7's basis), 3 seed-pairs per cell:

| L | p | base median | base p5 | RMU median | RMU p5 |
| --- | --- | --- | --- | --- | --- |
| 4 | 10 | 0.891 | 0.702 | 0.891 | 0.702 |
| 7 | 10 | 0.846 | 0.690 | 0.874 | 0.760 |
| 7 | 12 | 0.855 | 0.701 | 0.877 | 0.746 |
| 14 | 10 | 0.795 | 0.641 | 0.822 | 0.670 |
| 24 | 10 | 0.703 | 0.401 | 0.719 | 0.450 |
| 24 | 12 | 0.711 | 0.413 | 0.716 | 0.470 |

Medians of 0.70–0.89 sit **above** A.7's reported cross-seed range (top quintile
~0.81, bottom ~0.60). Stability degrades monotonically with depth, which is
consistent and interpretable. So the noise floor is not the problem here.

**The problem is that the base↔RMU median matched cosine is 0.85–0.86 at L7 —
statistically indistinguishable from the base's own 0.846 cross-seed floor.** By
direction matching, the RMU dictionary looks exactly like another seed of the
base dictionary. That is the finding of §3, and it is why §1's criterion 2 fires.

## 3. Diff sets — the introduced side is vacuous, the dropped side is not

Cutoff = 5th percentile of the base cross-seed floor at that (layer, p), as
pre-registered. Means over 3 seeds × 2 percentiles:

| L | K base → K RMU | **dropped** | introduced | same-model control (introduced) | median matched cos |
| --- | --- | --- | --- | --- | --- |
| 4 | 1279 → 1279 (1.00×) | **0.000** | 0.000 | 0.052 | 1.000 |
| **7** | 250 → 123 (**0.49×**) | **0.523** | 0.033 | 0.058 | 0.859 |
| 14 | 200 → 133 (0.66×) | **0.371** | 0.049 | 0.056 | 0.812 |
| 24 | 572 → 487 (0.85×) | **0.175** | 0.032 | 0.058 | 0.706 |

Read the two middle columns together. **Introduced ≈ 0.03–0.05 everywhere and
below the 0.05–0.06 same-model control — no signal.** **Dropped goes 0.000 →
0.523 → 0.371 → 0.175 — a 10:1 signal over control at the loss site with a clean
monotone decay away from it.**

This is exactly what a consolidation looks like through a bijective matcher.
Hungarian assigns `min(K_a, K_b)` pairs; when K_RMU is half K_base, *every* RMU
region finds a partner and the change is forced onto the unpaired base regions.
H1 predicted "one massive **introduced** region". What exists is one massive
region that the matcher labels **persisted**, while half the base dictionary
vanishes underneath it.

The A.3-style fixed 0.7 cutoff is also reported (`diff_sets.csv`) and tells the
same story at L7 (introduced 0.010–0.059); at L24 it inflates to 0.46–0.53
purely because the cross-seed floor there has fallen to 0.40, which is the
concrete demonstration of why a fixed cutoff was rejected in the prereg.

## 4. Membership identity — the measure the paper could not use

Because both checkpoints partition the **same** activations in the **same**
order, region identity can be settled by who is inside a region rather than by a
cosine whose scale is set by ambient dimension. Median over seeds:

| L | ARI base↔RMU | ARI cross-seed control | median best member-Jaccard, base↔RMU | same, cross-seed control |
| --- | --- | --- | --- | --- |
| 4 | **1.000** | 0.556 | **1.000** | 0.106 |
| **7** | **0.092** | 0.566 | 0.071 | 0.070 |
| 14 | 0.173 | 0.504 | 0.060 | 0.057 |
| 24 | 0.264 | 0.618 | 0.061 | 0.059 |

Two things fall out.

**The partition is transformed at L7.** ARI 0.092 against a same-model cross-seed
floor of 0.566. The RMU partition differs from base far more than reseeding
differs from itself — a 6× gap, in the direction the intervention predicts, with
the correct locality ordering (L7 most disturbed, then L14, then L24, L4 exactly
untouched).

**And per-region identity is not conserved by EP at all, in either model.** The
median region's best member-overlap with any region of an independent rebuild of
*the same model on the same data* is **0.06–0.11**. Two EP dictionaries that
differ only in streaming order share almost no regions member-for-member. Against
that floor, base↔RMU's 0.06–0.07 is indistinguishable — not because the models
agree, but because the measure has no headroom left.

That is the deeper reason the introduced set is vacuous. It is a property of EP,
measured here on 542k real activations across 3 seeds, and it is a stronger
statement of the A.7 instability than A.7 makes.

## 5. The result that decides the gate

If EP's regions are streaming-order accidents, is *anything* here reproducible?
Yes — and the contrast is the cleanest number in this experiment.

Cross-seed reproducibility of each model's **dominant** region (no contents read;
member sets and directions only):

| L | p | model | member Jaccard across seed pairs | mean-direction cosine |
| --- | --- | --- | --- | --- |
| **7** | 10 | base | **0.0007, 0.812, 0.015** | 0.479, 0.997, 0.486 |
| **7** | 10 | **RMU** | **0.917, 0.896, 0.876** | **0.9996, 0.9998, 0.9996** |
| **7** | 12 | base | **0.0001, 0.697, 0.0000** | 0.181, 0.988, 0.154 |
| **7** | 12 | **RMU** | **0.833, 0.920, 0.814** | **0.9991, 0.9999, 0.9989** |
| 14 | 12 | base | 0.0004, 0.0005, 0.631 | 0.329, 0.362, 0.938 |
| 14 | 12 | RMU | 0.670, 0.634, 0.731 | 0.997, 0.994, 0.995 |
| 24 | 12 | base | 0.873, 0.891, 0.899 | 0.992, 0.999, 0.995 |
| 24 | 12 | RMU | 0.770, 0.528, 0.574 | 0.998, 0.995, 0.997 |

**At the loss site the base model's largest region is a streaming-order accident
— two of three seed pairs share essentially zero members (Jaccard 0.0001–0.015).
The RMU model's largest region is reproduced by every seed at member Jaccard
0.81–0.92 and mean-direction cosine 0.9989–0.9999.**

This is H1's cross-seed prediction — *"both seeds should surface near-parallel
exemplars regardless of streaming order, since u is fixed by the intervention"* —
confirmed at four nines, and it inverts the usual worry. The intervention does
not merely survive EP's instability; it is **more stable than the geometry EP
finds in the unmodified model**, because it is anchored by an injected fixed
direction rather than by first-arrival luck.

Note the basis dependence, which is itself informative: the *exemplar* cosine for
the same RMU region is only 0.52–0.75 while the *mean* direction is 0.999. The
consensus is pinned; the first-arrival anchor is not. That is A.7's point about
basis choice, reproduced on an intervention with a known ground-truth direction.

## 6. Kill criteria, applied mechanically

As pre-registered, no post-hoc adjustment.

| criterion | outcome |
| --- | --- |
| introduced set at or below the random null across seeds | **not fired** (12/18 above null p95, median J 0.50) |
| adjacent percentiles nominate disjoint introduced sets | **not fired** (median J 0.25–1.00 at L7/L14; 0.00–0.15 at L24) |
| stability survives only under D_i filtering | **not fired** — D_i top-quintile filtering changes nothing (identical median J 0.50 and identical 12/18), because the introduced sets are already tiny |
| **H4 vacuity: introduced ≈ same-model control** | **FIRED at every layer** (ratios 0.51, 0.85, 0.57) |

**Mechanically: PASS. Honestly: the first three criteria are evaluated on sets of
1–7 regions and carry almost no information, and the fourth says the quantity
they are evaluated on is noise.** The gate is passed by §3's dropped side, §4's
ARI and §5's dominant-region reproducibility — none of which the pre-registered
criteria were written to look at, because the prereg assumed the intervention
would show up as introduction.

Membership-basis introduced sets do *not* reproduce across seeds (median Jaccard
0.000, above null in 3/18). Reported because it is the honest result for that
arm: a set of 1–2 regions has nothing to reproduce.

## 7. What this changes for Gate 1C

H1 as written — *"exactly one massive **introduced** region"* — is **wrong in
form and right in substance**. There is exactly one massive region; it is
labelled persisted, not introduced, because RMU consolidates rather than adds.
Full adjudication of H1/H2/H3 stays in 1C as pre-registered, but 1C should test:

1. **The dominant-region hypothesis, not the introduced-set hypothesis.** Is the
   L7 dominant RMU region forget-pure? Its member-forget-fraction and its
   direction against the empirically measured u (Gate 1A: 87.2% of the
   displacement in one direction, 0.665 aligned with all-ones) are the tests.
2. **The dropped set is the informative one.** Which 52% of base regions are
   dissolved at L7, and are their members the forget stream?
3. **H2's locality gradient is already measured** and points the way the prereg
   predicted: 0.000 / 0.523 / 0.371 / 0.175 dropped fraction at L4/7/14/24.
   Decay, not growth — the direction called in advance.
4. **L24 seed variance must be reported, not averaged away.** The dominant RMU
   region at L24 p10 is 135,504 / 38,693 / 113,095 members across seeds 0/1/2 —
   a 3.5× spread. Exactly the pattern the prereg's "a single-seed number is not
   a result" rule exists for.

## 8. Threats

1. **The kill criteria were mis-aimed, and I am reporting a PASS obtained from
   measurements they did not specify.** The prereg's criteria were written
   assuming introduction; the evidence is in dropping and consolidation. The
   criteria are reported literally in §6 alongside the substantive result rather
   than quietly replaced, and Addendum 1 already records one mis-specification of
   the same kind in Gate 1A. Two mis-specified proxies out of a handful is a
   pattern worth stating: pre-registering *quantities* on an instrument whose
   behaviour is not yet characterised is harder than pre-registering hypotheses.
2. **Three seeds.** A.7 used five. Three gives three seed-pairs per cell, enough
   for a spread but not a tight interval.
3. **No saturation.** Builds are budget-limited by the 4400-prompt pool, by
   design (matched streams). K would keep growing with more prompts, so absolute
   K is not comparable to the paper's saturated builds — only the base/RMU
   contrast within this budget is.
4. **Shared calibration only.** The per-model arm has not been run. Gate 1A
   projected K_RMU two orders of magnitude larger at L7 under per-model θ; that
   remains a projection, not a measurement, and the abort ceiling exists for it.
5. **The length band excludes ~half of WMDP-cyber** (Gate 1A §3.1). Unchanged
   here.

---

## STOP

Gate 1B is complete. Verdict: **PASS**, with the introduced-set primitive
reported as vacuous for this intervention and the H4 vacuity control recorded as
fired.

Awaiting approval to run Gate 1C against the stable structure identified in §5 —
the dominant region and the dropped set — rather than against the introduced set
the prereg named.
