# Is Exemplar Partitioning useful as a runtime model monitor?

Consolidated read across Gates 0B, 1B and 2. Model `google/gemma-2-2b-it`,
layer 20 unless stated, prebuilt hub dictionaries at revision
`0ec26618db2bd64cdbf00a9a1b0eb23fffdea12a`. Inference only throughout: no
gradient step was taken on the model, and every probe baseline is closed-form.

---

## The short answer

**No, not as a detector.** Across nine experiments and six labelled tasks, every
time EP was asked to compete at detection, something cheaper won — usually a
difference-in-means direction fit on sixteen labelled examples.

**But the partition is not empty.** It beats a matched-K random partition on
language identity by +12 sd under the strictest control available, it holds its
operating point under attack better than any continuous scorer tested, and it
exhibits the split-early / consolidate-late structure across depth that the
abstraction story predicts. Those are real properties. None of them is
detection accuracy.

The single most useful framing to carry forward: **EP is an unsupervised
description of activation space, and the evidence says it should be evaluated as
one.** Every result where it was scored as a classifier is negative; the results
where it is scored as a structural description are mixed-to-positive.

---

## What the matched-K random coreset control did to this programme

One control decides most of what follows, so it is worth stating separately.

A **matched-K coreset** is K activations drawn at random from the same build
stream EP's exemplars came from, used as if they were exemplars. It matches EP
on memory, on inference cost, on the space it operates in, and on the data it
saw. The only difference is that EP's K vectors were leader-clustered and the
coreset's were sampled.

| gate | EP vs. matched-K coreset |
|---|---|
| 0B (distance scorers) | coreset **wins** — EP's distance saturates inside the support |
| 1B (occupancy histogram) | EP **wins** (but loses to Mahalanobis and entropy) |
| 2 A0 (refusal flag) | **tie** at every percentile, both prompt formats |
| 2 A3 (4 concepts) | EP wins on **1 of 4** (language ID) |
| 2 A1 (label efficiency) | EP wins narrowly; both lose to the probe at every budget |
| 2 A2b (calibration drift) | EP **+0.014 vs +0.050** — EP more stable |
| 2 B (trajectories) | tie or coreset wins on ~half the scorers |
| 2 C2 (cross-layer) | coreset **wins**, −3.3 sd, stable across a 12× sample sweep |

Any EP result reported without this control is uninterpretable. It is the reason
Gate 0B's headline died and the reason A0's 95%-pure refusal region does not
support the conclusion it appears to.

---

## Findings that are about EP

### 1. The partition carves real language structure (positive, robust)

EP beats the matched-K null on Bulgarian-vs-English by +16.6 sd at p=1, at
**three consecutive percentiles** (p=1, 2, 4), so it survives the
single-percentile-artifact standard.

The obvious debunking — "Bulgarian is outside the Pile support, so this is
distance-to-support detection in disguise" — was tested and **fails**:

- distance alone barely separates the classes where EP wins (AUROC 0.545, 0.504,
  0.392 at p=1,2,4)
- forcing both classes onto a shared distance distribution costs EP almost
  nothing (+16.6 sd → **+12.5 sd**)
- broken out by distance decile, EP's advantage is **largest in the nearest
  stratum** (+0.091) and smallest in the farthest (+0.019) — the opposite of the
  OOD prediction

This is the strongest positive result in the programme.

**The mechanism, which explains all of Arm A in one sentence.** Count regions
that are ≥90% one class with ≥5 members, and ask what share of each class they
hold:

| task | p | partition | pure regions | % of positives covered | % of negatives covered |
|---|---|---|---|---|---|
| language | 4 | **EP** | 103 | **0.657** | 0.367 |
| language | 4 | coreset | 97 | 0.498 | 0.239 |
| code/math | 4 | EP | 9 | **0.126** | 0.162 |
| code/math | 4 | coreset | 11 | **0.655** | 0.390 |

EP allocates dedicated regions to Bulgarian and fails to allocate them to code
vs. math; a density-proportional random sample does the opposite.

This is the covering-vs-density asymmetry that has run through the whole
programme since Gate 0B. EP places exemplars to **cover the support**, so it
wins when a class occupies its own distinct volume of activation space
(a different language) and loses when the distinction lives *inside* a dense
region (code vs. math word problems, harmful vs. benign instructions), where
sampling ∝ density puts more exemplars in the contested area.

That yields a usable prediction: **EP region identity is informative about
distinctions that are volumetric, and uninformative about distinctions that are
directional within a shared volume.** Every result in Arm A fits it — and it is
also why refusal, the motivating case, was never going to work: harmful and
benign instructions are the same kind of text about different subjects.

### 2. A discrete flag holds calibration where a continuous probe does not (positive, thin)

Fit one rule on plain prompts, never retune, then measure the false-alarm rate
under eight jailbreak wrappers:

| scorer | plain FPR | mean FPR under attack | drift |
|---|---|---|---|
| ridge probe | 0.050 | 0.534 | **+0.484** |
| difference-in-means | 0.050 | 0.083 | +0.033 |
| **EP-FLAG p1** | 0.050 | 0.064 | **+0.014** |
| coreset flag p1 | 0.003 | 0.053 | +0.050 |

The ridge probe has AUROC ≈ 1.000 on every attack and is still the worst scorer
to actually deploy: its *ranking* survives the wrapper, its *absolute score*
does not, so a threshold fit on plain traffic produces a tenfold increase in
false alarms. A region flag has no threshold to drift — you are in a flagged
region or you are not.

EP also beats the random partition here (+0.014 vs +0.050), which makes this the
second EP-specific positive. **Thin evidence**: one coreset draw, eight attacks.
It should be replicated before being leaned on.

### 3. Sequential structure exists and is label-irrelevant (clean null)

Regions repeat across adjacent tokens **2.4–2.8× more than a within-prompt
shuffle** that preserves each prompt's region marginal exactly, at every
resolution. Real sequential structure.

It carries nothing about the label. The order gain — bigram surprise minus
unigram surprise, same table — is −0.036, −0.021, +0.001, +0.020 across p =
2, 4, 8, 10. Zero, and negative where the table is best resolved.

**One symbol at the final position gives AUROC 0.958. Twelve symbols of
trajectory give at most 0.66.** Averaging within a request destroys the signal
rather than accumulating it, because the distinction lives at the position where
the instruction has been integrated.

### 4. Cross-layer correspondence is *worse* than random (negative, well-supported)

This is the finding that removes the last fallback framing — that EP might be
worth having as a stable discrete coordinate system comparable across layers and
checkpoints.

Mutual information between a prompt's early region and its late region, against
matched-K coresets at both layers, swept over sample size:

| n | MI EP | MI null | margin | lift EP | lift null |
|---|---|---|---|---|---|
| 600 | 0.933 | 1.433 | −3.2 sd | +0.063 | +0.109 |
| 7460 | 0.730 | 1.088 | **−3.3 sd** | **+0.101** | **+0.143** |

Both MI estimates fall with n (the bias shrinking) while the margin holds flat —
so it is not a small-sample artifact. Held-out prediction accuracy is *higher*
for EP (0.464 vs 0.340) but only because EP's late partition is more
concentrated (majority baseline 0.363 vs 0.196); lift over that baseline favours
the random partition on both layer pairs at every sample size, and still does
after normalising for headroom (P2: 0.158 vs 0.179; P1: 0.187 vs 0.267).

Honest qualifier: under the normalised statistic the **primary** pair's margin
is 11% relative, not the 30% raw lift implies. The large gap is on P1, which is
the exploratory pair. The direction is consistent across three statistics, two
layer pairs and a 12× sample sweep; the magnitude on P2 is modest.

Mechanism: random coreset cells are sampled ∝ density, so both layers' partitions
track the same dominant modes and stay aligned. EP spreads exemplars to *cover*
the support, allocating resolution to sparse outlying areas that differ layer to
layer.

### 5. Region identity at usable K is a template detector (negative, decisive)

At p=10 (K=176), four of eight jailbreak wrappers put **all 600 prompts — harmful
and benign alike — into a single region**. At p=4, chat formatting puts 82% of
prompts in one region and drives cross-fit AUROC to 0.646.

The hub dictionaries were built on raw web text. A chat scaffold is a large
perturbation to the final-position activation and cosine-argmax against Pile
exemplars is dominated by it. This is the mechanism behind the whole of Arm A,
and it applies at exactly the resolutions a cheap deployed monitor would use.

---

## Findings that are about the model, not about EP

Worth separating, because they are robust and they are what EP was useful for
*discovering* even where it was not useful for *detecting*.

- **Refusal-relevant content is linearly readable at layer 20 with almost no
  labels.** Difference-in-means on 16 examples: AUROC 0.998.
- **Harmful prompts split early and consolidate late.** Effective region count
  ratio harmful/benign at the last instruction token: **4.35 at L4 → 2.05 at L12
  → 0.61 at L20**. Harmful prompts start scattered over 4.3× more regions than
  benign and end more concentrated. Only the L4 excess clears its null, so what
  EP adds over random is the early scatter, not the late consolidation.
- **The merge into the refusal region is strongly label-selective (+0.60)** —
  harmful members of a shared early region are 60 points more likely to reach
  the refusal region than their benign neighbours. A random early partition
  reproduces this exactly, so it is a fact about layer 20's separation, not
  about EP's structure.
- **Base64 and leetspeak defeat every detector tested**, including a probe with
  AUROC 1.000 on plain prompts (driven to 0.432 — *inverted*). Both rewrite
  tokens rather than adding a wrapper, so the content direction is not present
  at layer 20 to be read by anything.

- **Detectability is anti-correlated with attack efficacy, for every scorer.**
  Across eight attacks, the correlation between non-refusal rate and detection
  margin is **r = −0.94** for the best scorer (−0.94 excluding base64). The
  attacks that break refusal — leetspeak, base64 — are the ones nobody sees;
  the attacks every detector handles cleanly (payload split, GCG suffix) have a
  0% success rate. Wrapping attacks leave the content tokens intact, so refusal
  fires *and* the direction stays readable; rewriting attacks defeat both at
  once. **Any robustness table averaged over an attack suite without weighting
  by success rate overstates the monitor by construction** — including the ones
  earlier in this programme.

  This is the most consequential safety finding here and it is not about EP.

---

## What would change the conclusion

- A concept that is genuinely non-linearly separable, where the piecewise-constant
  lookup should structurally win. Nothing tested here was: every task had a
  linear probe above 0.99.
- A dictionary built on the traffic distribution rather than on Pile. Every
  negative here is partly a transfer result, and the one prior success
  (`gate.json`) used a dictionary built on the labelled prompts.
- Replication of the calibration-stability result with proper coreset draws.

## What would not

- More percentiles, more label budgets, or more attacks. A1 swept seven budgets
  × 20 draws × 5 tasks and found no crossover anywhere; A0/A3 swept five
  percentiles on five tasks.
- More trajectory data. Coverage of the transition table rose 2.3% → 57.3%
  while the repeat excess stayed flat and the order gain did not trend.

---

## Standing caveats

- **One model, one main layer.** Everything is gemma-2-2b-it at layer 20.
- **Arm C has no artifact check.** The hub carries one percentile per early
  layer, so the multi-percentile standard that validated the language result
  cannot be applied to any cross-layer claim.
- **L12's dictionary saturated on 131,072 activations at K=145**, an order of
  magnitude less build data than L20's, so a null at P2 is weak evidence of
  absence.
- **Prompt-level detection only** for the jailbreak work unless the generation
  pass (A4) says otherwise.

## Where the numbers live

```
docs/experiments/GATE2_RESULTS.md          full Gate 2 write-up
artifacts/runs/monitor/gate2_verdict.json  machine-readable verdict
artifacts/runs/monitor/gate2_a0_routing.csv        A0 hub sweep
artifacts/runs/monitor/gate2_a1_labelcurve.csv     A1 label efficiency
artifacts/runs/monitor/gate2_a2_jailbreak.csv      A2 attack AUROC
artifacts/runs/monitor/gate2_a2b_detection.csv     A2b detect/alarm split
artifacts/runs/monitor/gate2_a2b_calibration.json  A2b drift
artifacts/runs/monitor/gate2_a3_concepts.csv       A3 four concepts
artifacts/runs/monitor/gate2_a3b_ood.csv           A3b OOD diagnosis
artifacts/runs/monitor/gate2_a4_attack_success.csv A4 generation pass
artifacts/runs/monitor/gate2_b_trajectory.csv      B trajectories
artifacts/runs/monitor/gate2_c_crosslayer.json     C cross-layer
artifacts/runs/monitor/gate2_c2_scale.json         C2 scale sweep
```
