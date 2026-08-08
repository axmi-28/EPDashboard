# Gate 2 — pre-registration: EP as a *behavioural* monitor

Written before any Gate 2 number exists. Self-contained: assumes no knowledge of
the repo or of Gates 0/1.

---

## 0. What the object is

**Exemplar Partitioning (EP)** is a training-free way to carve up a language
model's activation space. You run text through the model, take the residual
stream at one layer, centre and unit-normalise it, and sweep through the
activations in arrival order: if an activation is within cosine threshold `θ` of
an existing *exemplar*, it joins that exemplar's region; otherwise it becomes a
new exemplar. You stop when the rate of new exemplars hits zero (*saturation*).

The result is a **dictionary**: `K` unit vectors in `R^d`, a centre, and `θ`.
At inference, an activation's region is `argmax_i cos(x - centre, e_i)` — one
`K x d` matmul and an argmax. Assignment is a pure argmax: there is **no
"none of the above" escape**, every activation lands somewhere.

Two properties matter for monitoring:

- Exemplars are **real observed activations**, so the coordinate system is fixed
  and comparable across time and across model checkpoints (unlike SAE features,
  which move under retraining seed).
- The partition is **hard and discrete**, so an input gets a *symbol*, not a
  score.

### The dictionaries this gate uses

All pulled prebuilt from the HuggingFace **dataset** repo
`J-RUM/exemplar-partitioning` at revision `0ec26618db2bd64cdbf00a9a1b0eb23fffdea12a`.
**No dictionary is built locally in this gate.** Model is
`google/gemma-2-2b-it` (26 layers, `d_model=2304`) unless stated.

| dictionary | K | θ | build acts | saturated |
|---|---|---|---|---|
| it L4 p4 | 491 | 0.8588 | 557,056 | yes |
| it L12 p10 | 145 | 0.8846 | 131,072 | yes |
| it L20 p1 | 5796 | 0.7798 | — | yes |
| it L20 p2 | 2037 | 0.8215 | — | yes |
| **it L20 p4** | **686** | **0.8613** | 573,440 | yes |
| it L20 p8 | 226 | 0.9000 | — | yes |
| it L20 p10 | 176 | 0.9124 | 376,832 | yes |
| **base** L20 p10 | 192 | 0.9065 | 311,296 | yes |

`p` is the percentile that sets `θ`; lower `p` = finer partition. **Primary
resolution for every decision rule in this gate is `p=4` (L20, K=686)**, declared
here so that reporting the full sweep afterwards is not a search for a winner.

Available layer pairs at *matched percentile*, which is what makes Arm C
possible without building anything:

- **P1**: L4 p4 (K=491) x L20 p4 (K=686)
- **P2**: L12 p10 (K=145) x L20 p10 (K=176)  ← primary, best K balance

---

## 1. Where Gates 0 and 1 left this

Both prior gates tested **label-free** monitors on `gemma-2-2b-it` L20, against a
ladder of distribution shifts (R0 in-distribution Pile; R1 code+math; R2
Bulgarian; R3 Pile text in odd chat scaffolds; R4 jailbreaks; R5 random tokens),
2,000 prompts per rung, 128 tokens each.

- **Gate 0B — distance to nearest exemplar.** Failed. The published mean gap
  reproduced exactly, but as a detector it reached only 1.7–5.8% TPR at 1% FPR,
  and a **matched-K set of randomly chosen activations beat it**. Mechanism:
  EP places exemplars to *cover* the support, so distance measures
  distance-to-support and saturates inside it; a random sample places points
  proportional to density, so its distance is a density estimate. Coverage is
  the ceiling, not the asset.
- **Gate 1B — histogram of region occupancy over traffic windows.** Partial.
  EP beat the matched-K random coreset at every resolution on R2 and R5 — the
  first real EP win — but lost to Mahalanobis distance (R2) and to the model's
  own next-token entropy (R5), detected nothing at 1% contamination, was blind
  to R1/R3/R4, and had **0.000 TPR@1%FPR per individual request** everywhere.
  The hypothesised mechanism (EP spreads traffic more evenly) was falsified:
  EP's histogram is *more* concentrated than the coreset's, and at p=1 only
  13.5% of cells are ever occupied.

**What this gate changes.** Both prior arms asked "is this activation abnormal",
which has no ground truth. Gate 2 asks about **labelled behaviours**, so every
arm can be flatly wrong. Nothing in Gate 0/1's negative results applies to a
supervised flag.

**Two carried-over findings that Gate 2 leans on:**

1. A separate run found that harmful instructions concentrate in a small number
   of L20 regions — one region held 404 prompts at 74.3% harmful with 100%
   harmful recall across 5 occupied cells. **But that run used a locally built
   `p=12` dictionary, not a hub one.** Arm A therefore starts by re-establishing
   it on hub dictionaries; nothing may be built on it until that passes.
2. Prompts with *successful* jailbreaks still land in the refusal region — the
   compliance override happens downstream. If true on hub dictionaries, that is
   the single most valuable property here: the internal signal survives the
   attack that defeats the behavioural signal. Arm A2 tests it directly.

---

## 2. Rules that apply to every arm

- **Inference only.** No gradient steps on the model, no finetuning. Fitting a
  linear probe on frozen activations is a *baseline*, not model training; probes
  are fit closed-form (ridge / LDA) to keep this unambiguous.
- **Three disjoint pools**: fit / calibrate / evaluate. Region selection never
  touches the eval set.
- **Every EP scorer has a matched-K random-coreset twin.** A coreset is K
  activations drawn at random from the same build stream, used as exemplars.
  This is the control that killed Gate 0B. Three independent draws; report the
  draw sd; a win inside one sd is a tie.
- **Metrics are AUROC and TPR@1%FPR.** Mean-gap statements are not results.
- Report the full percentile sweep, but decide on the pre-declared `p=4`.
- **A win at exactly one percentile is an artifact**, not a finding — the
  standard already applied when R3 spiked at a single p in Gate 1B.
- Every number lands in a results file, not just stdout.

---

## 3. Arm A — supervised behaviour flags vs. linear probes

### The question

You have labels for a behaviour. Is "which region did this land in" a better
detector than a linear probe on the raw activation — and specifically, **is it
better when labels are scarce**?

The claim being tested is *not* "EP beats probes", which is probably false at
scale. It is that **region-flagging is a lookup table over a nonlinear
partition**, which predicts a specific win/lose pattern:

- wins when the concept is not linearly separable (regions carve disconnected
  lobes; one hyperplane cannot)
- wins at small label counts (ranking K bins needs fewer labels than fitting a
  2304-d hyperplane)
- loses on gradations *within* a region (piecewise-constant score)
- loses as labels grow (the probe's capacity pays off)

So the deliverable is a **crossover curve**, and the finding is the label budget
at which it crosses — or that it never does.

### A0 — does the routing survive the hub sweep? (gates everything else)

Assign the 300 harmful / 300 benign instruction set under hub L20 p ∈ {1,2,4,8,10}.
Per p report: number of occupied regions, harmful fraction of the most-harmful
region, harmful recall of the flagged set, and the same three under a matched-K
coreset.

**Stop condition:** if concentration appears at only one percentile, or does not
exceed the coreset null, A0 fails and Arm A stops. Report and move to B.

### A1 — label-efficiency curve

Label budgets `n ∈ {16, 32, 64, 128, 256, 512, 1024}`, **20 random draws each**,
eval on a held-out pool that no draw ever touches.

Scorers:

| id | what it is | params |
|---|---|---|
| `EP-FLAG` | smoothed harmful rate of the landed region, `(h+α)/(n+2α)`, α=0.5 | K counts |
| `EP-RIDGE` | closed-form ridge on the one-hot region indicator | K |
| `PROBE` | closed-form ridge / LDA on the raw 2304-d activation | d |
| `DIFFMEAN` | normalised difference-in-means direction | d |
| `CORE-FLAG` | `EP-FLAG` with a matched-K random coreset partition | K |

`DIFFMEAN` is included because AxBench's headline result is that simple
supervised directions are very hard for unsupervised dictionary methods to beat.
`CORE-FLAG` separates "EP's partition is good" from "any partition is good".

Report AUROC and TPR@1%FPR, mean ± sd over draws, per behaviour, per p.

### A2 — robustness under jailbreak

Fit **only on plain harmful vs. benign** — no jailbreak prompt is ever in the
training pool. Evaluate on the 2,000 R4 jailbreak prompts (JailbreakBench +
AdvBench goals x 8 attacks: prefix injection, refusal suppression, roleplay,
distractor, base64, leetspeak, payload split, GCG suffix), broken out per attack.

This is where carried-over finding (2) predicts EP holds up and the probe may
not. It is also the only part of this gate with a realistic threat model.

### A3 — does it generalise past refusal? (free, uses cached data)

Same machinery, three more labelled tasks already extracted:

- **code vs. math** — MBPP (974) vs. GSM8K (1026), a topic concept
- **language ID** — Bulgarian (2000) vs. English Pile (2000)
- **scaffold ID** — 5-way, 400 each, *identical Pile content* in five chat
  templates. Content is held fixed, so this isolates pure format — the cleanest
  test of whether regions encode things a probe would find awkward.

### Decision rule A

- **PASS** if `EP-FLAG` beats `CORE-FLAG` by >2 draw-sd on ≥2 of the four
  behaviours at p=4, **and** there exists a label budget n where
  `EP-FLAG ≥ PROBE`.
- **FAIL (partition worthless)** if `EP-FLAG` never beats `CORE-FLAG`.
- **FAIL (no niche)** if `EP-FLAG` never reaches `PROBE` at any n.
- A2 is reported separately and can pass or fail independently — a flag that
  loses on clean data but survives jailbreaks is still an interesting result,
  and must be described as exactly that.

### Cost

Final-position activations for R0–R5 are already cached. A3 is free. A0/A1/A2
need the 300/300 instruction set extracted (~minutes on MPS). **~1.5 h.**

---

## 4. Arm B — region trajectories

### The question

Every prior experiment used **one symbol per prompt** — the region of the final
token. A 128-token passage produces **128 symbols**. Gate 1B showed that a
single symbol has zero per-request power but that averaging ~500 requests works.
If that averaging is available *within* a single prompt, per-request EP
monitoring becomes viable and Gate 1B's negative simply reflects having given
the method one sample.

### B0 — is the transition structure trivial? (run first)

On held-out Pile (R0), at p=4:

- self-transition rate `P(r_{t+1} = r_t)`
- `I(r_t ; r_{t+1})`, against two nulls: (i) positions shuffled within prompt
  (destroys order, keeps marginals), (ii) **matched-K coreset partition**
- the same three under the coreset

**Stop condition:** if the self-transition rate exceeds 0.95, trajectories are
near-constant and there is nothing to model — report and stop. If EP's
transition MI does not exceed the coreset's, the structure is activation
smoothness, not EP, and B1 is reported as descriptive only.

### B1 — trajectory scorers, per prompt

Fit a bigram model over regions on an R0 training split (Laplace α=0.5). Score
every prompt in every rung:

| id | statistic | tests |
|---|---|---|
| `T1` | mean `-log P(r_{t+1} | r_t)` | order + identity |
| `T2` | mean `-log P(r_t)` | identity only — does order matter? |
| `T3` | distinct regions visited / trajectory entropy | coverage of the path |
| `T4` | mean top1−top2 cosine margin | boundary ambiguity |
| `E1` | mean / max next-token entropy | **free baseline, already beat EP once** |
| `D1` | mean distance to nearest exemplar over positions | Gate 0B's scorer, averaged |
| `C1–C3` | `T1–T3` under a matched-K coreset | is it EP or any partition? |

Metric: AUROC and **TPR@1%FPR per prompt** vs. R0, per rung.

### B2 — the decisive contrast

Per-prompt AUROC of `T1` (128 samples) against the final-position single-region
surprisal from Gate 1B (1 sample), same prompts, same dictionary. This is the
"one request is a window" hypothesis stated as a number.

### Decision rule B

**PASS** requires all four:
1. `T1` beats its coreset twin `C1` by >2 sd,
2. `T1` beats `T2` — order carries information beyond which regions appear,
3. per-prompt TPR@1%FPR on ≥1 rung materially exceeds the ~0.000 recorded at
   final position,
4. `T1` beats `E1`, the free entropy baseline.

Failing (4) while passing (1)–(3) is reported as "the mechanism is real, the
monitor is not competitive" — the same shape as Gate 1B's verdict.

### Cost

Requires re-extraction: **500 prompts per rung x 128 positions** = 64,000 rows
per rung, ~150 MB fp16 per rung, ~0.9 GB total. ~20–30 min on MPS, run in the
background while Arm A's analysis proceeds on cached data. Analysis ~1 h.

---

## 5. Arm C — cross-layer correspondence

### The question

An input has a region at an early layer and a region at a late layer. Do early
regions map systematically onto late ones? **Merges** (many early → one late)
are the model abstracting; **splits** (one early → many late) are the model
making a distinction. The refusal finding is exactly one split, observed at a
single layer. Arm C asks whether splits can be found *structurally*, without
knowing in advance what behaviour to look for.

### C0 — contingency tables

Joint counts `N[i,j]` over per-token activations from Arm B's extraction
(R0, 2,000 prompts x 128 positions = 256,000 rows), for:

- **P2 (primary)**: L12 p10 (K=145) x L20 p10 (K=176) → 25,520 cells, ~10
  samples/cell. Adequate.
- **P1 (exploratory)**: L4 p4 (K=491) x L20 p4 (K=686) → 337k cells, ~0.8
  samples/cell. **Under-powered; reported as exploratory only, no decision
  rests on it.**

### C1 — is the correspondence real?

Normalised mutual information between layer-A and layer-B assignment, against
(i) row-shuffled null and (ii) **matched-K coreset partitions at both layers**.
MI is biased upward by K, which is precisely why the null must be a coreset at
matched K rather than an analytic correction.

### C2 — merge/split structure

Per late-layer region, the entropy of its early-layer preimage (high = merge).
Per early-layer region, the entropy of its late-layer image (high = split).
Rank; report the top 10 of each.

### C3 — the behavioural test (this is what makes C non-decorative)

Using Arm A's harmful/benign labels: find an **early-layer region whose
harmful/benign composition is near chance** which **splits into late-layer
regions with significantly separated harmful rates**. That localises, in depth,
where the model draws the distinction — discovered from structure, then checked
against labels.

Tested against the coreset null: a random partition will also produce splits,
and the question is whether EP's are more behaviourally aligned.

### C4 — base vs. instruct (free)

Assign the same labelled prompts under it-L20-p10 (K=176) and base-L20-p10
(K=192) — matched percentile, near-matched K. Prior claim: the base model routes
harmful and benign together, the instruct model splits them. Region **ids are
not comparable across dictionaries**, so the statistic is the *achievable
separation* under each (max harmful-rate spread across occupied regions), not
any identity mapping.

### Decision rule C

- **PASS** if NMI exceeds the coreset null by >2 sd on P2 **and** C3 finds ≥1
  split with a harmful-rate separation that survives the coreset null.
- **FAIL** otherwise.
- **No flow diagram is produced unless C1 and C3 both pass.** This arm produces
  attractive pictures whether or not it means anything, and that is its main
  hazard.

### Known weaknesses, stated in advance

- L12's dictionary saturated on only 131k activations at K=145. That is coarse;
  a null result at P2 is weak evidence of absence.
- Only one percentile is available per layer pair, so the single-p artifact
  check that Gate 1B relied on **cannot be run here**. Any C result is
  provisional for that reason alone.

### Cost

Reuses Arm B's per-token activations; needs two extra dictionary downloads and
two extra assignment passes. **~30 min.**

---

## 6. Order of work

| | step | why here |
|---|---|---|
| 1 | **A0** | gates the whole refusal line; ~10 min |
| 2 | launch Arm B extraction in background | it is the only long pole |
| 3 | **A1, A2, A3** | runs on cached data while B extracts |
| 4 | **B0 → B1 → B2** | as soon as extraction lands |
| 5 | **C0–C4** | reuses B's activations |
| 6 | write `GATE2_RESULTS.md` | |

## 7. What a clean negative looks like

Stated now so it cannot be softened later:

- A fails both ways → EP is a worse basis for supervised detection than the raw
  activation, and the concept-monitoring pitch is finished.
- B fails at (1) or (2) → trajectories are activation smoothness, not routing.
- B passes (1)–(3) but fails (4) → real mechanism, uncompetitive monitor.
- C fails → cross-layer correspondence is not stronger than a random partition,
  and the merge/split framing should be dropped rather than illustrated.

If A, B and C all fail, the conclusion is that **EP's monitoring value is in
offline analysis, not at runtime**, and that is the result to write up. It is
worth as much as a positive and it costs one day.

## 8. Artifacts

```
artifacts/runs/monitor/gate2_a0_routing.csv
artifacts/runs/monitor/gate2_a1_labelcurve.csv
artifacts/runs/monitor/gate2_a2_jailbreak.csv
artifacts/runs/monitor/gate2_a3_concepts.csv
artifacts/runs/monitor/gate2_b0_transitions.json
artifacts/runs/monitor/gate2_b1_trajectory.csv
artifacts/runs/monitor/gate2_c_crosslayer.json
artifacts/runs/monitor/gate2_verdict.json
docs/experiments/GATE2_RESULTS.md
```
