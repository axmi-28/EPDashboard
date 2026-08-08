# EP vs. baselines — a sparse-probing-style comparison methodology

Design document, not a result. Written against two papers:

- **KE25** — Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda, *Are Sparse
  Autoencoders Useful? A Case Study in Sparse Probing* (arXiv:2502.16681,
  ICML 2025). Code `github.com/JoshEngels/SAE-Probes`; packaged benchmark
  `github.com/sae-probes/sae-probes`.
- **TM25** — Tillman & Mossing (OpenAI), *Investigating task-specific prompts
  and sparse autoencoders for activation monitoring* (arXiv:2504.20271).

Both ask the question this repo has been asking since Gate 0B — *does the fancy
decomposition beat the boring baseline?* — and both answer "no, mostly". The
value here is not their verdict, it is their **evaluation contract**. This
document adopts that contract, states what our own Gates already settle under
it, and specifies the parts that are genuinely open.

---

## 1. What the two papers actually establish

### KE25 — the contract

| element | what they do |
|---|---|
| breadth | **113 binary probing datasets**, deliberately including hard ones (`26_headline_isfrontpage`, `136_glue_mnli_entailment`) |
| models | Gemma-2-9B (main), Llama-3.1-8B (replication); last-token residual stream, layer 20 |
| SAE side | Gemma Scope JumpReLU / Llama Scope TopK; probe on the top-`k` latents by **mean absolute activation difference between classes** (k ∈ {1,16,128}), L1 logistic regression |
| baselines | logistic regression (L2), PCA + LR, KNN, XGBoost, MLP — **10 hyperparameter values each**, tuned by validation AUC |
| regimes | standard; data scarcity (n ∈ [2,1024], 20 log-spaced); class imbalance (positive ratio 0.05→0.95, 19 values); label noise (0→0.5 corrupted, 11 values); covariate shift (8 OOD datasets: GLUE-X, translated, name-perturbed) |
| metric | test AUROC; ≥100 test examples per dataset, mean 1945 |
| **decision rule** | **"quiver of arrows"** — pick the best method by *validation* AUC, report that method's *test* AUC, with and without the new method in the pool. The deliverable is the **marginal ΔAUC of adding the arrow**, not a head-to-head win |
| exception | when validation is unfaithful to test (label noise, covariate shift), fall back to a fixed head-to-head |

Two results matter more than the headline for us:

1. **The pooling illusion (§5).** Max-pooled SAE probes beat *last-token*
   baselines 19.6% of the time vs. 2.2% for last-token SAE probes — and that
   advantage **halves to 8.7%** once an attention-pooled probe is added to the
   baseline pool. The apparent SAE win was an aggregation asymmetry, not a
   basis advantage. Any EP result that pools over tokens must be compared
   against a pooled baseline or it is uninterpretable.
2. **SAE-choice inflation.** Choosing the best SAE by test AUC would have
   manufactured a win. The quiver-of-arrows procedure exists to prevent this.
   **This is the principled replacement for our ad-hoc "a win at exactly one
   percentile is an artifact" rule** — θ/percentile becomes a hyperparameter
   selected on validation like any other.

### TM25 — the axis KE25 is missing

Same shape, but the independent variable is **compute budget**, not dataset
difficulty.

- Methods: zero-shot prompted model output (Yes/No logit diff); few-shot (1–32);
  **prompted probing** (prefix+suffix and suffix-only templates that name the
  monitored concept, then probe the last token); raw last-token probing;
  SAE max-pooled probing; SAE pre-activation probing; LAT Scan (unsupervised
  PCA); probe ⊕ model-output stacking.
- Tasks: moderation (2,610 expert-labelled passages; harassment, violence),
  SimpleQA hallucination (10k on-policy QA pairs, auto-labelled, split by
  *question* so none leak), RottenTomatoes sentiment (~10k).
- Protocol: single 80/20 split held **fixed** across all experiments; training
  set subsampled to vary data quantity; layer at ~75% depth; AUROC vs. number
  of positive training examples.
- Shift: English→non-English and single-turn→multi-turn, plotted as
  ID-vs-OOD AUROC across train-set sizes. Prompted probing generalises well;
  **SAE max-pooled probes generalise poorly**.
- Verdict framed as a **recommendation table keyed on constraints**: no
  inference budget → zero-shot; inference budget + data → prompted probing;
  neither → raw activation probing; train-time compute available but inference
  constrained → SAE probing.

**The framing to steal is the last one.** TM25 does not claim SAEs win; it
identifies the *resource cell* in which they are the right call. EP's only
plausible pitch is exactly that shape.

### Sibling prior art to cite and to baseline against

**ITDA** (Leask, Nanda, Al Moubayed, arXiv:2505.17769) greedily builds a
dictionary *of real observed activations* — no gradient step, 1% of SAE
training time and data — and its headline win is **cross-model comparison**
(Jaccard on ITDA dictionaries beats CKA/SVCCA). That is a training-free
observed-activation dictionary, i.e. EP's nearest published neighbour, and its
strongest result is in the same territory as our RMU model-diffing gates. It
must appear as a baseline arrow, and the write-up must position EP against it,
not only against SAEs.

---

## 2. What this repo has already settled under that contract

Gates 0B / 1B / 2 (`EP_MONITOR_SUMMARY.md`) are, in KE25 terms, a 5-dataset
single-model study. Under a matched-K random coreset control:

| our arm | KE25/TM25 analogue | outcome |
|---|---|---|
| 0B distance scorers | unsupervised OOD | **fail** — coreset wins; TPR@1%FPR 1.7–5.8% |
| 1B occupancy histogram | population drift | EP > coreset, but < Mahalanobis / next-token entropy |
| 2 A1 label curve | **KE25 data scarcity** | **no crossover at any budget on 5 tasks**; probe's lead is *widest* at n=16 |
| 2 A2/A2b jailbreak | **TM25 covariate shift** | EP-FLAG threshold drift **+0.014** vs. ridge **+0.484**, diffmean +0.033, coreset +0.050 |
| 2 A3 concepts | KE25 breadth (n=4) | EP > coreset on **1 of 4** (language ID, +12.5 sd after controlling for distance) |
| 2 B trajectories | TM25 pooling | 12 symbols lose to 1; order gain ≈ 0 |
| 2 C cross-layer | ITDA's cross-model claim | **cross-layer MI is below random**, −3.3 sd, stable over a 12× sample sweep |

**What that licenses:** EP is not a better basis for supervised detection than
the raw activation, on those tasks, at that layer, with those dictionaries.

**What it does not license, and this is the whole opening:**

1. **No headroom.** Every Gate 2 task had a linear probe above 0.99. KE25's
   suite averages ≈0.8–0.9 baseline AUC and was *built* to contain hard cases.
   A piecewise-constant lookup cannot show an advantage on a task a hyperplane
   already solves perfectly — the experiment was structurally incapable of a
   positive.
2. **One dictionary provenance.** Every negative used **Pile-built hub
   dictionaries applied to chat traffic**. Gate 2's own diagnosed failure mode
   (§"region identity at usable K is a template detector": at p=10, four of
   eight jailbreak wrappers put *all 600 prompts* in one region) is a
   provenance artifact, not a property of EP. The summary names this as one of
   three things that could change the verdict.
3. **One EP readout family.** Everything scored was `EP-FLAG` (region → smoothed
   positive rate) or one-hot ridge. Gate 1B already proved this matters more
   than any hyperparameter — switching from distance to occupancy flipped the
   verdict on the same dictionary and the same data. The graded,
   SAE-comparable readout has never been run.
4. **The metric.** Both papers report AUROC almost exclusively; KE25 explicitly
   praises it as "agnostic to classification thresholds". That is precisely
   what conceals our A2b result. Nobody deploys a monitor without a threshold.

---

## 3. The three factors this study varies that no prior EP work has

Everything below is organised as a factorial over:

- **F1 — readout** (7 EP arrows, §4)
- **F2 — dictionary provenance** (3 levels, §6)
- **F3 — regime** (7 levels, §7)

with the task suite (§5) providing the breadth that Gate 2 lacked, and the
matched-K coreset threaded through every cell as the mandatory control.

---

## 4. F1 — the EP arrows

Notation: centre `c`, exemplar matrix `E ∈ R^{K×d}` (unit rows), cosine
threshold `τ`, `x̂ = unit(x − c)`, `s = E x̂ ∈ R^K`, `r = argmax s`.

| id | definition | params | analogue |
|---|---|---|---|
| `EP-FLAG` | smoothed positive rate of region `r`: `(h+α)/(n+2α)`, α=0.5 | K counts | Gate 2 incumbent |
| `EP-ONEHOT` | L2 logistic on the K-dim one-hot of `r` | K | Gate 2 `EP-RIDGE` |
| **`EP-CODE`** | **`z_i = relu(s_i − τ)`**, then top-Q by mean-abs class difference, L1 logistic | Q | **direct SAE-probe analogue** |
| **`EP-MARGIN`** | `EP-FLAG` ⊕ `s_(1) − s_(2)` ⊕ `s_(1)` | K + 2 | **no SAE counterpart exists** |
| `EP-POOL` | `EP-CODE` max-pooled over tokens | Q | Bricken / TM25 |
| `EP-OCC` | window occupancy surprisal `−log p_ref(r)` | K | Gate 1B (population, not per-request) |

Three of these are new and each targets a specific diagnosed failure:

- **`EP-CODE` is the honest head-to-head.** `relu(s − τ)` is a JumpReLU encoder
  with a **fixed shared threshold**, an **untrained** encoder matrix whose rows
  are real observed activations, and a tied decoder. That makes EP an SAE-shaped
  object and the comparison to a Gemma Scope probe becomes apples-to-apples:
  same top-Q-by-mean-difference selection, same L1 logistic, same k. **Report
  measured L0** (median count of `s_i > τ`) — it is the comparability knob, and
  `ep-cell-shell` says ~40% of members sit within 0.02 of a rival, so L0 > 1 is
  the norm and must be quantified, not assumed.
- **`EP-MARGIN` is the differentiator argument.** Per `ep-cell-shell`, an SAE
  feature has no rival, so a member has nothing to be *contested by*. The
  top1−top2 margin is a quantity the SAE idiom cannot express. `ep-cell-shell`
  measures 39–46% of members within 0.02 of reassignment, so the channel is
  live, not a formality.

### Rejected arrow, recorded so it is not re-proposed

`one-hot(r) ⊕ (x̂ − e_r)` ("region plus how far off-centre") is **not** a
capacity increase. With a linear head it collapses:

```
a_r + w·(x̂ − e_r)  =  w·x̂ + (a_r − w·e_r)
```

— the ordinary probe with a free intercept per region. The hyperplane never
rotates, so it cannot separate anything a single direction cannot, and the
exemplar's arbitrariness (`ep-region-not-a-direction`: the exemplar is a
first-arrival accident) is absorbed into a free parameter and contributes
nothing. Getting real capacity would need a *separate* hyperplane per region,
which needs labels per region — the exact scarcity Gate 2 A1 diagnosed.

A per-region intercept is, however, a piecewise-constant **calibration**
correction. Carry it into regime 6 (threshold drift), not the accuracy arm.

---

## 5. Baselines — three tiers, all mandatory

**Tier 1 — KE25's quiver** (the accuracy bar): logistic regression (L2), PCA+LR,
KNN, XGBoost, MLP, each with 10 hyperparameter settings tuned on validation;
plus **attention-pooled probe** (KE25 Eq. 2) whenever any pooled arrow is in
play; plus **difference-in-means**, which is the method that actually beat
EP-FLAG at n=16 in Gate 2 A1 and which AxBench found hard for unsupervised
dictionaries to beat.

**Tier 2 — partition controls** (the EP-specific bar, and the one that decides
most cells):

- **matched-K random coreset**, ≥5 independent draws, report draw sd; *a win
  inside one sd is a tie*. This is stricter than any control in either paper:
  it matches memory, inference FLOPs, the space, and the build data exactly —
  the only difference is leader-clustering vs. sampling.
- **k-means at matched K.** `ep-occupancy-monitor` found EP wins Gate 1B with a
  *worse*-conditioned histogram (gini 0.509 vs. 0.398), so the active
  ingredient is boundary placement, not balance. k-means is the other
  boundary-placement method and is now the sharp control, not a redundant one.
- **ITDA at matched K** — the published training-free observed-activation
  dictionary. If ITDA matches EP everywhere, the contribution is a comparison
  of selection rules, and the write-up should say so.

**Tier 3 — free / no-dictionary** (the humility bar, all of which have already
beaten EP at least once): max next-token entropy, Mahalanobis, prompt length as
a triviality control, and TM25's **zero-shot prompted model output**.

Gate 0B's lesson stands: prompt length gave AUROC 0.998 on two rungs. Every
table carries a triviality control or it is not reportable.

---

## 6. F2 — dictionary provenance (the factor most likely to move the verdict)

This is EP's version of TM25's prompting axis and of Bricken's
fine-tune-the-SAE-on-task-data. Three levels, built identically apart from the
activation stream:

| level | build stream | cost |
|---|---|---|
| **P-PILE** | hub dictionaries, raw web text (status quo — every prior negative) | free |
| **P-TRAFFIC** | chat-formatted, task-agnostic traffic | one extraction + build |
| **P-PROMPTED** | activations under TM25's prefix+suffix prompted-probing template | one extraction + build per template |

**Preregistered prediction:** EP's coreset-relative margin rises monotonically
P-PILE < P-TRAFFIC < P-PROMPTED, and the template-detector pathology
(single-region collapse under chat wrappers) disappears at P-TRAFFIC. Rationale:
Gate 2 finding 5 is a distribution-mismatch artifact, and the one prior EP
success (`gate.json`, 404 members at 74.3% harmful) used a dictionary built on
the labelled prompts.

**If the prediction fails, that is the strongest possible negative** and it
retires the "it was only a transfer result" defence permanently. Build it either
way. EP builds require no gradient step, which is the entire reason this factor
is affordable for EP and would not be for SAEs.

---

## 7. F3 — regimes

KE25's four, plus TM25's shift axis, plus two that neither paper runs and that
target where EP's evidence actually points.

1. **Standard** — full data, balanced. Expect EP to lose; run it for the
   quiver ΔAUC baseline and to detect pipeline bugs.
2. **Data scarcity** — n log-spaced [2, 1024], 20 draws. *Gate 2 A1 already ran
   this on 5 saturated tasks and found no crossover; the point of re-running is
   the suite in §8, where there is headroom.*
3. **Class imbalance** — positive ratio 0.05→0.95, train and test matched.
   **Never run on EP.** Prediction below.
4. **Label noise** — 0→0.5 flipped, clean test set. **Never run on EP.**
   Head-to-head (validation is corrupted, per KE25).
5. **Covariate shift** — two families: KE25's dataset transforms (translation,
   name perturbation, GLUE-X) and TM25's boundaries (English→non-English,
   single-turn→multi-turn), plus our jailbreak suite. Head-to-head.
6. **Threshold drift (new, EP's best card).** Fit at a fixed operating point on
   clean data, freeze, measure **FPR under shift**. Neither paper measures
   this; AUROC is defined to hide it. Gate 2 A2b: ridge +0.484, diffmean
   +0.033, EP-FLAG +0.014, coreset +0.050 — EP won against both the probe and
   the null, on thin evidence (one coreset draw, eight attacks) that this study
   is sized to replicate properly.
7. **Build variance (new, mandatory, adversarial to EP).** EP's claimed
   differentiator over SAEs is that exemplars are observed activations, so the
   basis does not move under retraining seed. `ep-occupancy-monitor` falsified
   this: two builds **from the same 60k activations in different order** give
   per-request surprisal correlated at only **0.15–0.44**, and K itself grows
   with build size. Report every arrow's score correlation and *decision*
   agreement across ≥3 stream orders. Coverage/θ-ball overlap is a vacuous
   stability statistic (it returns 1.000) — do not use it.

Regimes 6 and 7 are the ones that make this a contribution rather than a
replication with a different dictionary.

---

## 8. Task suite

**Adopt KE25's 113 datasets wholesale** via `sae-probes/sae-probes`. This is the
single highest-leverage change relative to everything in Gates 0–2, and it costs
one extraction pass rather than any new design work.

- **Primary model: Gemma-2-9B, layer 20**, matching KE25 so their published
  baseline and SAE numbers are directly comparable and we are not re-deriving
  their quiver. Gemma Scope gives a real SAE arm at the same layer.
- **Secondary: `gemma-2-2b-it` L20**, where all our hub dictionaries, coresets
  and eight-attack machinery already exist. Every Gate 0/1/2 number is a free
  prior for it.
- **Stratify the suite before any EP number is computed**, on the baseline LR
  AUC at n=1024: `saturated` (≥0.98), `headroom` (0.85–0.98), `hard` (<0.85).
  The headroom and hard strata are where any EP claim must live; the saturated
  stratum exists to prove we reproduce Gate 2's negative.
- **Add TM25's SimpleQA hallucination task.** On-policy, auto-labelled,
  question-disjoint splits — and it is the one task where TM25 found *every*
  probing method beat the model's own zero-shot answer, i.e. the internal state
  genuinely carries something the output does not. Best available candidate for
  a non-linear concept.

---

## 8b. Pre-vetting — which concepts is EP even eligible for?

A first-class arm, not a post-hoc filter. It is what turns "EP wins on 9 of
113" from cherry-picking into a claim.

### The hypothesis

Region identity predicts a label only when the concept **occupies its own
volume** of activation space, and carries nothing when the distinction is a
**direction inside a shared volume**. Gate 2's measured basis, at p=4, counting
regions ≥90% one class with ≥5 members:

| task | partition | % of positives covered |
|---|---|---|
| language ID | **EP** | **0.657** |
| language ID | coreset | 0.498 |
| code vs. math | EP | **0.126** |
| code vs. math | coreset | 0.655 |

EP allocates dedicated regions to Bulgarian and fails to allocate any to
code-vs-math; a density-proportional random sample does the reverse.

### Scope — this criterion governs the classifier arm only

It is a property of the *concept*. It gates `EP-FLAG`/`EP-CODE`/`EP-ONEHOT` —
anything asking a region to predict a label. It does **not** gate `EP-OCC`
(population drift) or the calibration regime, which partition the input
distribution and never ask a region to carry a label.

### Candidate criteria — all label-free, all computed on activations alone

Each is a scalar per dataset, computed from `X` (and, where noted, the training
labels only — never the test labels, never any EP output):

1. **1-NN vs. linear** — test AUC of a 1-nearest-neighbour classifier minus
   that of an L2 logistic probe, both at n=256. Positive ⇒ the concept is not
   one direction. The most direct operationalisation.
2. **Class multi-modality** — number of connected components of each class's
   k-NN graph, or BIC-selected Gaussian-mixture component count per class.
   Volumetric concepts should be lumpy.
3. **Scatter ratio** — trace of between-class covariance over trace of
   within-class covariance. Low ⇒ the classes overlap in bulk and differ by
   direction.
4. **Region purity headroom** (EP-flavoured but label-cheap) — with 64 labels
   only, the fraction of labelled positives falling in regions that are ≥90%
   positive. This is the mechanism itself, measured at a budget too small to
   constitute the result.

Criterion 4 is the informative one and also the riskiest: it touches the EP
partition, so it must be computed on the fit pool and reported separately from
1–3, which are EP-agnostic.

### Protocol — split-half, declared before any EP score is seen

1. Randomly split the 113 datasets into **vet-fit (57)** and **vet-test (56)**,
   seed recorded here before extraction.
2. On vet-fit only: compute criteria 1–4, compute EP's coreset-relative margin,
   fit a single-threshold rule per criterion (and one logistic combination).
3. **Freeze the rule.**
4. On vet-test: predict eligible / ineligible per dataset *before* looking at
   EP's score, then reveal.
5. Report EP's performance on **both** predicted strata. A table showing only
   the eligible stratum is not a result.

### Decision rule 8b

- **PASS** — on vet-test, the eligible stratum's mean EP-vs-coreset margin
  exceeds the ineligible stratum's by more than the coreset draw sd, **and**
  the eligible stratum is at least 8 datasets (below that the rule is selecting
  noise).
- **FAIL** — no criterion separates the strata out of sample. Then EP's wins,
  if any, are unpredictable in advance, and the correct write-up says exactly
  that: a method that sometimes wins and cannot be told when is not deployable
  and its wins should be treated as sampling luck until shown otherwise.

### What a PASS buys

The headline stops being an accuracy claim and becomes:

> On the *N* of 113 datasets flagged in advance by a label-free criterion,
> a training-free partition reaches within *X* AUROC of a tuned probe at
> ~1% of an SAE's training cost; on the remaining datasets it is *Y* behind,
> as predicted.

That is a claim about *when partition methods work*, it generalises past EP to
every hard-assignment method, and it is the kind of result KE25 explicitly says
the field lacks.

---

## 9. Metrics and the cost table

Per cell, report **all five**:

1. **Test AUROC** — comparability with both papers.
2. **TPR@1%FPR** — deployment. Gate 0B's AUROC 0.65 was TPR 0.03; reporting one
   without the other is how that result nearly survived.
3. **Quiver ΔAUC** — the decision-theoretic number: best-by-validation test AUC
   with the EP arrow in the pool minus without. Percentile/θ is selected on
   validation like every other hyperparameter.
4. **FPR drift** at a frozen operating point (regime 6).
5. **Cross-build decision agreement** (regime 7).

And a **cost table**, TM25-style, because it is the axis on which EP is
strongest and it is the honest home for a method that will not win on accuracy:

| method | train-time compute | labels for the basis | inference cost | basis stability |
|---|---|---|---|---|
| SAE probe | SAE pretraining (GPU-days) | none | encoder matmul | seed-dependent |
| ITDA | ~1% of SAE | none | matching pursuit | — |
| **EP** | **one forward pass + O(NK) clustering, no gradient** | **none** | **one K×d matmul + argmax** | **order-dependent — measure it** |
| LR probe | seconds | n per task | one dot product | closed-form, deterministic |
| zero-shot | none | none | **extra forward pass** | n/a |

A claim of the form "EP reaches X% of the probe's AUROC at Y% of the training
cost, on tasks with property Z" is defensible and publishable. A claim that EP
beats probes is not, and three gates of evidence say so.

---

## 10. Protocol and controls

- **Three disjoint pools** — fit / validate / evaluate. Region selection, θ
  selection and arrow selection all happen on validate; eval is touched once.
- **Quiver of arrows for standard / scarcity / imbalance**; **fixed
  head-to-head for label noise and covariate shift** (validation is unfaithful
  there — KE25's own carve-out).
- **≥5 coreset draws everywhere**, sd reported, ties declared. Gate 0B's single
  draw credited a baseline for sampling luck; Gate 2 A2b's single draw is the
  reason the best EP result in the programme is still called thin.
- **Any pooled arrow requires the attention-pooled baseline in the same table.**
  KE25 §5 is the cautionary tale and it applies verbatim.
- **Triviality control** (prompt length, class prior) in every table.
- **Preregister decision rules before extraction**, in this file, as the
  Gate 2 plan did. The single-percentile rule is now subsumed by validation
  selection, so state it as: *θ is chosen on validation; the full sweep is
  reported; no claim rests on a percentile that validation did not pick.*
- **Report the stratum**, always. "EP wins 8 of 113" is uninterpretable without
  knowing whether those 8 are hard or saturated.

### Preregistered decision rules

- **PASS (niche established)** — on the `headroom`+`hard` strata, at P-TRAFFIC
  or P-PROMPTED, some EP arrow has **quiver ΔAUC > 0** with a CI excluding 0,
  **and** beats the matched-K coreset by >2 draw-sd, **and** the winning
  stratum is predicted in advance by the volumetric/directional criterion (§11).
- **PASS (deployment niche)** — EP loses on AUROC but wins regime 6 (FPR drift)
  against ridge, diffmean **and** coreset, replicated over ≥5 draws and ≥8
  shifts. This is a real result and must be written as *exactly* that: a
  calibration property, not detection accuracy.
- **FAIL (partition worthless)** — no EP arrow ever beats its coreset twin.
- **FAIL (no niche)** — EP arrows beat the coreset but quiver ΔAUC ≤ 0
  everywhere. This is KE25's own verdict on SAEs and is a perfectly good
  outcome to publish.
- **VOID** — if regime 7 shows cross-build decision agreement is low wherever
  EP wins, the win is a property of one build and no verdict is claimable. Run
  regime 7 on any positive **before** writing it up.

---

## 11. Where EP might actually have promise

Ranked by strength of existing evidence. Each is stated as a falsifiable
prediction, not a hope.

### 1. Volumetric concepts — the one mechanism with a measured basis

Gate 2's covering-vs-density asymmetry is the most useful thing the programme
produced:

> EP places exemplars to **cover the support**, so it wins when a class occupies
> its own distinct volume of activation space, and loses when the distinction
> lives *inside* a dense region, where sampling ∝ density puts more exemplars in
> the contested area.

Measured: on language ID at p=4, EP's ≥90%-pure regions cover **65.7%** of
positives vs. the coreset's 49.8%; on code-vs-math the ordering inverts
(**12.6%** vs. 65.5%).

**This is a prediction, and the 113-dataset suite is the instrument that can
test it.** Protocol, criteria and decision rule are §8b. A confirmed predictor
of *when a partition beats a hyperplane* is a better contribution than a win —
it is exactly the kind of result KE25 says the field is missing, and it
generalises past EP to every hard-assignment method.

Corollary that keeps this honest: `115_nyc_borough_Manhattan`,
`125_world_country_Italy`, `155_athlete_sport_basketball` are entity/geography
tasks — plausibly volumetric. Sentiment and entailment are plausibly
directional. If the criterion just recovers "EP does topic, not semantics",
say so plainly.

### 2. Fixed-threshold calibration stability under attack

The strongest deployment-relevant asymmetry we have, and neither paper measures
it. A ridge probe with **AUROC ≈ 1.000 on every attack** still ten-times its
false-alarm rate when the threshold is fit on plain traffic (+0.484); a region
flag has no threshold to drift (+0.014, beating both diffmean and the coreset).

Why this is EP-shaped rather than incidental: a discrete symbol has no scale, so
a monotone distortion of the activation that shifts every score cannot move the
decision unless it crosses a boundary. That argument applies to *any* hard
partition, so the coreset and k-means controls are what determine whether it is
EP or hardness that wins — and EP already beat the coreset once here.

Needs: proper draws, more shift families than jailbreaks, and TM25's
English/non-English and single-turn/multi-turn boundaries, where an attack
framing does not apply and the shift is benign.

### 3. Label noise and class imbalance — untested, and the priors favour EP

`EP-FLAG` is a smoothed count estimator: high bias, low variance,
`(h+α)/(n+2α)`. Label noise at rate ε shrinks every bin's estimate toward 0.5
without changing the *ranking* of bins — a monotone contraction, and AUROC is
rank-based. A logistic hyperplane, by contrast, has its decision boundary
rotated by noisy points. Same argument for imbalance: bin ranking is invariant
to the class prior, whereas an unregularised logistic fit at 5% positives is not.

This is the clearest untested case for the lookup, it is two of KE25's four
regimes, it costs nothing beyond the extraction we are already paying for, and
it has a clean failure mode: if EP's advantage under noise is matched by the
coreset, it is a property of binning and the correct write-up is "discretisation
helps under label noise", not "EP helps".

### 4. Zero-training-compute dictionaries at frontier scale

TM25's recommendation table has a cell — *train-time compute unavailable,
inference-time compute constrained* — where SAEs are excluded by cost and
zero-shot is excluded by latency. EP needs **no gradient step**, and this repo
has already built dictionaries on a 27B model with a target-K search
(`ep-target-k-search`) and on Modal (`ep-modal-dicts-27b`). ITDA is the
incumbent in that cell and its selling point is precisely 1%-of-SAE cost.

The claim to test is not accuracy but **accuracy per unit of training compute**,
with the cost table of §9 as the deliverable, and ITDA as the method to beat.

### 5. Interpretability-side utility, evaluated the way KE25 evaluates it

KE25 §4 finds SAE latents surface spurious correlations and dataset
mislabelling — and then shows a logistic-regression probe applied to Pile tokens
finds the same things. The lesson is that the interpretability pitch needs the
same baseline discipline as the accuracy pitch.

EP's dashboard already ships two channels with **no SAE counterpart**: the
contested-membership margin (`ep-cell-shell`) and the runner-up competition
graph (`ep-competition-graph`: 3.08× semantic enrichment under an IDF-weighted,
frequency-matched null; 37× structural enrichment at K=5190; and *not* the
geometric neighbour — agreement 3.1–4.5%, median geometric rank 30–68). Those
survive K ≪ d, unlike everything in `ep-voronoi-highdim`.

Test them KE25-style: can EP's margin channel flag the mislabelled CoLA
examples, the AI-vs-human punctuation artifact, and the `living_room`
English-only latent failure — **and can a logistic probe do it too**? If the
probe matches EP everywhere, report that; it is the same finding KE25 reported
about SAEs and it is worth stating.

### 6. Cross-model / cross-checkpoint diffing (adjacent, and better supported)

Not monitoring, but worth naming because ITDA's headline win is here and this
repo already has the positive control: `ep-rmu-diff-gate1a/1b` found a 50:1
region-formation signal and a stability inversion under RMU unlearning. Note
the hard constraint from Gate 2 C2: **cross-layer** correspondence is *below*
random, so any cross-model claim must be same-layer, on a shared activation
stream, with a same-model reseed control run through the identical procedure.

---

## 12. Where EP will not have promise — do not spend here

- **Beating probes on accuracy in standard conditions.** Gate 2 A1: no
  crossover on 5 tasks at any of 7 budgets over 20 draws. KE25 found the same
  for SAEs in its easiest regime. Run the standard arm only for ΔAUC context.
- **A stable discrete coordinate system across layers.** Gate 2 C2 killed it:
  MI below the coreset at −3.3 sd, stable across a 12× sample sweep.
- **Distance-to-exemplar as a novelty score.** Gate 0B: it flips sign across
  equally-novel templates (0.885 → 0.226) and detects character soup while
  sitting at chance on semantic attacks — backwards for safety use.
- **Per-request drift detection from occupancy.** Gate 1B: TPR@1%FPR is 0.000
  in every cell. Population instrument only; frame it as such.
- **Multi-token trajectories.** Gate 2 B: one symbol at the final position gives
  0.958, twelve symbols give ≤0.66, and the order gain is zero. If pooling is
  revisited it should be `EP-POOL` (max over per-token codes, TM25-style), not
  sequence modelling.
- **θ-ball overlap as a stability statistic.** Returns 1.000 by construction.

---

## 13. Staging

| stage | content | gate |
|---|---|---|
| 0 | Port `sae-probes` extraction; stratify the 113 by baseline LR AUC; **record the 8b vet-fit/vet-test split seed**; reproduce KE25's Gemma-2-9B L20 baseline quiver | our baselines must match theirs before any EP number counts |
| 1 | `EP-CODE` + `EP-FLAG` + `EP-MARGIN` at P-PILE, standard + scarcity, headroom/hard strata, ≥5 coreset draws | if no arrow beats its coreset anywhere → FAIL (partition worthless); stop |
| 2 | Build P-TRAFFIC and P-PROMPTED dictionaries; repeat stage 1 | the monotonic provenance prediction of §6 |
| 3 | **Pre-vetting (§8b)**: criteria on vet-fit, freeze rule, predict vet-test, reveal | decision rule 8b |
| 4 | Label noise + class imbalance (§11.3) | cheap, untested, priors favour EP |
| 5 | Covariate shift + **threshold drift** on gemma-2-2b-it, where the attack machinery exists | replicate A2b properly |
| 6 | Build variance (regime 7) on every positive so far | VOID any win that does not survive |
| 7 | Cost table, write-up | |

Stages 0–1 decide whether the rest is worth running. Stage 5 is not optional:
`ep-occupancy-monitor` has already caught one EP "win" that was a property of a
single build ordering.

---

## 13b. Out of scope — considered and dropped

**EP as scaffolding around a probe** (coverage-based selection of which
examples to label; region-balanced or worst-region reweighting; region-grouped
cross-validation splits). Each is a real technique, but in each the load-bearing
ingredient is *having a partition*, not having EP's partition — k-means on the
same activations supplies groups equally well, more cheaply, and with a
standard citation. Two further problems: uncertainty-based active learning
dominates one-shot diversity sampling except at cold start, where k-center
already occupies the niche; and `ep-occupancy-monitor` measured EP's grouping as
*more* lopsided than random (gini 0.509 vs. 0.398), which is the wrong direction
for group-balancing to want. The matched-K coreset would almost certainly tie.

Dropped because the defensible claim it supports — "a dictionary you built for
interpretability also throws these in" — is a convenience argument, not a
finding, and pursuing it would dilute the head-to-head this document exists to
run.

---

## 14. The honest summary of the bet

Three gates say EP loses as a detector. Both papers say the same about SAEs, and
KE25 in particular says it after 113 datasets and five tuned baseline families —
which is why its negative result is publishable and ours currently is not.

The bet this design makes is that **the prior negatives were run on tasks with no
headroom, with a dictionary built on the wrong distribution, using the one
readout that discards the graded information EP has**, and that fixing those
three things either produces a narrow, characterised, mechanistically predicted
niche — volumetric concepts, noisy labels, frozen thresholds, zero training
compute — or produces the strongest negative result available about
training-free partition methods.

### Stated in advance, so it cannot be softened later

**The expected outcome is that EP does not beat a tuned probe on accuracy,
anywhere.** Three gates say so and nothing in this design changes the
underlying arithmetic: a probe needs its direction only roughly right to
*rank*, which 16 labels supply, while a lookup needs its bins populated, which
scales with K. The graded readout narrows this; it does not close it. Any
version of this study whose success condition is "EP wins" is mis-specified.

The success conditions are therefore: **(i)** §8b separates eligible from
ineligible concepts out of sample; **(ii)** the eligible stratum lands inside a
pre-declared tolerance band of the tuned probe; **(iii)** EP holds its
threshold under shift where the probe does not.

**Fix the tolerance band now, in this file, before extraction** — proposal:
*median gap ≤ 0.03 AUROC to the tuned-quiver winner on the eligible stratum,
with EP > matched-K coreset by >2 draw-sd on the same datasets.* Without a
number written down first, "cheap alternative" becomes the place every negative
result goes to be reframed, and three gates of careful work lose their
credibility retroactively.

### Scope the cheapness claim correctly

**EP is not cheaper than a probe.** A probe is seconds of CPU and needs no
dictionary at all. EP is cheaper than an *SAE*, and the comparison is only
meaningful when the dictionary's other outputs — regions, attached member
texts, the competition graph — are wanted anyway. Every occurrence of "cheap
alternative" in the write-up must read "cheap alternative to a trained
dictionary", never "cheap alternative to a probe", which is false and
disposable in one sentence by a reviewer.

Both outcomes are worth the extraction pass. Neither requires EP to win.
