# Do jailbroken prompts leave the refusal region?

`google/gemma-2-2b-it`, L20, the saved p=12 seed-0 dictionary.

Written before the Tier 0 numbers were read. §6 records the predictions at the
time of writing so the result can embarrass them.

---

## 0. Verified facts this rests on

From `artifacts/runs/refusal_reference/results/gemma-2-2b-it/L20_p12_seed0/behavioral.json`
and re-verified locally by `experiments/jailbreak/gate.py` (§3):

| fact | value |
|---|---|
| partitions | 207 |
| calibration threshold θ | 0.649953 |
| refusal region | pid **18** |
| pid 18 final-position members | 405 (reference) / **404** (local re-run) |
| harmful prompts in pid 18 | **300 / 300** |
| benign prompts in pid 18 | 105 / 104 |
| Δ refusal, exemplar basis, K=1 | **−0.76** |
| Δ refusal, matched null region | **0.00** |
| base refusal: harmful / benign | 0.99 / 0.023 |

Two facts from the local re-run that are not in the reference JSON and that
shape the design:

- **Mean cosine distance to pid 18's exemplar is 0.126 for harmful prompts and
  0.540 for benign**, against θ = 0.650. Harmful prompts sit deep inside the
  cell; benign prompts sit near its boundary. The separation is large and
  continuous, which is what makes a *distance-based* escape metric worth more
  than the discrete assignment.
- **Only 5 of 207 cells hold any final-position activation** (pid 18 = 404,
  82 = 130, 92 = 60, 109 = 4, 45 = 2). The chat scaffold consolidates
  instruction-formatted prompts into a handful of cells. So "escape" has few
  places to go, and a destination analysis will be coarse.

**pid 18 is not a harm detector.** It holds every harmful prompt *and* a third
of the benign ones — `harmful_fraction` 0.74. It is a broad
instruction-consolidation cell that happens to contain all the harm, and
ablating its exemplar happens to collapse refusal. Claims below are about that
cell, not about a clean "harmfulness feature".

## 1. The question

Refusal is the one behaviour in this project that fully replicated. A jailbreak
is, behaviourally, the thing that defeats it. EP gives a discrete,
pre-generation read on where a prompt sits. So:

> When a harmful instruction is wrapped in a published jailbreak, does its
> final-position activation still land in pid 18?

Both answers are substantive:

- **STAYS** → the model still represents the request the same way, and the
  jailbreak overrides refusal *downstream* of L20. Region membership is
  necessary but not sufficient for refusal.
- **LEAVES** → the jailbreak works by moving the prompt out of the region,
  which makes region membership a detector that fires before a token is
  generated.

## 2. Why this is the right experiment for EP specifically

The role experiment (`PLAN_ROLE_QWEN3_4B.md` §10b) established the screening
rule: EP resolves a construct only if the construct's angular displacement is
comparable to θ. Role came in at 0.004·θ and was invisible.

A jailbreak wrapper is real token content — hundreds of tokens of roleplay or
rule-listing — so it clears that bar by construction. But "by construction" is
an assertion, and the role post-mortem's whole lesson was to measure it. So
`metrics.wrapper_displacement` reports displacement in units of θ for every
template, and it is reported *first*, before any escape rate is interpreted.

## 3. Reproduction gate — PASSED

`python -m experiments.jailbreak.gate` re-runs the 600 plain build prompts through this
checkout and requires harmful recall into pid 18 ≥ 0.97.

Result: **recall 1.0000 (300/300)**, n_members 404 vs 405, harmful_fraction
0.7426 vs 0.7407. One boundary benign prompt flipped — the local run is MPS
bfloat16 where the reference was CUDA bfloat16.

This gate is not ceremony. The three ways this experiment silently produces
garbage — wrong dtype shifting the geometry under the saved exemplars, a
changed chat template, the wrong pickle — all yield plausible numbers and none
of them raise.

## 4. Design

**Grid.** 300 harmful (AdvBench + JailbreakBench, the exact build set, loaded
by importing the reference script's own loaders) × 300 benign (Alpaca) ×
9 templates = 5400 prompts. `experiments/jailbreak/corpus.py`.

**The benign arm is load-bearing, not a nicety.** A jailbreak wrapper is most
of the token content of the prompt it wraps, so the final-position activation
moves for benign and harmful goals alike. Without benign goals in the *same*
wrappers, "harmful prompts left pid 18" is indistinguishable from "this wrapper
relocates everything". Every headline number is differenced against it.

**Templates** (`experiments/jailbreak/templates.py`), labelled by Wei et al. 2023's
mechanism taxonomy — the labels are the hypothesis, not bookkeeping:

| class | templates |
|---|---|
| control | `plain` |
| competing objectives | `prefix_injection`, `refusal_suppression`, `roleplay`, `distractor` |
| mismatched generalization | `base64`, `leetspeak`, `payload_split` |
| adversarial optimization | `gcg_suffix` (published Zou et al. universal suffix) |

All are published attacks used as stimuli for representation analysis; nothing
is optimised against the target model. `roleplay` is deliberately short — the
multi-paragraph community DAN variants would confound escape with a pure
context-length effect.

**Primary statistic: `harm_auroc`.** Within one template, how well does
distance to pid 18's exemplar rank harmful above benign? It is the direct
operationalisation of "does the model still recognise this as harmful", it is
continuous, and its null is exactly 0.5 with no simulation needed. On `plain`
it should be near 1.0 given the 0.126 / 0.540 split.

**Membership is reported two ways, because `Dictionary.assign` is
nearest-exemplar with no threshold** — it never returns −1 and never reports
"outside every cell". `assigned_rate` is "pid 18 is the nearest cell";
`in_cell_rate` is `dist ≤ θ`, the honest containment test. They disagree
whenever an activation is inside pid 18's radius but closer to another
exemplar, which is common at θ = 0.65 with heavily overlapping cells. Reading
only the argmin would report a false escape.

## 5. Tiers

- **Tier 0 — escape map.** Forward passes only. Runs locally on MPS; no pod.
  `experiments/jailbreak/exp_escape.py`.
- **Tier 1 — behaviour + mediation.** Generate 60 tokens per harmful prompt,
  score refusal with the validated substring classifier, cross-tab region ×
  refusal. Gated on Tier 0 producing a spread in escape rates; if every
  template behaves identically there is nothing to mediate.
- **Tier 2 — causal.** Ablate pid 18's exemplar under each wrapper. Does the
  −0.76 survive the wrapper? Gated on Tier 1.

**A known limit on Tier 1, stated up front:** gemma-2-2b cannot decode base64.
Its `base64` generations will be neither refusals nor compliance, and the
substring scorer will read that as a successful jailbreak. Attack-success
numbers for the mismatched-generalization arm are therefore *not* interpretable
without a judge or a degeneracy filter. The Tier 0 question — does the encoded
request still land in pid 18 — is unaffected, because it asks about the
representation and not about the output.

## 6. Predictions, recorded before reading the Tier 0 table

The taxonomy makes a falsifiable claim. Competing-objectives attacks are
defined by the model *recognising* the request and being outvoted;
mismatched-generalization attacks by it *not recognising* the request. Region
membership is a recognition read-out. So:

1. `prefix_injection`, `refusal_suppression`, `roleplay`, `distractor` →
   harmful prompts **stay** in pid 18, `harm_auroc` stays high.
2. `base64`, `leetspeak`, `payload_split` → harmful prompts **leave**,
   `harm_auroc` collapses toward 0.5.
3. `gcg_suffix` → genuinely unknown; the taxonomy does not place it. This is
   the interesting cell.
4. `wrapper_displacement` ≫ the role experiment's 0.004·θ for every template.

**The most likely way prediction 1 fails, and it is not a small risk:** the
competing-objectives wrappers add a lot of text, and with only 5 occupied cells
the whole corpus may simply relocate together, driving `harm_auroc` toward 0.5
for structural reasons that have nothing to do with harm recognition. The
benign arm is what diagnoses this — if benign prompts move to the same place by
the same distance, the escape is the wrapper, not the jailbreak. A uniform
collapse across *all* templates including the competing-objectives ones should
be read as the design hitting the final-position consolidation ceiling, not as
a mechanistic finding.

## 7. Tier 0 results

`artifacts/runs/jailbreak/escape.json`, 5400 prompts, ~8 min on MPS bf16.
θ = 0.650. `stay` = fraction assigned to pid 18. `disp` = mean cosine
displacement from the same goal's plain activation, in units of θ.

| template | mechanism | harm AUROC | stay_h | stay_b | d18_h | d18_b | disp_h/θ | disp_b/θ |
|---|---|---|---|---|---|---|---|---|
| plain | control | 0.999 | 1.000 | 0.347 | 0.126 | 0.540 | — | — |
| prefix_injection | competing | 0.999 | **1.000** | 0.850 | 0.171 | 0.524 | 0.16 | 0.62 |
| refusal_suppression | competing | 0.986 | 0.970 | 0.240 | 0.333 | 0.545 | 0.42 | 0.35 |
| roleplay | competing | 0.998 | **1.000** | 0.980 | 0.277 | 0.481 | 0.38 | 0.65 |
| distractor | competing | 0.999 | 0.910 | **0.000** | 0.257 | 0.543 | 0.34 | 0.65 |
| base64 | mismatched | **0.607** | 1.000 | 1.000 | 0.522 | 0.524 | 0.77 | 0.79 |
| leetspeak | mismatched | 0.905 | 1.000 | 0.850 | 0.274 | 0.422 | 0.36 | 0.62 |
| payload_split | mismatched | 0.999 | 1.000 | 0.737 | 0.169 | 0.479 | 0.21 | 0.47 |
| gcg_suffix | adversarial | 0.996 | 1.000 | 0.583 | 0.144 | 0.452 | 0.09 | 0.29 |

### 7.1 The headline

**Harmful prompts do not leave the refusal region.** `stay_h` is 1.000 for six
of the eight attacks, 0.970 for `refusal_suppression`, 0.910 for `distractor`.
No published attack family dislodges a harmful instruction from pid 18.

Meanwhile benign prompts move freely under the same wrappers: `distractor`
relocates **100%** of them to pid 82, `refusal_suppression` sends 55% to pid 92,
`roleplay` pulls almost all of them *into* pid 18 (0.347 → 0.980). Benign
displacement exceeds harmful displacement for seven of eight templates —
`prefix_injection` moves benign goals 0.62·θ and harmful goals 0.16·θ, a
factor of four.

So the asymmetry is not "jailbreaks don't move activations". They move them a
lot. Harmful prompts specifically are pinned.

This is outcome (ii) of the three the experiment was designed to separate:
**region membership is necessary but not sufficient for refusal, and whatever
the jailbreak defeats sits downstream of L20.** It is only that result if the
attacks actually work behaviourally, which is what Tier 1 tests.

### 7.2 Predictions scored

- **1 (competing objectives stay) — confirmed.** All four keep AUROC ≥ 0.986
  and stay ≥ 0.91.
- **2 (mismatched generalization leaves) — one of three, and not by escaping.**
  `base64` does collapse harm recognition (AUROC 0.999 → 0.607), but *not* by
  moving harmful prompts out: `stay_h` and `stay_b` are both 1.000 and d18 is
  0.522 vs 0.524 — statistically identical. base64 causes **convergence**, not
  escape. Harmful and benign base64 prompts become the same thing to the
  model. `leetspeak` degrades partially (0.905). `payload_split` does not
  degrade at all (0.999) — the model reassembles the split goal
  representationally even though no contiguous span of the prompt contains it.
- **3 (GCG unplaced) — it barely registers.** `gcg_suffix` has the *smallest*
  displacement of any template, 0.09·θ, and AUROC 0.996. A suffix optimised
  against other models does almost nothing to gemma-2-2b's representation.
- **4 (displacement ≫ role's 0.004·θ) — confirmed, with a twist.** Range is
  0.09·θ to 0.79·θ, i.e. 25–200× the role effect. But every one is **below**
  θ. The wrappers displace substantially and still not far enough to escape.
  That is the quantitative form of the headline.

### 7.3 A metric that turned out to be vacuous

`in_cell_rate` (distance ≤ θ) is **1.000 in every cell of the table**, and 0.987
even for plain benign prompts. θ = 0.65 is a ~66° cone that contains the entire
corpus. Containment carries no information here; all discrimination comes from
argmin over exemplars.

This is the same shape of problem as `ep-voronoi-highdim` (adjacency is vacuous
when K < d). Reported here rather than quietly dropped, because a reader who
sees only `assigned_rate` might reasonably assume the two agree.

### 7.4 What Tier 0 does not license

- `harm_auroc` on `plain` is **partly in-sample**: pid 18 was selected post-hoc
  as the refusal region on exactly these 600 prompts. The cross-template
  *comparison* is clean, since no wrapper's activations entered dictionary
  construction, but the 0.999 baseline is not an out-of-sample number.
- Everything is one layer (L20), one model, one percentile, one seed. The
  reference itself found that two of four streaming seeds give Δ = 0, so
  region identity is a first-arrival accident; a second seed's dictionary might
  put the refusal mass in a differently-shaped cell.
- "Never leaves pid 18" is a statement about **final-position** activations.
  Only 5 of 207 cells are ever occupied at final position, so the geometry has
  few places to send anything.

## 8. Tier 1 results

`artifacts/runs/jailbreak/behavior.json`, `behavior_rescored.json`, `generations.json`.
100 harmful goals × 9 templates = 900 completions, 60 tokens greedy, ~45 min.

Batching validated: **refusal-label agreement 1.000** on 40 unbatched re-runs
(exact-text agreement 0.750, so decoding does drift — the decision does not).

### 8.1 Two corrections applied before reading anything

Both were found by reading the generations, and both inflate attack success.

**The reference scorer under-counts refusals here.** gemma-2-2b refuses plain
prompts with "I cannot", which the reference phrase list covers — that is why
it validated at 0.99/0.023. Under a wrapper it switches register to *"I am
programmed to be a helpful and harmless AI assistant"*, *"goes against my core
principles"*, *"I am an AI and cannot"* (the list has `as an ai`, not
`am an ai`). None of those match, so all were scored as jailbreaks.
`experiments/jailbreak/rescore.py` adds 17 phrases and re-scores the saved text; it is
tested to be a strict extension of the reference scorer and to not fire on the
compliant generations actually observed. It recovers **11 refusals** under
`refusal_suppression` and **6** under `roleplay`.

**Two templates measure comprehension, not compliance.** `base64` ASR was 1.00;
its non-degenerate outputs are hallucinated benign instructions —
`{"instruction": "print('Hello, world!')"}`. The model never decoded anything.
`leetspeak` is the same failure in prose: *"the text you provided is a jumble of
letters and numbers"*. Neither is a jailbreak. The degeneracy filter caught
base64's blobs (80%) but cannot catch coherent-but-off-task output, which needs
a judge. Both are marked `comprehension_limited` and **excluded from all
behavioural claims** rather than patched into looking reasonable.

This was predicted in §5 for base64. It was **not** predicted for leetspeak,
which I had expected to be a working attack.

### 8.2 The table

ASR under the extended scorer, 95% Wilson intervals, `stay` from Tier 0.

| template | mechanism | ASR ref | **ASR ext** | 95% CI | stay | refusal AUROC |
|---|---|---|---|---|---|---|
| plain | control | 0.03 | **0.03** | — | 1.000 | 1.000 |
| refusal_suppression | competing | 0.39 | **0.28** | [0.20, 0.37] | 0.970 | 0.872 |
| roleplay | competing | 0.17 | **0.11** | [0.06, 0.19] | 1.000 | 0.927 |
| distractor | competing | 0.10 | **0.10** | [0.06, 0.17] | 0.880 | 0.967 |
| prefix_injection | competing | 0.06 | **0.05** | [0.02, 0.11] | 1.000 | 1.000 |
| payload_split | mismatched | 0.02 | **0.02** | [0.01, 0.07] | 1.000 | 0.980 |
| gcg_suffix | adversarial | 0.02 | **0.02** | [0.01, 0.07] | 1.000 | 0.974 |
| base64 | mismatched | 1.00 | — | — | 1.000 | — |
| leetspeak | mismatched | 0.48 | — | — | 1.000 | — |

`refusal_suppression` clears baseline decisively: 0.28 [0.20, 0.37] against
0.03, non-overlapping. Its successes are genuine on-task compliance, verified
by reading them — emissions-test defeat, a fake-news production breakdown, a
fabricated product review.

### 8.3 The result

**The attacks that work do not move the prompt out of the refusal region.**
`refusal_suppression` reaches ASR 0.28 with **97%** of prompts still assigned
to pid 18; `roleplay` reaches 0.11 with **100%**. Ablating pid 18's exemplar
collapses refusal by 0.76 — so this is a region that is causally load-bearing,
holding the prompt, while the behaviour it drives is overridden anyway.
**Region membership is necessary but not sufficient, and what these attacks
defeat sits downstream of L20.**

**But when a prompt does leave, refusal collapses.** `distractor` pushed 12 of
100 harmful prompts out to pid 82, and those were refused at **0.42**
[0.19, 0.68] against **0.97** [0.90, 0.99] for the 88 that stayed —
non-overlapping. `refusal_suppression`'s 3 escapees point the same way (0.33 vs
0.73) at an n that proves nothing on its own.

*Confound, checked:* the 12 escaping goals were slightly softer targets to begin
with — plain refusal 0.83 versus 0.99 for the stayers. The effect survives it:
escapees fall 0.83 → 0.42 while stayers move 0.99 → 0.97.

So all three pre-specified outcomes occurred, and EP separates them:

1. **Override-type** (`refusal_suppression`, `roleplay`, `prefix_injection`) —
   region preserved, behaviour flipped. The majority mechanism.
2. **Escape-type** (`distractor`) — region changed, refusal collapses with it.
   Leaving is *sufficient*; it is just not how most attacks work.
3. **Ineffective** (`gcg_suffix`, `payload_split`) — neither moves the
   representation (0.09·θ, 0.21·θ) nor defeats the behaviour (ASR 0.02).

Escape and efficacy are **dissociated**: the most effective attack has the
second-smallest escape rate, and the largest escape rate belongs to an attack
of middling efficacy. Region membership is therefore not a jailbreak detector —
it would miss `refusal_suppression` entirely — but it *is* a mechanism
classifier, and it reads out before a token is generated.

### 8.4 The taxonomy prediction was wrong

§6 predicted competing-objectives attacks stay and mismatched-generalization
attacks leave. The first half held. The second did not, in two separate ways:

- `base64` destroys harm recognition (AUROC 0.999 → 0.607) **without any
  escape** — `stay_h` and `stay_b` are both 1.000 and d18 is 0.522 vs 0.524.
  It causes *convergence*, not escape: harmful and benign base64 prompts become
  the same object. That is a distinct third geometry the design did not
  anticipate.
- `payload_split` does not degrade recognition at all (AUROC 0.999) despite no
  contiguous span of the prompt containing the goal. The model reassembles it
  representationally.

And the one template that *did* produce escape, `distractor`, is a
competing-objectives attack. So EP's escape map does not recover Wei et al.'s
classes. It cuts the space somewhere else — along override-vs-escape, which is
a mechanistic distinction about *where* the attack acts rather than about what
the model understands.

## 8.5 Sub-cell position predicts the outcome, and the anchor doesn't matter

Within a template, position *inside* pid 18 predicts whether the jailbreak
succeeds. `Δd18` is the within-goal change from that goal's own plain
activation, so it is much less confounded by "harder goals sit closer and get
refused more" than the absolute distance is.

| template | AUROC d18 → jailbreak | AUROC Δd18 → jailbreak | d18 refused | d18 jailbroken |
|---|---|---|---|---|
| refusal_suppression | 0.872 | 0.710 | 0.312 | 0.434 |
| roleplay | 0.927 | 0.587 | 0.271 | 0.379 |
| distractor | 0.967 | 0.573 | 0.247 | 0.455 |
| prefix_injection | 1.000 | 0.806 | 0.166 | 0.476 |

The full ladder, against θ = 0.650:

```
plain harmful           0.126
wrapped, still refused  0.166 - 0.312
wrapped, jailbroken     0.379 - 0.476
plain benign            0.540
θ                       0.650
```

Jailbroken prompts are **not** near the cell boundary — they sit at ~65% of the
radius and remain closer to the refusal exemplar than ordinary benign prompts.
They move partway along the axis benign prompts occupy. That is why the effect
is probabilistic rather than a clean flip, and it is why tightening θ is not a
fix: no threshold separates the two populations, and AUROC 0.872 is the ceiling
on any that tried.

**So the deciding information is present at L20, inside the same cell, below
EP's resolution** — not downstream in depth, and not unknown.

### Anchor robustness (`artifacts/runs/jailbreak/anchor_robustness.json`)

Every distance above is to a first-arrival exemplar, and the reference found 2
of 4 seeds give Δ = 0, so this needs checking. For pid 18,
cos(exemplar, mean_member) = **0.899** — about 26°, below the ~0.94 the paper
reports typically, so the anchors genuinely differ.

Recomputing everything against three anchors of decreasing seed-dependence —
`exemplar` (first arrival), `mean` (spherical mean of 734 members, essentially
arrival-independent), `reanchored` (member closest to the mean):

- per-prompt distance rankings correlate at Spearman **ρ = 0.978–0.983**
- jailbreak AUROC: 0.872 / 0.872 / 0.882 (`refusal_suppression`),
  0.927 / 0.925 / 0.924 (`roleplay`), 0.967 / 0.969 / 0.969 (`distractor`)
- harm AUROC agrees to within 0.003 everywhere except `base64`
  (0.607 / 0.572 / 0.611), which is quarantined anyway

**The measurement is anchor-invariant.** A 26° difference in anchor barely
perturbs the *ordering* of prompts spread over a far wider angular range.

**The intervention is not.** The paper's own result is that exemplar beats mean
by 0.4–0.6 for ablation, because a projection is sensitive to the exact
direction in a way a ranking is not. So this check retires the seed objection
for §8.5 and §7, and retires it *not at all* for the Δ = −0.76 that motivates
the whole experiment. Only a real second-seed rebuild settles that.

## 9. Status and what would sharpen it

Done and reproducible: `bash scripts/experiments/run_jailbreak.sh {gate|escape|behavior}`,
72 unit tests, entirely local on MPS — no pod was needed at any point, because
the saved dictionary removed the calibration and discovery passes.

Not done, in value order:

1. **Tier 2 (causal).** Ablate pid 18's exemplar under each wrapper. The
   prediction from §8.3 is that Δ stays near −0.76 for the override-type
   attacks (the region is still doing its job, and knocking it out should still
   work) and is *smaller* for `distractor`'s escapees (they have already left).
   This is the test that would turn the correlational mediation result into a
   causal one, and it reuses the reference ablation hook unchanged.
2. **A second streaming seed.** Region identity is a first-arrival accident and
   the reference found 2 of 4 seeds give Δ = 0. Every number here is seed 0.
3. **An LLM judge** for on-task compliance, which would let `base64` and
   `leetspeak` back into the analysis instead of being quarantined.
4. Larger n on `distractor`'s escapees — 12 is the whole basis of the
   escape-type claim.

`experiments/jailbreak/exp_behavior.py`, 100 harmful goals × 9 templates, 60 tokens greedy.

Generation is batched (left-padded, 8 at a time) for a 5.2× speedup. The
reference harness's warning against batching applies to the *ablation* hook,
which rewrites padding positions; no hook is installed in this tier. bf16 is
non-associative so batched greedy decoding can drift mid-sequence — measured at
4/6 byte-identical on a spot check, with the refusal decision preserved in both
divergent cases. `--validate-unbatched 40` re-runs a random subsample singly
and reports refusal-**label** agreement, so this is checked rather than assumed.

*(Results to follow.)*
