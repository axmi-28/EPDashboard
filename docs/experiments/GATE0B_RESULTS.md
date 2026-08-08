# Gate 0B — results

**H0 stands. EP's nearest-exemplar distance does not beat a random coreset at
matched memory budget, and is not usable as a runtime monitor at any resolution
tested.**

Run 2026-07-30. `gemma-2-2b-it` L20, final token position, bf16 on MPS.
12,000 eval prompts (6 rungs x 2000), 204,800 per-position reference
activations. Hub dataset revision `0ec26618db2bd64cdbf00a9a1b0eb23fffdea12a`;
per-p blob SHAs in `artifacts/runs/monitor/dicts/*.npz`. Full table:
`artifacts/runs/monitor/gate0b_results.csv` (150 rows), per-source breakdown
`gate0b_by_source.csv` (750 rows), paper-reproduction column
`gate0b_paper_repro.csv`. Figures `artifacts/figures/gate0b_auroc.png`,
`artifacts/figures/gate0b_tpr.png`. Verdict `artifacts/runs/monitor/gate0b_verdict.json`.

## 1. The decision rule, applied

Pre-registered in `docs/experiments/PLAN_EP_MONITOR.md` before any number was seen, and
evaluated mechanically by `evaluate_decision_rule` (14 dedicated tests).

| p | K | R2 | R3 | R4 |
|---|---|---|---|---|
| 1 | 5796 | LOSS | LOSS | win |
| 2 | 2037 | LOSS | win | win |
| 4 | 686 | LOSS | LOSS | win |
| 8 | 226 | LOSS | win | win |
| 10 | 176 | win | win | win |

**Rule 1: H0 stands.** EP beats the matched-K coreset on all three decision
rungs at exactly one of five percentiles. That is a win only under a search
over p, which the pre-registration forbids.

And the single winning configuration does not survive inspection. On R2 at
p=10 EP's margin over the coreset is **+0.016 against a coreset draw
standard deviation of 0.060** — a quarter of one sd, i.e. noise. The honest
statement is not "EP wins at p=10" but "at p=8 and p=10 EP ties the coreset,
and at p=1, 2 and 4 it loses by 16-24 draw-sd."

| R2 (cleanest rung) | p=1 | p=2 | p=4 | p=8 | p=10 |
|---|---|---|---|---|---|
| coreset − EP best | **+0.257** | **+0.290** | **+0.345** | +0.010 | −0.016 |
| in coreset draw sd | 23 | 24 | 16 | 1 | 0 |

**Rule 2: the monitoring framing is not dead by Mahalanobis.** S4 wins R2
(0.808) but is *below chance* on R3 (0.421), R4 (0.393) and R5 (0.138). A
covariance fitted on Pile finds random-token activations closer to the mean
than Pile itself. So S4 does not dominate; it is simply a different, also-poor
detector.

**Rule 3: rungs where everything fails.** R3 (template shift) is effectively
dead — the best score by any scorer at any p is S5 at 0.622, and EP's best is
0.574. Nothing separates a chat scaffold from raw Pile at the final token.

## 2. The full AUROC table

Negative class is R0 (Pile held-out, 2000) throughout.

| rung | scorer | p=1 | p=2 | p=4 | p=8 | p=10 |
|---|---|---|---|---|---|---|
| **R1** code+math | S1 EP nearest | 0.531 | 0.545 | 0.634 | 0.635 | 0.481 |
| | S2 EP margin | 0.535 | 0.549 | 0.579 | 0.476 | 0.528 |
| | S3 coreset kNN | 0.644 | 0.547 | 0.426 | 0.339 | 0.319 |
| | S4 Mahalanobis | 0.529 | — | — | — | — |
| | S5 entropy | 0.153 | — | — | — | — |
| | **S0 length** | **0.002** | — | — | — | — |
| **R2** Bulgarian | S1 EP nearest | 0.545 | 0.504 | 0.392 | 0.716 | 0.712 |
| | S2 EP margin | 0.488 | 0.456 | 0.405 | 0.621 | 0.608 |
| | **S3 coreset kNN** | **0.801** | **0.793** | **0.750** | **0.726** | 0.696 |
| | S4 Mahalanobis | 0.808 | — | — | — | — |
| | S5 entropy | 0.540 | — | — | — | — |
| | S0 length | 0.500 | — | — | — | — |
| **R3** scaffolds | S1 EP nearest | 0.574 | 0.532 | 0.328 | 0.476 | 0.483 |
| | S2 EP margin | 0.527 | 0.473 | 0.430 | 0.448 | 0.662 |
| | S3 coreset kNN | 0.578 | 0.482 | 0.470 | 0.428 | 0.384 |
| | S4 Mahalanobis | 0.421 | — | — | — | — |
| | S5 entropy | 0.622 | — | — | — | — |
| | S0 length | 0.499 | — | — | — | — |
| **R4** jailbreak | S1 EP nearest | 0.643 | 0.630 | 0.672 | 0.553 | 0.539 |
| | S2 EP margin | 0.568 | 0.558 | 0.648 | 0.574 | 0.584 |
| | S3 coreset kNN | 0.619 | 0.545 | 0.461 | 0.392 | 0.385 |
| | S4 Mahalanobis | 0.393 | — | — | — | — |
| | S5 entropy | 0.426 | — | — | — | — |
| | **S0 length** | **0.004** | — | — | — | — |
| **R5** random | S1 EP nearest | 0.652 | 0.683 | 0.735 | 0.767 | 0.787 |
| | S2 EP margin | 0.593 | 0.654 | 0.634 | 0.657 | 0.679 |
| | S3 coreset kNN | 0.780 | 0.669 | 0.519 | 0.736 | 0.719 |
| | S4 Mahalanobis | 0.138 | — | — | — | — |
| | **S5 entropy** | **1.000** | — | — | — | — |
| | S0 length | 0.500 | — | — | — | — |

S4/S5/S0 do not depend on K, so they are shown once.

## 3. TPR at 1% FPR — EP is unusable as a monitor

This is the number that matters operationally and it is not close.

| rung | EP best (any p) | coreset best | S5 entropy |
|---|---|---|---|
| R1 | 0.017 | 0.003 | 0.001 |
| R2 | 0.058 | **0.223** | 0.001 |
| R3 | 0.021 | 0.001 | 0.004 |
| R4 | 0.049 | 0.003 | 0.001 |
| R5 | 0.043 | 0.044 | **1.000** |

At a 1% false-positive budget — already generous for a monitor that would fire
on production traffic — EP's nearest-exemplar distance catches **between 1.7%
and 5.8%** of shifted inputs. On the language-shift rung the random coreset
catches four times as many.

## 4. We reproduce the paper's statistic, and it does not mean what it is used for

This is the substantive point about §7 / appendix C.

**The mean nearest-exemplar distance gap reproduces.** The paper reports
0.04–0.08 random-vs-Pile at L20. We measure R5 − R0:

| p | 1 | 2 | 4 | 8 | 10 |
|---|---|---|---|---|---|
| mean-distance gap, R5 − R0 | +0.0416 | +0.0464 | +0.0499 | +0.0524 | +0.0556 |

Squarely inside the published band. So we are looking at the same geometry the
paper described.

**The within-threshold rate is vacuous.** The paper's other reported quantity —
the fraction of activations inside a cell — is **1.000 for every rung at every
percentile**, including random tokens at p=1 where θ = 0.7798:

| p=1, θ=0.780 | R0 | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| within-threshold rate | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 |

Not a single rung is ever "outside the dictionary". The repo's walkthrough
already concedes this for loose p; it holds at the tightest published
resolution too. Any monitor built on θ would have a recall of zero.

**Between those two facts is the whole finding.** A mean-distance gap of 0.05
is real and reproducible, and it corresponds to an AUROC of 0.65 and a TPR of
0.03 at 1% FPR. A gap between distribution means is not a separability claim,
and reporting the gap without a separability metric — which is what §7 and
appendix C do — overstates what the signal can do by a wide margin.

## 5. EP's distance is not measuring distribution shift

The per-source breakdown is where the result stops being "EP is weak" and
becomes "EP is measuring something else". **R3 is the clean experiment**: five
scaffolds, identical Pile content, 128 tokens for 1997 of 2000 prompts (three
`chatml` prompts land at 127 on a tokenizer boundary), so length and content are
both held fixed and only the template varies. The length control confirms it:
S0 on R3 is AUROC 0.4993, chance, and the three short prompts sit in the
*highest*-scoring scaffold, not the lowest.

| R3 scaffold (S1, p=1) | n | AUROC | tokens |
|---|---|---|---|
| llama2_inst `[INST] <<SYS>>…` | 400 | **0.885** | 128.0 |
| chatml `<|im_start|>…` | 400 | 0.806 | 128.0 |
| api_frame (JSON) | 400 | 0.513 | 128.0 |
| caps_system | 400 | 0.441 | 128.0 |
| xml_frame `<request>…` | 400 | **0.226** | 128.0 |

The rung-level 0.574 is an average over scaffolds that disagree about the
*sign*. Two novel templates are detected well; a third is detected in reverse —
XML-wrapped Pile lands measurably **closer** to the exemplars than raw Pile
does. A distribution-shift detector cannot have that property. Whatever
nearest-exemplar distance tracks, it is a property of the specific surface form
relative to whatever happened to open a cell during the build, not novelty.

R4 tells the same story from the other side, including the same sign flip
(n = 250 per attack, AUROC weighted across goal sources):

| R4 attack (S1, p=1) | n | AUROC | tokens |
|---|---|---|---|
| gcg_suffix | 250 | **0.969** | 35.9 |
| leetspeak | 250 | 0.783 | 66.6 |
| prefix_injection | 250 | 0.743 | 33.9 |
| base64 | 250 | 0.709 | 92.4 |
| payload_split | 250 | 0.615 | 40.8 |
| refusal_suppression | 250 | 0.546 | 79.9 |
| roleplay | 250 | 0.543 | 47.9 |
| distractor | 250 | **0.236** | 48.9 |

EP detects the attacks that are character soup — a GCG suffix is essentially
random tokens, which is R5 — is at chance on the *semantic* attacks that read
as ordinary English, and runs backwards on `distractor`, where a jailbroken
prompt looks *more* in-distribution than held-out Pile. That ordering is the
opposite of what a safety monitor needs, and the 0.24-to-0.97 spread across
attacks on a single rung repeats the R3 finding: the score is surface-form
specific, not novelty-sensitive.

It also lines up with the earlier result in this project that working
jailbreaks stay inside the refusal region and that `distractor` is precisely
the template that moves prompts to a *different* cell
(`docs/experiments/PLAN_JAILBREAK_GEMMA2_2B.md`, [[ep-jailbreak-override]]).

## 6. Two rungs are length artifacts, and the control caught them

S0 is prompt token count and is not a monitor. Its AUROC:

| rung | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|
| S0 AUROC | **0.002** | 0.500 | 0.499 | **0.004** | 0.500 |

R1 and R4 are separated *near-perfectly* by counting tokens (0.002 and 0.004 are
0.998 and 0.996 with the sign flipped) because MBPP/GSM8K/jailbreak prompts are
39 and 56 tokens against R0's 128. **No claim about R1 or R4 can be attributed
to representation geometry**, EP's or anyone's. R2, R3 and R5 are exactly
length-matched at 128 tokens and are the only rungs that carry information —
which is why R2 is treated as the decisive one above.

Without S0 in the table, EP's 0.672 on R4 at p=4 would have looked like a
result.

## 7. What actually works

**S5, the model's own next-token entropy, gets AUROC 1.000 and TPR@1%FPR
1.000 on random tokens** — perfect separation, no dictionary, no build stream,
no stored activations, one forward pass the model was doing anyway. It is also
the best scorer on R3 (0.622). It is useless on R4 (0.426), which is the
honest limitation.

The ranking on the only clean, non-trivial rung (R2): Mahalanobis 0.808 ≈
coreset 0.801 > entropy 0.540 ≈ EP 0.545. EP is last.

## 8. Anomalies worth recording

- **EP's p-dependence is non-monotonic and sometimes inverted.** S1 on R2
  across p=1,2,4,8,10 is 0.545, 0.504, **0.392**, 0.716, 0.712 — below chance
  at p=4, then a 0.32 jump at p=8. The mean-distance gap flips sign with it
  (−0.018 at p=4, +0.043 at p=8). The coreset degrades smoothly with K, as a
  memory-budget baseline should. Adding 460 exemplars between p=8 and p=4
  should not reverse a distribution-shift signal.
- **The paper predicts the opposite direction.** README and walkthrough both
  say the OOD signal is strongest at tight p (p=1, p=2). On R2 and R5, EP's
  AUROC is *highest at the loosest* dictionary (p=10, K=176).
- **S4 below chance on random tokens (0.138)** is a genuine curiosity, not a
  bug: random-token activations sit closer to the Pile mean under the Pile
  covariance than Pile activations do. Consistent with random inputs producing
  low-norm, low-structure residuals.

## 9. Threats to this conclusion

Stated because they are real, not to soften the verdict.

1. **Extractor mismatch (the main one).** The hub dictionaries are built
   `per-position`; this eval is final-token. That is the hazard flagged in
   `GATE0A_FINDINGS.md` §6.1 and accepted as option (b). It cannot explain the
   *comparison* — S3 and S4 read the same per-position reference pool and are
   queried with the same final-token activations, so the mismatch is symmetric
   and the coreset beats EP under it. It could depress all absolute numbers.
   **A per-position arm is the one follow-up that could change the reading**,
   and it is cheap: the reference-pool extractor already exists.
2. **One model, one layer, one seed.** gemma-2-2b-it L20 only. The dictionaries
   are all seed 0; earlier work in this project found 2 of 4 streaming seeds
   give Δ = 0 on the refusal ablation, so seed sensitivity is documented for
   this construction.
3. **S4 is not memory-matched** (5.3M floats regardless of K: 13x S1's budget
   at K=176, 0.4x at K=5796). It is a reference point for rule 2, not a
   competitor, and rule 2 did not fire anyway.
4. **bf16 on MPS.** Validated against a CPU forward: cosine 0.99995, relative
   Frobenius error 1.0%, entropy max-abs-diff 0.030 nats
   (`artifacts/runs/monitor/mps_validation.json`). Immaterial for a cosine method.

## 10. Recommendation

The gate was a falsification test and it falsified. On its own terms the
project stops here: **EP's construction does not earn its keep as a runtime
monitor**, and the published OOD claim rests on a mean-distance gap that we
reproduce exactly and that carries almost no separability.

The one experiment worth running before closing is the **per-position arm**
(threat 1), because it is the only threat that could move the verdict rather
than merely qualify it. If EP still loses to a matched-K coreset when queried
in the same regime it was built in, the conclusion is unconditional.
