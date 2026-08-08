# PLAN — Gate 1: does EP's *partition* work as a monitor?

Status: **Arm B complete — see `docs/experiments/GATE1B_RESULTS.md`. Arm A not started.**
Written before any Gate 1 number was computed.

Arm B outcome in one line: rule B1 **passes** (EP cells beat matched-K coreset
cells at all five percentiles on R2 and R5 — the first robust EP win in either
gate), rule B2's balance mechanism is **falsified** (EP's histogram is *more*
concentrated, so the win is boundary placement, not balance), and rule B3's
cross-build comparability **fails** on the tighter reading-agreement test
(surprisal correlates 0.15–0.44 across arrival orders) even though the
pre-registered θ-overlap criterion passed. Both wins are dominated by
Mahalanobis or by free max-token entropy.

Gate 0B closed the **distance** family: EP's nearest-exemplar distance (S1), its
boundary margin (S2), and the paper's within-threshold rate all fail to beat a
matched-K random coreset. See `docs/experiments/GATE0B_RESULTS.md`. That verdict stands and
Gate 1 does not revisit it.

Gate 1 tests a **different mechanism**: the cell *identity* an activation is
assigned to, not the distance to it. This is not a rescue of the Gate 0B
hypothesis — it is a separate hypothesis with its own pre-registered rule, and
it can fail independently.

## Why this is not the forbidden search

The Gate 0B pre-registration forbids searching over configurations for one where
EP wins *at the same scorer*. Changing p, layer, or threshold to rescue S1 would
be that search. Changing from `d(x, e_i)` to `argmin_i` is a different statistic
computed from a different property of the construction — the partition rather
than the metric. The distinction is only honest if the new rule is fixed in
advance, which is what this file is for.

## Arm A — categorical routing monitor

**The claim.** Appendix A.6: at L20 the base model routes 569/600 harmful and
benign instruction prompts into a single final-position region at chance harmful
rate, while the instruction-tuned model splits them across five regions, one
carrying 405 prompts at 74% harmful / 75% refusal. Monitor primitive: one argmax
over a K x D matrix, then "is this cell flagged".

**Already replicated here** (`artifacts/runs/jailbreak/gate.json`, 2026-07-29): 404
members, harmful fraction 0.7426, 5 occupied cells, harmful recall 1.0. Arm A
does not re-derive this. It asks whether it is a monitor.

**A0 — does the structure survive the hub sweep?** The replication is at p=12,
locally built. The hub has {1,2,4,8,10}. At p=1, K=5796 for 600 prompts, so
concentration is arithmetically impossible. Report occupied-cell count, largest-
cell harmful fraction and recall at every p. *If the routing structure exists
only at p=12, say so plainly; it becomes a claim about one hand-picked
resolution, not about EP.*

**A1 — the monitor.** Score(x) = empirical harmful rate of the cell x is
assigned to, estimated on a **calibration** pool of labelled prompts disjoint
from eval. Unassigned/unseen cells take the pool prior. This gives a continuous
score and therefore an ROC; single-cell membership does not.

**Matched baselines.** All fitted on the *same* calibration labels:

| id | scorer | matched on |
|---|---|---|
| A-EP | harmful rate of assigned EP cell | K exemplars |
| A-CORE | harmful rate among k nearest of K random build-stream activations | K, same labels |
| A-MEAN | projection on the closed-form mean-difference (harmful − benign) direction | same labels |
| A-LDA | shrunk-covariance LDA score | same labels |
| A-CELL1 | membership in the single best cell (the paper's primitive) | K |

A-MEAN and A-LDA are **closed form**: no gradient steps, so the inference-only
constraint holds. A logistic probe would violate it and is not run.

**The claim actually under test is label efficiency, not label freedom.** The
partition is discovered without labels, but flagging a cell requires them. So
the decisive plot is AUROC vs number of calibration labels N, swept over
N ∈ {25, 50, 100, 200, 400, 800}. EP's case is that the partition is a
low-dimensional basis that needs fewer labels than a direction fitted in D=2304.

**Decision rule A (fixed in advance).**
1. If A-EP does not beat A-CORE at matched K and matched N across the sweep,
   the partition adds nothing over random anchors for routing. Say so.
2. If A-EP does not beat A-MEAN at **any** N, the categorical framing is dead —
   a two-mean difference is the cheapest possible baseline and needs no
   dictionary.
3. EP wins on label efficiency only if it beats A-MEAN at **small N** (<= 100)
   even if it loses at large N. That is a real and reportable result, and it is
   the only outcome that would justify the construction here. It must be stated
   as label efficiency, not as accuracy.
4. Report TPR@1%FPR alongside AUROC. Gate 0B's lesson is that AUROC alone hides
   an unusable operating point. The known single-cell point is TPR 1.00 at
   FPR 0.347; if nothing reaches a usable FPR, that is the headline.

**Eval task.** Harmful vs benign instructions, final position, gemma-2-2b-it
L20. This is content classification, not OOD — the Gate 0B R0-R5 ladder is the
wrong eval and is not reused for Arm A.

## Arm B — occupancy-histogram drift monitor

**The claim.** Exemplars are observed activations with fixed provenance, so
occupancy histograms are comparable across time windows with no labels and no
retraining; watch for distributional shift in traffic.

**The comparison that actually discriminates.** Not SAEs. A **frozen random
coreset** has an equally fixed coordinate system at zero cost, and Gate 0B
showed it is hard to beat. The hypothesis that could favour EP is *occupancy
balance*: threshold-based leader clustering spreads mass across cells, whereas
random-coreset Voronoi cells are dominated by a few high-density winners, and a
multinomial two-sample test has more power against a flatter reference.

**B1 — power curves.** Reference histogram from held-out in-distribution
traffic. Test windows of N ∈ {50, 100, 200, 500, 1000} requests containing a
contamination fraction eps ∈ {0.01, 0.05, 0.1, 0.25, 1.0} drawn from each Gate
0B rung. Statistic: chi-square / total-variation against the reference, with the
null calibrated by resampling clean windows so the false-alarm rate is fixed at
1% by construction rather than assumed. Output: detection power vs (N, eps) per
rung, EP vs matched-K coreset vs a Mahalanobis-score histogram.

**B2 — is the coordinate system actually stable?** The paper's differentiator
assumes exemplar identity is reproducible. Our own prior finding is that the
exemplar is a first-arrival accident. Build two dictionaries from the same
stream in different orders and report exemplar overlap and histogram
correlation. *If overlap is low, "comparable across checkpoints" holds only for
a frozen dictionary — which is a property of any frozen reference set, not of
EP.*

**B3 — balance diagnostic.** Gini / entropy of the occupancy histogram, EP vs
matched-K coreset. This is the mechanism behind B1; report it whichever way B1
comes out, because it explains the result rather than just scoring it.

**Decision rule B (fixed in advance).**
1. If EP's power curve does not dominate the matched-K coreset's at some
   (N, eps) that is operationally meaningful (eps <= 0.1, N <= 500), the drift
   framing adds nothing. Say so.
2. If EP wins B1 **and** B3 shows its histogram is flatter, the mechanism is
   confirmed and the result is about occupancy balance, not about EP per se —
   any balanced quantiser would do. State that, and note that a k-means or
   coverage-maximising coreset would be the next baseline.
3. If B2 shows low exemplar overlap across build orders, the
   cross-checkpoint-comparability claim is not supported and must be reported as
   false regardless of B1.

## Hygiene (carried from Gate 0)

- Three disjoint pools: dictionary build stream / calibration (label + flag
  setting) / eval. Flag selection never touches eval.
- Hub dictionaries as primary (revision `0ec26618`, blob SHAs in
  `GATE0A_FINDINGS.md` §3). The local p=12 build is carried only as the
  replication anchor and is labelled as such.
- Inference-only: closed-form baselines only, no gradient steps.
- AUROC **and** TPR@1%FPR everywhere. Mean gaps are not an output.
- Every number to a results file.
- Coreset baselines averaged over independent draws with the draw sd reported,
  so a win inside one sd is called a tie.

## Known hazards

1. **p=12 is not on the hub.** If A0 shows the structure is p=12-only, Arm A's
   headline changes from "EP routes harmful prompts" to "one hand-tuned
   resolution does". Pre-registering this stops it being reframed later.
2. **600 prompts over 5796 cells** is a sparsity problem, not a monitor result.
   Per-cell rate estimates need enough members; report members-per-cell and
   treat cells below a minimum count as prior.
3. **Refusal rate and harmfulness are different labels.** A.6 reports both. A
   cell that predicts *refusal* predicts model behaviour, not input harm, and
   the two come apart exactly on jailbreaks — which is what
   `artifacts/runs/jailbreak/` already found. Score both and keep them separate.
4. **Final-position vs per-position** extractor mismatch, carried from Gate 0A
   §6.1 and still unresolved. Symmetric across arms, but it caps absolute
   numbers.
