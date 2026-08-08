# Gate 1 Arm B — occupancy-histogram drift monitor

Pre-registration: `docs/experiments/PLAN_EP_MONITOR_GATE1.md`. Data: `gemma-2-2b-it` L20,
final token, hub dictionaries (revision `0ec26618`), p ∈ {1,2,4,8,10},
K ∈ {5796,2037,686,226,176}. No forward passes — everything reads artifacts
built in Gate 0B. Results in `artifacts/runs/monitor/gate1b_*.csv|json`, code
`experiments/monitor/occupancy.py`, `monitor/run_gate1b*.py`, 27 tests.

## Headline

**The categorical monitor is a genuinely different scorer from the distance
monitor, and it beats the matched-K random coreset — which the distance monitor
never did.** Decision rule B1 passes. But **both of its robust wins are
dominated by a baseline that needs no dictionary at all**, its proposed
mechanism is falsified, and the property that was supposed to differentiate it
from SAEs does not hold.

## 1. Decision rule B1 — EP cells vs matched-K coreset cells

Detection power at a 1% false-alarm rate fixed by resampling clean windows.
N=500 requests, contamination eps=0.25. EP / coreset:

| rung | p=1 | p=2 | p=4 | p=8 | p=10 | verdict |
|---|---|---|---|---|---|---|
| R2 language | 0.55/0.02 | 0.91/0.12 | 1.00/0.00 | 0.79/0.00 | 0.83/0.00 | **EP wins at every p** |
| R5 random | 0.80/0.44 | 0.95/0.43 | 0.45/0.33 | 1.00/0.14 | 0.78/0.02 | **EP wins at every p** |
| R3 template | 0.00/0.06 | 0.00/0.49 | 0.27/0.35 | 0.00/0.29 | 1.00/0.08 | single-p spike, not counted |
| R4 jailbreak | 0.06/0.03 | 0.08/0.20 | 0.02/0.02 | 0.00/0.23 | 0.00/0.00 | EP fails |
| R1 domain | 0.00/0.34 | 0.06/0.48 | 0.00/0.09 | 0.00/0.39 | 0.00/0.19 | EP loses at every p |

**This is the first time in either gate that EP beats the coreset robustly.**
On R2 and R5 it wins at all five percentiles, not one. Gate 0B's verdict was
that leader clustering adds nothing over random anchors; in the *categorical*
regime that is false. The construction does buy something — just not through
distance.

**R3 is excluded by the standard applied in Gate 0B.** 0.00, 0.00, 0.27, 0.00,
1.00 is not a trend with a hole; it is one cell surrounded by zeros. Reporting
p=10 alone would be exactly the search over five configurations the
pre-registration forbids. It is noted because R3 was the rung where *every*
Gate 0B scorer failed, so it may be worth a targeted look — but not as a result.

## 2. The wins are dominated by baselines with no dictionary

Power at N=500, eps=0.1, best p for each scorer:

| rung | EP cells | coreset cells | EP dist | coreset dist | Mahalanobis | max-entropy |
|---|---|---|---|---|---|---|
| R2 | 0.62 | 0.02 | 0.07 | 0.39 | **0.94** | 0.01 |
| R5 | 0.39 | 0.14 | 0.28 | 0.24 | 0.00 | **1.00** |
| R3 | 0.43* | 0.10 | 0.08 | 0.05 | 0.01 | 0.05 |
| R4 | 0.02 | 0.06 | **0.19** | 0.09 | 0.00 | 0.00 |
| R1 | 0.05 | **0.16** | 0.14 | 0.16 | 0.04 | 0.00 |

\* single-p, see above.

R2 goes to **Mahalanobis** (0.94 vs 0.62) — a covariance fitted on the build
stream, no partition. R5 goes to **max next-token entropy** (1.00 vs 0.39) —
one forward pass, no build stream at all, AUROC 1.000 per request. So EP wins
the comparison it was given and loses the comparison that matters
operationally. Both of the rungs where EP robustly beats the coreset already
have a cheaper scorer that beats EP.

**Nothing detects 1% contamination anywhere.** At eps=0.01 every scorer, every
rung, every p sits at the false-alarm rate. If the realistic operating regime is
a small fraction of shifted traffic, none of this works.

**Per-request TPR@1%FPR is 0.000 for the cell scorer in every cell of the
table.** The categorical monitor is a population instrument only. It cannot flag
an individual request, which is what "runtime monitor" usually means.

## 3. B3 — the proposed mechanism is falsified

The hypothesis was that threshold-based leader clustering spreads mass more
evenly than random anchors, giving a multinomial test more power. Occupancy of
1000 clean R0 requests:

| p | K | EP gini | coreset gini | EP entropy ratio | coreset entropy ratio |
|---|---|---|---|---|---|
| 1 | 5796 | 0.891 | 0.867 | 0.754 | 0.775 |
| 2 | 2037 | 0.779 | 0.731 | 0.820 | 0.846 |
| 4 | 686 | 0.626 | 0.562 | 0.881 | 0.905 |
| 8 | 226 | 0.518 | 0.431 | 0.913 | 0.940 |
| 10 | 176 | 0.509 | 0.398 | 0.915 | 0.946 |

**EP's histogram is *more* concentrated than the random coreset's at every
percentile**, on every measure, without exception. The mechanism runs backwards.

This makes the B1 win *more* interesting, not less: EP wins with a worse-
conditioned histogram, so the advantage comes from *where* the boundaries sit,
not from how evenly they divide traffic. Pre-registered rule B2 ("if EP wins B1
and B3 shows a flatter histogram, the result is about balance and any quantiser
would do") therefore does not fire. A balanced quantiser is not the explanation
and k-means is not the obvious next baseline.

Note also the reason the mechanism fails: cells are calibrated on the
*per-position* build distribution but queried on *final-token* traffic, which
occupies them very unevenly — 13.5% of cells occupied at p=1.

## 4. B2 — the SAE differentiator does not hold

The claim: exemplars are observed activations with fixed provenance, so the
coordinate system is comparable across time and checkpoints, unlike SAE
features which move under retraining seed.

**Pre-registered test (exemplar overlap) passes.** Two dictionaries built from
the same 60k build-stream activations in different arrival orders, at the hub θ:

| p | K_a | K_b | frac a matched | frac b matched | median nn distance |
|---|---|---|---|---|---|
| 1 | 1639 | 1600 | 1.000 | 1.000 | 0.655 |
| 4 | 419 | 402 | 1.000 | 1.000 | 0.758 |
| 10 | 129 | 132 | 1.000 | 1.000 | 0.821 |

**But that test is nearly vacuous and I should not have pre-registered it as
decisive.** "Matched" means "within θ", and the median counterpart sits at
84–90% of θ — out at the cell edge, not near the centre. This is the same
high-dimensional artifact recorded earlier in this project: a θ-ball criterion
is satisfiable almost anywhere when K ≪ d.

The test that decides the claim is whether two builds give the **same reading**:

| p | corr(surprisal_A, surprisal_B) | power_A | power_B (R2, N=500, eps=0.1) |
|---|---|---|---|
| 1 | 0.442 | 0.06 | 0.08 |
| 4 | 0.150 | 0.11 | 0.12 |
| 10 | 0.332 | 0.01 | **0.15** |

Same data, same θ, different arrival order → per-request scores correlate at
**0.15–0.44**, and at p=10 one build detects the R2 drift while the other does
not. **Arrival-order dependence in EP is the direct analogue of seed dependence
in an SAE.** The stated differentiator does not survive.

What does survive: if you **freeze one dictionary** and reuse it, comparability
over time is exact — but that is a property of any frozen reference set,
including the random coreset, and costs nothing. Note too that K grows with
build size (1639 at 60k vs 5796 at 3.6M activations, p=1), so histograms from
different-sized builds are not comparable at all.

I am flagging the goalpost move explicitly: my pre-registered criterion returned
a pass and the tighter test returned a fail. Both are reported. The tighter one
is the one I believe, and the reason is stated above rather than discovered
after the fact — the θ-ball criterion was already known in this project to be
weak in high dimensions, and I should have caught it when writing the plan.

## 5. What this changes

Gate 0B: EP's **distance** adds nothing over a random coreset. Unchanged.

Gate 1 Arm B: EP's **partition** does add something over random anchors — a
robust, all-five-percentiles win on language shift and random tokens, achieved
with a *worse*-conditioned histogram, which points at boundary placement as the
active ingredient. That is a real and previously untested positive.

It is not, however, a usable monitor: both wins are beaten by Mahalanobis or by
free next-token entropy, nothing sees 1% contamination, per-request TPR@1%FPR is
zero, and the cross-build comparability that motivated the framing does not
hold.

## 6. Threats

1. **Final-token queries against a per-position build.** Unresolved since Gate
   0A §6.1, and §3 shows it is doing visible damage here — 13.5% cell occupancy
   at p=1 means most of the partition is never exercised. This is the one threat
   that could raise EP's absolute numbers.
2. **R1 and R4 remain length-confounded** (Gate 0B S0 AUROC 0.002 / 0.004), so
   EP's failures there are uninterpretable, not informative.
3. **B2 used 60k build rows, not 3.6M.** Smaller builds are noisier, so the
   0.15–0.44 correlation is an upper bound on disagreement at hub scale. The
   direction of the finding is safe; the magnitude is not.
4. **One reference window of 1000 R0 requests** defines every histogram. At
   K=5796 that is sparse and the smoothing constant is doing real work.

## 7. Status of Arm A

Not started. `artifacts/runs/jailbreak/gate.json` already replicates the Appendix A.6
routing structure (404 members, harmful fraction 0.7426, 5 occupied cells) but
at **p=12, locally built** — not a hub percentile. Given that R3 here spiked at
exactly one percentile and that Gate 0B found p-dependence non-monotonic, A0
(does the routing survive the hub sweep) should run before anything is built on
top of it.
