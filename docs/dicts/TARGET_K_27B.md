# Retraining the Qwen3.6-27B dictionary to a target region count

Goal: a Neuronpedia-importable EP source for `Qwen/Qwen3.6-27B` L55 with as
close to **16,000 regions** as possible.

## Why the percentile knob cannot do this

`sweep_p` takes a percentile `p`, calibrates `theta` as the p-th percentile of
within-batch pairwise cosine distance, and leader-clusters at that `theta`.
Region count `K` is a consequence, never an input.

Measured on the full 27B L55 cache (2,991,091 activations, d=5120, Pile,
ctx 128):

| p | theta | K |
|---|--------|--------|
| 1 | 0.6584 | 109883 |
| 4 | 0.8218 | 5190 |
| 8 | 0.8870 | 779 |
| 12 | 0.9163 | 280 |
| 16 | 0.9347 | 144 |

Two regularities fall out, and together they rule `p` out as the knob:

1. **`theta` is linear in `ln p`** — `theta ~= 0.1107 ln p + 0.6617`, residuals
   within ±0.005 over `p` in [1, 8].
2. **`K ~ (1 - theta)^b`** with `b` ≈ 4.8 / 4.6 / 4.2 across the
   0.66→0.74→0.82→0.89 segments (the exponent coarsens as `theta` rises).

So a ±0.005 error in `theta` — which is just the fit noise of regularity 1 —
is `4.6 × 0.005 / 0.227` ≈ **±10% in K**. Picking a `p` and hoping cannot land
on 16,000. `p` sets the neighbourhood; only `theta` sets `K`.

## Where 16,000 sits

Interpolating the two regularities: K=16000 falls between p=2 and p=4, at

    theta* ~= 0.7726   (equivalent p ~= 2.7)

Sensitivity there: ±10% in K is only ∓0.005 in theta. That number is a *seed*,
not an answer — it leans on an interpolated p=2 point, and the sub→full-cache
multiplier is itself p-dependent (×3.35 at p=1, ×1.75 at p=4, ×1.39 at p=8),
so it should not be trusted to better than ~20%. The search measures its way
in from there.

## The search

`qwen_ep/target_k.py`. Leader clustering is deterministic given
(center, threshold, stream order), so `K(theta)` is a monotone non-increasing
step function that can simply be searched.

- The calibration **center is computed once and cached** to `center.npz` in the
  cache dir. It does not depend on `p`, and recomputing it per trial would let
  float-summation order shift the geometry, so `K` differences would no longer
  be attributable to `theta` alone.
- Each trial is a **counting-only pass**: no per-region prompt heaps, no member
  reservoir. It reports one integer.
- Steps are **secant in `(ln(1-theta), ln K)`**, where the curve is locally
  straight with slope `b`. The first step uses `b = 4.6`; later steps use the
  slope through actual observations, taken from the adjacent pair nearest the
  latest trial (since `b` drifts with `theta`).
- A trial **aborts at `--abort-factor` × target** (default 3×). A theta guessed
  too low then costs a fraction of a pass instead of a full one plus a full
  disk. Its `K` is recorded as a lower bound and the step direction is still
  correct.
- Once a bracket exists, a secant step that leaves it falls back to bisection
  in `log(1 - theta)`.
- Every trial is appended to `search.json` in the run dir, so a killed pod
  resumes without repeating a measured `theta`.

Validated on a synthetic cache: converges to +0.33% of target from a seed 13×
over target (2 aborted + 3 full trials), and the final build reproduces the
winning trial's `K` exactly — confirming determinism.

## Memory: the member reservoir

`Partition.sample_members` holds up to 30 member directions per region —
`30 × 5120 × 4` = **614 KB per region** at d=5120:

| K | reservoir |
|---|-----------|
| 5190 | 3.2 GB |
| 16000 | 9.8 GB |
| 48000 (3× abort ceiling) | 29.5 GB |

At the abort ceiling that alone would OOM a pass that was only going to report
an integer, so counting trials disable it. Measured on the synthetic run it is
**83% of the pickle** (11.7 MB → 2.0 MB at cap 0), matching the 88% figure
already recorded in `experiments/monitor/dicts.py`. `K` is identical at every
cap — it does not affect clustering.

Nothing in `epdashboard` or `member_scan` reads it, but
`experiments/refusal.py`, `experiments/jailbreak/anchor_robustness.py` and the
legacy dashboard do. It therefore stays **on by default** (`EP_RESERVOIR=30`,
~11 GB pickle at K=16000); set `EP_RESERVOIR=0` for a ~1.2 GB pickle if those
experiments will not be re-run against this dictionary.

## Running it

The 27B activation cache was deliberately not kept, so this is a re-extract.
Phase 1 is the only GPU-bound step and dominates cost.

```bash
EP_MAXTOK=200000 ./scripts/dicts/run_27b.sh    # smoke first
./scripts/dicts/run_27b.sh                     # full: K=16000
EP_TARGET_K=16384 ./scripts/dicts/run_27b.sh   # if the import wants 2^14
```

Knobs: `EP_TARGET_K` (default 16000, empty to skip), `EP_SEED_THETA`
(0.7726), `EP_TOL` (0.01), `EP_RESERVOIR` (30), `EP_PCTS` (empty — the
p=4/8/12/16 dictionaries were built on the previous cache).

Cost: re-extract ~40 min GPU, then each counting trial is a full clustering
pass. p=4 (K=5190) took 198 s; cost scales with average `K` over the stream,
so a K=16000 trial is roughly 3× that. Budget 3–4 trials plus one final build.

`theta` can also be replayed directly without a search:

```bash
python -m qwen_ep.sweep_p --cache-dir <cache> --thresholds 0.7726
```

## RESULT (run 2026-07-31, Modal)

Re-extracted 2,991,091 activations — **byte-identical in count to the original
RunPod run**, and the calibration ladder reproduced to 6 decimals
(p=1 → 0.658426, p=4 → 0.821784), so this cache is interchangeable with the
one the p=4/8/12/16 dictionaries were built from.

**Closest attainable to 16,000 is K=15,950** (−0.31%) at
`theta = 0.772642189816015`, equivalent `p = 2.724`. The pre-run prediction was
theta* ≈ 0.7726 / p* ≈ 2.723 — essentially exact.

### 16,000 is not merely hard to hit, it is unreachable

The search ran 14 trials and localised a genuine **discontinuity of 105
regions** straddling the target:

| theta | K |
|-------|---|
| 0.772628127 | 16,051 |
| 0.772631643 | 16,052 |
| 0.772632302 | 16,055 |
| 0.772632467 | **16,055** |
| 0.772632522 | **15,950** |
| 0.772633401 | 15,943 |
| 0.772642190 | 15,950 |
| 0.772656251 | 15,928 |

105 regions vanish across `dtheta = 5.5e-8`. **No theta yields any K in
(15,950, 16,051).** The two nearest attainable values are 15,950 (−50) and
16,051 (+51).

Cause: leader clustering is sequential, so one early activation crossing the
threshold changes whether it seeds a cell, and that single flipped decision
cascades through every later assignment. The same mechanism shows up as
**non-monotonicity** — K *rises* from 16,051 to 16,055 as theta *increases*
across 0.772628 → 0.772632. K(theta) is therefore not a clean monotone step
function but carries roughly ±7 regions of order-dependent jitter, on top of
the large cascade jumps.

This bounds what any theta search can do. Tolerances tighter than ~±0.5% are
not meaningful near K=16,000; the search will simply spend trials proving the
gap exists (trials 8–14 above did exactly that).

### Two dictionaries were built

| run dir | K | theta | note |
|---------|---|-------|------|
| `qwen3_6-27b_L55_k16000_ctx128_cache_pile` | 15,950 | 0.772642189816015 | closest to 16,000 |
| `qwen3_6-27b_L55_k16051_ctx128_cache_pile` | 16,051 | 0.772628 | only route to *exactly* 16,000 |

They are near-tied on distance (50 vs 51), but they are not equally useful:
only the one **above** 16,000 can be trimmed to exactly 16,000 by dropping its
51 smallest regions. You can trim down, never up. Pick the 16,051 build if the
import wants a round 16,000; otherwise the 15,950 build is the honest closest.

Cost: ~22 min A100-80GB extraction + 78 min of search (14 trials, ~5.5 min
each) + two ~8 min final builds + two member scans.

### Pulling the dictionaries back: verify, do not trust

`modal volume get` silently corrupted the 9.37 GB `dictionary.pkl` on the first
attempt — two zero-filled holes (0.4 MiB at 52% of the file, 8.4 MiB at 99.9%)
with the **correct total size**, a valid pickle header and a valid STOP byte.
The command exited 0. Size checks and head/tail inspection all pass on a file
that cannot be unpickled.

Use `scratchpad/fetch_verify.sh <run-dir>`, which re-downloads until an actual
`pickle.load` succeeds and cross-checks `len(partitions)` and `threshold`
against `metadata.json`. Also fetch `metadata.json` on its own — the bulk get
writes it as 0 bytes.
