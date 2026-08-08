# Handoff: replicate the EP refusal experiment on gemma-2-2b-it

Self-contained brief for a fresh session. Goal: reproduce the paper's causal
refusal-ablation result on `google/gemma-2-2b-it`, then use the same machinery
to localize and ablate *other* behaviors in the same model.

A parallel session is working on scaling this to `gemma-2-27b-it`. Don't
duplicate that; this brief is the 2B track.

---

## 1. What the result is

Paper: **"Exemplar Partitioning for Mechanistic Interpretability"**, Jessica
Rumbelow, arXiv **2605.14347**. Repo already cloned at
`exemplar-partitioning/` (MIT, `github.com/jessicarumbelow/exemplar-partitioning`).
The PDF text is extracted at `/tmp/.../ep_paper.txt` if still present;
otherwise re-fetch. The refusal experiment is §4.1 plus **Appendix A.2**, which
has all the detail — read A.2, not just §4.1.

EP builds a feature dictionary by leader-clustering streamed activations on the
centered unit sphere within a calibrated distance threshold. Each region is
anchored by an **observed exemplar** — a real activation — which serves as both
membership criterion and intervention direction. Dictionary size is not
specified in advance; it's whatever the geometry yields at that threshold.

The refusal claim: in gemma-2-2b-it at L20, refusal concentrates in one region,
and projecting activations off that region's first-arrival exemplar collapses
held-out refusal by up to **Δ = −0.96**, while a size-and-coherence-matched
non-refusal region gives **Δ = 0**.

## 2. Exact protocol

Implemented in `exemplar-partitioning/scripts/exp_behavioral.py` (906 lines,
read it — it is the ground truth, not the README).

1. **Prompts.** 300 harmful (AdvBench 520 via GitHub raw CSV + JailbreakBench
   `JBB-Behaviors` 100, deduped) + 300 benign (Alpaca `tatsu-lab/alpaca`,
   instructions with empty `input`). Chat-formatted with Gemma's template.
2. **Build.** Calibrate on *this corpus* (cache key `{model}__behavioral`), not
   on the Pile. `extract_per_position`, 100k calibration tokens, then
   `discover()` to saturation. Labels do **not** enter construction.
3. **Score.** Assign **final-position** activations to regions. Generate 60
   tokens greedy for each of the 600 build prompts, classify refusal by
   substring match (`REFUSAL_PHRASES`, first 300 chars, lowercased). Per region,
   mean member refusal rate.
4. **Select.** Candidates = regions with `n_members ≥ 5` and
   `refusal_rate ≥ 0.3`; rank by `refusal_rate × n_members`; take top 5.
5. **Null control.** Among regions with `refusal_rate ≤ 0.05` and `n ≥ 5`, pick
   the one closest to the target in `(log10 member_count, member_coherence)`.
6. **Ablate.** On 50 held-out harmful prompts (same pool, build prompts
   removed — a within-distribution test, not OOD), install a forward hook at
   `blocks.20.hook_resid_post` that, **at every token position during
   generation**, subtracts the calibration center, projects off the
   QR-orthonormalized span of the top-K region directions, and adds the center
   back. Sweep K = 1..5.
7. **Bases.** Three: `mean` (spherical mean of member directions), `exemplar`
   (first arrival), `exemplar_reanchored` (member with highest cosine to the
   region mean).

## 3. Numbers you must hit

Baseline held-out refusal is **0.98** on every run. Appendix A.2, Table 3
(K=1, n=50 held-out, four streaming seeds; build prompts and held-out set fixed
across seeds, only construction order varies):

| p | K | top n | top rate | Δ exemplar (seeds 0–3) | null Δ |
|---|---|---|---|---|---|
| 8 | 373–395 | 173–313 | 0.57–1.00 (frag) | {0, −0.10, 0, 0} | {–, 0, 0, 0} |
| 10 | 308–317 | 288–406 | 0.74–0.99 | {−0.64, −0.98, 0} (3/4) | {0, 0, 0} |
| **12** | **189–207** | **279–405** | **0.75–0.99** | **{−0.74, −0.96, 0, 0}** | {0, 0, –, 0} |
| 16 | 113–118 | 315–459 | 0.66–0.95 | {−0.74, −0.96, −0.12, 0} | {0, 0, 0, 0} |
| 18 | 82–94 | 305–595 | 0.51–0.92 | {0, 0, −0.12, 0} | {0, –, 0, 0} |
| 20 | 74–77 | 456–579 | 0.52–0.66 (cont) | {−0.02, 0, −0.12} (3/4) | {0, 0, 0} |

Single-seed elsewhere: p=5 → −0.68, p=6 → −0.68, p=15 → −0.74.

**Two of four seeds legitimately give Δ = 0 at every working percentile.** A
single seed returning 0 is not a failed replication. Run seeds 0–3 and check
the *distribution*.

Other invariants:
- **Exemplar beats mean by 0.4–0.6** across the working range. This is the
  paper's central mechanistic claim.
- **cos(mean, exemplar) ≈ 0.94.** Available before any generation — cheapest
  early signal. The paper's argument: projecting off a direction at angle θ to
  the true axis leaves sin²θ of the signal; at 0.94 that's ~12%, enough to
  drive refusal, which is why the on-axis exemplar wins.
- **Null Δ = 0.00 wherever a matched null is selectable.**
- The refusal region is huge: 279–459 of 600 build prompts, 60–200× the average
  partition. The IT chat scaffold consolidates instruction-formatted prompts
  into a few dominant final-position activations; refusal is a direction
  *within* that consolidation, not a separately bounded cluster.

Two failure modes bound the working range: **p=8 fragmentation** (cell smaller
than the cluster, splits across sub-cones) and **p=20 contamination** (cell
broader than the cluster). The paper is explicit that p=8's failure is "a
discretisation accident at that radius rather than a monotonic effect."

## 4. Environment

`transformer-lens` is **not currently installed** in `.venv`. Install it:

```bash
.venv/bin/pip install transformer-lens==3.6.0
```

No conflict: TL 3.6.0 requires `transformers>=5.9.0` and `torch>=2.6`; the venv
has `transformers==5.14.1`, `torch==2.13.0`. (TL 2.x needed transformers 4.x —
that constraint is gone. Ignore older advice about needing a second venv.)

Also needed: `datasets`, and network access — the script pulls AdvBench from a
GitHub raw URL and JailbreakBench + Alpaca from HF. `gemma-2-2b-it` is gated:
accept the licence and set `HF_TOKEN`.

## 5. How to run it

**Preferred — Modal**, already wired up in `modal/refusal.py` at the repo root.
Its `--smoke` path *is* this experiment:

```bash
modal run modal/refusal.py --smoke              # gemma-2-2b-it L20 p12 seed 0
modal run modal/refusal.py --smoke --seeds 0,1,2,3
```

Setup is in `MODAL_REFUSAL.md` (§0). It shells out to `exp_behavioral.py`
unmodified. ~20 min, a couple of dollars.

**Local** is viable for a 2B (~5 GB bf16) but slow: the script generates one
prompt at a time, ~1550 generations per seed. On Apple Silicon pass
`--device mps`; TL + MPS + bfloat16 is a known rough edge, so fall back to
`--device cpu` or float32 if you hit dtype errors. Expect hours, not minutes.

```bash
cd exemplar-partitioning
../.venv/bin/python -m scripts.exp_behavioral \
    --model google/gemma-2-2b-it --model-short gemma-2-2b-it \
    --layer 20 --percentile 12 --seed 0 --device mps
```

**`--percentile 12` is not optional.** The script defaults to `8.0`, which the
paper reports as the fragmentation *failure* case. The upstream default is the
one value you specifically do not want.

`scripts/` has no `__init__.py` and relies on namespace-package resolution, so
`python -m scripts.exp_behavioral` only works with cwd at the repo root.

## 6. Reading the output

`behavioral.json` in `--output-dir`. Check in this order:

1. **`base_refusal_rates`** — harmful ≈ 0.98, benign ≈ 0.03. If harmful is low,
   the substring classifier isn't firing and every Δ below is meaningless.
2. **`ablation.sweep_by_basis.exemplar[0].delta`** — the headline.
3. **exemplar vs mean** — exemplar should win by 0.4–0.6.
4. **`cos_mean_exemplar_per_partition`** — expect ~0.94.
5. **`null_ablation`** — must be ~0.00, or the effect isn't specific.

Also useful: `scripts/make_fig_refusal.py` reproduces the per-percentile figure.

## 7. Known traps

- **`--percentile 8.0` default.** See above. Most likely single cause of a
  failed replication.
- **Per-position vs final-position build.** The paper builds with
  `extract_per_position`. In this project's separate Qwen port, building on
  final-position activations gave Δ = **exactly 0.00** at every layer and both
  bases. The upstream calibration docstring warns that mixing per-position
  calibration with final-position discovery "silently produces meaningless
  cells." Don't change the extractor.
- **Batched generation.** The reference generates one prompt at a time. If you
  add batching for speed, note the ablation hook rewrites *every* position
  including left-padding. Verify against `gen_batch=1` before trusting results.
- **n=50 held-out is noisy** — binomial error ≈ ±0.07. Fine for confirming a
  −0.96; useless for distinguishing −0.15 from 0. For new claims, raise
  `--n-held-out-harmful`; the harmful pool is ~620, so ~200 held-out is
  available alongside 300 build prompts.
- **Calibration cache key is (model, hook, percentile)** — no seed component.
  That's intentional: seeds share one calibration and vary only streaming order.
  `EP_CALIBRATION_CACHE` overrides the default `~/.cache/ep/calibration`.

## 8. Extending to other behaviors

This is the actual goal after replication, and the paper states the
generalization directly (A.2):

> The same protocol applies to any behaviour with an automatic scorer
> (substring match, LLM judge, probe): the geometry does the discovery work and
> labels do only the region selection.

So to localize a new behavior, fork `exp_behavioral.py` and swap exactly three
things — everything else is unchanged:

1. **`_load_harmful` / `_load_benign`** → your two contrastive prompt sets.
2. **`_is_refusal`** → your scorer. Must be automatic and reasonably precise;
   a noisy scorer caps your measurable effect size, since it corrupts both
   region selection *and* the Δ measurement.
3. **`--min-refusal-rate`** → the threshold separating "loaded" regions from
   baseline. The 0.3 default is ~10× the ~3% benign baseline; recalibrate to
   your behavior's base rate.

Keep the **null control** unchanged. Region selection is post-hoc on the build
set, so without a size-and-coherence-matched null you cannot distinguish "this
behavior's direction" from "any high-density direction at this layer."

Upstream scripts worth knowing before building anything new:

- **`label_dictionary.py`** — LLM autointerp over partitions, formatting sample
  prompts with per-token importance highlighting. The natural tool for
  *discovering* what behaviors are even present before picking one to ablate.
- **`exp_partition_steering.py`** — steers with hand-picked partitions whose
  meaning is read straight off `partition.sample_prompts`. The paper notes the
  cosine-contrast selection variant (`exp_concept_steering.py`) had selection
  failures, so prefer this one.
- **`exp_patching.py`** — activation patching with EP partitions as the swap
  unit; tests whether regions are causally load-bearing at specific
  (layer, position) pairs.

**Temper expectations on positive steering.** The paper tested the symmetric
intervention (add α·exemplar to benign prompts at L20) and it did *not* work
cleanly: α ∈ {50, 100} was indistinguishable from baseline; α = 200 produced
apologetic prefixes that trip the substring classifier without coherent refusal
content; α = 400 degenerated into single-token loops. Conclusion: the exemplar
direction is causally **necessary** for refusal but not **sufficient** as a
single-direction injection. Expect the same asymmetry for other behaviors, and
don't treat a steering null as evidence against a localization result.

## 9. Context from the parallel Qwen work

Background, not a task. This project previously ran a Qwen port of this
experiment (`experiments/refusal.py`, results in `artifacts/runs/behavioral/qwen3_5-4b/`) on
Qwen3.5-4B. Results diverged sharply from the paper:

- Best Δ was **−0.22** (mean basis, L27 p8); exemplar capped at −0.04.
- **Mean beat exemplar everywhere** — inverted vs the paper.
- cos(mean, exemplar) ≈ 0.83 vs the paper's 0.94.
- Null was clean (0.00) throughout, and probe AUROC was 1.0 at every layer, so
  harmful/benign separability was never the bottleneck.

Whether that's a Qwen-geometry finding, a scale effect, or an artifact of the
port is exactly what the 2B replication settles. **If gemma-2-2b-it reproduces
here, the port is the remaining suspect and the Qwen numbers become a real
finding.** Report the outcome back — the 27B track depends on it.

One hypothesis being carried into the 27B work: the reference script ablates at
**one layer only**, whereas Arditi et al. (2024), whose result the paper
benchmarks against, project the refusal direction out of *every* layer's
residual stream. That may not matter at 26 layers and may matter a lot at 46.
If you want a cheap contribution from the 2B side, adding an all-layer ablation
mode and checking it doesn't *change* the 2B result would de-risk the 27B run.
