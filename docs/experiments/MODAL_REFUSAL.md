# Replicating the EP refusal experiment on Modal (gemma-2-27b-it L35)

Everything lives in [modal/refusal.py](../../modal/refusal.py). Read
[MODAL.md](../epdashboard/MODAL.md) first if you have not used Modal here before — the volume
and `--detach` mechanics are identical and are not repeated.

## What this runs

The upstream `scripts/exp_behavioral.py`, **unmodified**, shelled out to from a
Modal GPU function. No `qwen_ep` code is involved. That is the point: the aim
is to find out what the reference harness does at scale, so the only inputs are
CLI flags.

Target is `google/gemma-2-27b-it` — the largest same-generation scale-up of the
paper's `gemma-2-2b-it`. Same training recipe, same tokenizer, same model
family, so it isolates *scale* with nothing else moving.

| | paper | this run |
|---|---|---|
| model | gemma-2-2b-it | gemma-2-27b-it |
| layers | 26 | 46 |
| d_model | 2304 | 4608 |
| ablation layer | L20 (77% depth) | **L35** (76% depth) |
| percentile | 12 | 12 |
| build prompts | 300 harmful + 300 benign | same |
| held-out harmful | 50 | same |
| seeds | 0–3 | same |

## 0. One-time setup

Both Gemma checkpoints are gated. Accept the licence for whichever you are
running — <https://huggingface.co/google/gemma-2-2b-it> for the 2B track,
<https://huggingface.co/google/gemma-2-27b-it> for the 27B — then create a
**read** token and hand it to Modal:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

The secret must be named exactly `huggingface`; that is what
`Secret.from_name` in [modal/refusal.py](../../modal/refusal.py) looks up. A
workspace may already list similar names (`huggingface-secret`) belonging to
other members — those are not it, and the job fails at container start if the
name is missing.

Both volumes (`ep-hf`, `ep-refusal`) are declared `create_if_missing=True`, so
no `modal volume create` is needed. `ep-hf` is shared with the dashboard job,
so Gemma weights land next to the Qwen ones and download exactly once.

## 1. Smoke test — and simultaneously, the thing that matters most

```bash
modal run modal/refusal.py --smoke
```

This runs the **paper's own configuration**: gemma-2-2b-it, L20, p=12, seed 0.
It is doing two jobs at once.

As a smoke test it proves the image builds, TransformerLens loads, AdvBench and
JailbreakBench download, and results reach the volume — for a couple of dollars
and ~20 minutes instead of finding out three hours into a 27B run.

It runs on a right-sized container (A100-40GB / 32 GB RAM) rather than the
27B's A100-80GB / 192 GB, since the 2B replication is now a track in its own
right and gets run four-plus times. The tradeoff is that it no longer validates
the 27B's memory request — that one is only exercised by the real run, where
the failure mode to watch for is a host OOM shortly after the weight download.

More importantly, it is the **harness validation**. Look at
`behavioral.json → ablation.sweep_by_basis.exemplar[0].delta`. The paper
reports, at p=12 across four seeds, `{-0.74, -0.96, 0, 0}`. Seed 0 should be
one of those. Baseline held-out refusal should be 0.98.

- **Land in that distribution** → the harness reproduces, and every later
  number is interpretable.
- **Get something else** → stop. Do not run the 27B. Something is wrong with
  the environment or the datasets, and it is 100× cheaper to find out here.

Because two of the paper's four seeds legitimately give Δ=0, a single seed
returning 0 is ambiguous. If seed 0 gives 0, run `--seeds 0,1,2,3` at 2B (still
cheap) and check the *distribution* matches rather than any single value.

## 2. The real run

```bash
modal run --detach modal/refusal.py --seeds 0,1,2,3
```

`--detach` matters — this is a multi-hour job and you will close the laptop.
Follow it with `modal app logs ep-refusal`.

The four seeds run as **concurrent containers** via `starmap`. Same total
GPU-seconds as running them serially, but ~3 h of wall clock instead of ~12.

### Rough shape and cost

Per seed, roughly: weight load and TL conversion ~15 min (first run adds the
54 GB download), calibration and build ~10 min, then generation dominates.

The reference script generates **one prompt at a time** — no batching. Per
seed that is 600 labelling + 50 baseline + 750 ablation (3 bases × K=1..5 × 50)
+ 150 null = **1550 generations** of 60 tokens. At a realistic 10–15 tok/s for
an unbatched 27B under TL, call it 4–6 s each, so ~2–2.5 h of pure generation.

**≈3 h per seed, ≈12 A100-hours total, on the order of $30.** Treat that as an
estimate, not a quote.

If you want it cheaper, `--top-k 3` removes 6 of the 15 ablation passes and
cuts roughly 40% of the generation time. The paper's headline table is K=1
throughout, so the tail of the K-sweep is the first thing to trade away.

## 3. Pull the results

```bash
modal volume get ep-refusal 'results/**' ./runs/refusal_reference
```

Each seed directory holds `behavioral.json` (config, partition loadings, the
K-sweep across all three bases, the null ablation) plus the pickled dictionary.

## What to look at, in order

1. **`base_refusal_rates`** — harmful should be ~0.98, benign ~0.03. If harmful
   is low, the substring classifier is not firing on Gemma-2-27b's refusal
   register and every Δ below is meaningless. Check the `examples` blocks.
2. **`ablation.sweep_by_basis.exemplar[0].delta`** — the headline. Paper gets
   down to −0.96 at 2B.
3. **exemplar vs mean.** The paper's central claim is exemplar beats mean by
   0.4–0.6. Your Qwen-4B runs invert this. Which way 27B falls is the most
   diagnostic single number in the output.
4. **`cos_mean_exemplar_per_partition`** — ~0.94 in the paper, ~0.83 on your
   Qwen builds. This predicts (3) via the paper's sin²θ argument and is
   available *before* any generation, so it is the cheapest early signal.
5. **`null_ablation`** — must be ~0.00. If the size-and-coherence-matched
   non-refusal region also collapses refusal, the effect is not specific and
   nothing else in the file means what it appears to.

## Things worth knowing before you spend money

- **192 GB of host RAM is not padding.** TransformerLens does not stream
  weights: `from_pretrained_no_processing` materialises the full HF model on
  CPU, builds a converted state dict *alongside* it, and only then moves to
  GPU. Peak host RAM is ~2× the 54 GB of bf16 weights. The dashboard job's
  64 GB request would OOM here, and it would do so after the download.
- **p=12, not the script's default.** `exp_behavioral.py` defaults to
  `--percentile 8.0`, which the paper reports as a fragmentation *failure* on
  gemma-2-2b-it. The upstream default is the one value you specifically do not
  want. `modal/refusal.py` overrides it; if you invoke the script by hand, do
  not forget.
- **Calibration is shared across seeds by design.** The cache key is
  (model, hook, percentile) with no seed component, and `EP_CALIBRATION_CACHE`
  points at the volume. This is correct — the paper fixes build prompts and
  held-out set across seeds and varies only streaming order.
- **L35 is a starting point, not a commitment.** It is depth-matched to the
  paper's L20. But L20 was also where Gemma-2-2B's chat scaffold consolidates
  instruction-formatted prompts into a few dominant final-position regions, and
  there is no guarantee that co-occurs with 76% depth at 27B. If the top
  refusal region does not absorb ~250–450 of the 600 build prompts, try L30 or
  L40 before concluding anything about the method.
- **n=50 held-out is the paper's protocol, and it is noisy.** Binomial error on
  Δ is roughly ±0.07 there. Fine for replicating a −0.96, useless for
  distinguishing −0.15 from 0. Once you are past replication and into new
  claims, raise `--n-held-out-harmful` — the AdvBench + JailbreakBench pool is
  ~620 prompts, so with 300 build prompts there is room for ~200.
