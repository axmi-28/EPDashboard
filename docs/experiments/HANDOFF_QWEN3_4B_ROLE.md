# Handoff: localizing *role* in EP space on Qwen3-4B

Self-contained brief for a fresh session. Goal: take the role-confusion
construct from arXiv **2603.12277** and ask whether "role" is a **direction**
(what that paper assumes) or a **region** (what Exemplar Partitioning would
say), using the EP harness that has already been validated on this machine.

This follows a **successful** replication of the EP refusal result on
gemma-2-2b-it. Read §4 before designing anything — the refusal run left behind
two structural facts about EP geometry that directly constrain this experiment.

---

## 1. Use `Qwen/Qwen3-4B`. Not Qwen3.5-4B.

The task was originally scoped as "Qwen3.5-4B". **Do not use it.** Checked
directly:

```python
import transformer_lens.loading_from_pretrained as L
[n for n in L.OFFICIAL_MODEL_NAMES if '3.5' in n and 'qwen' in n.lower()]  # -> []
```

TransformerLens 3.6.0 has no Qwen3.5 support. Qwen3.5 is
`Qwen3_5ForConditionalGeneration` — a VLM whose text backbone is a hybrid of
Mamba-style linear attention and periodic full attention. Its config does not
even expose `num_hidden_layers` at the top level. This is exactly why
`qwen_ep/adapter.py` exists in this repo: it hooks the raw HF module tree
because TL cannot load the model.

That matters enormously here, because the validated harness
(`exemplar-partitioning/scripts/exp_behavioral.py`) is built entirely on
`HookedTransformer.from_pretrained_no_processing`. On Qwen3.5 you would be
forced onto the `qwen_ep` port — **and that port is the prime suspect for the
one anomaly in this project's results** (see §4). Do not build a new experiment
on top of unvalidated infrastructure.

`Qwen/Qwen3-4B` is strictly better for this task:

| | value |
|---|---|
| TL support | ✅ in `OFFICIAL_MODEL_NAMES` |
| layers / d_model | 36 / 2560 |
| **mid-layer** (paper's default depth) | **L18** |
| 77% depth (refusal-run analogue) | L28 |
| relation to paper | same generation as its `Qwen3-30B-A3B` |
| gated | no — no HF licence step |

And critically, **all five of the paper's roles have real dedicated tokens**:

```
<|im_start|> 151644   <|im_end|>  151645
<think>      151667   </think>    151668
<tool_call>  151657   </tool_call> 151658
<tool_response> 151665  </tool_response> 151666
```

This is why gemma-2-2b-it was abandoned for this experiment: it has **two**
roles (`user`, `model`), no system role at all (`TemplateError: System role not
supported`), no tool role, and no reasoning channel — so CoT Forgery, the
paper's headline attack, is not hard on gemma, it is *impossible*. Qwen3-4B has
the full role inventory in native tokens, which is what the paper's methodology
requires.

## 2. What the paper claims

**"Prompt Injection as Role Confusion"**, Charles Ye, Jasmine Cui, Dylan
Hadfield-Menell. ICML 2026, PMLR 306. arXiv 2603.12277. Code:
`github.com/role-confusion/prompt-injection-as-role-confusion`.

Thesis: LLMs infer *who is speaking* from **how text sounds**, not from its
role tag. Untrusted text that imitates a role inherits that role's authority.
"To the model, sounding like a role is indistinguishable from being one."

**Role probes** are the instrument. Construction (their Figure 5, and the
methodological core — do not deviate):

1. Sample **non-instruct** text from pretraining corpora (C4, DOLMA3).
2. Wrap **identical content** in each of the five role tags.
3. Train a linear probe per layer on the token's hidden state to predict role.
4. `Userness(t) := P(user | h_t)`, likewise CoTness / Assistantness / Toolness.

Holding content constant is the whole point: tags and style are perfectly
correlated in real conversation data, so a probe trained on real chat logs
learns the confound. Theirs can *only* have learned the tag's geometric
signature. Trained on 1,250 sequences per model, max seq len 1024. They report
**mid-layer** by default (peak semantic salience).

Headline results you can lean on:

- **Table 1** — Userness / Toolness of user-style text, correctly tagged
  `<user>` vs re-wrapped as `<tool>` (the canonical agent injection):

  | model | Userness (`<user>`) | Userness (`<tool>`) | Toolness (`<tool>`) |
  |---|---|---|---|
  | gpt-oss-20b | 99.7% | 87.6% | 9.3% |
  | gpt-oss-120b | 88.2% | 85.2% | 10.1% |
  | Nemotron-3 | 88.1% | 78.7% | 18.2% |
  | **Qwen3-30B-A3B** | **83.6%** | **75.7%** | **19.5%** |

  Userness stays **76–88%** under a `<tool>` tag; Toolness never exceeds 20%.
  Tags never reassert control at any depth.
- CoT-style text re-wrapped entirely in `<user>` tags stays at **85% CoTness /
  2% Userness**. Style hijacks the representation outright.
- **CoT Forgery**: 56–70% ASR against frontier models, near-zero baseline.
  Destyling (same argument, model's reasoning style stripped) collapses ASR
  **61% → 10%** — style is causal. Absurd vs plausible justifications are
  indistinguishable (60% vs 63%): the model does not scrutinize forged
  reasoning, it executes it.
- Role confusion **predicts ASR before a single token is generated**.

Real-conversation data comes from **OASST1** and **ToxicChat** (200
conversations).

## 3. The opening for EP — stated by the paper itself

§7.4, Open Questions:

> "linear probes assume roles occupy directional subspaces; we validate this
> through downstream prediction (confusion predicts ASR) and through
> convergence"

That assumption is precisely what EP relaxes. Their entire construct is
`P(role | h_t)` from a linear probe — role-as-direction. EP makes no
directional assumption: it hard-partitions the centered unit sphere into
regions anchored on **real observed activations**, and the anchor doubles as the
intervention direction.

So the question is well-posed and unanswered: **is role a direction, or a
region?** Two things EP buys you that a probe cannot:

1. **Discreteness.** "87.6% Userness" is a soft statement. "N% of re-tagged
   tokens land in the *same EP region* as genuinely-tagged user tokens" is a
   hard one.
2. **A causal handle.** The region's exemplar is a real activation you can
   project off. On gemma-2-2b-it refusal, the exemplar basis beat the mean basis
   by **0.58** — that asymmetry is EP's central mechanistic claim and it
   reproduced.

## 4. What this project already established — read this

**The EP refusal replication PASSED** on gemma-2-2b-it L20 p=12 seed 0
(2026-07-29), via the unmodified upstream harness on Modal:

| quantity | got | paper |
|---|---|---|
| n_partitions | 207 | 189–207 |
| baseline held-out refusal | 0.98 | 0.98 |
| base rates harmful / benign | 0.99 / 0.023 | 0.98 / ~0.03 |
| **Δ exemplar (K=1)** | **−0.76** | {−0.74, −0.96, 0, 0} |
| Δ mean | −0.18 | exemplar wins by 0.4–0.6 → here 0.58 |
| cos(mean, exemplar) | 0.90 | ~0.94 |
| null Δ (all 3 bases) | 0.00 | 0.00 |

Results at
`artifacts/runs/refusal_reference/results/gemma-2-2b-it/L20_p12_seed0/behavioral.json`;
summarize with `scripts/experiments/summarize_refusal.py`.

**Consequence for you:** the *reference harness is trustworthy*. A separate
Qwen3.5-4B port (`experiments/refusal.py`) gave best Δ −0.22 with **mean beating
exemplar** — inverted vs the paper — and cos ≈ 0.83. Since the reference
reproduces, that port is the suspect, not the method. **Fork
`exp_behavioral.py`; do not build on `experiments/refusal.py`.**

### Two structural findings that constrain your design

Decomposing the gemma seed-0 region loadings:

```
  pid  n_mem  harm_frac  n_harmful  n_benign  refusal
   18    405     0.7407      300.0     105.0     0.75
   82    130     0.0000        0.0     130.0     0.00
   92     59     0.0000        0.0      59.0     0.00
  109      4     0.0000        0.0       4.0     0.00
   45      2     0.0000        0.0       2.0     0.00
  sum                        300.0     300.0
```

1. **All 300 harmful prompts land in a single region.** Not most — all, exactly,
   with zero harmful members anywhere else. Its 75% "refusal rate" is just
   300/405. So the "refusal region" is really "the region every harmful
   instruction collapses into."
2. **Only 5 of 207 regions receive *any* final-position activation** (member
   counts sum to exactly 600). The other 202 exist only from mid-sequence
   tokens during the per-position build.

**Therefore: label per-token, not final-position.** At the final position,
gemma's L20 geometry was nearly degenerate; a role experiment run that way will
produce a 5-region answer that is trivially separable and tells you nothing.
Role is a property of *every token in a turn*, so per-token labelling is both
more faithful and vastly higher-statistics. Verify the same degeneracy holds (or
doesn't) on Qwen3-4B at L18 before committing — it may not.

## 5. Environment

Already set up and verified in `.venv` (py3.12):

```
torch 2.13.0   transformers 5.14.1   transformer-lens 3.6.0
datasets 5.0.0   scikit-learn 1.9.0   pypdf 6.14.2
ep 0.1.0 (editable, from exemplar-partitioning/)
```

`ep` is `pip install -e exemplar-partitioning`. `scripts/` has no
`__init__.py` and relies on namespace-package resolution, so `python -m
scripts.X` **only works with cwd at the upstream repo root**.

**Do not run this locally on MPS.** Measured on an M5: gemma-2-2b at **3.3
tok/s**, and TransformerLens emits
`UserWarning: MPS backend may produce silently incorrect results (PyTorch
2.13.0)` (TransformerLens#1178). For an experiment whose output is a number
compared against published values, a backend flagged for silent numerical error
is disqualifying. Use Modal.

## 6. How to run — Modal

`modal/refusal.py` at the repo root is wired and working. Read
`docs/experiments/MODAL_REFUSAL.md`; `docs/epdashboard/MODAL.md` covers volume mechanics.

- Secret `huggingface` exists and is valid (a previous attempt failed because
  the token had 3 stray characters — 40 chars instead of 37).
- Volumes `ep-hf` (weights) and `ep-refusal` (results) both
  `create_if_missing=True`.
- The 2B path is right-sized via `.with_options(gpu="A100-40GB", memory=32768)`;
  **Qwen3-4B needs its own sizing** — 36 layers × 2560, ~8 GB bf16, so
  A100-40GB with ~48 GB host RAM is ample (TL peaks at ~2× weights on CPU).

Pull results (note: no glob — `modal volume get` rejects glob suffixes now, and
the path has an extra `results/` level):

```bash
modal volume get --force ep-refusal results ./runs/refusal_reference
```

## 7. The experiment

The refusal handoff's recipe is "fork `exp_behavioral.py` and swap three
things." That holds, but one swap is much bigger than it looks.

**Tier 0 — validate the construct before spending anything.** Train a plain
linear role probe on Qwen3-4B and reproduce the paper's Table 1 row for the
Qwen family: user-style text at ~84% Userness under `<user>`, staying ~76%
under `<tool_response>`. **If Qwen3-4B shows no role confusion, there is
nothing for EP to localize and the rest is moot.** This is the
smoke-test-before-spend pattern that has worked twice in this project. It needs
no generation — Userness is a forward pass.

**Tier 1 — is role a region?** Build an EP dictionary at L18 on the paper's
Figure 5 corpus: identical neutral C4/DOLMA text wrapped in `user` vs
`assistant` (start with two classes, extend to five).

Then the three swaps:

1. **Prompt sets** (`_load_harmful` / `_load_benign`) → the two tag-wrapped
   versions of the same content.
2. **Scorer** (`_is_refusal`) → **you don't need one.** Because you wrapped the
   text yourself, role is ground truth, so the existing `harmful_fraction`
   field *already computes userness fraction* — no probe, no generation, no
   judge. The ~950 generations per seed collapse to ~600 forward passes:
   minutes, not an hour, cheap enough to sweep percentiles and many seeds.
3. **Threshold** (`--min-refusal-rate`) → **this is the big one.** Refusal had a
   ~3% base rate, so 0.3 was ~10× baseline. A balanced two-class role setup has
   a **50%** base rate, so 0.3 is *below chance* and selects everything.
   Recalibrate to purity against 50% (think `|userness − 0.5|`). The null
   control inverts too: a matched null is a region at **~50%** userness
   (role-mixed), not ~0%.

The comparison that answers the question: linear-probe AUROC (their method) vs
EP region membership as a role classifier. If region ID predicts role as well as
the probe, role is region-like. If the probe wins decisively, role really is
directional and EP adds nothing here — **that is a publishable negative and you
should report it as one.**

**Tier 2 — the confusion experiment, sharpened.** Take real user-style text
(OASST1 / ToxicChat, as the paper does), re-wrap it in `<tool_response>`, and
ask whether the **region assignment** follows style or tag. This is where EP is
strictly more informative than the probe: you get a discrete answer instead of a
probability, plus the exemplar anchoring the user-role region.

**Tier 3 — causal ablation, the refusal analogue.** Project off the user-role
region's exemplar during generation and measure whether the model still treats
user text as an *instruction* rather than as text to continue. Score with
**verifiable-constraint prompts** (IFEval-style: "answer in one word",
"translate to French") so the scorer is exact rather than an LLM judge. The
claim on offer: *the user-role region is causally necessary for instruction
framing.* Most interesting result available, least certain.

Keep the **null control** in every tier. Region selection is post-hoc on the
build set; without a size-and-coherence-matched null you cannot distinguish
"the role direction" from "any dense direction at this layer."

## 8. Traps

Carried over, all of them cost real time already:

- **`--percentile` defaults to 8.0**, which the EP paper reports as a
  fragmentation *failure*. gemma needed 12. **Do not assume 12 transfers** —
  the threshold is calibrated on *your* corpus, and neutral tag-wrapped C4 is a
  very different corpus from chat-formatted instructions. Sweep it.
- **Calibration cache key is `(model, hook, percentile)` with no seed.** But
  `_calib_iter` shuffles with `args.seed`. So `starmap`-ing all seeds at once
  makes each compute its own threshold and race to write the same path,
  confounding calibration variance with streaming order. **Run seed 0 alone
  first, then the rest.**
- **Don't change the extractor.** The build must use `extract_per_position`.
  Building on final-position activations gave Δ = exactly 0.00 at every layer in
  the Qwen port; the upstream docstring warns that mixing per-position
  calibration with final-position discovery "silently produces meaningless
  cells."
- **`model_info()` is not a gate/auth test.** It returns 200 for gated repos
  with no access. Use `whoami()` plus an actual file fetch. (Moot for Qwen3-4B —
  ungated — but it cost a container on gemma.)
- **`_load_harmful` has a silent fallback** to 17 embedded templates if both
  network paths fail, producing a structurally valid but meaningless run.
  `modal/refusal.py` now preflights this; keep the guard when you fork.
- **n=50 held-out is noisy** (binomial ±0.07). Fine for confirming −0.76,
  useless for distinguishing −0.15 from 0. Raise it for new claims.
- **Temper expectations on positive steering.** On refusal, ablation worked but
  adding α·exemplar did not: α ∈ {50,100} indistinguishable from baseline,
  α=200 gave apologetic prefixes that trip the classifier without coherent
  content, α=400 degenerated into token loops. Necessary ≠ sufficient. Expect
  the same asymmetry and don't read a steering null as evidence against
  localization.

New to this experiment:

- **Mid-layer vs late-layer.** The role paper reports **mid-layer** (L18 of 36);
  the refusal result was at **77% depth** (L28 equivalent). These are different
  claims about different depths — build at L18 first because that is where the
  paper says role lives, but a depth sweep is cheap here given no generation.
- **Style vs tag must be separable in your corpus.** The paper's whole
  contribution rests on holding content constant. If you build EP on real chat
  data instead of tag-wrapped neutral text, you learn the confound and the
  result is uninterpretable.

## 9. What success looks like

Not "EP finds a role region." Either of these is a result:

- **Role is region-like** — one EP region concentrates user-tagged tokens,
  re-tagged injected text lands in it anyway (reproducing the paper's confusion
  discretely), and projecting off its exemplar breaks instruction framing with a
  matched null at 0.00.
- **Role is directional** — the linear probe beats EP region membership, and the
  paper's assumption is vindicated against a method built to challenge it.

Report whichever happens. The refusal track's value came from the reference
harness reproducing cleanly; the same standard applies here.
