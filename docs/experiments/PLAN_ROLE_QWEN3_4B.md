# Plan: how does *role* live in EP space? — Qwen3-4B, L18

Execution plan for [HANDOFF_QWEN3_4B_ROLE.md](HANDOFF_QWEN3_4B_ROLE.md).

**The handoff's framing is abandoned deliberately.** It asks whether one EP
region concentrates user-tagged tokens. With content held constant across role
conditions, the dominant source of variance in a token's activation is *which
word it is*, so EP at token resolution partitions primarily by token semantics
and "the user region" almost certainly does not exist. That is not a negative
result about EP — it is the wrong unit of analysis. This plan treats **the
partition as a coordinate system**, not the region as an object.

Four instruments follow from that, and none of them require a user region to
exist. §1 is the new object. §5 is where EP is expected to earn its keep.

---

## 0. Verified facts this plan is built on

Checked directly against the tokenizer and the installed TransformerLens, not
taken from the handoff:

- `Qwen/Qwen3-4B` is in `OFFICIAL_MODEL_NAMES`. 36 layers, d_model 2560,
  mid-layer L18, ungated.
- Role tokens exist as claimed: `<|im_start|>` 151644, `<|im_end|>` 151645,
  `<think>` 151667, `</think>` 151668, `<tool_response>` 151665,
  `</tool_response>` 151666; `user`/`assistant`/`system`/`tool` are single
  tokens (872 / 77091 / 8948 / 14172).
- **Qwen3 has no top-level `tool` role.** `apply_chat_template` renders a tool
  message as `<|im_start|>user\n<tool_response>\n…</tool_response><|im_end|>` —
  tool output *is* a user turn. The paper's flat five-tag abstraction does not
  hold on this template, so §2 runs both the native nested form and a
  hypothetical flat `<|im_start|>tool\n…`.
- The assistant turn always injects an empty `<think>\n\n</think>\n\n` block.
- `bos_token` is `None`, so TransformerLens sets `bos = eos = <|im_end|>` and
  prepends it. `pad_token` stays `<|endoftext|>` (151643), which is **different**
  — so `extract_per_position`'s `lengths = (tokens != pad_id).sum()` is correct.
  Had they collided, every sequence length and final position would have been off
  by one, silently. The prepended `<|im_end|>` is an artifact at position 0,
  skipped by the extractor and identical across all conditions; noted, not fixed,
  because fixing it means touching the extractor.

---

## 1. The displacement map — the primary measurement

Every content token appears once per role condition, so we have
`r_user(d,i)` and `r_asst(d,i)` for identical content. Three outcomes,
distinguishable in one forward-only pass:

| outcome | signature | interpretation |
|---|---|---|
| **shared** | `r_user == r_asst` for most `(d,i)` | role is a within-region offset; directional, EP adds nothing |
| **coherent displacement** | regions differ, but `δ = e_{r_asst} − e_{r_user}` points the same way for every content token | role is **one direction acting on a content-partitioned space**; the probe finds it too |
| **conditional displacement** | regions differ, `δ` incoherent, but the coupling `r_user → r_asst` is stable | role is **content-conditional**; the probe's 84% is an average over incompatible local geometries, and EP is the only instrument that can say so |

Statistics:

- **Flip rate** `P(r_a ≠ r_b | content fixed)` for all 15 condition pairs.
- **Displacement coherence** `‖Σδ‖ / Σ‖δ‖ ∈ [0,1]` per ordered pair. 1 = one
  direction; ~0 = no shared direction. Compare against the coherence of
  displacements between two *random* region pairs of matched distance — the null
  is not zero, because any two regions at fixed cosine distance have nonzero
  expected alignment in 2560 dimensions.
- **Coupling stability**: the empirical `r_user → r_asst` transition matrix.
  Is it sparse? Row-stochastically concentrated? Does `user→assistant` share
  edge structure with `user→tool`? Same object as the competition-graph result,
  so ask it the same questions.

The third outcome is the one worth chasing and the one nothing in the paper can
express. **Honest prior: I expect coherent displacement** — role close to one
direction, EP re-deriving it. In that world EP's contribution is §5, not §1, and
that should be stated rather than discovered late.

## 2. Corpus — the component most likely to be silently wrong

Everything rests on holding content constant, so this is unit-tested locally
before any GPU is provisioned.

**Source.** `allenai/c4`, config `en`, streaming, fixed seed — the paper's
corpus. Reuse `qwen_ep/data.py`'s pattern: take a document, keep the first
`n_content` (default 96) tokens, decode back to a string, use that as the
invariant content `X`. `monology/pile-uncopyrighted` is a documented robustness
corpus, not the primary.

**Six conditions.**

| id | wrapper |
|---|---|
| `system` | `<\|im_start\|>system\n{X}<\|im_end\|>` |
| `user` | `<\|im_start\|>user\n{X}<\|im_end\|>` |
| `assistant` | `<\|im_start\|>assistant\n{X}<\|im_end\|>` |
| `cot` | `<\|im_start\|>assistant\n<think>\n{X}</think><\|im_end\|>` |
| `tool_native` | `<\|im_start\|>user\n<tool_response>\n{X}</tool_response><\|im_end\|>` |
| `tool_flat` | `<\|im_start\|>tool\n{X}<\|im_end\|>` |

Every suffix begins with a **special token**, never with a newline — a trailing
`\n` before `</think>` would merge with the last content token and break the
constancy invariant for that condition only. Every prefix ends with `\n`, so the
prefix→content boundary is the same character in all six.

Bare single turn, matching the paper's Figure 5 construction, even though a lone
`<|im_start|>assistant` block is off-distribution. A variant with a fixed
neutral preceding user turn is a robustness arm.

**Content-span location.** Not by prefix length — BPE can merge across the
boundary. Tokenize `X` standalone with `add_special_tokens=False`, then search
for that exact token-id subsequence in each wrapped tokenization. Keep the
document only if it is found in **all six** conditions at identical length. Drop
the rest and log the rate. This makes constancy a checked invariant.

**Scaffold tokens never enter a labelled statistic.** Tag tokens differ across
conditions by construction, so a region separating them detects token identity,
not role. They are fed to the EP build (as upstream does) but masked out of every
metric, and reported separately as a diagnostic.

**Content length must be exact, not nominal.** Measured on C4: taking the first
96 tokens of a document, decoding, and re-encoding returns **95 or 96** tokens,
not always 96. Constancy across conditions survives that (the string is what is
held fixed), but the paired assignment array `A[d, c, j]` that every metric
consumes has to be rectangular along `j`. `stream_contents` therefore skips
documents that do not re-encode to exactly `n_content` tokens, and
`experiments/role/preflight.py` fails if any raggedness survives. Making the metrics ragged
instead would have been strictly more code for the same answer.

**Scale.** 600 docs × 96 content tokens × 6 conditions ≈ 350 k labelled
activations, plus scaffold. Calibration wants 100 k. Split **by document**,
400 train / 200 test, so no content string appears on both sides. Measured at
200 docs: 1200 prompts, median prompt length 100 tokens, 115 k labelled content
activations + 6 k scaffold.

## 3. Occupancy and region polarity

Per condition, the occupancy distribution `p_role(r)` over all K regions.
`JS(p_user ‖ p_asst)` against a **shuffled-label null** answers "are user tokens
generally not where assistant tokens are" at the distributional level, which is
the right level. Per region,

```
λ(r) = log( p_user(r) / p_assistant(r) )
```

turns the partition into a labelled dictionary. λ is the only supervision used
downstream, and it comes from tags we applied ourselves — no probe.

**The role-NMI null must preserve the pairing.** Found in the CPU dry run: each
content token appears exactly once per condition, so at a low flip rate every
region receives an almost perfectly balanced condition mix, the empirical joint
factorizes, and the plug-in MI is essentially *unbiased*. Permuting condition
labels globally destroys that balance and manufactures bias from nothing — on 76
regions it produced a null of 0.105 against an observed 0.003, which would have
"proved" the absence of role information regardless of the data. The correct null
permutes conditions **within each (doc, position) group**, which keeps the design
and each region's member set intact and destroys only which condition landed in
which region (`metrics.normalized_mi_null_paired`). Global permutation stays
correct for the *content* label, which is not paired.

Regions with |λ| large and small member counts are noise; report λ with a
Jeffreys prior and a member-count floor, and check the λ ranking is stable
across streaming seeds (the exemplar-is-a-first-arrival-accident finding says
region *identity* is seed-dependent, so a λ ranking that reshuffles across seeds
is not a real object).

## 4. PCA — three of them, with in-repo precedent

`docs/experiments/ASSISTANT_AXIS_EP.md` already ran this protocol on a 4B and got
`cos(PC1, Axis) = 0.87`, with distance-to-anchor-cell recovering the linear
ordering at ρ = 0.86–0.92. Reuse it:

1. **PCA on the exemplar matrix `E` (K × d)** — the dictionary's own
   low-dimensional structure. Does λ load on the top PCs? If role polarity is a
   2–3 dimensional property of the partition, that *is* the answer, in EP's
   terms. Report variance explained and `corr(λ, PC_j)`.
2. **PCA on occupancy profiles** — per-(doc, condition) region histograms,
   `6N × K`. PCA in *region* space, no activations involved; this representation
   exists only because the partition is hard. Does PC1 separate roles?
3. **The EP role axis** `a = Σ_r λ(r) · e_r` — a difference-of-means in region
   space over unit-norm real activations weighted by discrete occupancy. Report
   `cos(a, PC1)` and `cos(a, probe direction)`. If `cos(a, probe) ≈ 0.95`, EP
   re-derived their axis training-free: modest, clean. If it is ~0.6 and `a`
   ablates better, more interesting.
4. **The m-dim role subspace**: top-m regions by |λ|, stack exemplars, QR.
   `exp_behavioral.py:551` already does exactly this
   (`np.linalg.qr(ablation_dirs[:k].T)`); the only changes are ranking by λ
   instead of refusal rate, and letting m run to 32–64 instead of 5.
5. **Distance-to-region-set** as a continuous scalar (`Dictionary.distances`,
   dictionary.py:684), giving EP a graded score without abandoning the
   partition — the bridge the Assistant Axis work validated at ρ ≈ 0.9.

## 5. Causal — two levels; the second is the EP-specific claim

**(a) Head-to-head subspace ablation.** Project off the m-dim EP role subspace
during generation, versus the linear probe direction (their method), versus a
dimension-matched null subspace drawn from λ ≈ 0 regions, versus the region-mean
basis. Score with **verifiable constraints** (IFEval-style: "answer in exactly
one word", "reply in French", "do not use the letter e") so the scorer is exact
string logic — not an LLM judge and not the refusal substring list. The question:
does user text still get *obeyed*, or does the model continue it as text? Run all
three bases (`mean`, `exemplar`, `exemplar_reanchored`) so the
exemplar-beats-mean asymmetry, EP's central mechanistic claim, is tested.

**(b) Region-gated intervention — the headline.** A projection is
unconditional: it hits every token at that layer. A hard partition lets you
intervene *only when the token's region is λ-polarized*. Compare on a
**selectivity frontier**:

- y: effect on role attribution (constraint-following rate),
- x: collateral damage — KL / perplexity increase on unrelated text,
- curves: global projection (sweep strength) vs region-gated projection (sweep
  the λ threshold), at matched effect.

If gating achieves the same effect while touching ~20% of tokens and leaving
perplexity intact, **that is a result a direction cannot produce** — and it holds
even in the boring §1 outcome where role is one coherent direction. Selectivity,
not effect size, is where a hard partition should beat a linear one. This is the
headline causal claim, not a stretch goal.

Expect ablation to work and positive steering not to, per the refusal run
(α ∈ {50,100} indistinguishable from baseline, α=400 degenerate). A steering
null is not evidence against localization.

**Practical payoff, free.** If λ-polarized regions are identifiable, then "tool
or retrieved text is occupying user-polarized regions" is a discrete,
thresholdable prompt-injection detector available before a single token is
generated — the paper's own claim that role confusion predicts ASR, but with
hard region IDs instead of a probability.

## 5b. Tier 0 RESULT (2026-07-30) — gate re-specified, post-hoc

Ran on the RunPod A100. **The gate as originally written FAILED, and the
threshold was mis-specified, not the model refuted.** Recording this in full
because the re-specification below was decided *after* seeing the data.

**Per-layer role-probe accuracy at 6 classes (chance 0.167), L18 corpus:**

| L | 0 | 3 | 6 | 9 | 12 | 15 | **18** | 21 | 24 | 27 | 30 | 33 | **35** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc | .352 | .423 | .333 | .374 | .391 | .379 | **.459** | .427 | .406 | .366 | .350 | .458 | **.664** |

**Bimodal, not the paper's mid-layer peak.** Local peak at L18 (so the paper's
depth is defensible), a **trough at L27–L30**, and a global peak at the final
layer (AUROC .903). Two consequences: L35's strength is discounted in
interpretation because the last block sits against the unembedding, where the
residual is dominated by next-token statistics rather than by "who is speaking";
and **role does not co-locate with refusal** — the gemma refusal effect was at
77% depth, which is exactly this trough.

**Why the gate failed.** `userness_under_user = 0.299` on OASST1 against a 0.70
threshold. But that threshold was transplanted from the paper's 0.836, which
comes from a probe near ~85% accuracy on a 20B–120B model. A 6-way probe at
0.450 accuracy assigns the true class ~0.3 by construction, so **0.70 was
unreachable and the gate tested the instrument, not the model.**

**What `experiments/role/diagnose_gate.py` established** (one L18 harvest):

- **Not a transfer failure.** `userness_under_user` C4 0.312 vs OASST1 0.260;
  4-way 0.416 vs 0.425. The train-neutral/test-real protocol transfers fine.
- **Not the position confound.** A 4-way probe over
  {system, user, assistant, tool_flat} — all 3-token prefixes, no positional cue
  — still gets 0.552 vs 0.250 chance.
- **Class dilution was not the real problem either.** Chance-normalised, the
  6-way probe is the *most* informative about `user` (1.87×) versus 4-way
  (1.66×) and binary (1.26×). Removing near-duplicate classes raised the raw
  probability without improving discriminability. The stable summary across every
  slicing is **AUROC ≈ 0.75–0.77**.
- **The paper's central claim reproduces.** Its headline "Toolness never exceeds
  20%" is, for a 5-way probe, *exactly chance* — Qwen3-30B 0.195 against a 0.200
  floor = 0.98× chance. Here: 0.260 against 0.250 = **1.04× chance**. The tag
  asserts nothing. Userness retention 0.425 → 0.324 = **76%** (paper 91%).
  Binary: user text keeps P(user) 0.449 under an `assistant` tag vs 0.656
  correctly tagged.
- Incidental: **`system` is far better identified** than any other role (0.724
  in-distribution vs 0.257–0.422), and role mass leaks preferentially to
  *structurally similar* conditions — user-tagged text puts 0.166 on
  `tool_native`, confirming that "tool output is a user turn" has a geometric
  consequence.

**Re-specified gate (post-hoc, approved 2026-07-30).** Absolute per-class
probability is not comparable across probes of different strength. Gate instead
on: (a) role linearly decodable, macro-AUROC > 0.70; (b) Userness retention under
a wrong tag > 0.60; (c) Toolness under the tool tag < 1.3× chance. Qwen3-4B
passes all three (0.774 / 0.76 / 1.04×).

**Why proceeding is the right call, stated plainly:** role *is* decodable at L18
(2.7× chance) so there is structure to localise, and the confusion effect is
present. Moreover AUROC 0.77 rather than 0.99 means the linear account is
incomplete — which is the regime where a method that makes no directional
assumption has room to contribute. That said, weak linear decodability is also
consistent with weak role structure of *any* kind, so §1's flip rates may still
come out low and the `SHARED` branch remains live.

**Layer sweep revised** to {18, 24, 30, 33, 35}: L18 primary, L30 as the trough
contrast, L33/L35 where signal actually peaks. The original {9, 14, 18, 22, 28}
was anchored on the paper's mid-layer and the refusal depth, and L28 is now known
to be the worst layer for role.

## 6. Tier 0 — the gate, run first

If Qwen3-4B shows no role confusion there is nothing to localize. Per-layer
multinomial logistic role probes on the tag-wrapped C4 train documents, evaluated
on test documents (**split by document** — a token-level split leaks content and
will report ~99% at every layer), then the paper's Table 1 row for the Qwen
family on real user-style text from `OpenAssistant/oasst1`:

| quantity | paper (Qwen3-30B-A3B) | gate |
|---|---|---|
| Userness under `<user>` | 83.6% | > 70% |
| Userness under `<tool_response>` | 75.7% | > 60% |
| Toolness under `<tool_response>` | 19.5% | < 35% |

One forward pass with 36 hooks captures every layer at once: a single
~15-minute container. `lmsys/toxic-chat` is a secondary source; OASST1 alone
suffices for the gate.

**If the gate fails, stop and report it.** A 4B not exhibiting a 30B's role
confusion is worth writing down, and it makes §1–§5 unanswerable rather than
negative.

## 7. Comparison arms — capacity matching

A linear probe has 2560 supervised parameters per class; EP region membership is
one unsupervised categorical. "Probe wins" is therefore the expected result of a
naive comparison and says nothing about EP. So every predictive comparison
carries a third arm: **k-means at matched K** on the same activations,
unsupervised, same discretization budget. Then EP ≈ k-means is a real negative,
and EP > k-means is a real claim about EP's partition specifically.

## 8. Sweeps and diagnostics

Percentile first at fixed L18: p ∈ {4, 8, 12, 16}, chosen on `n_partitions` and
the saturation curve, **not** on the metrics. Then layers at the chosen p: L9,
L14, **L18** (paper's mid-layer), L22, L28 (77%-depth analogue of the gemma
refusal result). Forward-only, so the grid is affordable.

Also record the gemma degeneracy check: how many regions receive any
final-position activation at L18. Cheap, and it is a direct claim about whether
Qwen3-4B's chat scaffold collapses the way gemma's L20 did.

## 9. Code layout

The vendored `exemplar-partitioning/` stays **untouched** — its value comes from
being the unmodified reference the refusal replication validated. New code is a
`role/` package at the repo root importing `ep` as a library:

```
experiments/role/corpus.py           # §2: six wrappers, span location, invariant checks
experiments/role/metrics.py          # §1, §3, §4: displacement, occupancy, λ, PCA, NMI
experiments/role/probe.py            # §6 Tier 0
experiments/role/exp_role.py         # driver: calibrate -> discover -> assign -> metrics
experiments/role/exp_role_causal.py  # §5, forked from scripts/exp_behavioral.py
role/tests/              # §10, no GPU
modal_role.py            # runner, mirrors modal/refusal.py
```

`modal_role.py` mounts `EP/ep` → `/root/ep`, `EP/scripts` → `/root/scripts`,
`role/` → `/root/role`, cwd `/root`, matching the refusal job's namespace
resolution.

**Calibration cache key.** Upstream keys on `(model, hook, percentile)` with no
seed, which the handoff flags as a write race across concurrent seeds. Two
fixes: the model-name field becomes `{model}__role` so it cannot collide with the
refusal calibration already on the volume, and `extras` carries
`{"corpus": ..., "n_content": ...}`. Seed 0 runs alone before the rest.

**Container.** `A100-40GB`, `memory=49152`. Qwen3-4B is ~8 GB bf16 and
TransformerLens peaks at ~2× weights on the host, so 48 GB is ample and the
27B's 192 GB is waste.

## 10. Local, before any GPU

Pure numpy / tokenizer, no model weights:

1. Six-wrapper construction on real C4 documents: assert content token ids are
   **byte-identical across all six conditions**, assert span indices are exact,
   report the drop rate.
2. Scaffold-mask correctness: assert no `<|im_start|>`, `<|im_end|>`, `<think>`,
   `</think>`, `<tool_response>`, `</tool_response>` or bare role-word token
   survives into the labelled set.
3. Metrics against synthetic assignments with analytically known NMI,
   displacement coherence, and flip rate.
4. Egress preflight in the `_load_harmful` spirit: assert the C4 and OASST1
   loaders actually returned data rather than silently falling through.

Tiers 0–4 are forward-pass dominated: single-digit dollars. §5 is the only
generation-bound tier.

## 10b. RESULTS (2026-07-30) — role is a sub-cell perturbation

Run on a RunPod A100-80GB. Eight configurations, ~30 min of compute. Raw output
in `artifacts/runs/role/`, logs in `artifacts/logs/role/`.

```
  L   p      K   flip   cos_d  thresh  ratio  beyond  NMIrole  NMIcont  AUROC finpos
 18   4   7197  0.036  0.0008  0.1837  0.004  0.0000   0.0011   0.8046  0.504      1
 18   8   5218  0.034  0.0008  0.1993  0.004  0.0000   0.0009   0.7596  0.504      1
 18  12   4168  0.035  0.0008  0.2094  0.004  0.0000   0.0008   0.7326  0.505      1
 18  16   3474  0.037  0.0008  0.2173  0.004  0.0000   0.0013   0.7128  0.507      1
 24   4   3830  0.029  0.0011  0.3832  0.003  0.0000   0.0010   0.7643  0.505      1
 30   4   4276  0.036  0.0017  0.5894  0.003  0.0000   0.0012   0.7268  0.505      1
 33   4   3041  0.040  0.0023  0.6709  0.003  0.0000   0.0011   0.6899  0.506      1
 35   4   5118  0.060  0.0037  0.6365  0.006  0.0000   0.0143   0.7284  0.570      2
```

`cos_d` = mean cosine distance between the two tagged copies of the *same*
content token; `thresh` = the calibration threshold, i.e. the cell radius;
`ratio` = cos_d / thresh; `beyond` = fraction displaced past the threshold;
`AUROC` = EP region membership as a role classifier (chance 0.5); `finpos` =
regions receiving any final-position activation.

### The finding

**Role displaces an activation by 0.3–0.6% of a cell radius.** Across four
resolutions and five depths, `ratio` never exceeds 0.006, and `beyond` is
**0.0000 in every configuration** — of ~920 k paired tokens, essentially none is
displaced past the threshold. This is what makes the result **resolution- and
depth-independent**: no percentile can make role a region, because the
displacement is two to three orders of magnitude below the cell scale. Getting
flips would need a threshold near 0.001, which puts roughly one activation in
each cell.

Everything else follows from that. Region membership predicts role at chance
(AUROC 0.504–0.507) while content NMI runs 0.69–0.80 — the partition is a
*content* partition. Occupancy JS sits on its null for every pair. λ never
exceeds 0.85. PC1 of the exemplar matrix explains 3.1%, so there is no
low-dimensional dictionary structure for λ to load onto (§4 answered, in the
negative).

### The magnitudes are internally consistent, which is the reassuring part

All-pair displacement at L18/p4 (threshold 0.1837), mean cosine distance:

| pair | cos_d | ratio | flip |
|---|---|---|---|
| user → system | 0.00288 | 0.016 | 0.072 |
| user → cot | 0.00262 | 0.014 | 0.061 |
| assistant → cot | 0.00143 | 0.008 | 0.041 |
| user → tool_native | 0.00111 | 0.006 | 0.035 |
| tool_flat → tool_native | 0.00094 | 0.005 | 0.035 |
| user → assistant | 0.00077 | 0.004 | 0.036 |
| user → tool_flat | 0.00075 | 0.004 | 0.034 |

The *ordering* tracks the Tier 0 probe's confusability exactly: `system` is the
most geometrically distinct role in EP space and also the best-identified by the
probe (0.724 diagonal vs 0.257–0.422 for the rest); `user` and `tool_flat` are
the least separable in both. **EP is measuring the right quantity — it is simply
below its resolution by ~250×.** That is a scale mismatch, not a broken metric.

L35 is the only config that moves at all (flip 0.060, NMI 0.0143 ≈ 13× the rest,
AUROC 0.570, 2 final-position regions) — the layer where the probe also peaked
(AUROC 0.903). Discounted in interpretation: the last block sits against the
unembedding, where the residual encodes next-token statistics rather than
speaker identity.

**`finpos` = 1 in seven of eight configs.** One region out of thousands receives
any final-position activation, versus gemma's 5/207. A final-position experiment
here would have been literally vacuous; the handoff's per-token instruction was
load-bearing, not a refinement.

### What this does and does not license

Supported: **role is linearly decodable and angularly tiny.** The paper's
directional assumption is vindicated on its own terms — AUROC 0.774 at L18, 0.903
at L35 — and a hard partition built on unsupervised geometry cannot see a feature
whose angular contribution is 250× below the threshold calibrated on that
geometry's dominant variance.

**Not supported: "EP is the wrong tool for roles" in general.** The design
*guaranteed* this outcome by construction. Holding content constant is what makes
this a clean test of whether the *tag* forms a region — and it also strips out
precisely the variance EP partitions on. Real user turns and real assistant turns
differ in content and style, and EP would very likely separate those; it just
would not be measuring role. The narrow, defensible claim is: **the tag's
contribution to the residual stream does not form EP regions.**

**The generalisable lesson is about EP's operating range**, not about roles. EP
resolves features that dominate local geometry — on refusal, all 300 harmful
prompts collapsed into a *single* region, an enormous effect — and is blind to
perturbations far below its calibration threshold. Before applying EP to a
construct, measure the construct's angular magnitude against the threshold. That
check is now `metrics.paired_displacement_magnitude` and costs one forward pass.

### Tier 3 (causal) cut — reasoning

The §5b region-gated intervention was the intended headline: intervene only on
λ-polarized regions and beat a global projection on the selectivity frontier.
**Tier 1 removed its object.** Gating needs λ-polarized regions; |λ| max is 0.85
with role NMI 0.001 and region-membership AUROC at chance. A subspace built from
that ranking would ablate an arbitrary direction and yield an uninterpretable
number, at the cost of the only generation-bound tier. Cut on the merits, not for
time. Seeds 1–3 cut for the same reason — λ-ranking stability is undefined when
there is no λ signal to be stable about. The k-means arm was cut because it is
informative only if EP beats chance, and EP does not.

## 11. What gets reported

- **Conditional displacement** (§1 outcome 3) — role is content-conditional,
  no single direction exists, and the probe's number is an average over
  incompatible geometries. Strongest available result.
- **Coherent displacement + EP gating wins** (§1 outcome 2 + §5b) — role is
  approximately one direction that EP re-derives training-free, and the
  partition buys *selectivity* the direction cannot. Narrower, still causal,
  still not something the paper can say. **Most likely outcome.**
- **Shared occupancy** (§1 outcome 1) — role never moves the region assignment.
  EP adds nothing at token resolution at this layer; report as a negative and
  say at which layers and percentiles it holds.

Numbers are interpretable only if §6 passes and the §10 invariants hold.
