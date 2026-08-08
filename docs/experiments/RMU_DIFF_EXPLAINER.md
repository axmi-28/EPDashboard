# EP for model diffing — the RMU positive control, explained

Figures: `artifacts/figures/rmu_diff/fig01…fig12.png`. Full reports:
`GATE1A_RMU_DIFF.md`, `GATE1B_RMU_DIFF.md`, `GATE1C_RMU_DIFF.md`.

---

## What Exemplar Partitioning is, in one paragraph

Run text through a model and collect the residual-stream activation at every
token. Normalise them onto a sphere and carve the sphere into cells. The trick:
each cell is anchored on an **actually observed activation** — the first one to
arrive that didn't fit in any existing cell — rather than on a learned feature.
Stream data through; whenever an activation lands more than a radius θ from every
existing anchor, it opens a new cell. A "dictionary" is the resulting few hundred
regions, each with a real example at its centre. No training, very few tokens.

## Why it looks promising for model diffing

Because regions are anchored on observed activations, two models' dictionaries
are directly comparable — no alignment step, no matching of learned features.
Build one for model A and one for model B, match the regions, read off which are
**new**, which **vanished**, which **persisted**. That's a diff, and unlike a
supervised probe it needs no labels and returns a named object you can inspect.

---

## The experiment

**Models.** `HuggingFaceH4/zephyr-7b-beta` (base) vs `cais/Zephyr_RMU` (the
official RMU-unlearned checkpoint from the WMDP paper, arXiv 2403.03218).
Mistral-7B architecture, 32 blocks, d_model 4096. Revisions pinned.

**Why this pair.** This is a **positive control, not a discovery run.** The edit
is already characterised, so we know the right answer before we start:

- Diffing the weights shows only **3 of 291 tensors changed** —
  `mlp.down_proj.weight` in blocks 5, 6, 7 (fig 1).
- RMU's mechanism is known from its source: it drives hazardous-context
  activations at block 7 to **equal a single fixed random vector** `c·u`,
  ‖c·u‖ = 6.5, while regularising benign activations back toward the original
  model. Not "add a direction" — a point collapse.
- Prior work already found it by hand: activation norms jump at the loss site,
  and projecting out one direction recovers ~71% / ~45% of the WMDP-bio /
  WMDP-cyber capability gap (AlignmentForum, "Unlearning via RMU is mostly
  shallow").

If EP can't recover *that*, it won't recover anything subtle.

**Dataset.** 4,400 four-way multiple-choice prompts, 50/50:
- **hazardous / "forget"** — WMDP-bio + WMDP-cyber (`cais/wmdp`), the knowledge
  RMU removed
- **benign / "retain"** — MMLU (`cais/mmlu`), the knowledge it was regularised
  to keep
- Length-banded to 48–256 tokens and histogram-matched, because WMDP-cyber runs
  to 2,503 tokens against MMLU's median 112 and prompt length would otherwise be
  a confound.

### The steps

1. Confirm the intervention is present in *this* prompt set: 4-way MCQ accuracy
   for both checkpoints, in two prompt formats. RMU drops to chance on WMDP,
   MMLU is untouched (fig 2).
2. One forward pass of each model over all 4,400 prompts, taking the
   residual stream at **blocks 4, 7, 14 and 24**, at **every token position**
   (not just the last). Block 4 is upstream of every edited weight, so its
   activations must be bit-identical — a null control with a known answer.
3. Build EP dictionaries from those activations: 2 models × 4 depths ×
   2 cell radii (p = 10, 12) × **3 data orderings** = **48 dictionaries**.
   Both models see byte-identical prompts in byte-identical order.
4. Use **shared calibration** — one centre and radius, measured on the base
   model, applied to both. Under per-model calibration RMU's radius halves and
   the two dictionaries stop being comparable at all.
5. **Kill gate first, before reading any region.** Match base regions to RMU
   regions and count introduced / dropped / persisted. Run the *identical*
   procedure on two data orderings of the same base model — that is the noise
   this has to beat.
6. Only then look inside: label every member activation by which prompt it came
   from and ask what the changed regions actually contain.

---

## What we found

**The standard diff primitive found nothing (fig 5).** "Which regions are new?"
returned 3–5% at every depth — *below* the same-model reshuffle control (5–6%).
On its own that reads as "no signal".

**The other half of the same computation found it clearly (fig 5).** "Which
regions vanished?" returned **52%** at the edited block against 5% noise,
decaying with distance from the edit and exactly **0%** above it. RMU
**merges** regions rather than adding them, and a one-to-one matcher can only
express a merge as disappearance. The dictionary collapses from ~200 regions to
~100 and one region swallows 39% of all activations (fig 7).

**The measure you pick decides the answer (fig 4).** Asking "did the activation
move further than θ?" says 2% — nearly invisible. Asking "did it end up further
than θ from *every* anchor?" says 50%, against 1% for benign prompts. A cell
boundary can be crossed by a small move if you already sat near it.

**What is in the region (fig 8).** RMU's giant region is **93–96% hazardous
prompts** and holds **73% of all hazardous activations** in the run. The base
model's largest region sits at **0.47–0.50 — chance**. Base has no region that
separates hazardous from benign; RMU creates one. No labels were used to build
any dictionary.

**Where the dissolved regions went (fig 9).** Of the members of vanished base
regions, **86–93% of the hazardous ones** are funnelled into that single new
region — against **6–15% of the benign ones**. The regions weren't merged, they
were *sorted*, by exactly the property RMU was trained on.

**It is the injected direction, not an artifact (fig 10).** We re-ran both models
to estimate RMU's control vector from the activations. The region's stored
direction sits at cosine **0.73** to it, versus **0.30** to the "anti-mean"
direction that would indicate a centring artifact — stable to ±0.005 across six
independent builds. EP didn't just detect a change; the object it returns is the
thing that was injected.

### The uncomfortable finding

**EP's regions are largely artifacts of data ordering.** Rebuild the same
dictionary on the same model and the same data, changing only the streaming
order, and typical region-to-region member overlap is **~7%**. Whole-partition
agreement is ~0.55 (fig 11). This is why per-region matching was hopeless:
base-vs-RMU similarity was *identical* to base-vs-itself similarity — not because
the models agree, but because the measure has no headroom left.

### The finding that rescues it (fig 6)

At the edited block, the **base** model's largest region shares almost **zero**
members between two orderings. The **RMU** region is reproduced by every ordering
at **81–92%** shared membership. The injected structure is more stable than
anything EP finds in the untouched model — and that contrast is what made it
findable at all.

---

## What to take away

1. **The positive control passes.** Unsupervised, EP returns a single region that
   is 93–96% hazardous, holds 73% of the hazardous stream, and points at the
   injected vector.
2. **Don't use "which regions are new".** Use consolidation — dropped regions,
   partition agreement, dominant-region membership — measured on a shared
   activation stream.
3. **Always run the same-model reshuffle control.** Without it you cannot tell
   signal from ordering noise, and the noise is larger than intuition suggests.
4. **This was the easy case** — an edit that moves 38% of activation space.
   Nothing here shows a subtle diff is recoverable, and the ~7% ordering-noise
   floor is the reason to doubt it.

## Honest caveats

- Purity is measured on a **balanced 50/50 pool**; on a realistic stream with 1%
  hazardous content the same region would look very different.
- Labels are **prompt-level**, so benign filler tokens inside a WMDP question
  count as "hazardous" — purity is a lower bound on selectivity.
- At block 24, results swing **3–3.5×** on streaming order alone (fig 12). Only
  the edited block is tight enough to trust from few runs.
- The **chem transfer control was not run**. RMU had no chemistry forget corpus,
  so if the region also absorbs WMDP-chem prompts it is "hazardous technical
  content", not "what RMU removed". Cheapest remaining check on specificity.
- Prior work already found this direction by plotting activation norms. **EP's
  value-add over that simpler method is not yet demonstrated** — it would have to
  come from the region's readable contents, which nobody has inspected
  qualitatively yet.
