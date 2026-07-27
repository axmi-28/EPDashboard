# EPDashboard — decisions & methodology log

A running log of the choices behind the tool, kept honest as it grows. The
reference point throughout is **SAEDashboard**
(github.com/jbloomAus/SAEDashboard); where a decision diverges from the SAE
idiom it is because EP geometry is genuinely different, and the reason is
recorded here.

## What it is

A tool that turns a built EP dictionary (`dictionary.pkl` + `metadata.json`
from a run dir) into:

1. **Raw data**: `header.json` + `regions_NNN.json` batches (default 256
   regions per file, mirroring SAEDashboard's `n_features_at_a_time` batching)
   under `<out>/<run_name>/`.
2. **Local HTML**: a self-contained `index.html` (dictionary-level region
   table) + one `regions_NNN.html` per batch (region cards). No server, no
   external assets — pages open from `file://`.

Two levels are planned: **dictionary level** and **region level** (the
SAEDashboard-feature-card analogue). The region level is built first; the
`regionTable` in `header.json` is deliberately the seed data for the future
dictionary level.

## The region card ↔ feature card mapping

| SAEDashboard (feature)      | EPDashboard (region)                                        |
|-----------------------------|-------------------------------------------------------------|
| top activating examples     | closest members by cosine distance to exemplar              |
| quantile-interval samples   | random samples from distance bands (near/mid/far) + a uniform random draw |
| per-token activation color  | per-token projection ⟨h−c, e⟩ onto the region direction     |
| activation histogram        | projection histogram (members vs corpus sample) **and** distance-to-exemplar histogram |
| logits (pos/neg tokens)     | logit lens of exemplar & mean direction (promoted/suppressed) + J-lens + verbalizability |
| frequency / sparsity        | member count, density (member share of scanned tokens), coherence |
| correlated features         | nearest regions by full-space cosine between exemplars      |

## Architecture: two passes (2026-07-26)

Same shape as SAEDashboard's gather-then-extract:

- **Pass 1** streams every activation once: center, normalise, assign
  (`argmax` cosine vs every exemplar), and update streaming accumulators —
  ranked/sampled example slots (vectorised per-region top-k, adapted from
  `qwen_ep.member_scan`), moments, fixed-bin histograms. Memory is
  O(K × slots), never O(tokens).
- **Pass 2** revisits only the prompts that won a slot (a few thousand) and
  recovers their *full per-position* activations, which per-token sequence
  coloring needs. In forward mode this is a second, tiny forward pass; in
  cache mode a second filtered shard read. Storing per-token data for every
  streamed prompt in pass 1 would be the alternative, and it is exactly the
  ~80 MB-per-dictionary mistake the old brainstorming dashboard measured.

Activation sources are pluggable behind one interface (`source.py`):
`ForwardSource` (any HF text dataset, streamed and shuffled, decoded
128-token windows — identical to dictionary construction so examples land on
the same tokens) and `CacheSource` (`qwen_ep.extract_cache` shards; no model,
no GPU — mandatory at 27B scale where the forward pass is the entire cost).

**Dependency seams** on `qwen_ep` (to vendor if the tool is spun out):
`adapter.QwenModel` (residual-stream hook), `lens_weights`/`jlens_weights`
(unembedding + Jacobian lens fetching), and the `extract_cache` shard format.

## p is a run parameter, not a rebuild

Multiple run dirs may share one build (`--run-dirs a,b,c`) as long as they
share model and layer. The activation stream is consumed once and every
dictionary's accumulators update in the same loop — so sweeping `p` costs one
forward pass total. This mirrors how the dictionaries themselves were built
(`sweep_p` over one cache).

## Honesty choices (what makes an EP dashboard different)

These come from measured findings in the exploratory phase; they shape the
default panels:

- **"Closest members" is labeled exactly that — not "top activating".** The
  exemplar is the *first-arrival* activation of the region, so any ranking
  against it inherits an accident of stream order, and top-projection ranking
  is a mild perturbation of closest-ranking (order correlation +0.84 with
  cosine sim). The card leads with closest for SAEDashboard familiarity, but
  the band/random groups sit directly beneath as the honesty check.
- **Distance bands are fixed thirds of [0, θ], not quantiles of the observed
  distances.** Quantile bands would always look balanced; fixed bands expose
  the real geometry — the median member sits ~90% of the way to the wall, so
  a typical region shows a thin "near" band and a crowded "far" band. That
  emptiness is the finding, not a bug. Samples within a band are uniform
  (bottom-k on a random key = reservoir without replacement).
- **A uniform random draw is always shown.** Measured on qwen27b-L55-p8:
  top examples oversell coherence badly (random median d=0.795 vs θ=0.887).
- **Per-token coloring is projection, membership is marked separately.**
  Tokens are colored by ⟨h−c, e_region⟩ (diverging: blue positive, red
  negative, alpha ∝ |proj|); tokens that actually *belong* to the region
  (argmax + within θ) get a dotted underline, the firing token a solid one.
  Projecting onto a direction and belonging to the cell are different facts
  in EP, and conflating them is the "region is a direction" fallacy.
- **Two histograms, not one.** The projection histogram (member reservoir vs
  a shared uniform corpus subsample, peak-normalised overlay) answers the SAE
  question "how does this direction fire over the corpus?". The
  distance-to-exemplar histogram over [0, θ] answers the EP question "how
  tight is this cell?" — it has no SAE analogue and is the region's shape in
  one glance.
- **Density replaces sparsity.** density = members-in-scan / all scanned
  tokens. Unlike an SAE feature, every token belongs to exactly one region,
  so densities sum to ~1 over the dictionary.

## Projection and magnitude

EP works exclusively on unit directions; magnitude is discarded at build
time. The scan recovers it for free: assignment already computes
`dirs @ E.T`, and `proj = cos · ‖h−c‖`. All projection numbers in the
dashboard are ⟨h−c, e⟩ in the *centered* space (center from the dictionary's
calibration), matching the intervention direction the exemplar defines.

## Lens methodology

- Directions are RMS-normalised before unembedding (final RMSNorm weights ×
  tied/untied W_U fetched per-tensor via HTTP range requests, npz-cached), so
  softmax temperature is canonical and entropies are comparable across
  regions. **Verbalizability = 1 − H/ln|V|.**
- Both **promoted and suppressed** tokens are shown (SAEDashboard's pos/neg
  logits), for both the exemplar direction and the mean member direction —
  their disagreement is itself a seed-representativeness signal.
- **J-lens** columns appear only when a Jacobian lens exists for the
  model/layer (`payload.lens.jlens` flag; the 27B lens exists — check the HF
  repo file list before assuming absence). The fit budget (`jNPrompts`) is
  printed on the card because verbalizability is **not comparable across
  models with different fit budgets** (4B-it median 0.78 vs 27B median 0.15).
- Untied-embedding gotcha: on Qwen3.6-27B the unembedding is `lm_head.weight`,
  not `embed_tokens` — pointing at the wrong tensor silently yields a wrong
  lens. Model entries live in `qwen_ep.lens_weights.SPECS`, keyed off the HF
  id here.

## Replay check

Pass 1 re-streams the corpus with the same seed, but any drift (dataset
mutation, tokenizer change, different budget) would silently attach examples
to a different activation distribution than the dictionary was built from. So
the scan-vs-stored **member-share correlation** (scale-free; counts only
match at full budget) is computed every run and printed on every page header.
Low correlation ⇒ the examples describe a different stream — rebuild or fix
the seed/dataset before trusting the cards.

## Defaults (and why)

| knob | default | rationale |
|------|---------|-----------|
| dataset | `monology/pile-uncopyrighted`, 128-token windows | matches dictionary construction & the paper |
| n_prompts | 24 576 | SAEDashboard/Neuronpedia's `n_prompts_total` ballpark (~3.1M tokens ≈ paper build budget) |
| position 0 | skipped | attention-sink slot, matches dictionary construction |
| n_closest / n_per_band / n_random | 10 / 5 / 10 | SAEDashboard-ish card length |
| buffer | (10, 5) tokens | SAEDashboard `buffer` concept; asymmetric because the left context is what makes the firing token readable |
| hist bins | 40 | SAEDashboard convention |
| reservoir | 256/region | backs quantiles + member histogram + random draw |
| bg_sample | 8 192 tokens | corpus background costs bg_sample × K floats of RAM — cap accordingly at large K |
| regions_per_batch | 256 | keeps single JSON/HTML files in the tens of MB |

## Output format notes

- `header.json` carries provenance (config snapshot in `<out>/config.json`,
  source description, replay stats), the batch manifest, and the
  `regionTable` summary used by both the index page and the future
  dictionary-level view.
- Sequence records ship **only windowed tokens** (`tok` strings + `act`
  per-token projections + `mb` member indices + `fi` firing index), never full
  prompts — the payload lesson from the old dashboard. `act` is `null` at
  position 0 (no activation exists there).
- JSON is embedded into HTML with `</` escaped (`<\/`) so text containing
  `</script>` can't break the page.

## HTML

Self-contained pages, no server, light/dark via `prefers-color-scheme`;
palette and chart rules follow the dataviz reference palette (blue =
member/positive, red = negative pole, gray = corpus background; single-hue
histograms; tooltips on tokens and histogram bars). One page per batch with
an in-page region selector + prev/next that hop across batch files via plain
links; `#r<i>` anchors deep-link a region.

## Validation (2026-07-26 smoke run)

`--run-dirs runs/qwen3_5-2b_L12_p8p0_… --n-prompts 400` on the M5 (MPS):
50,800 activations, **member-share corr 0.973** at 12% of the build budget —
the replay check works and the rescanned mix matches. Checks that now gate
changes:

- `scratch/check_epdash.py`-style JSON assertions (hist lengths, firing token
  always has an activation value, batch manifest consistency, HTML payload
  round-trip after `<\/` unescaping);
- a jsdom DOM smoke (both pages execute their scripts; region page renders
  2 histograms, 5 sequence groups, colored tokens, one `fire` token per
  sequence; selector navigation re-renders; index default-sorts members
  desc) — the old dashboard's jsdom gotchas (matchMedia, rAF) don't apply
  because this template uses neither;
- headless-Chrome screenshots, eyeballed. Region 0 came out as the known
  "Q:" interrogative region with J-lens " Asking/ wondering/ questioning" —
  panels mutually consistent.

Content sanity from the card itself: closest members all sit at d≈0 (several
prompts tie with the exemplar's token), the far band and random draw are
dominated by d≈0.7–0.84 members — the "top examples oversell coherence"
finding, now visible on every card by construction.

Perf note: streaming-dataset startup (shuffle-buffer fill) dominates small
runs (~8 min for 400 prompts, of which forward was ~1 min). Don't extrapolate
smoke-run throughput — same lesson as the 27B work.

## Open / deferred

- Neuronpedia import compatibility: explicitly out of scope for now.
- Dictionary-level dashboard (level 1): next; consumes `header.json`.
- Modal deployment for big models: the tool is already GPU-agnostic
  (`--cache-dir` path); a Modal runner would wrap `ForwardSource` on an A100
  box and ship `<out>/` back.
- Quantile-band alternative (`--band-mode quantile`) if fixed bands prove too
  empty at small p.
