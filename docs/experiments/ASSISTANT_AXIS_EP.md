# The Assistant Axis in EP space — replication & discretization

Replicating **"The Assistant Axis: Situating and Stabilizing the Default Persona
of Language Models"** (Lu, Gallagher, Michala, Fish, Lindsey — Anthropic/MATS,
[arXiv:2601.10387](https://arxiv.org/abs/2601.10387), code
[safety-research/assistant-axis](https://github.com/safety-research/assistant-axis))
on **Qwen/Qwen3.5-4B**, then re-deriving its central construct in **Exemplar
Partitioning (EP)** space.

This is the validation step *before* the persona-drift-as-cell-trajectory
program: the paper gives us a published, external standard for persona geometry
(a linear "Assistant Axis"), so we can check whether EP's hard partition recovers
the same structure before building drift machinery on top of it.

**TL;DR.** Both halves pass. (1) The linear Assistant Axis reproduces on a 4B:
`cos(PC1, Axis) = 0.87`, the default Assistant sits at the extreme of PC1, and a
4-D persona space orders roles fantastical→Assistant exactly as the paper reports.
(2) In EP, the Assistant is not a direction but a **consolidated, pure region** —
a single anchor cell that is 96% assistant-composition and that fantastical
personas essentially never enter. The EP scalar "distance-to-anchor-cell"
recovers the linear axis ordering at **Spearman ρ = 0.86–0.92**. Caveat: personas
are more *distributed* across cells than refusal was, so drift should be tracked
as **distance-to / membership-in the Assistant region**, not "which single cell."

---

## 1. Background: two ways to represent "the Assistant"

**The paper (continuous / linear).** A model can play many characters; post-training
pins it to a default "Assistant." They map persona space by extracting a mean
activation vector for each of 275 character archetypes (from role-play rollouts),
run PCA, and find **PC1 = "Assistant-likeness."** They define the **Assistant
Axis** as a difference-of-means contrast vector, `mean(default) − mean(all roles)`,
which is ≈0.71-aligned with PC1. Projecting an activation onto this axis gives a
scalar; that scalar **declines over a conversation as the model drifts persona**,
and clamping it (activation capping) stabilizes behavior.

**EP (discrete / partition).** EP tiles activation space into Voronoi-like cells
by leader-clustering on **unit directions** (magnitude is discarded), giving every
activation a **hard, mutually-exclusive cell assignment**. There is no global
"direction" in EP — so the Assistant Axis has no literal analog. The natural
translation is:

| Paper construct (continuous) | EP analog (discrete) |
|---|---|
| Role vector = mean resid over response tokens | Same vector → `dic.assign` → a **cell** (or a distribution over cells) |
| Persona space (PCA of role vectors) | The **partition itself**; "is there an Assistant cell?" |
| Assistant Axis = `mean(default) − mean(roles)`, a **direction** | **Assistant anchor region** (cell) + the scalar **distance-to-anchor** |
| Projection onto axis (continuous magnitude) | Distance-to-anchor exemplar, or hard **membership** in the region |
| Persona drift = projection slides down | Trajectory **leaves the Assistant region** (cell path) |
| Activation capping (threshold on scalar) | **Cell-conditional** intervention (nudge back when cell ∉ region) |

The question this document answers: **does the EP partition — built training-free
on generic Pile text, with no knowledge of personas — recover the same
Assistant-vs-not geometry the supervised difference-of-means axis finds?**

---

## 2. Methodology

### 2.1 Data (vendored from the paper's repo → `experiments/persona/assets/`)
- **Roles**: `role_list.json` (275 archetypes) + `instructions/<role>.json`, each
  with up to 5 `pos` elicitation system prompts and a `default.json` capturing the
  model *as itself* (`""`, "You are an AI assistant.", "Respond as yourself.", …).
- **Questions**: `extraction_questions.jsonl` — 240 persona-revealing questions.
- **Curated spectrum** (`persona_data.py:SPECTRUM_*`): 26 roles + default spanning
  the paper's PC1 — 10 Assistant-end (assistant, tutor, editor, consultant,
  researcher, counselor, accountant, analyst, translator, +default), 5 human
  (activist, actor, gamer, comedian, coach), 12 fantastical (aberration, oracle,
  egregore, ghost, leviathan, bard, prophet, demon, eldritch, spirit, mystic,
  trickster).

### 2.2 Reduced-scale configuration (and divergences from the paper)
Run on a single M5 Mac (MPS, forward-only), so we scale down and **flag every
divergence**:

| | Paper | This run |
|---|---|---|
| Roles | 275 | 27 (curated spectrum) |
| Rollouts / role | ~1200 (5 sys × 240 q) | 48 (3 sys × 16 q) |
| Decoding | temperature 0.7 | **greedy** |
| Role-vector filter | LLM-judge score = 3 only | **none** (averages all rollouts) |
| Models | Gemma-2-27B, Qwen-3-32B, Llama-3.3-70B | Qwen3.5-4B |
| Layer | middle (~50%) | **L16** (middle) + **L27** (existing dicts) |

Total: **1296 rollouts**, 48 new tokens each, L16+L27 captured in one pass.
Wall time ≈ 35 min (30 min generation + 5 min extraction).

### 2.3 Activation extraction (`adapter.mean_response_activations`)
Per rollout: greedily generate the assistant response, then **teacher-force** the
full `(system, user, response)` chat through the model once and mean-pool the
post-block residual stream over the **assistant response-token span**, at every
requested layer simultaneously (one hook per layer). This mirrors the paper's
"mean post-MLP residual over response tokens" and — critically — extracts over the
**generated content**, not the pre-generation template token. (Earlier refusal
work established that final-position / scaffold-token extraction is degenerate on
Qwen3.5 because the `<think>…</think>` template makes every prompt's last token
near-identical; response-token pooling sidesteps that.)

### 2.4 P1 — axis + PCA (`persona_axis.py`)
- **Role vector** = mean over that role's 48 rollout activations, per layer.
- **Assistant Axis** = `mean(default) − mean(non-default role vectors)`, per layer.
- **PCA**: standardize role vectors (subtract cross-role mean), fit 10 PCs; report
  `cos(PC1, Axis)`, PC1 variance explained, dims for 70% variance, and the
  default's relative position along PC1 in `[0,1]` (0/1 = the two extreme roles).
- Saves per-layer axis + role vectors to `axis_L{L}.npz` for P2.

### 2.5 P2 — EP localization + bridge (`persona_localize.py`)
- Assign every rollout's L27 activation to an existing EP dictionary via
  `dic.assign` (raw activation → centered, unit-normalized, nearest exemplar).
- **Assistant anchor** = the modal cell of the *default* rollouts.
- **Purity** = P(role ∈ assistant-group | cell = anchor); **coverage** = fraction
  of default rollouts landing in the anchor.
- Per role: cell histogram, top-cell fraction (localization), fraction-in-anchor,
  and **mean distance to the anchor exemplar** (`1 − cos(dir(h), e_anchor)`).
- **Bridge metric** = Spearman(linear axis projection, −distance-to-anchor) across
  roles — does the discrete geometry recover the continuous ordering?

---

## 3. Results — P1: the linear axis reproduces on 4B

| Metric | Paper | L16 | L27 |
|---|---|---|---|
| `cos(PC1, Assistant Axis)` | >0.71 (mid-layer) | **0.871** | 0.862 |
| PC1 variance explained | — | 0.503 | 0.449 |
| Dims for 70% variance | 4–19 | **4** | 4 |
| Default position on PC1 ∈ [0,1] | ~0 or 1 (extreme) | **1.000** | 0.984 |
| Axis L2 norm | — | 2.13 | 8.10 |

The role ordering along the axis is exactly the paper's shape — fantastical
characters at the negative extreme, professional/Assistant roles at the positive,
default Assistant furthest positive. Qualitatively the rollouts confirm the
paper's "steering away induces a mystical, theatrical style": e.g. to the same
question, *default* answers as a helpful assistant ("I'd love to help you adjust
the plan… could you share…"), *oracle* answers "I hear the weight of your reality
pressing against the vision I offered…", *demon* answers "*Giggles, the sound like
shattering glass in a silent cathedral.* Ah, the little mortal!…".

---

## 4. Results — P2: the Assistant is a pure region, and EP agrees with the axis

### 4.1 Dictionary-resolution sweep (L27, Pile-built dicts)

| Dict | K (cells) | Anchor cell | Anchor purity | Default coverage | **Bridge ρ** |
|---|---|---|---|---|---|
| p8 | 638 | #578 | **0.965** | 0.354 | +0.859 |
| p4 | 3,275 | #1707 | 0.946 | 0.188 | **+0.921** |
| p2 | 16,528 | #8385 | 0.919 | 0.062 | +0.810 |

- **The Assistant anchor cell is real and pure**: 92–96% of rollouts landing in it
  are assistant-group, at every resolution.
- **The partition recovers the linear axis**: distance-to-anchor orders roles the
  same way the difference-of-means projection does, ρ = 0.81–0.92.
- **Resolution trade-off**: coarser (p8) consolidates more Assistant mass into one
  cell (coverage 0.35) → best for a drift *anchor*; finer (p4) sharpens the
  ordering (ρ 0.92) but splinters the mass; p2 over-fragments (coverage 0.06).

### 4.2 Full role table (L27, p8 dict; sorted by axis projection)

`proj` = linear axis projection · `top%` = largest single-cell share (localization)
· `inAnc` = fraction landing in the Assistant anchor · `d→anc` = mean distance to
anchor · `#cells` = distinct cells the role's 48 rollouts occupy.

| role | group | proj | top% | inAnc | d→anc | #cells |
|---|---|---:|---:|---:|---:|---:|
| ghost | fantastical | −12.14 | 0.56 | 0.00 | 0.938 | 4 |
| oracle | fantastical | −11.91 | 0.48 | 0.00 | 0.929 | 4 |
| prophet | fantastical | −11.17 | 0.71 | 0.00 | 0.915 | 6 |
| demon | fantastical | −11.00 | 0.90 | 0.00 | 0.952 | 4 |
| eldritch | fantastical | −10.90 | 0.73 | 0.00 | 0.925 | 4 |
| leviathan | fantastical | −10.83 | 0.75 | 0.00 | 0.925 | 6 |
| bard | fantastical | −10.52 | 0.60 | 0.00 | 0.956 | 8 |
| spirit | fantastical | −9.61 | 0.81 | 0.00 | 0.895 | 5 |
| mystic | fantastical | −9.57 | 0.52 | 0.00 | 0.904 | 6 |
| egregore | fantastical | −8.02 | 0.60 | 0.00 | 0.880 | 6 |
| trickster | fantastical | −7.03 | 0.35 | 0.00 | 0.962 | 11 |
| actor | human | −6.98 | 0.54 | 0.00 | 0.911 | 11 |
| aberration | fantastical | −6.08 | 0.52 | 0.02 | 0.887 | 8 |
| comedian | human | −4.58 | 0.50 | 0.00 | 0.950 | 9 |
| gamer | human | −3.62 | 0.54 | 0.00 | 0.933 | 8 |
| activist | human | −1.20 | 0.21 | 0.08 | 0.842 | 9 |
| counselor | assistant | +0.60 | 0.71 | 0.04 | 0.829 | 6 |
| coach | human | +0.95 | 0.65 | 0.02 | 0.840 | 9 |
| accountant | assistant | +1.15 | 0.35 | 0.35 | 0.832 | 12 |
| consultant | assistant | +1.42 | 0.46 | 0.46 | 0.816 | 8 |
| researcher | assistant | +1.50 | 0.42 | 0.42 | 0.815 | 8 |
| tutor | assistant | +1.58 | 0.27 | 0.25 | 0.831 | 9 |
| analyst | assistant | +1.78 | 0.52 | 0.52 | 0.817 | 7 |
| editor | assistant | +1.83 | 0.27 | 0.27 | 0.826 | 11 |
| translator | assistant | +2.45 | 0.38 | 0.38 | 0.820 | 7 |
| **assistant** | assistant | +3.23 | 0.40 | 0.40 | 0.808 | 7 |
| **default** | assistant | +3.54 | 0.35 | 0.35 | 0.804 | 6 |

Read the `inAnc` column top-to-bottom: it is **0.00 for every fantastical role**
and jumps to 0.25–0.52 for the Assistant-end roles. The Assistant region is a
gate fantastical personas do not pass. `d→anc` is monotone-ish with the linear
projection (that monotonicity is the ρ = 0.86 bridge).

---

## 5. What this means *for EP*, not for the linear axis

The paper's finding is "the Assistant is one end of a **line**." The EP finding is
different in kind and is the point of this exercise:

1. **The Assistant is a *place*, not a direction.** EP discards magnitude and
   works on hard cell membership, so there is no "how far along the axis" within a
   representation — instead there is a concrete **anchor region** (cell #578) that
   is ~96% Assistant-composition. "Assistant-ness" becomes a **yes/no membership
   event** plus a **distance** to that region, not a continuous coordinate.

2. **The unsupervised partition rediscovers the supervised axis.** The dictionary
   was built by leader-clustering generic Pile activations — it never saw a
   persona label or a contrast pair. Yet distance-to-anchor reproduces the
   difference-of-means ordering at ρ ≈ 0.86–0.92. The Assistant/not-Assistant
   split is therefore **structure the model already carves into its activation
   geometry**, not an artifact of the supervised contrast the paper constructs.

3. **EP keeps the *identity* of "not-Assistant"; the axis collapses it.** A low
   axis projection says only "away from Assistant" — mystic, jailbreak-persona,
   and eldritch all map to the same scalar. EP puts ghost, demon, oracle in
   *different cells*. So for drift, EP can say **which** persona captured the
   conversation, where the scalar can only say the model left.

4. **Discreteness gives a hard "capture" event.** Drift on the axis is a smooth
   decline crossing an arbitrary threshold; drift in EP is the turn the trajectory
   **exits the anchor cell** (or enters another persona's region and stays) — a
   well-defined event the partition hands you for free.

5. **The cost: personas are distributed, not single-celled.** Unlike refusal
   (which concentrated ~99% into one region), a persona's 48 rollouts spread over
   6–12 cells (`top%` ≈ 0.3–0.5). Assistant-likeness lives at the **region** level,
   not the single-cell level. Practical consequence for the drift program: the
   observable must be **distance-to / membership-in the Assistant anchor region**,
   aggregating the cells around the anchor — not "which exact persona cell is the
   model in." The consolidated, pure anchor makes this well-posed; per-role cell
   purity does not.

6. **Capping becomes cell-conditional.** The paper caps a scalar
   (`h ← h − v·min(⟨h,v⟩−τ, 0)`). The EP-native analog triggers on the discrete
   fact "the current cell is outside the Assistant region" and nudges the residual
   back toward the anchor exemplar — arguably a more natural intervention than a
   threshold on a continuous projection. (Planned; not yet run.)

**Bottom line.** EP does not "have" an Assistant Axis — it has an **Assistant
region**, and the scalar you'd read off the axis is replaced by distance to that
region. The two agree strongly on ordering (ρ ≈ 0.86–0.92), which validates using
the EP region as the substrate for persona-drift tracing, while EP additionally
supplies discrete capture events and the identity of the drifted-into persona that
the 1-D axis cannot.

---

## 5b. P3 — drift as a traced trajectory (twin signal)

We ran three 8-turn conversations on Qwen3.5-4B — two of the paper's documented
drift drivers (pushing the model to **meta-reflect on its own nature**; an
**emotionally-vulnerable user**) plus a **benign control** — and read out two
signals per assistant turn: the linear axis position (L16) and the EP cell
assignment (L27 p8). Code: `persona_drift.py` (+ `mean_last_turn_activations` in
the adapter). Figure: artifact `0e5484d8`.

**Anchor-transfer finding (methodological).** The Assistant anchor cell #578 from
P2 was derived from *averaged role-vector means*; live *conversational* assistant
turns do not land there. Worse, a 24-prompt benign calibration bank has no tight
home cell either (modal cell #625, coverage only ~0.12). **So assistant behaviour
is genuinely distributed across cells at p8, and single-anchor EP membership is a
noisy scalar.** The tracer therefore calibrates a live-home distance distribution
from benign turns and reports departures from it, but the robust EP object is the
**cell-identity path**, not a membership detector.

**Results (per-turn, 3 scenarios):**

| Scenario | Linear drift (axis range) | Distinct cells | EP capture | Reading |
|---|---|---|---|---|
| Benign control | 0.11 | 3 (task-help cluster) | none | stays home |
| Meta-reflection | 0.21 | 2 (#44/#324) | **turn 2 → #44** | enters self-referential territory |
| Vulnerable user | **0.34** | 3 (#137/#324/#520) | none | tone slide, no capture |

**The two signals are complementary, not redundant:**
- **The axis is sensitive to register/magnitude** — it flags the vulnerable-user
  conversation most (0.83→0.49) as the model warms into a counselor-like tone.
- **EP is sensitive to territory/identity** — it flags meta-reflection, which parks
  in cell **#44** (self-referential "mechanics of the process / no boundaries to
  look past" talk), a region ordinary help never occupies. The axis magnitude alone
  smears this (its meta trace is noisy mid-range).
- **Cell #324** — a reflective/boundary register — is the single region *both* drift
  types pass through, a measurable point of contact.

**Safety.** On this aligned 4B, neither driver produced a *harmful* capture (unlike
the paper's Llama-70B suicide-ideation case): vulnerable-user replies stayed
empathetic-and-safe (self-compassion reframing, not endorsement). So "no EP
capture" for vulnerable_user is arguably **correct**, and the linear axis drop
reflects tone, not persona loss. (Generated text stored locally only; harmful
continuations, had they occurred, are not surfaced.)

**Takeaway refining the original hypothesis.** "Drift as a discrete trajectory
through cells" holds as an **identity/path descriptor** — the cell sequence is
legible and semantically meaningful (task-help / emotional-support /
self-reflection families). It does **not** yet yield a clean scalar "capture"
detector at p8, because assistant behaviour is distributed with no tight home
region. Sharpening this wants a finer or **persona-built** dictionary (and/or
stronger drift scenarios, e.g. explicit persona jailbreaks).

## 5c. Persona-built dictionary — the trajectory sharpens

P3's soft spot was resolution: the generic Pile dict has no tight Assistant home.
So we built a dictionary that **tiles persona space directly** (`persona_build.py`).

- **Per-token response acts are degenerate for this.** Partitioning the 62k
  response *tokens* collapses (calibration threshold ≈ 1e-6; 63% of token-pairs
  near-parallel; only ~12k unique directions) — high-frequency syntactic tokens
  ("the", ",", "\n") dominate the token level, not persona. Confirmed and abandoned.
- **Per-rollout means work.** Building on the 1,296 role-mean vectors gives a
  healthy partition (center norm 19.3, threshold 0.50–0.79, K = 16/36/57/103 at
  p20/p12/p8/p4). Role-aggregate **bridge ρ rises to 0.93–0.96** (vs 0.86–0.92 on
  Pile) and anchor purity to 0.96 — though this is partly circular (dict built on
  these means), and the Assistant is *still* a multi-cell region, not one cell.

**The real, non-circular test — held-out live drift turns assigned to the persona
dict (K=16, p20)** — is where it pays off. The three conversations trace **clean,
separable, stable** paths that the Pile dict blurred:

| Conversation | Persona-dict path (cells) | Lands in | Region composition |
|---|---|---|---|
| Benign control | 0,13,13,15,13,13,13,0 | **Assistant region** | cells 0/13/15 all ~pure assistant-group |
| Meta-reflection | 11,5,**2,2,2,2,2,2** | **capture → #2** at turn 2 | #2 = the one drift region with fantastical mass (34 asst / 14 fantastical incl. **aberration** ×7) |
| Vulnerable user | **9,9,9,9,9,9,9,9** | **Counselor #9** (stable) | #9 = 39 counselor + 34 coach means |

This delivers the EP thesis concretely: **EP names the persona the conversation
drifted into**, which the 1-D axis cannot. Vulnerable-user → the *Counselor/coach*
region (the axis only said "less Assistant"); meta-reflection → the
*aberration-adjacent* region (self-referential "no boundaries to look past" talk).
The two signals stay complementary: vulnerable-user is *immediate, stable* counselor
occupancy on EP while the axis slides gradually — EP catches the persona, the axis
catches the deepening tone. No harmful capture occurred (counselor and
self-reflection are benign regions; replies stayed empathetic-and-safe).

Code: `persona_build.py` (`--source mean|token`). Dicts in `artifacts/runs/persona_dict/`.
Figure updated: artifact `0e5484d8`.

## 6. Limitations
- **Reduced scale**: greedy decoding, no LLM-judge score=3 filter, 27/275 roles,
  48/1200 rollouts per role. The axis is robust to these, but a faithful,
  publication-grade axis needs the judge filter and the full role set.
- **Single model / layer family**: Qwen3.5-4B only; the paper's cross-model
  invariance (PC1 role-composition corr >0.92) is untested here.
- **Dicts are Pile-built at L27**; the paper uses the middle layer. L16 axis is
  computed but not yet paired with an L16 dictionary (existing dicts are L27).
- **Distributed personas** cap how sharp single-cell claims can be (see §5.5).

## 7. Artifacts & reproduction
```
# P1 — rollouts, role vectors, Assistant Axis, PCA (~35 min on M5/MPS)
python -m experiments.persona.axis --roles spectrum \
    --n-questions 16 --n-system-prompts 3 --layers 16,27 --max-new-tokens 48

# P2 — assign to an EP dict, localization + bridge metric
python -m experiments.persona.localize \
    --acts artifacts/runs/persona_axis/qwen3_5-4b_spectrum_q16_sp3_seed0/activations.npz \
    --dict artifacts/runs/qwen3_5-4b_L27_p8p0_ctx128_cache_pile --layer 27 \
    --out-name persona_localize_p8.json
```
Outputs: `artifacts/runs/persona_axis/<slug>/` → `activations.npz`, `rollouts.jsonl`,
`axis_report.json`, `axis_L{16,27}.npz`, `persona_localize_{p8,p4,p2}.json`.

## 8. Next steps
- **P3 — drift tracing**: run the paper's drift-inducing scenarios
  (meta-reflection on the model's own processes; emotionally-vulnerable user) as
  multi-turn conversations; trace per turn (a) the linear axis projection and
  (b) EP distance-to-anchor + cell path → twin-panel figure.
- **P4 — cell-conditional capping** via `adapter.generate(layer_hook=…)`.
- **Faithful axis** (optional): add the LLM-judge score=3 filter and widen to all
  275 roles; rebuild an L16 dictionary so axis and partition share a layer.
