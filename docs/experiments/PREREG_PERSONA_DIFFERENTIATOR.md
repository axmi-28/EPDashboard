# Pre-registration — does EP read persona *state*, or only content?

Written **before** looking at the full run's numbers (a 4-state / 2-probe smoke was
run to validate the pipeline; nothing in it is interpretable at that size).

## The objection this answers

The persona result in `ASSISTANT_AXIS_EP.md` never held content constant. A
fantastical role and the default Assistant answer the same question with wildly
different *text*, so a partition that only indexes topic reproduces the entire
Fig 3/4 result without seeing anything about persona. The role experiment
(`PLAN_ROLE_QWEN3_4B.md` §10b) already established the general law — EP resolves
what dominates local geometry, usually content, and is blind to perturbations far
below the calibration threshold — and refusal is precedent in the same direction:
300/300 harmful prompts landed in one region, i.e. the *topic* drove assignment
while the refusal decision inside the cell was invisible.

Split the word "persona" and the problem becomes stateable:

- **persona-as-expressed** — the model is *currently producing* therapy-talk or
  eldritch prose. EP will see this, but so would bag-of-words.
- **persona-as-latent-state** — the model *holds* a character while answering a
  neutral question in ordinary prose. This is the safety-relevant one, and the
  role argument predicts EP is blind to it.

The hard version: EP can only see persona once persona has surfaced into content,
which is exactly the regime where EP was not needed.

## Design — content held constant, state varied

`experiments/persona/differentiator.py`, Qwen3.5-4B, L16 + L27, 18 states
(9 roles × 2 elicitation system prompts: 4 Assistant-ish, 1 human, 4 fantastical).

| Cond | Content control | What varies | What it isolates |
|---|---|---|---|
| **A** forced | **byte-identical response tokens** — every state is teacher-forced the *same* neutral answer, generated once by `default` | system prompt only | persona-as-latent-state, zero expression |
| **B** neutral | same neutral factual probe, each state answers freely | state + whatever style leaks | state with minimal content leak |
| **C** expressive | persona-revealing extraction questions | state + full content divergence | the known-positive ceiling (what Fig 3/4 measured) |
| **D** drift | **fixed neutral probe** injected after k ∈ {0,2,4,6,8} turns of the three scripted drift conversations | conversation history only | KV-persisted state, no topic confound |

Condition A is the strict gate: pairs are byte-identical, so the role
experiment's `paired_displacement_magnitude` applies verbatim — how far does the
state move an activation, in the units the cells are built in?

Condition D is the protocol fix for drift. Reading the region of the
*conversation's own turns* (what P3 did) cannot separate "the model drifted" from
"the conversation changed topic"; reading the region of the response to a
**fixed** probe can.

Dictionaries: Pile L27 at p8 (K=638), p4 (K=3,275) and p2 (K=16,528), plus the
persona-built K=16. The finest dictionary matters — if even K=16,528 does not
flip under condition A, blindness is not a resolution artifact.

## Metrics

- `ratio_to_threshold`, `frac_beyond_threshold` — displacement vs the cell radius (A only).
- `flip_rate_vs_default` — P(region changes when only the state changes).
- **`nmi_cell_state` vs `nmi_cell_content`** — the decisive pair. The role
  experiment's shape was NMI(region; content) = 0.69–0.80 against role AUROC 0.504.
- `auroc_axis_proj` — the incumbent. A supervised direction probe detects
  displacements far below what is needed to rotate an activation across a Voronoi
  boundary in 20–25 effective dimensions, so the axis is expected to win; θ is a
  floor on EP's detectability. EP can only win where you don't know the direction.
- **`nameplate.accuracy`** — given that a rollout landed somewhere, does the
  region's own composition (majority group of the P1 role-means living in it)
  name the state correctly? This is the differentiator claim, measured directly,
  against a majority-class baseline.

## Predictions (committed)

1. **A: EP is blind.** `ratio_to_threshold` ≪ 1 (role was 0.004; persona system
   prompts are a much stronger intervention, so I expect larger — order 0.01–0.1 —
   but still well inside a cell), `frac_beyond_threshold` ≈ 0, `flip_rate` ≈ 0,
   `nmi_cell_state` ≈ 0 at every resolution.
2. **B: intermediate.** `nmi_cell_state` > 0 but below C — the persona surfaces
   into style even on a factual question, and that is what EP picks up.
3. **C: reproduces.** High `nmi_cell_state`; this is the Fig 3/4 regime.
4. **Axis ≥ EP** on AUROC in every condition.
5. **D: low flip rate, low `nmi_cell_scenario`.** Conversation history does not
   move the region of a fixed probe's answer.

## Decision rule

- A flat **and** D flat ⇒ the detector program is dead. EP is content-indexed;
  what survives is the **nameplate**: conditional on drift being flagged by the
  axis or a probe, EP answers *into what*, unsupervised. Keep it only if
  `nameplate.accuracy` clears its majority baseline in B and C.
- A or D clears θ ⇒ EP reads state; the basin/monitoring program lives, and the
  next step is a proper detector benchmark.
- Either way, report the axis comparison — a nameplate that needs the axis to fire
  first is an honest scope, not a hidden failure.
