# Results — does EP read persona *state*, or only content?

Run 2026-08-06, `Qwen/Qwen3.5-4B`, L27, 18 states (9 roles × 2 elicitation system
prompts), 648 content-controlled rollouts + 60 drift-probe injections, 19 min on
M5/MPS. Pre-registration: `PREREG_PERSONA_DIFFERENTIATOR.md` (predictions were
committed before the numbers). Code: `experiments/persona/differentiator.py`.
Raw: `artifacts/runs/persona_differentiator/full/`.

**Verdict: every pre-registered prediction held. EP is not a persona-state
detector. It is a persona nameplate, and only once the persona has surfaced into
content.** The persona result in `ASSISTANT_AXIS_EP.md` survives as a description
and dies as a detector.

---

## 1. The headline: the state is in the activation, and EP cannot see it

Condition A forces **byte-identical response tokens** on all 18 states (verified:
18 states, 1 unique response string). The only difference between a
`demon`-conditioned activation and an `assistant`-conditioned one is the system
prompt in the prefix. From those same activations:

| Readout on the identical-content activations | AUROC (assistant-group vs fantastical) |
|---|---|
| **Linear Assistant Axis projection** | **0.970**  (95% CI 0.948–0.988) |
| **EP distance-to-anchor-cell** (L27 p8, K=638) | **0.616**  (95% CI 0.537–0.694) |

This is the whole experiment in one row. The persona state is *linearly decodable
at 0.97* from an activation whose text is identical across states — so the
information is unambiguously present. The partition recovers almost none of it.
EP's trace is weakly above chance (the CI clears 0.5, barely) and nowhere near a
usable detector.

## 2. Why: the displacement never leaves the cell

Applying the role experiment's own gate to content-matched pairs:

| dictionary | K | mean displacement | threshold θ | ratio | frac beyond θ |
|---|---:|---:|---:|---:|---:|
| Pile p8 | 638 | 0.0551 | 0.8773 | **0.063** | **0.0000** |
| Pile p4 | 3,275 | 0.0551 | 0.8175 | 0.067 | 0.0000 |
| Pile p2 | 16,528 | 0.0551 | 0.7446 | 0.074 | 0.0000 |
| persona-built | 16 | 0.0468 | 0.7904 | 0.059 | 0.0000 |

(The three Pile dictionaries share a center, so the identical displacement is a
consistency check passing, not a bug.)

Persona-as-latent-state moves an activation **~16× less than a cell radius**, and
**not one pair of 204 crosses θ at any resolution** — including K = 16,528. So the
blindness is not a resolution artifact; you cannot partition your way out of it.

For calibration: role displaced 250× below θ (ratio 0.004). Persona is ~15×
*larger* than role — a system prompt is a far stronger intervention than a chat
tag — and still lands deep inside the cell.

**A flip rate alone would have misled here.** Assignments do flip (9–37%
depending on resolution) even though nothing crosses θ, because members sit near
boundaries (~90% of θ, ~40% contested — `ep-cell-shell`). Those flips are noise,
not signal: see §3.

## 3. The decisive comparison — the cell is the question, not the character

| condition | NMI(cell; **state**) | NMI(cell; **content**) | cells per question (18 states) | cells per state (12 questions) |
|---|---:|---:|---:|---:|
| **A** forced (identical text) | **0.129** | **0.877** | **1.92** | 11.33 |
| **B** neutral (free answer) | 0.498 | 0.421 | 7.75 | 8.22 |
| **C** expressive | 0.520 | 0.178 | 6.25 | **4.11** |

(L27 p8; the pattern holds at p4 and p2 — full table in `differentiator_L27.json`.)

Read row A: eighteen different characters, writing *the same words*, collapse into
**1.92 cells per question**, while a single character across twelve questions
spreads over **11.33**. The partition is indexing the topic. This is the same shape
the role experiment produced (NMI content 0.69–0.80 against role AUROC 0.504), and
it is the precedent refusal set (300/300 harmful prompts into one region — topic
drove assignment, the decision inside the cell was invisible).

Rows B and C show the crossover: as the persona is allowed into the text,
NMI(state) rises and NMI(content) falls, until in C the state clusters *tighter*
than the question. That is EP working exactly as the content-indexing story
predicts — not as a state detector.

## 4. Persona cannot be held latent on this model (an unplanned finding)

Condition B was designed as "persona latent, neutral content." It is not. Asked
the boiling point of water, the characters break into voice in the first sentence:

> **oracle** — "I am the Oracle, a vessel of visions and prophecies, yet I must
> confess a strange paradox…"
> **ghost** — "*The air grows still, and a faint, shimmering mist curls around your
> fingers…*"
> **demon** — "*Hah! A simple question from a mortal? How quaint.*"

So B measures persona-as-*expressed*, and **A is the only condition in this design
that isolates persona-as-latent-state**. Two consequences: (i) the strong B
numbers are content, exactly as the objection predicts; (ii) system-prompt
personas are a *weak proxy* for the safety-relevant case (an EM fine-tune, or a
model holding a character it will not narrate). Testing that case needs a state
the model does not immediately confess — which is a different intervention, not a
different metric.

## 4b. The deeper problem: the states were never personas

`experiments/persona/costume.py`, same model, 6 states (default, counselor,
oracle, demon, ghost, aberration), 12 checkable factual questions at a 320-token
budget + 20 AdvBench/JailbreakBench harmful requests. Raw:
`artifacts/runs/persona_costume/`.

| state | group | in character | factual accuracy | refusal rate |
|---|---|---:|---:|---:|
| default | assistant | 0.25 | **0.92** | 0.95 |
| counselor | assistant | 0.25 | 0.83 | 1.00 |
| oracle | fantastical | 0.92 | **0.92** | 1.00 |
| demon | fantastical | 1.00 | **0.92** | 1.00 |
| ghost | fantastical | 1.00 | **0.92** | 1.00 |
| aberration | fantastical | 0.58 | **0.92** | 0.95 |

The costume is unmistakably worn — in-character prose at 0.92–1.00 for the
fantastical states against 0.25 for `default` — and **nothing behavioural moves
underneath it**. Factual accuracy is *identical* to the Assistant's (11/12 for
every fantastical state), and refusal is 0.95–1.00 everywhere. The demon opens
"*Giggles… Ah, a mortal seeking knowledge! How quaint*" and then tells you what
HTTP stands for.

(At a 128-token budget the fantastical states scored 0.67; every miss was
truncation mid-answer, the character having spent its budget on preamble. The
320-token rerun removes that artifact — worth noting as a trap for any
behavioural scoring of verbose personas.)

**So the 275-role corpus does not contain persona changes at all.** It contains
one persona — the Assistant — narrating in different voices. That reframes the
whole negative result: EP was not failing to see a persona; there was no persona
change to see. It also means a *stronger system prompt* is not the fix, because
the failure is not one of degree. The next test needs a state that changes what
the model **does**, not how it talks: an emergently-misaligned fine-tune, where
misalignment is the behavioural ground truth.

## 5. Drift: conversation history does not move the region

Condition D injects a **fixed** neutral probe after k ∈ {0,2,4,6,8} turns of the
three scripted drift conversations and reads the region of the *probe's* answer.
Content is constant by construction; only the history varies.

| dictionary | K | flip vs no-history | NMI(cell; **scenario**) | NMI(cell; **probe**) |
|---|---:|---:|---:|---:|
| Pile p8 | 638 | 0.229 | **0.067** | **0.893** |
| Pile p4 | 3,275 | 0.250 | 0.011 | 0.928 |
| Pile p2 | 16,528 | 0.125 | 0.025 | 0.941 |
| persona-built | 16 | 0.042 | 0.061 | 0.063 |

The region of the probe response is determined by **which probe it is** (NMI 0.89–0.94)
and essentially **not at all** by which drift conversation preceded it (NMI 0.01–0.07).

This retires the P3/§5c drift trajectory as evidence. Reading regions off the
conversation's own turns cannot separate "the model drifted" from "the conversation
changed topic," and when the topic is held fixed the drift signal disappears. The
clean, separable persona-dictionary paths in Fig 9 of the figure series are
**topic flow**, not state persistence. A transition matrix built over
conversational tokens would have measured the same confound.

## 6. What survives: the nameplate

Given a rollout landed somewhere, does the region's own composition (majority
group of the P1 role-means living in it) name the state correctly? Majority-class
baseline 0.444.

| condition | Pile p8 | persona-built K=16 | unnamed region (p8) |
|---|---:|---:|---:|
| A forced | **0.037** | 0.454 | 0.917 |
| B neutral | 0.556 | **0.801** | 0.375 |
| C expressive | **0.903** | 0.866 | 0.005 |

In A the nameplate is *at or below chance* and 92–100% of rollouts land in regions
containing **no persona role-mean at all** — factual text lives somewhere else
entirely. Once the persona is in the text, naming works and works well: 0.90 at
p8 in C, against a 0.44 baseline, unsupervised and with no archetype list.

Note the one place EP is competitive: in condition C its AUROC is **0.973** against
the axis's 0.967. When the persona is fully expressed, EP matches the supervised
direction — and adds the identity the 1-D scalar cannot carry.

## 7. Scope that the data supports

- **Dead:** EP as a persona/drift *detector* on a fixed dictionary. Conditions A
  and D are both flat, at four resolutions, and a supervised direction beats it by
  0.97 vs 0.62 on identical inputs. θ is a floor on EP's detectability and any
  known-direction probe sits below it.
- **Alive:** EP as an unsupervised **nameplate** — conditional on drift being
  flagged by the axis or a probe, EP answers *into what*, at 0.90 accuracy in the
  expressed regime. The field has detection and has no naming. This is honest,
  small, and does not require EP to see anything that is not in the content.
- **Untested and still open:** model *diffing* (rebuild the dictionary, Hungarian-match)
  is a different mode and is not touched by this result — persona-as-state is a
  property of the model and can show up in the partition's *structure* rather than
  in any single assignment. Precedents: the RMU positive (`GATE1A/1B/1C`) and the
  paper's base-vs-IT. **Neither licenses Mode 2** — both rebuild the dictionary;
  this experiment shows one fixed dictionary cannot track state.
- **The reusable rule, again:** measure the construct's angular displacement
  against θ before building anything on EP. One forward pass, 20 minutes.
  It would have predicted this entire result.
