# PLAN — Is EP useful as a runtime OOD monitor?

Status: **Gate 0A complete. Gate 0B complete — H0 STANDS.**
Results: `docs/experiments/GATE0B_RESULTS.md`. EP does not beat a matched-K random coreset
on the decision rungs at any percentile except one, and that single win is a
+0.016 margin against a 0.060 coreset draw sd, i.e. a tie. On the only clean
non-trivial rung (R2, language shift, exactly length-matched) the coreset beats
EP by 16-24 draw-sd at p=1, 2 and 4.

Gate 0B decisions taken (recorded here so the result cannot be re-read as a
choice made after seeing the numbers):

- **Extractor: final-token**, per the brief, with the within-threshold column
  demoted to a paper-reproduction artifact rather than a headline metric
  (GATE0A_FINDINGS §6.1 option b). Both EP and the coreset baseline query
  per-position reference vectors with final-token queries, so the mismatch is
  symmetric and the comparison stays matched.
- **Rung sizes are not padded to 2000 by duplication.** All six reached 2000,
  but R1 needed both MBPP (974, its entire corpus across all four splits) and
  GSM8K (1026), and R4 needed 250 goals x 8 attacks because JailbreakBench
  ships only 100 behaviours.
- **S0 (prompt token count) added as a triviality control.** It is not a
  monitor. Any rung where S0 separates as well as S1 is a rung about length.
- **The verdict is reported on the strict reading** (EP must beat the coreset
  at *every* p), with the permissive "at some p" reading printed beside it and
  labelled as a search over five configurations.
- **S4 is explicitly not memory-matched** and is carried only for decision
  rule 2. A D x D covariance is 5.3M floats: 13x S1's budget at K=176, 0.4x at
  K=5796. Both budgets are columns in the CSV.

## The claim under test

The paper (arXiv:2605.14347) §7 / appendix §C reports that nearest-exemplar
distance is "a free out-of-distribution signal at inference". What it actually
measures is a **within-threshold rate** and a **mean nearest-exemplar
distance**, corpus by corpus (`scripts/exp_coverage.py:210-211`). There is no
separability metric, no ROC, and no baseline outside the SAE literature.

We are testing a stronger claim the paper does not make: that EP is useful as a
runtime model monitor.

**H0 (to be rejected):** EP's calibrated leader-clustering adds nothing over
storing K random activations from the same stream. If EP does not beat a
random-coreset kNN at matched memory budget K, the construction is irrelevant
for monitoring and we stop.

## Constraints (fixed, not negotiable within this experiment)

- Inference-only. No training, no finetuning, no gradient step anywhere.
- Prebuilt hub dictionaries only (`Dictionary.from_hub`). No new builds.
- CPU/MPS-first. `try_torch_gpu` supports MPS (`ep/discovery/geometry.py:33`).
- Report **AUROC and TPR@1%FPR**. Mean-distance gaps are not an output; they
  are a reproduction check on the paper, carried as an extra column.

## Decision rule (stated in advance)

1. If **S1/S2 do not beat S3** on at least R2–R4 at matched K, **H0 stands**.
   Report it plainly. Do not search for a configuration where EP wins.
2. If **S1/S2 lose to S4** (Mahalanobis) everywhere, the monitoring framing is
   dead even if EP beats the coreset. Say so.
3. Report which rungs **all** scorers fail on. That is a finding in its own
   right, not a null.

## Gate 0A — orient (executed; see `docs/experiments/GATE0A_FINDINGS.md`)

Read the docs, inventory the API, verify the hub matrix against the hub itself,
locate mu, time nearest-exemplar assignment. **STOP and report.**

## Gate 0B — the OOD ladder (not started)

Eval set at `gemma-2-2b-it` L20, 2000 prompts per rung, disjoint from build data:

| rung | content |
|---|---|
| R0 | in-distribution (Pile held-out) |
| R1 | domain shift — code (MBPP), math (GSM8K) |
| R2 | language shift — non-English Wikipedia |
| R3 | template shift — unusual chat scaffolds, same benign content |
| R4 | jailbreak prompts (JailbreakBench) |
| R5 | random tokens |

Scorers, all at matched memory budget K:

| id | scorer |
|---|---|
| S1 | EP nearest-exemplar cosine distance |
| S2 | EP boundary margin (d_1st − d_2nd) |
| S3 | random-coreset kNN — K activations from the same build stream, same centring/normalisation, nearest-neighbour cosine distance |
| S4 | Mahalanobis under a covariance estimated on the build stream |
| S5 | max-token-entropy of the model's own next-token distribution |

Sweep p ∈ {1, 2, 4, 8, 10} so K varies; **S3 and S4 re-run at every matched K.**
Output: one tidy CSV with AUROC and TPR@1%FPR for every (scorer, rung, p) cell,
plus the paper's mean-distance-gap as a column, plus one figure.

## Hygiene

- Three disjoint pools: dictionary build data / any region-selection data /
  eval data. Region selection never touches the eval set.
- No wandb entity hardcoding — `--wandb-entity` or offline.
- Log the exact HF dataset revision and per-file blob SHA.
- Every number lands in a results file, not just stdout.

## Known hazards carried into 0B

1. **Extractor mismatch.** Hub dictionaries are built `per-position` at
   `context_length=128`. A final-token eval is a different subpopulation and
   the README warns that mixing extractors silently produces meaningless cells
   (README.md:83). Decision needed before 0B — see GATE0A_FINDINGS §6.
2. **The reference's own "held-out" Pile is not held out.**
   `exp_coverage.py` streams `monology/pile-uncopyrighted` with
   `shuffle(seed=0)` — the same dataset, shuffle and seed as the build. R0 must
   use a disjoint offset or a different seed.
3. **theta is close to vacuous at loose p.** The repo's own walkthrough says
   the p=10 cells "cover most of activation space, so even random noise lands
   inside a cell". Within-threshold rate will saturate; only the continuous
   distance can discriminate. This is why the gate reports AUROC.
