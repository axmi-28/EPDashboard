# Running the role experiment on RunPod

Companion to [PLAN_ROLE_QWEN3_4B.md](PLAN_ROLE_QWEN3_4B.md), which is the *why*.
This is the *how*. RunPod over Modal this time, so the job is an interactive SSH
session rather than a container spec — which means the environment is your
problem and the ordering below is not decorative.

## 0. Pod sizing

`Qwen/Qwen3-4B`: 36 layers, d_model 2560, ~8 GB bf16, **ungated** (no licence
click, unlike the Gemma track).

| | value | why |
|---|---|---|
| GPU | A100-40GB, or any 24 GB+ card | 8 GB weights; activations are the rest |
| host RAM | **≥ 48 GB** | TransformerLens does not stream weights: `from_pretrained_no_processing` materialises the HF model on CPU, builds a converted state dict *alongside* it, then moves to GPU. Peak host RAM ≈ 2× weights. |
| disk | ~40 GB | 8 GB weights + C4 shards + OASST1 |

A smaller card works for Tiers 0–2 (forward passes only, batch 16). The causal
tier generates unbatched, so it is bandwidth-bound and a faster card pays for
itself there.

## 1. Bring the box up — before pulling weights

```bash
git clone <this repo> && cd EP_Qwen3.5_2B
bash scripts/experiments/runpod_role_setup.sh
```

The stock RunPod PyTorch image ships **torch 2.4.1**, and four things break. All
four fail *after* a model download if you hit them serially, which is why setup
runs first and ends with a weights-free corpus preflight:

1. **transformers 5.x needs torch ≥ 2.5** — `DTensor` moved to
   `torch.distributed.tensor`. The symptom is an `ImportError` deep inside a
   `modeling_*` module, not a clean version error. → `torch==2.13.0`.
2. **torchvision is then ABI-broken** (`operator torchvision::nms does not
   exist`) and transformers may import it. We only do text → uninstall it.
3. **`zstandard` for the Pile** — `monology/pile-uncopyrighted` ships
   `.jsonl.zst`. C4 is `.json.gz` and does not need it, but the Pile is the
   robustness corpus, so install it now rather than mid-sweep.
4. **`ep` installs `--no-deps`** — its `transformer-lens` pin would drag in an
   older `transformers` that does not know Qwen3. We install TL ourselves.

Differences from the 27B recipe: transformer-lens **is** required here (it is the
validated harness), and `flash-linear-attention` is **not** (Qwen3-4B is dense,
with no linear-attention layers).

Setup ends with:

```
python -m experiments.role.preflight --n-docs 200 --dataset c4 --check-oasst
```

which must print `PREFLIGHT OK`. It checks that the loaders actually returned
data (not a silent fallback), that content token ids are identical across all six
role conditions, that no scaffold token leaked into a labelled span, and that
content lengths are exactly rectangular. None of it needs a GPU or weights.

## 2. Tier 0 — the gate

```bash
bash scripts/experiments/run_role.sh gate
```

~15 min. Trains per-layer linear role probes on tag-wrapped C4 and reproduces the
paper's Table 1 row for the Qwen family on real OASST1 user text.

Exit code **0 = pass, 2 = fail**, and the gate is on the *flat* tool tag, not the
native one — Qwen3 renders a tool message as a `user` turn wrapping
`<tool_response>`, which is the easy case for the paper's claim, so gating on it
would be gating on nothing.

| cell | paper (Qwen3-30B-A3B) | gate |
|---|---|---|
| Userness under `<user>` | 83.6% | > 70% |
| Userness under flat `<tool>` | 75.7% | > 60% |
| Toolness under flat `<tool>` | 19.5% | < 35% |

**Read the per-layer accuracy column before reading the gate.** If probe accuracy
is near chance (1/6 = 0.167), the probe failed and the gate says nothing at all.
If accuracy is high but re-tagged Userness collapses, Qwen3-4B genuinely
re-asserts tags — that is a finding, and it means Tiers 1–5 are unanswerable
rather than negative. Stop and report either way.

## 3. Tiers 1–2 — the EP measurements

```bash
bash scripts/experiments/run_role.sh sweep-p        # p in {4, 8, 12, 16} at L18
bash scripts/experiments/run_role.sh sweep-l 12     # L in {9,14,18,22,28} at chosen p
bash scripts/experiments/run_role.sh seeds 18 12    # streaming seeds 1,2,3
```

Choose the percentile on `n_regions` and the saturation curve, **not** on the
metrics — picking resolution by whichever value maximises the headline number is
how you manufacture a result. Upstream defaults to 8.0, which the EP paper
reports as a fragmentation *failure*; gemma needed 12; tag-wrapped C4 is a
different corpus again.

**Do not run stages in parallel.** The EP calibration cache is keyed on
(model, hook, percentile) with **no seed component**, while `_calib_iter`
shuffles with the seed. Concurrent seeds each compute their own threshold and
race to write one path, confounding calibration variance with streaming order.
Seed 0 runs alone, then the rest.

Each run writes `artifacts/runs/role/Qwen3-4B/L{L}_p{P}_seed{S}/`:

- `role.json` — every metric
- `role_subspace.npz` — λ, the role axis, the m-dim subspace, top-λ region ids
- `Qwen3-4B_L{L}.pkl` — the dictionary
- `displacement_{a}__{b}.npy` — mean displacement directions

### What to read, in order

1. **`n_regions`** and `final_position_degeneracy`. Gemma's L20 gave 5 of 207
   regions any final-position activation. If Qwen3-4B is similarly degenerate,
   per-token labelling was necessary; if not, that is worth recording.
2. **`displacement[user→assistant]`** — the whole question. The log line
   `VERDICT (user->assistant): …` classifies it:
   - `flip_rate < 0.2` → **SHARED**: the tag barely moves the region. EP adds
     nothing at this layer; report as a negative with layer and percentile.
   - `coherence_z > 3` and `coherence > 0.3` → **COHERENT**: role ≈ one direction
     acting on a content-partitioned space. The likely outcome; EP's contribution
     is then the gating in §5b, not the direction.
   - otherwise → **CONDITIONAL**: no single role direction, and the probe's 84%
     is an average over incompatible local geometries. Strongest result on offer.
3. **`information`** — `nmi_region_role` vs `nmi_region_content`. If content
   dominates, the partition is tracking token semantics, which is the honest
   prior. Compare role NMI against **`nmi_region_role_null_paired`**, never
   against `nmi_region_role_null_global_wrong` — the latter is recorded only to
   show how badly the naive null misleads. Because every content token appears
   once per condition, a low flip rate makes each region's condition mix almost
   perfectly balanced, the joint factorizes, and the plug-in MI is essentially
   unbiased; global label permutation destroys that balance and manufactures bias
   from nothing. On the dry run it gave a null of 0.105 against an observed
   0.003, which would have "proved" no role information regardless of the data.
4. **`classifiers`** — `ep_region` vs `kmeans` at matched K. The probe beating EP
   is uninformative (2560 supervised parameters vs one unsupervised categorical);
   EP beating k-means is the real claim.
5. **`polarity.top_user_pids` across seeds.** Region identity is a first-arrival
   accident, so a λ ranking that reshuffles between seeds is not a real object and
   nothing may be built on it. `metrics.polarity_rank_stability` is the number.

## 4. Tier 3 — causal, gated on the above

```bash
bash scripts/experiments/run_role.sh causal \
  artifacts/runs/role/Qwen3-4B/L18_p12_seed0/role_subspace.npz
```

Generation-bound and the only expensive tier. Two arms: head-to-head subspace
ablation (EP subspace vs the probe direction vs a dimension-matched null), and
the region-gated selectivity frontier, which is the claim a direction cannot
make.

**Qwen3's reasoning channel is closed by default here** (`enable_thinking=False`
in the chat template). With it open, the model spends the whole 48-token budget
inside `<think>`, every verifiable constraint scores as violated *at baseline*,
and the ablation Δ is identically zero for a reason that has nothing to do with
role. `--thinking` re-enables it, and then `--max-new-tokens` needs to be ~512.

Scoring is exact string logic (`experiments/role/constraints.py`), not an LLM judge and not
the refusal substring list — "I can't" has nothing to do with role. Check
`baseline.rate` first: it needs to be high, or ablation has no room to fall.
Check `per_family` too: an effect living entirely in one constraint family is a
scorer artifact, which is why the breakdown is always reported.

Expect ablation to work and positive steering not to. On refusal, α ∈ {50,100}
were indistinguishable from baseline and α=400 degenerated into token loops.
Necessary ≠ sufficient; a steering null is not evidence against localization.

## 5. Getting results off the pod

```bash
# from your laptop
rsync -avz --exclude '*.pkl' pod:/workspace/EP_Qwen3.5_2B/runs/role/ ./runs/role/
```

The dictionaries are the large files; pull them only for the configuration you
intend to analyse. `artifacts/logs/role/` has the full stdout of every stage.

## 6. Local-only checks, for reference

All of this runs on a laptop with no GPU and no weights, and should be green
before the pod exists:

```bash
python -m pytest role/tests/ -q          # 107 tests
python -m experiments.role.preflight --n-docs 200 --dataset c4 --check-oasst
```
