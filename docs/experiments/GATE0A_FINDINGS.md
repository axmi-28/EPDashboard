# Gate 0A — orientation findings

Machine: Darwin 25.5.0, Apple Silicon, torch 2.13.0, python 3.12.13,
`huggingface_hub` 1.24.0. Run 2026-07-30.

## 1. Docs read

`README.md`, `scripts/README.md`, `notebooks/walkthrough.ipynb`.

Two passages matter for this gate.

- **README.md:83** — "Calibration and discovery must use the same extractor.
  The threshold is calibrated against the distribution of activations the
  extractor produces; mixing per-position calibration with final-position
  discovery (or different context lengths) silently produces meaningless
  cells."
- **walkthrough.ipynb cell 11** — "on the loose p=10 build loaded here, the 203
  cells cover most of activation space, so even random noise lands inside a
  cell. To see the OOD geometry, load `percentile=2`."

The second is the repo conceding, in its own tutorial, that the within-
threshold rate is resolution-dependent to the point of vacuity at loose p. It
is also **not reproducible as written** — see §7.

## 2. API surface (exact signatures, verified against source)

Everything below is re-exported at `ep.*` (`ep/__init__.py`).

```python
# ep/discovery/dictionary.py
Dictionary.from_hub(
    cls, model_short: str = "gemma-2-2b", layer: int = 12, percentile: int = 10,
    *, repo_id: str = "J-RUM/exemplar-partitioning", cache_dir: str | None = None,
) -> Dictionary                                                        # :738

Dictionary.assign(self, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]   # :670
    # vecs are RAW activations (N, D). Centering + normalisation applied
    # internally. Returns (partition_ids (N,) int64, distances (N,) float32).

Dictionary.distances(self, x: np.ndarray) -> np.ndarray                # :684
    # RAW activations in, (N, K) cosine distances clamped to [0, 2] out.

Dictionary.to_directions(self, x: np.ndarray) -> np.ndarray            # :168
Dictionary._nearest_exemplar(self, batch_dirs) -> (min_dists, best_idxs)  # :193
Dictionary.finalize(self, min_members: int = 1) -> None                # :649
Dictionary.distance_distributions(self, min_members: int = 2) -> dict  # :712

Partition.closest_prompts  -> list[(dist, prompt, position)] nearest-first   # :77
Partition.farthest_prompts -> list[(dist, prompt, position)] farthest-first  # :91

# ep/discovery/extraction.py
extract_per_position(model, prompts, hook_name,
                     max_positions_per_prompt=None, batch_size=128) -> ExtractionResult
extract_final_position(model, prompts, hook_name, batch_size=256) -> ExtractionResult
# ExtractionResult(x (N,D), prompt_ids, position_ids, n_forward_passes, n_tokens)

# ep/discovery/pipeline.py
calibrate_pipeline(model, texts, hook_name, *, n_tokens=200_000,
                   percentile=10.0, extract_fn=None, extract_kwargs=None,
                   prompt_batch_size=16, seed=0, cache_model_name=None,
                   cache_extras=None, force_recalibrate=False) -> Calibration

# ep/discovery/calibration.py
load_or_calibrate(model_name, hook_name, activation_batches_fn, *,
                  n_tokens=200_000, percentile=10.0, extras=None,
                  force=False) -> Calibration
Calibration(center: np.ndarray, threshold: float,
            n_activations: int, percentile: float)     # frozen dataclass
```

**Per-region object** — `Partition` (`dictionary.py:55`), a plain dataclass:

| field | meaning |
|---|---|
| `exemplar_direction` (D,) f32 | first-arrival unit direction in centred space; the membership criterion and the intervention direction |
| `mean_member_direction` (D,) f32 | spherical mean of member directions |
| `member_coherence` float | c_i = ‖Σ member dirs‖ / N_i ∈ [0,1] |
| `member_count` int | N_i |
| `sum_dist_to_exemplar`, `sum_sq_dist_to_exemplar` | for intra-cell mean/var |
| `sample_members` list[(D,) f32] | reservoir, cap 30 |
| `sample_prompts` / `boundary_prompts` | heaps, cap 10; use the sorted properties |
| `label` | `None` on all hub dictionaries |

### Three API traps, verified numerically

1. **`assign` has no threshold and never returns −1.** It is pure argmin over
   exemplars; the only −1 path is an empty dictionary (`dictionary.py:677`).
   "Outside every cell" must be derived by the caller as
   `dist > dictionary.threshold`. Confirmed: `assign` and `distances().argmin(1)`
   agree exactly, and no `-1` appears.
2. **`_nearest_exemplar` takes directions, `assign`/`distances` take raw
   activations.** Passing raw activations to `_nearest_exemplar` returns a
   different, silently-wrong answer (verified: assignments disagree). Any
   custom scorer must call `to_directions` first.
3. **`try_torch_gpu` accepts MPS** (`geometry.py:33`), so the GPU matmul path is
   live on this machine and `EP_FORCE_CPU=1` disables it. Relevant because
   `distances()` materialises the full (N, K) matrix on host.

## 3. Prebuilt dictionaries — verified against the hub, not the docs

`J-RUM/exemplar-partitioning`, dataset revision **`0ec26618db2bd64cdbf00a9a1b0eb23fffdea12a`**,
last modified 2026-05-17. 13 dictionaries, 26 files.

**The README matrix is accurate.** Every (model, layer, p) it claims exists on
the hub, and there is nothing extra. No over-claim this time.

Our target, `gemma-2-2b-it` L20, has the full sweep. Blob SHAs logged for
provenance:

| p | K | θ | build activations | pkl size | blob SHA |
|---|---|---|---|---|---|
| 1 | **5796** | 0.779816 | 3,620,864 | 1.77 GB | `bc4bc3a05e0908246a2d44d00fec5918eda920ef` |
| 2 | 2037 | 0.821454 | 1,589,248 | 624 MB | `a1d7f472d4590176f079ff0051739717dd078d1c` |
| 4 | 686 | 0.861254 | 573,440 | 216 MB | `a5545f48c0f11f44fcaf60613d251e5aadda9095` |
| 8 | 226 | 0.900035 | 294,912 | 71 MB | `dd01e87e44455b718f71aaa380cad2d70d92eba5` |
| 10 | 176 | 0.912395 | 376,832 | 57 MB | `88a2c83b6f3ec42ee3a529d063ebc21f35768263` |

K = 5796 at p=1 matches the brief's "K~5796" exactly. **The memory budget for
matched-K baselines therefore spans 176 → 5796, a 33× range.**

All five were built with identical settings: `google/gemma-2-2b-it`, layer 20,
`context_length=128`, `sampling_mode="full"`, `extractor="per-position"`,
`merge_close=False`, `seed=0`, `calibration_tokens=200000`, device cuda. Build
corpus is `monology/pile-uncopyrighted` split `train`, streamed, with
`ds.shuffle(seed=0, buffer_size=10000)` (`scripts/build_partitions.py:164-166`).

Note p=1 used `max_tokens=100_000_000` where every other p used `10_000_000`,
though all five report `saturated: true`.

## 4. Where mu lives

**mu ships inside the pickle.** `Dictionary.center` (D,) float32 and
`Dictionary.threshold` are instance attributes set at construction and
serialised by `__getstate__` (only torch handles and per-batch scratch are
stripped, `dictionary.py:269`). Loading from the hub gives you the exact build
mu with no calibration cache, no model load, and no recompute. The disk cache
at `~/.cache/ep/calibration/{model}__{hook}__pct{p}{extras}.npz` (override
`EP_CALIBRATION_CACHE`) is a **build-time** artifact only — irrelevant at
inference, and we do not have the authors' copy.

Any monitor reuses mu by calling `d.assign(raw_acts)` / `d.distances(raw_acts)`,
or explicitly `ep.discovery.geometry.centered_unit(x, d.center)`.

**Verified: mu is bit-identical across the p sweep.** For `gemma-2-2b-it` L20,
p=1, p=8 and p=10 all give `np.array_equal(...) == True`, max abs diff 0.0,
‖center‖ = 226.9555 for all three. This is the expected result — the centre is
computed before the percentile is applied — but it had to be checked, and it
means **S1/S2/S3/S4 can all share one centring across the whole sweep**, so the
matched-K comparison is not confounded by a moving mu.

## 5. Timing probe

`assign` on 1000 activations, D=2304, best of 5 after warm-up, MPS vs forced CPU:

| K | p | MPS (ms/1k) | CPU (ms/1k) | µs/activation (best) | extrapolated to 20k |
|---|---|---|---|---|---|
| 176 | 10 | 4.25 | **1.68** | 1.68 | **0.03 s** |
| 686 | 4 | 3.19 | 3.22 | 3.19 | 0.06 s |
| 2037 | 2 | 5.25 | 6.99 | 5.25 | 0.11 s |
| 5796 | 1 | **10.04** | 19.48 | 10.04 | **0.20 s** |

The p=1 row is the **real downloaded dictionary** (K=5796, θ=0.7798). Rows for
K = 686 and 2037 were run against exemplar matrices of the true shape while
p=1 downloaded; the synthetic K=5796 row gave 10.12 ms against the real
10.04 ms, so the shape-only probe is accurate to ~1%.

**Assignment is free.** 0.2 s for the full 20k eval set at the largest K. MPS
only wins above K≈700; below that the host↔device transfer dominates and CPU is
faster, so the runner should pick per-K rather than assume GPU.

**The real cost of 0B is forward passes**, not scoring: 6 rungs × 2000 prompts
at ctx 128 through `gemma-2-2b-it`. Every scorer reads the same activations, so
extract once, cache to disk, and score all five S1–S5 off the cache.

Memory, not latency, is the constraint that bites: `distances()` returns the
full (N, K) host array — 2000 × 5796 × 4 B = 46 MB per rung, 278 MB for the
whole eval set. Fine if batched per rung, and **required** for S2, which needs
the top-2 gap and cannot use `assign`.

## 6. Hazards found — decisions needed before 0B

### 6.1 Extractor mismatch (needs your call)

Hub dictionaries are `per-position` at ctx 128. The brief specifies a
**final-token** eval. These are different activation subpopulations, and θ was
calibrated on the former. README.md:83 warns this "silently produces
meaningless cells" — that warning is about building, but it applies with equal
force to a θ-based decision at inference.

Options:
- **(a) per-position eval to match the build.** Faithful to θ. Changes the unit
  of analysis from "prompt" to "token", so 2000 prompts × ~127 positions =
  254k activations per rung. AUROC is then per-token, and a per-prompt score
  needs an explicit pooling rule (mean / max / last).
- **(b) final-token eval as specified.** Keeps the unit as "one prompt, one
  score" and matches how a real monitor would be deployed. θ is then
  off-distribution, so the within-threshold rate is uninterpretable — but the
  gate reports AUROC and TPR@1%FPR, which only need the ranking, and S3/S4
  are built on final-token activations too, so the comparison stays matched.
- **(c) both**, per-position as the faithful arm and final-token as the
  deployment arm.

I recommend **(b)** with the within-threshold column dropped, and (c) only if
the p-sweep result turns out to hinge on it. Under (b) the paper-reproduction
column in §7 has to be run per-position separately, since that is what the
paper measured.

### 6.2 The reference's own "held-out" Pile is not held out

`exp_coverage.py:46-49` streams `monology/pile-uncopyrighted` with
`shuffle(seed=0, buffer_size=10000)` and `--seed` defaults to 0 — the same
dataset, same shuffle, same seed as the build (`build_partitions.py:164-166`).
The paper's in-distribution rung therefore re-reads the documents the
dictionary was built on, from the top of the same stream. R0 must skip past the
build consumption or use a different seed; I will do both and record the offset.

### 6.3 S2 may be dead on arrival, for a reason worth reporting

The top-2 distance gap over random probes is tiny and **shrinks as K grows**:

| p | K | margin mean | min | max |
|---|---|---|---|---|
| 10 | 176 | 0.0084 | 0.0004 | 0.0318 |
| 1 | 5796 | 0.0058 | 0.0000 | 0.0317 |

The exemplars are nearly equidistant, which is what K ≪ D forces, and adding
exemplars packs them tighter. This is the same geometry that made adjacency
vacuous in earlier work here. S2 should still be run, but a near-chance AUROC
for S2 is a property of high-dimensional Voronoi cells, not a bug, and I will
report it as such rather than as a failure of EP.

## 7. The notebook's random-noise claim does not reproduce

Cell 11 asserts p=10 cells are so loose "even random noise lands inside a
cell". Drawing `N(0, 30²)` noise **around the dictionary centre** gives
nearest-exemplar distances in [0.919, 1.097] against θ = 0.9124 — a
within-threshold rate of **0.000**. Same at p=1: mean distance 0.9233 against
θ = 0.7798, within-threshold rate **0.000**. The notebook's own snippet uses
`rng.standard_normal((100, D))` with no centre offset, so after centring every
sample points along −mu and they collapse into one cell; that is an artifact of
the probe, not a property of the dictionary. Flagging it because the same
mistake would silently corrupt the R5 random-tokens rung. R5 will use random
**token ids** through the model (as `exp_coverage.py` does), never synthetic
Gaussians.

## 8. Logistics

- `p8` (71 MB) and `p10` (57 MB) downloaded at ~5 MB/s. `p1` (1.77 GB) goes
  through the Xet path (`hf_xet` 1.24.0 installed), which buffers in RAM and
  leaves the `.incomplete` blob at 0 bytes until reconstruction — do not read
  that file as a progress indicator. Observed ~3 MB/s and ~2 GB peak RSS.
  p=1 took **500 s** wall for 1.77 GB (~3.5 MB/s).
- Total download for the full sweep: **2.7 GB**. Disk has 591 GB free.
- Loading is fast (p=10: 0.5 s, p=8: 1.3 s, p=1: 1.0 s warm) but p=1 peaks at
  **2.71 GB RSS**. Measured: 168,164 stored `sample_members` × 2304 float32 =
  **1.55 GB of the 1.77 GB file** — i.e. 88% of the payload is a visualisation
  reservoir that **no scorer needs**. The (5796, 2304) exemplar matrix that
  actually does the work is 53 MB.
  Recommend a one-time pass per p that saves `{exemplars, mean_members,
  member_count, member_coherence, center, threshold}` to a lean `.npz`, then
  never touching the pickles again. Keeps the 0B runner under ~200 MB.

## 9. Nothing has been run beyond orientation

No eval set built, no experiment code written, no dictionary rebuilt. Awaiting
approval of the §6.1 decision before Gate 0B.
