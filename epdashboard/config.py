"""Run configuration for EPDashboard.

Field names deliberately parallel SAEDashboard's ``SaeVisConfig`` /
``NeuronpediaRunnerConfig`` where a concept transfers (dataset path, context
length, prompt budget, per-batch feature count, sequence buffer); EP-specific
knobs (distance bands, background projection sample) are documented inline.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

PILE = "monology/pile-uncopyrighted"


@dataclass
class EPVisConfig:
    # ----------------------------------------------------------- dictionary
    #: Run directories, each holding ``dictionary.pkl`` (+ ``metadata.json``).
    #: Several may be given (e.g. the same layer at different ``p``): they share
    #: one activation pass, which is where the whole cost lives.
    run_dirs: list[str] = field(default_factory=list)
    #: HF model id. Default: read from each run's metadata.json.
    model_id: str | None = None
    #: Residual-stream layer the dictionary was built at. Default: metadata.
    layer: int | None = None
    #: Token index offset between harvested positions and tokenizer offsets
    #: (1 for models that prepend BOS, e.g. Gemma; 0 for Qwen).
    bos_offset: int = 0

    # ---------------------------------------------------- activation source
    #: HF dataset streamed for activations (SAEDashboard:
    #: ``huggingface_dataset_path``). Ignored when ``cache_dir`` is set.
    dataset: str = PILE
    dataset_split: str = "train"
    dataset_column: str = "text"
    #: Tokens per prompt window (SAEDashboard: ``context_length`` / ``n_ctx``).
    context_length: int = 128
    #: Prompts streamed in pass 1 (SAEDashboard: ``n_prompts_total``).
    #: Activation count ≈ n_prompts × (context_length − 1); position 0 is
    #: skipped as the attention-sink slot, matching dictionary construction.
    n_prompts: int = 24_576
    #: Read activations from an ``extract_cache`` shard dir instead of running
    #: the model. Preferred when the cache still exists — at 27B scale the
    #: forward pass is the entire cost of the job.
    cache_dir: str | None = None
    batch_size: int = 16            # prompts per forward sub-batch
    prompt_batch_size: int = 64     # prompts per assignment chunk (forward mode)
    chunk_size: int = 8192          # activations per assignment chunk (cache mode)
    seed: int = 0
    device: str | None = None       # None = auto (cuda > mps > cpu)

    # ------------------------------------------------------ region selection
    #: Region indices to build (None = all K). Mirrors SAEDashboard
    #: ``features``.
    regions: list[int] | None = None

    # ---------------------------------------------------------- panel sizes
    n_closest: int = 10             # sequences in the "closest members" group
    n_bands: int = 3                # distance bands over [0, θ]: near/mid/far
    n_per_band: int = 8             # random sequences kept per band
    n_random: int = 16              # sequences in the uniform random draw
    #: Tokens shown left/right of the firing token (SAEDashboard: ``buffer``).
    buffer: tuple[int, int] = (10, 5)
    n_neighbors: int = 8            # nearest regions by exemplar cosine
    lens_k: int = 10                # tokens per logit-lens / J-lens list
    hist_bins: int = 40
    #: Shared uniform token subsample whose projections back the grey
    #: "full corpus" background in every projection histogram. Memory is
    #: bg_sample × K floats, so cap accordingly at large K.
    bg_sample: int = 8192
    #: Uniform member reservoir per region (backs quantiles, the member
    #: projection histogram, and the random-draw sequence group).
    reservoir: int = 256

    # ------------------------------------------------- EP-specific accumulators
    # These have no SAEDashboard analogue: they describe the *partition* rather
    # than a direction, and every one of them needs the assignment stream, so
    # they are collected on every run whether or not a panel displays them yet
    # (adding a panel later is a re-render; adding an accumulator is a rescan).
    #: Runner-up competition graph is a dense (K, K) int32 count matrix, so it
    #: is skipped above this K (8192 → 268 MB; 27B L55 p4 is K=5190).
    comp_max_k: int = 8192
    #: Competitors kept per region in the written record.
    n_competitors: int = 8
    #: A member is "contested" when its runner-up cell is within this fraction
    #: of θ of its own — i.e. it sits near a Voronoi bisector, not in a core.
    contested_eps: float = 0.1

    # --------------------------------------------------------------- output
    out_dir: str = "epdash_out"
    #: Regions per JSON/HTML batch file (SAEDashboard:
    #: ``n_features_at_a_time``).
    regions_per_batch: int = 256
    html: bool = True
    #: Cache dir for unembedding/J-lens npz files (default: <out_dir>/.lens).
    lens_cache: str | None = None

    # ------------------------------------------------------------- plumbing
    def out_path(self) -> Path:
        return Path(self.out_dir)

    def lens_cache_path(self) -> Path:
        return Path(self.lens_cache) if self.lens_cache else self.out_path() / ".lens"

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["buffer"] = list(self.buffer)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "EPVisConfig":
        d = json.loads(Path(path).read_text())
        if "buffer" in d:
            d["buffer"] = tuple(d["buffer"])
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)
