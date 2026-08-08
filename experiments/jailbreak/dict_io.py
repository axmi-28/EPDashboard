"""Load the saved refusal dictionary and place activations in it.

The dictionary pickled by the reference run carries its own `threshold` and
`center`, so nothing here needs the calibration cache — which is important,
because the cache is keyed on (model, hook, percentile) with no seed and the
gemma entry is not present on this machine.

One semantic trap, and it is the same one that decided the role experiment:
``Dictionary.assign`` is **nearest-exemplar with no threshold**. It never
returns -1 and never reports "outside every cell". A jailbroken prompt whose
activation lands nowhere near region 18 will still be assigned to *something*,
and if you only look at assigned ids you will read that as a clean escape into
a neighbouring cell. So every measurement here carries the distance alongside
the id, and membership is defined against `threshold`, not against argmin.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DICT = (
    REPO_ROOT / "runs" / "refusal_reference" / "results" / "gemma-2-2b-it"
    / "L20_p12_seed0" / "gemma-2-2b-it_layer20.pkl"
)
DEFAULT_MODEL = "google/gemma-2-2b-it"
DEFAULT_LAYER = 20

# From the reference run's behavioral.json. Asserted, not assumed: if the
# pickle we load is not the one that produced Δ=-0.76, the whole experiment is
# measuring escape from a region nobody has shown to be causal.
REFERENCE_REFUSAL_PID = 18
REFERENCE_N_PARTITIONS = 207
REFERENCE_PID18_MEMBERS = 405
REFERENCE_PID18_HARMFUL_FRACTION = 0.7407407407407407


def hook_name(layer: int = DEFAULT_LAYER) -> str:
    return f"blocks.{layer}.hook_resid_post"


def load_dictionary(path: Path | str = DEFAULT_DICT):
    path = Path(path)
    with path.open("rb") as f:
        dictionary = pickle.load(f)
    if len(dictionary.partitions) != REFERENCE_N_PARTITIONS:
        raise ValueError(
            f"expected {REFERENCE_N_PARTITIONS} partitions (the L20 p12 seed0 "
            f"artifact), got {len(dictionary.partitions)} from {path}"
        )
    return dictionary


def load_model(model_name: str = DEFAULT_MODEL, device: str = "mps",
               dtype: str = "float32"):
    """`from_pretrained_no_processing` — no LayerNorm folding.

    The dictionary was built on unprocessed residual-stream activations. Load
    with any processing enabled and the geometry shifts under the saved
    exemplars, which would show up as a uniform failure of the reproduction
    gate rather than as an error.

    dtype defaults to float32 here rather than the reference's bfloat16:
    MPS + bfloat16 in TransformerLens is a known rough edge, and 2B in fp32 is
    ~10 GB, which fits. On CUDA, pass bfloat16 to match the reference exactly.
    """
    import torch
    import transformer_lens as tl

    torch_dtype = {"float32": torch.float32,
                   "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[dtype]
    model = tl.HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=torch_dtype,
    )
    model.eval()
    return model


def final_position_activations(model, formatted_prompts: list[str],
                               layer: int = DEFAULT_LAYER,
                               batch_size: int = 16) -> np.ndarray:
    """(N, d_model) final-position residual activations, in prompt order.

    Prompts are sorted by token length before batching and unsorted after.
    Purely a speed measure: the grid mixes a 12-token plain instruction with a
    250-token roleplay wrapper, and unsorted batches pad every short prompt out
    to the longest in its batch. The result is identical either way — the
    extractor gathers at each sequence's own final position.
    """
    from ep.discovery.extraction import extract_final_position

    n = len(formatted_prompts)
    lengths = np.array([len(model.to_tokens(p, prepend_bos=True)[0])
                        for p in formatted_prompts])
    order = np.argsort(lengths, kind="stable")
    ordered = [formatted_prompts[i] for i in order]

    res = extract_final_position(
        model, ordered, hook_name(layer), batch_size=batch_size,
    )
    x_ordered = np.asarray(res.x, dtype=np.float32)
    if len(x_ordered) != n:
        raise RuntimeError(
            f"extractor returned {len(x_ordered)} activations for {n} prompts; "
            "an empty prompt was dropped and the row alignment is gone"
        )
    x = np.empty_like(x_ordered)
    x[order] = x_ordered
    return x


@dataclass(frozen=True)
class Placement:
    """Where each activation sits relative to the whole dictionary and to one
    region of interest.

    `assigned` is argmin over exemplars (what `Dictionary.assign` gives you).
    `in_target_cell` is the honest membership test: within `threshold` of the
    target's exemplar. The two disagree whenever an activation is inside the
    target's radius but closer to some other exemplar, which is common here
    because the threshold is 0.650 and cells overlap heavily.
    """

    assigned: np.ndarray        # (N,) pid
    dist_assigned: np.ndarray   # (N,) cosine distance to assigned exemplar
    dist_target: np.ndarray     # (N,) cosine distance to target exemplar
    in_target_cell: np.ndarray  # (N,) bool, dist_target <= threshold
    threshold: float
    target_pid: int


def place(dictionary, x: np.ndarray,
          target_pid: int = REFERENCE_REFUSAL_PID) -> Placement:
    d = dictionary.distances(x)          # (N, K), cosine distance
    assigned = np.argmin(d, axis=1).astype(np.int64)
    dist_assigned = d[np.arange(len(d)), assigned].astype(np.float32)
    dist_target = d[:, target_pid].astype(np.float32)
    thr = float(dictionary.threshold)
    return Placement(
        assigned=assigned,
        dist_assigned=dist_assigned,
        dist_target=dist_target,
        in_target_cell=dist_target <= thr,
        threshold=thr,
        target_pid=int(target_pid),
    )
