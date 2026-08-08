"""Adapter onto the packaged KE25 sparse-probing benchmark (`sae-probes` on PyPI).

Three things this module exists to do:

- **Stub `sae_lens`.** `sae_probes/__init__.py` imports `run_sae_evals`, which
  imports `sae_lens`, at package import time. We never call the SAE path — our
  arrows are EP, and KE25's own SAE numbers are published as CSVs — but the
  import fires anyway. Installing `sae_lens` would drag `transformers<5` into a
  venv running 5.14, so instead we put a dummy module in `sys.modules` first.
  Touching anything under `sae_probes.run_sae_evals` will raise, which is the
  behaviour we want.
- **Pin the split.** KE25 splits with `seed=42`, `num_train=1024`, positives
  forced to 50% of train, test = everything left. Every arrow in this study has
  to see the same indices or the comparison is not head-to-head, so the split
  comes from their `get_train_test_indices` and nowhere else.
- **Read their published results.** `results/baseline_probes_{model}/` in the
  paper repo carries one row per (dataset, method) with `val_auc` and
  `test_auc`. That is the quiver we have to match before any EP number counts.

The 113 raw text datasets ship inside the wheel (43 MB, zstd CSVs), so nothing
here needs the Dropbox mirror the paper README points at.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- sae_lens stub, before the first `sae_probes` import ------------------

if "sae_lens" not in sys.modules:
    _stub = types.ModuleType("sae_lens")

    class _Unavailable:
        def __init__(self, *a, **k):
            raise ImportError(
                "sae_lens is deliberately not installed; the SAE arm of "
                "sae-probes is unused (see experiments/probes/benchmark.py)"
            )

    _stub.SAE = _Unavailable  # type: ignore[attr-defined]
    sys.modules["sae_lens"] = _stub

from sae_probes.utils_data import (  # noqa: E402
    get_binary_df,
    get_numbered_binary_tags,
    get_train_test_indices,
    read_numbered_dataset_df,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts" / "runs" / "probes"

# KE25's standard condition. `MAX_AMT` caps each dataset before splitting, so a
# 5,000-row dataset contributes 1,024 train + 3,975 test, not 1,024 + 3,976.
NUM_TRAIN = 1024
MAX_AMT = 5000
SPLIT_SEED = 42
POS_RATIO = 0.5

# Their headline configuration; `layer20_results.csv` is the CSV we reproduce.
HEADLINE_MODEL = "gemma-2-9b"
HEADLINE_HOOK = "blocks.20.hook_resid_post"


def dataset_tags() -> list[str]:
    """The 113 binary tags, in the master CSV's order."""
    return get_numbered_binary_tags()


def read_master_binary() -> pd.DataFrame:
    """The master CSV rows for the 113 binary datasets.

    `Dataset save name` is the on-disk path the activation generator wants, so
    the extraction driver maps tag -> path through here.
    """
    return get_binary_df()


def dataset_frame(tag: str) -> pd.DataFrame:
    """Raw text + label for one dataset. Columns include `prompt` and `target`."""
    return read_numbered_dataset_df(tag)


def labels(tag: str) -> np.ndarray:
    """Label-encoded targets, capped at `MAX_AMT`, matching `get_xyvals`."""
    from sklearn.preprocessing import LabelEncoder

    df = dataset_frame(tag)
    y = LabelEncoder().fit_transform(df["target"].values)
    return np.asarray(y)[:MAX_AMT]


def texts(tag: str) -> list[str]:
    """Prompts in the same order and cap as `labels`."""
    return dataset_frame(tag)["prompt"].astype(str).tolist()[:MAX_AMT]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    num_train: int


def standard_num_train(tag: str) -> int:
    """KE25's train size for one dataset: `min(size - 100, 1024)`.

    Not a flat 1,024. `size` here is the *uncapped* row count
    (`get_dataset_sizes`), while the activations are capped at `MAX_AMT`, so
    the two differ only for datasets above 5,000 rows -- where the cap is not
    binding on `num_train` anyway. Reproduced rather than simplified because
    27 of the 113 datasets are small enough for the `- 100` term to bite, and
    a flat 1,024 would quietly change their train size and every AUC with it.
    """
    return min(len(dataset_frame(tag)) - 100, NUM_TRAIN)


def split(tag: str, num_train: int | None = None) -> Split:
    """KE25's standard split for one dataset.

    `num_test` is their default (`n - num_train - 1`), so the scarcity regime,
    which varies `num_train`, keeps drawing its test set from the same pool
    rather than silently growing it.
    """
    y = labels(tag)
    if num_train is None:
        num_train = standard_num_train(tag)
    num_test = len(y) - num_train - 1
    tr, te = get_train_test_indices(
        y, num_train, num_test, pos_ratio=POS_RATIO, seed=SPLIT_SEED
    )
    return Split(train=tr, test=te, num_train=num_train)


def load_activations(
    cache: Path, tag: str, model: str = HEADLINE_MODEL, layer: int = 20,
    ood: bool = False,
) -> np.ndarray:
    """Last-token activations for one dataset, as fp32 numpy.

    `map_location="cpu"` is not optional. KE25 saved these straight off the
    GPU, so a good fraction of the `.pt` files carry CUDA storage tags and
    `torch.load` raises on a machine without CUDA -- and it raises *per file*,
    so a run can get most of the way through the suite before failing.
    """
    import torch

    suffix = "_OOD" if ood else ""
    name = f"{tag}{'_OOD' if ood else ''}_blocks.{layer}.hook_resid_post.pt"
    path = cache / f"model_activations_{model}{suffix}" / name
    t = torch.load(path, weights_only=False, map_location="cpu")
    return t.float().numpy()


# --- published results ----------------------------------------------------


def published_baselines(results_root: Path, model: str, layer: int) -> pd.DataFrame:
    """KE25's own baseline table: one row per (dataset, method).

    `results_root` is a checkout of github.com/JoshEngels/SAE-Probes. The CSV
    has already been reduced over each method's 10-setting hyperparameter
    sweep, so `val_auc` is the winning setting's validation score.
    """
    path = (
        results_root
        / f"results/baseline_probes_{model}/normal_settings/layer{layer}_results.csv"
    )
    return pd.read_csv(path)


def quiver(results: pd.DataFrame) -> pd.DataFrame:
    """KE25's quiver-of-arrows rule: per dataset, take the method with the best
    validation AUC and report *its* test AUC.

    Ties on `val_auc` are common (many datasets saturate validation), and
    `idxmax` resolves them by row order. That is deterministic given the CSV,
    but it means a tie is broken by the order methods happen to appear in, not
    by anything principled -- so a marginal-gain number computed off a quiver
    with many ties is weaker than it looks.
    """
    best = results.loc[results.groupby("dataset")["val_auc"].idxmax()]
    return best.set_index("dataset").sort_index()
