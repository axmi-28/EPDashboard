"""The EP readout families of PLAN_EP_VS_BASELINES §4, plus their controls.

Notation, fixed once here and used everywhere downstream:

    c           the dictionary's centre (d,)
    E           exemplar matrix (K, d), rows unit-norm
    tau         the build's cosine threshold
    x_hat       unit(x - c)
    s = E x_hat cosine similarity to every exemplar (K,)
    r           argmax s, the region a point falls in

The arrows differ only in what they read off `s`. That is the whole point of
the study: the dictionary is held fixed and the readout is varied, so a win or
loss attaches to the readout rather than to the partition.

    EP-FLAG    smoothed positive rate of region r          -- Gate 2 incumbent
    EP-ONEHOT  L2 logistic on one-hot(r)                   -- Gate 2 EP-RIDGE
    EP-CODE    relu(s - tau), top-Q by mean-abs class diff -- the SAE-probe analogue
    EP-MARGIN  EP-FLAG (+) s_(1) - s_(2) (+) s_(1)         -- no SAE counterpart

Two facts about EP that shape the implementations:

- **Assignment is pure argmax.** There is no "none of the above". A point
  arbitrarily far from every exemplar still lands in some region, so at test
  time EP-FLAG is a lookup that always returns something, and its errors on
  out-of-distribution input are silent. `s_(1)` is carried in EP-MARGIN
  precisely to expose that.
- **`relu(s - tau)` is a JumpReLU with a shared, untrained threshold.** EP-CODE
  is therefore an SAE probe whose encoder rows are real observed activations
  and whose decoder is tied. It is the arrow that makes the KE25 comparison
  apples-to-apples, and it is also the one with the least reason to work: the
  code is dominated by one large entry at r and a tail of near-threshold
  neighbours.

The **matched-K coreset** is not an ablation, it is the control that decides
the study. It draws K activations at random from the same build stream and
uses them as exemplars, matching memory, inference FLOPs and build data. Every
prior EP verdict in this programme turned on it, and an EP arrow that does not
beat it has shown nothing about *partitioning* -- only about having K
directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dictionary:
    """A lean EP dictionary: the parts a readout actually touches."""

    exemplars: np.ndarray  # (K, d) unit rows
    center: np.ndarray  # (d,)
    threshold: float
    percentile: int | None = None
    layer: int | None = None
    source: str = ""

    @property
    def K(self) -> int:
        return self.exemplars.shape[0]

    @classmethod
    def load(cls, path: str | Path) -> "Dictionary":
        z = np.load(path, allow_pickle=True)
        return cls(
            exemplars=np.ascontiguousarray(z["exemplars"], dtype=np.float32),
            center=np.asarray(z["center"], dtype=np.float32),
            threshold=float(z["threshold"]),
            percentile=int(z["percentile"]) if "percentile" in z.files else None,
            layer=int(z["layer"]) if "layer" in z.files else None,
            source=str(path),
        )


def unit(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Centre and L2-normalise rows.

    fp32 throughout, per the programme's dtype policy: activations are stored
    bf16 but every cosine is computed in fp32. At d=3584 a bf16 dot product
    loses enough precision to move argmax on points near a bisector, and the
    contested-shell measurements say roughly 40% of members sit there.
    """
    x = np.asarray(x, dtype=np.float32) - center.astype(np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def similarities(x: np.ndarray, d: Dictionary, chunk: int = 8192) -> np.ndarray:
    """s = E x_hat for every row of x. (n, K), fp32.

    Chunked because (334k, 5796) fp32 is 7.7 GB at the p=1 dictionary, and we
    only ever need one dataset's worth at a time.
    """
    out = np.empty((len(x), d.K), dtype=np.float32)
    E = d.exemplars.T
    for i in range(0, len(x), chunk):
        out[i : i + chunk] = unit(x[i : i + chunk], d.center) @ E
    return out


def regions(s: np.ndarray) -> np.ndarray:
    """argmax assignment. Ties broken by index, as in the build sweep."""
    return np.argmax(s, axis=1)


def top2(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(s_(1), s_(2)): best and runner-up cosine. Used by EP-MARGIN."""
    part = np.partition(s, -2, axis=1)
    return part[:, -1], part[:, -2]


# --- arrows ---------------------------------------------------------------


def ep_flag(
    r_train: np.ndarray, y_train: np.ndarray, r_test: np.ndarray, K: int,
    alpha: float = 0.5,
) -> np.ndarray:
    """Smoothed positive rate of the training region. `(h + a) / (n + 2a)`.

    Jeffreys smoothing (a = 0.5), not Laplace. It matters here because most
    regions are empty at n = 1024 << K: an unvisited region must score 0.5,
    and with a = 0.5 it does, so unseen regions are neutral rather than
    evidence of the majority class.
    """
    h = np.bincount(r_train[y_train == 1], minlength=K).astype(np.float64)
    n = np.bincount(r_train, minlength=K).astype(np.float64)
    rate = (h + alpha) / (n + 2 * alpha)
    return rate[r_test]


def ep_onehot(
    r_train: np.ndarray, y_train: np.ndarray, r_test: np.ndarray, K: int,
    seed: int = 0,
):
    """L2 logistic on the K-dim one-hot of the region.

    Differs from `ep_flag` only in fitting the per-region logits jointly with
    shrinkage instead of independently with a fixed prior, so a gap between the
    two is a statement about regularisation, not about EP.
    """
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse

    def enc(r):
        return sparse.csr_matrix(
            (np.ones(len(r), np.float32), (np.arange(len(r)), r)), shape=(len(r), K)
        )

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(enc(r_train), y_train)
    return clf.predict_proba(enc(r_test))[:, 1]


def ep_code(s: np.ndarray, threshold: float) -> np.ndarray:
    """z = relu(s - tau). The JumpReLU code EP-CODE and EP-POOL are built on.

    Note the code is *not* one-hot: every exemplar within tau of x_hat fires,
    which at a coarse threshold is a handful and at a fine one is often just
    the winner.
    """
    return np.maximum(s - threshold, 0.0)


def select_topq(
    z_train: np.ndarray, y_train: np.ndarray, q: int
) -> np.ndarray:
    """KE25's latent selection: top-q by mean absolute difference between the
    class-conditional means. Applied to the *training* split only.

    This is the step that leaks if it is ever run on the full data. Kept as its
    own function so the leak has one place to not happen.
    """
    pos = z_train[y_train == 1].mean(axis=0)
    neg = z_train[y_train == 0].mean(axis=0)
    return np.argsort(-np.abs(pos - neg))[:q]


def ep_margin_features(
    r: np.ndarray, s: np.ndarray, K: int, flag_scores: np.ndarray
) -> np.ndarray:
    """EP-FLAG (+) (s_(1) - s_(2)) (+) s_(1), as columns.

    The two extra scalars are the ones EP has and an SAE does not: how
    decisively the point won its region, and how close it is to any exemplar at
    all. `s_(1)` is the abstention signal -- low means the point is in nobody's
    neighbourhood and the flag is a guess.
    """
    s1, s2 = top2(s)
    return np.column_stack([flag_scores, s1 - s2, s1]).astype(np.float32)


# --- the control ----------------------------------------------------------


def coreset_dictionary(
    build_stream: np.ndarray, K: int, center: np.ndarray, threshold: float,
    seed: int,
) -> Dictionary:
    """Matched-K random coreset: K activations drawn from the build stream.

    Matches EP on memory, inference FLOPs and build data, and differs only in
    *where the exemplars sit* -- EP's cover the support, a random draw's sample
    the density. Draw several seeds: the between-draw sd is the yardstick every
    EP margin has to clear, and a single draw understates it badly.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(build_stream), size=K, replace=False)
    E = unit(build_stream[idx], center)
    return Dictionary(
        exemplars=np.ascontiguousarray(E, dtype=np.float32),
        center=center,
        threshold=threshold,
        source=f"coreset(seed={seed},K={K})",
    )
