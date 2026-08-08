"""Storage dtype for cached activations — bf16 without a new dependency.

The forward pass runs in bfloat16, but the cache used to be written as
``float16``. Those are not the same 2 bytes: bf16 spends 8 bits on the exponent
(fp32's full range) and 7 on the mantissa, fp16 spends 5 and 10. So fp16 caps
at **65504** and anything above it becomes ``inf`` — and one ``inf`` makes that
activation's unit direction ``nan``, which does not raise, it just quietly
spawns a junk singleton region. Deep-layer residual streams have massive
activation outliers, so that ceiling is a real hazard rather than a theoretical
one. bf16 costs the same 2 bytes and cannot overflow.

numpy has no native bfloat16 and ``ml_dtypes`` is not in the Modal image, so
bf16 is stored as ``uint16`` holding the top half of the fp32 bit pattern —
the same representation ``lens_weights._to_f32`` already decodes for BF16
safetensors. Rounding is round-to-nearest-even rather than truncation; naive
truncation biases every value toward zero, which on a spherical mean over
millions of activations is a systematic error, not a wash.

Only *storage* changes. Everything downstream stays float32: the threshold
search resolves K differences across dtheta ~ 5e-8, and the bf16 ulp near
theta=0.77 is 0.0039 — 5 orders of magnitude too coarse to represent the
search space at all.
"""

from __future__ import annotations

import numpy as np

#: Manifest value used by caches written before the dtype was recorded.
LEGACY_DTYPE = "fp16"
SUPPORTED = ("bf16", "fp16", "fp32")


def encode(x: np.ndarray, dtype: str) -> np.ndarray:
    """Pack float32 activations for on-disk storage."""
    if dtype == "fp32":
        return x.astype(np.float32, copy=False)
    if dtype == "fp16":
        return x.astype(np.float16)
    if dtype != "bf16":
        raise ValueError(f"unsupported activation dtype {dtype!r}; use {SUPPORTED}")
    f32 = np.ascontiguousarray(x, dtype=np.float32)
    bits = f32.view(np.uint32)
    # Round-to-nearest-even on the 16 bits being dropped: add 0x7FFF plus the
    # low bit of the surviving half, then truncate.
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    # NaNs must stay NaN: carrying can turn a NaN payload into an infinity.
    out = (rounded >> 16).astype(np.uint16)
    nan = np.isnan(f32)
    if nan.any():
        out[nan] = (bits[nan] >> 16).astype(np.uint16) | 0x0040
    return out


def decode(x: np.ndarray, dtype: str) -> np.ndarray:
    """Unpack stored activations back to float32."""
    if dtype == "bf16":
        if x.dtype != np.uint16:
            raise ValueError(
                f"manifest says bf16 but shard holds {x.dtype}; the cache was "
                "written by a different version — re-extract or fix the manifest")
        return (x.astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    return x.astype(np.float32)


def from_manifest(manifest: dict) -> str:
    """Storage dtype of a cache, defaulting to fp16 for pre-dtype manifests."""
    return manifest.get("activation_dtype", LEGACY_DTYPE)


def read_shard_x(data, manifest: dict) -> np.ndarray:
    """Decode a shard's ``x`` to float32 using the manifest's dtype."""
    return decode(data["x"], from_manifest(manifest))
