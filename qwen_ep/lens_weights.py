"""Fetch just the tensors needed for logit lens — unembedding + final norm —
from a hub safetensors file via HTTP range requests, skipping the rest of the
checkpoint.

``embed`` names whichever tensor *is* the unembedding for that checkpoint. On
the Qwen3.5 2B/4B and Gemma-2 that is ``embed_tokens`` (they set
``tie_word_embeddings=true``); on checkpoints that untie it — Qwen3.6-27B —
it is the separate ``lm_head.weight``. Pointing this at ``embed_tokens`` for an
untied model silently yields a wrong lens rather than an error, so check the
config before adding an entry.

Cached as .npz (float16) under --cache-dir so the download runs once.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_url

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def _fetch_range(url: str, start: int, end: int) -> bytes:
    """[start, end) byte range; follows the hub's CDN redirect."""
    r = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=600)
    r.raise_for_status()
    if len(r.content) != end - start:
        raise IOError(f"range fetch returned {len(r.content)} bytes, wanted {end - start}")
    return r.content


def _read_header(url: str) -> tuple[dict, int]:
    n = struct.unpack("<Q", _fetch_range(url, 0, 8))[0]
    header = json.loads(_fetch_range(url, 8, 8 + n))
    return header, 8 + n


def _to_f32(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        arr = u16.view("<f4")
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError(f"unhandled dtype {dtype}")
    return arr.reshape(shape).astype(np.float32)


def _load_local(repo_id: str, names: list[str]) -> dict[str, np.ndarray] | None:
    """Read tensors from the locally cached safetensors shards (handles
    sharding via the index). Returns None if the model isn't fully cached."""
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from safetensors import safe_open
    except Exception:
        return None
    # locate each tensor's shard from the index (or the single-file checkpoint)
    idx = try_to_load_from_cache(repo_id, "model.safetensors.index.json")
    shard_of: dict[str, str] = {}
    if isinstance(idx, str):
        wmap = json.loads(Path(idx).read_text())["weight_map"]
        for n in names:
            if n not in wmap:
                return None
            shard_of[n] = wmap[n]
    else:
        single = try_to_load_from_cache(repo_id, "model.safetensors")
        if not isinstance(single, str):
            return None
        for n in names:
            shard_of[n] = "model.safetensors"
    out: dict[str, np.ndarray] = {}
    for shard in set(shard_of.values()):
        path = try_to_load_from_cache(repo_id, shard)
        if not isinstance(path, str):
            return None
        # framework="pt": torch reads bf16 (numpy can't), then cast to f32.
        with safe_open(path, framework="pt") as f:
            for n, s in shard_of.items():
                if s == shard:
                    out[n] = f.get_tensor(n).float().numpy()
    return out


def fetch_tensors(repo_id: str, filename: str | None, names: list[str],
                  cache_dir: Path) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / (repo_id.replace("/", "--") + ".npz")
    if out_path.exists():
        with np.load(out_path) as z:
            return {k: z[k].astype(np.float32) for k in z.files}

    # Prefer reading from the local HF cache (no re-download); fall back to
    # per-tensor HTTP range requests against the hub.
    tensors = _load_local(repo_id, names)
    if tensors is None:
        tensors = {}
        # tensors may live in different shards; resolve each independently.
        for name in names:
            fn = filename or resolve_filename(repo_id, name)
            url = hf_hub_url(repo_id, fn)
            header, base = _read_header(url)
            info = header[name]
            s, e = info["data_offsets"]
            print(f"  {name}: {info['dtype']} {info['shape']} "
                  f"({(e - s) / 1e6:.0f} MB, range-fetch {fn})", flush=True)
            raw = _fetch_range(url, base + s, base + e)
            tensors[name] = _to_f32(raw, info["dtype"], info["shape"])

    np.savez(out_path, **{k: v.astype(np.float16) for k, v in tensors.items()})
    return tensors


SPECS = {
    "qwen": dict(
        repo_id="Qwen/Qwen3.5-2B-Base",
        filename="model.safetensors-00001-of-00001.safetensors",
        embed="model.language_model.embed_tokens.weight",
        norm="model.language_model.norm.weight",
        norm_plus_one=False,  # standard RMSNorm: x_hat * w
    ),
    "qwen4b": dict(
        repo_id="Qwen/Qwen3.5-4B",
        filename=None,  # sharded; embed + norm live in different shards
        embed="model.language_model.embed_tokens.weight",
        norm="model.language_model.norm.weight",
        norm_plus_one=False,
    ),
    "qwen4b-base": dict(
        repo_id="Qwen/Qwen3.5-4B-Base",
        filename=None,
        embed="model.language_model.embed_tokens.weight",
        norm="model.language_model.norm.weight",
        norm_plus_one=False,
    ),
    "qwen27b": dict(
        repo_id="Qwen/Qwen3.6-27B",
        filename=None,
        # tie_word_embeddings=false — the unembedding is its own tensor.
        embed="lm_head.weight",          # (248320, 5120) = 2.5 GB as fp16
        norm="model.language_model.norm.weight",
        norm_plus_one=False,
    ),
    "gemma": dict(
        # ungated mirror of google/gemma-2-2b (identical weights)
        repo_id="unsloth/gemma-2-2b",
        filename=None,  # resolved from the index at runtime
        embed="model.embed_tokens.weight",
        norm="model.norm.weight",
        norm_plus_one=True,  # gemma RMSNorm: x_hat * (1 + w)
    ),
}


def resolve_filename(repo_id: str, tensor: str) -> str:
    """Find which shard holds ``tensor`` (single-file repos included)."""
    from huggingface_hub import hf_hub_download
    try:
        idx = hf_hub_download(repo_id, "model.safetensors.index.json")
        return json.load(open(idx))["weight_map"][tensor]
    except Exception:
        return "model.safetensors"


def load_lens(model_key: str, cache_dir: Path) -> dict:
    spec = SPECS[model_key]
    # Pass filename through as-is (None for sharded repos): fetch_tensors
    # resolves each tensor's shard independently — embed and norm can differ.
    t = fetch_tensors(spec["repo_id"], spec["filename"],
                      [spec["embed"], spec["norm"]], cache_dir)
    w_u = t[spec["embed"]]           # (V, D)
    w_norm = t[spec["norm"]]         # (D,)
    if spec["norm_plus_one"]:
        w_norm = 1.0 + w_norm
    return {"W_U": w_u, "w_norm": w_norm}


def lens_topk(directions: np.ndarray, lens: dict, k: int = 8,
              chunk: int = 512) -> list[list[int]]:
    """RMSNorm-scale each direction, unembed, return top-k vocab ids."""
    import torch
    w_u = torch.from_numpy(lens["W_U"])          # (V, D)
    w_norm = torch.from_numpy(lens["w_norm"])    # (D,)
    ids: list[list[int]] = []
    for s in range(0, len(directions), chunk):
        d = torch.from_numpy(directions[s:s + chunk].astype(np.float32))
        d = d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-6) * w_norm
        logits = d @ w_u.T                       # (B, V)
        top = torch.topk(logits, k, dim=-1).indices
        ids.extend(top.tolist())
    return ids


def lens_topk_stats(directions: np.ndarray, lens: dict, k: int = 8,
                    chunk: int = 256) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """Like :func:`lens_topk` but also softmax the full vocab distribution.

    Returns ``(ids, entropy, topk_mass)`` — entropy in nats over the whole
    vocabulary and the summed probability of the top-k tokens. The RMSNorm
    scaling fixes the direction to RMS 1 before unembedding, so the softmax
    temperature is canonical and entropies are comparable across regions.
    """
    import torch
    w_u = torch.from_numpy(lens["W_U"])          # (V, D)
    w_norm = torch.from_numpy(lens["w_norm"])    # (D,)
    ids: list[list[int]] = []
    ents: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for s in range(0, len(directions), chunk):
        d = torch.from_numpy(directions[s:s + chunk].astype(np.float32))
        d = d / torch.sqrt((d * d).mean(-1, keepdim=True) + 1e-6) * w_norm
        logits = d @ w_u.T                       # (B, V)
        top = torch.topk(logits, k, dim=-1).indices
        ids.extend(top.tolist())
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ents.append((-(p * logp).sum(-1)).numpy())
        masses.append(p.gather(-1, top).sum(-1).numpy())
    return ids, np.concatenate(ents), np.concatenate(masses)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=list(SPECS))
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    lens = load_lens(args.model, Path(args.cache_dir))
    print(f"{args.model}: W_U {lens['W_U'].shape}, norm {lens['w_norm'].shape}")


if __name__ == "__main__":
    main()
