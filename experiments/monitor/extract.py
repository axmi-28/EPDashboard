"""Final-position activations, next-token entropy, and token counts in one pass.

Why a local extractor rather than `ep.extract_final_position`:

- S5 needs the model's own next-token entropy, which means the logits. The
  reference extractor discards them. Running a second forward pass to get them
  would double the only expensive part of the experiment.
- R5 is raw token ids with no text form (see `corpora.py`), and the reference
  extractor only accepts strings.
- We need the per-prompt token count as a scorer in its own right (S0), to
  catch rungs where prompt length alone separates the classes.

The gather is identical to `ep/discovery/extraction.py:213-216`: right-pad,
index at `lengths - 1`.

**Never ask for `return_type="logits"`.** gemma-2's vocabulary is 256,000, so
the logits for one batch are (16, 128, 256000) = 1.05 GB in bfloat16. On MPS
that lands in unified memory and the caching allocator does not hand it back
between batches; across 12,000 prompts it exhausted a 30 GB swap file and put
the process into uninterruptible wait. Instead we hook the final normalised
residual and do the unembed ourselves, 16 positions at a time (262 MB), then
drop each chunk. `output_logits_soft_cap` has to be reapplied by hand because
we are bypassing the code path that normally applies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENTROPY_POS_CHUNK = 16     # positions per unembed chunk; 16x16x256000x4B = 262 MB
EMPTY_CACHE_EVERY = 10     # batches between MPS allocator releases


def _entropy_from_resid(model, normalized, chunk: int = ENTROPY_POS_CHUNK):
    """Next-token entropy per position, unembedding a few positions at a time.

    Replicates `HookedTransformer.forward`'s tail exactly. Two steps are easy
    to miss and both were caught by diffing against `return_type="logits"`:

    - `ln_final.hook_normalized` fires on ``x / scale``, *before* the learned
      RMS weight is applied (`RMSNorm.forward`), so ``ln_final.w`` has to be
      multiplied back in here.
    - gemma-2 soft-caps the logits at 30 before the softmax. Skipping the cap
      changes the effective temperature and inflates every entropy.
    """
    import torch

    B, T, _ = normalized.shape
    out = torch.empty((B, T), dtype=torch.float32, device=normalized.device)
    cap = float(getattr(model.cfg, "output_logits_soft_cap", 0.0) or 0.0)
    w = getattr(model.ln_final, "w", None)
    # Keep the unembed in the model dtype. `model.W_U.float()` would allocate
    # 2304 x 256000 x 4B = 2.36 GB per call, once per position chunk, which
    # cost 2.6x throughput before it was spotted. The matmul accumulates in
    # fp32 internally; only the (B, chunk, V) result is upcast.
    for s in range(0, T, chunk):
        h = normalized[:, s:s + chunk]
        if w is not None:
            h = h * w
        # TransformerLens' RMSNorm returns float32 even on a bf16 model, so
        # cast down to the unembed dtype rather than up.
        lg = (h.to(model.W_U.dtype) @ model.W_U + model.b_U).float()
        if cap > 0.0:
            lg = cap * torch.tanh(lg / cap)
        lp = torch.log_softmax(lg, dim=-1)
        out[:, s:s + chunk] = -(lp.exp() * lp).sum(-1)
        del lg, lp, h
    return out


@dataclass
class Extracted:
    x: np.ndarray               # (N, D) float32 final-position activations
    entropy_max: np.ndarray     # (N,) max over positions of next-token entropy
    entropy_final: np.ndarray   # (N,) entropy at the final position
    n_tokens: np.ndarray        # (N,) int32, non-pad token count incl. BOS


def load_model(name: str = "google/gemma-2-2b-it", device: str = "mps",
               dtype: str = "bfloat16"):
    import torch
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained_no_processing(
        name, device=device, dtype=getattr(torch, dtype),
    )


def _to_token_batch(model, items, device):
    """Build a right-padded (B, T) id tensor from texts and/or raw id lists."""
    import torch

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    seqs: list[list[int]] = []
    for it in items:
        if it.token_ids is not None:
            seqs.append(list(it.token_ids))
        else:
            # One text at a time, so the result is never padded and there is
            # nothing to strip. Filtering pad ids here would silently delete
            # any real token that happens to share the pad id.
            ids = model.to_tokens([it.text], prepend_bos=True,
                                  padding_side="right")[0]
            seqs.append([int(v) for v in ids])
    lengths = [len(s) for s in seqs]
    T = max(lengths)
    out = np.full((len(seqs), T), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return (torch.from_numpy(out).to(device),
            torch.tensor(lengths, device=device))


def extract(model, items, *, layer: int = 20, batch_size: int = 16,
            max_tokens: int | None = 128, verbose: bool = True) -> Extracted:
    """Forward `items` and return final-position activations plus entropy.

    `max_tokens` truncates every sequence to at most that many tokens, matching
    the build's context length. Truncation happens in token space so the final
    position is well-defined for every rung.
    """
    import torch

    hook = f"blocks.{layer}.hook_resid_post"
    # Read the device off the parameters, not cfg: the MPS-vs-CPU validation
    # moves the model and cfg.device does not follow it.
    device = next(model.parameters()).device
    acts: dict = {}

    def fwd(a, hook):
        acts["x"] = a

    def fwd_final(a, hook):
        acts["final"] = a

    xs, e_max, e_fin, n_tok = [], [], [], []
    model.eval()
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        tokens, lengths = _to_token_batch(model, chunk, device)
        if max_tokens is not None and tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
            lengths = torch.clamp(lengths, max=max_tokens)

        model.reset_hooks()
        model.add_hook(hook, fwd, "fwd")
        model.add_hook("ln_final.hook_normalized", fwd_final, "fwd")
        with torch.no_grad():
            model(tokens, return_type=None)
        model.reset_hooks()

        B, T = tokens.shape
        with torch.no_grad():
            ent = _entropy_from_resid(model, acts["final"])

        # Pad positions carry a real (meaningless) entropy; mask before max.
        pos = torch.arange(T, device=device)
        valid = pos[None, :] < lengths[:, None]
        ent = ent.masked_fill(~valid, float("-inf"))

        final = lengths - 1
        bidx = torch.arange(B, device=device)
        xs.append(acts["x"][bidx, final].float().cpu().numpy())
        e_max.append(ent.max(dim=1).values.cpu().numpy())
        e_fin.append(ent[bidx, final].cpu().numpy())
        n_tok.append(lengths.cpu().numpy())

        acts.clear()
        del ent, tokens, lengths
        b_idx = start // batch_size
        if torch.backends.mps.is_available() and b_idx % EMPTY_CACHE_EVERY == 0:
            torch.mps.empty_cache()
        if verbose and b_idx % 25 == 0:
            print(f"  extract {start + len(chunk)}/{len(items)}", flush=True)

    return Extracted(
        x=np.concatenate(xs).astype(np.float32),
        entropy_max=np.concatenate(e_max).astype(np.float32),
        entropy_final=np.concatenate(e_fin).astype(np.float32),
        n_tokens=np.concatenate(n_tok).astype(np.int32),
    )


def extract_per_position_pool(model, texts, *, layer: int = 20,
                              batch_size: int = 16,
                              context_length: int = 128,
                              verbose: bool = True) -> np.ndarray:
    """Per-position activations for the S3/S4 reference pool.

    EP's exemplars are per-position Pile activations, so the coreset it is
    compared against must be drawn from the same population — otherwise the
    baseline is handicapped by construction rather than by the method.
    Positions 1..L-1 are kept, skipping BOS, exactly as
    `ep/discovery/extraction.py:107-114` does at build time.
    """
    import torch

    hook = f"blocks.{layer}.hook_resid_post"
    device = next(model.parameters()).device
    acts: dict = {}

    def fwd(a, hook):
        acts["x"] = a

    out: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        tokens = model.to_tokens(chunk, prepend_bos=True, padding_side="right")
        if tokens.shape[1] > context_length + 1:
            tokens = tokens[:, :context_length + 1]
        pad_id = model.tokenizer.pad_token_id or 0
        lengths = (tokens != pad_id).sum(dim=1)

        model.reset_hooks()
        model.add_hook(hook, fwd, "fwd")
        with torch.no_grad():
            model(tokens, return_type=None)
        model.reset_hooks()

        pos = torch.arange(tokens.shape[1], device=device)
        keep = (pos[None, :] < lengths[:, None]) & (pos[None, :] >= 1)
        out.append(acts["x"][keep].float().cpu().numpy())
        acts.clear()
        del tokens, lengths
        if (torch.backends.mps.is_available()
                and (start // batch_size) % EMPTY_CACHE_EVERY == 0):
            torch.mps.empty_cache()
        if verbose and (start // batch_size) % 25 == 0:
            print(f"  refpool {start + len(chunk)}/{len(texts)} texts, "
                  f"{sum(len(o) for o in out)} activations", flush=True)

    return np.concatenate(out).astype(np.float32)
