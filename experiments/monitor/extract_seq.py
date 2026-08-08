"""Per-position activations at several layers in one forward pass.

Arms B and C both need activations that `extract.py` throws away — B needs
every position rather than the last, C needs layers 4 and 12 alongside 20. Both
are captured in the same pass, because the forward is the expensive part and
the hooks are free.

Two things this records that the final-position extractor does not, and both
matter downstream:

- **Prompt boundaries.** Activations are stored as one flat (N, D) block with a
  `lengths` array, so a trajectory can be sliced back out. A flat pool with no
  boundaries (which is what `refpool.npy` is) cannot answer any question about
  order.
- **Token ids.** The gemma chat scaffold contributes a fixed token prefix and
  suffix to every prompt, so those positions carry identical activations across
  the whole corpus and would manufacture spurious "shared structure". Keeping
  the ids lets the scaffold be located empirically and masked.

Storage is fp16: 1,500 Pile documents at 128 positions is 885 MB in fp16 and
1.77 GB in fp32, and the scoring path upcasts to fp32 anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMPTY_CACHE_EVERY = 10


@dataclass
class Sequences:
    """Flat activations plus the offsets needed to slice them per prompt."""

    x: dict[int, np.ndarray]     # layer -> (total_positions, D) float16
    lengths: np.ndarray          # (n_prompts,) int32, positions kept per prompt
    token_ids: np.ndarray        # (n_prompts, max_len) int32, -1 where padded

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)

    def slice(self, layer: int, i: int) -> np.ndarray:
        o = self.offsets
        return self.x[layer][o[i]:o[i + 1]]


def shared_affix(token_ids: np.ndarray, lengths: np.ndarray) -> tuple[int, int]:
    """Length of the token prefix and suffix common to every prompt.

    For chat-formatted prompts this recovers the scaffold without hardcoding
    the template: `<start_of_turn>user\\n` on the left, `<end_of_turn>\\n
    <start_of_turn>model\\n` on the right. Returns (0, 0) for raw text.
    """
    n, _ = token_ids.shape
    lo = int(lengths.min())
    pre = 0
    while pre < lo and len(np.unique(token_ids[:, pre])) == 1:
        pre += 1
    suf = 0
    while suf < lo - pre:
        col = np.array([token_ids[i, lengths[i] - 1 - suf] for i in range(n)])
        if len(np.unique(col)) != 1:
            break
        suf += 1
    return pre, suf


def extract_sequences(model, texts, *, layers=(20,), batch_size: int = 16,
                      max_tokens: int = 128, skip_bos: bool = True,
                      verbose: bool = True) -> Sequences:
    """Every position of every prompt, at every requested layer.

    BOS is dropped by default, matching `ep/discovery/extraction.py:107-114`:
    the build stream never saw position 0, so its activation has no region that
    was fitted to it.
    """
    import torch

    device = next(model.parameters()).device
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    acts: dict[int, object] = {}
    hooks = {L: f"blocks.{L}.hook_resid_post" for L in layers}

    def make(L):
        def fn(a, hook):
            acts[L] = a
        return fn

    chunks: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    lengths: list[int] = []
    id_rows: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True, padding_side="right")
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
        n_real = (tokens != pad_id).sum(dim=1)

        model.reset_hooks()
        for L in layers:
            model.add_hook(hooks[L], make(L), "fwd")
        with torch.no_grad():
            model(tokens, return_type=None)
        model.reset_hooks()

        pos = torch.arange(tokens.shape[1], device=device)
        keep = (pos[None, :] < n_real[:, None])
        if skip_bos:
            keep = keep & (pos[None, :] >= 1)
        for L in layers:
            chunks[L].append(acts[L][keep].to(torch.float16).cpu().numpy())

        kept = keep.sum(dim=1).cpu().numpy()
        lengths.extend(int(v) for v in kept)
        ids = tokens.cpu().numpy()
        for i in range(len(batch)):
            row = ids[i][1:int(n_real[i])] if skip_bos else ids[i][:int(n_real[i])]
            id_rows.append(row.astype(np.int32))

        acts.clear()
        del tokens
        b = start // batch_size
        if torch.backends.mps.is_available() and b % EMPTY_CACHE_EVERY == 0:
            torch.mps.empty_cache()
        if verbose and b % 25 == 0:
            print(f"  seq {start + len(batch)}/{len(texts)} prompts, "
                  f"{sum(lengths)} positions", flush=True)

    max_len = max(len(r) for r in id_rows)
    id_mat = np.full((len(id_rows), max_len), -1, dtype=np.int32)
    for i, r in enumerate(id_rows):
        id_mat[i, :len(r)] = r
    return Sequences(
        x={L: np.concatenate(chunks[L]) for L in layers},
        lengths=np.array(lengths, dtype=np.int32),
        token_ids=id_mat,
    )
