"""Activation sources for the two dashboard passes.

Pass 1 streams ``(X, prompt_gid, position)`` chunks over the whole budget so
the region scan can rank every token. Pass 2 revisits only the prompts that
won a sequence slot and returns their *full* per-position activations, which
is what per-token projection coloring needs.

Two implementations share that interface:

``ForwardSource``
    Streams a HF dataset (default ``monology/pile-uncopyrighted``), decodes
    fixed ``context_length``-token windows, and runs the model with a residual
    hook via ``qwen_ep.adapter.QwenModel`` — the only model-touching seam.
    Pass 2 is a second forward over the winning prompts only, which is cheap
    (a few thousand prompts) relative to pass 1.

``CacheSource``
    Replays the shards ``qwen_ep.extract_cache`` wrote. No model, no GPU.
    Pass 2 is a second read of the same shards filtered to the winning gids.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

Chunk = tuple[np.ndarray, np.ndarray, np.ndarray]  # X (n,d), gid (n,), pos (n,)


def stream_dataset_texts(dataset: str, split: str, column: str, tokenizer,
                         context_length: int, seed: int,
                         min_chars: int = 200) -> Iterator[str]:
    """Yield decoded ``context_length``-token windows from any HF text dataset.

    Matches dictionary construction (`qwen_ep.data.stream_pile_texts`): shuffle
    the stream, skip short documents, keep the leading window of each. The
    yielded object is a *string*, re-tokenised downstream — that round-trip is
    what dictionary construction did, so examples land on the same tokens.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for item in ds:
        text = item.get(column, "")
        if len(text) < min_chars:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < context_length:
            continue
        yield tokenizer.decode(ids[:context_length])


class ForwardSource:
    def __init__(self, cfg, layer: int):
        from qwen_ep.adapter import QwenModel

        self.cfg = cfg
        self.layer = layer
        self.model = QwenModel(cfg.model_id, device=cfg.device)
        self.prompts: list[str] = []

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def describe(self) -> dict:
        return {"mode": "forward", "model_id": self.cfg.model_id,
                "layer": self.layer, "dataset": self.cfg.dataset,
                "split": self.cfg.dataset_split,
                "context_length": self.cfg.context_length,
                "n_prompts": self.cfg.n_prompts, "seed": self.cfg.seed}

    def _extract(self, batch: list[str]):
        return self.model.extract_per_position(
            batch, layer=self.layer, batch_size=self.cfg.batch_size,
            skip_first=True)

    def pass1(self) -> Iterator[Chunk]:
        cfg = self.cfg
        texts = stream_dataset_texts(cfg.dataset, cfg.dataset_split,
                                     cfg.dataset_column, self.tokenizer,
                                     cfg.context_length, cfg.seed)
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(self.prompts) + len(batch) >= cfg.n_prompts \
                    or len(batch) >= cfg.prompt_batch_size:
                res = self._extract(batch)
                gid = res.prompt_ids.astype(np.int64) + len(self.prompts)
                self.prompts.extend(batch)
                batch = []
                yield res.x.astype(np.float32), gid, res.position_ids.astype(np.int64)
                if len(self.prompts) >= cfg.n_prompts:
                    return
        if batch:
            res = self._extract(batch)
            gid = res.prompt_ids.astype(np.int64) + len(self.prompts)
            self.prompts.extend(batch)
            yield res.x.astype(np.float32), gid, res.position_ids.astype(np.int64)

    def pass2(self, gids: list[int]):
        """Yield ``(gid, X (T,d), positions (T,))`` per winning prompt.

        A generator so the caller can fold each prompt into its per-region
        caches and drop the raw activations — materialising all winners at
        once is ~1 MB/prompt and was the other half of the measured OOM.
        """
        gids = sorted(set(gids))
        for s in range(0, len(gids), self.cfg.prompt_batch_size):
            part = gids[s:s + self.cfg.prompt_batch_size]
            res = self._extract([self.prompts[g] for g in part])
            for i, g in enumerate(part):
                m = res.prompt_ids == i
                yield (g, res.x[m].astype(np.float32),
                       res.position_ids[m].astype(np.int64))


class CacheSource:
    def __init__(self, cfg, layer: int | None = None):
        self.cfg = cfg
        self.dir = Path(cfg.cache_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.layer = self.manifest["layer"]
        if layer is not None and layer != self.layer:
            raise ValueError(f"cache is layer {self.layer}, dictionary wants {layer}")
        self.prompts: list[str] = []
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.manifest["model_id"])
        return self._tok

    def describe(self) -> dict:
        m = self.manifest
        return {"mode": "cache", "cache_dir": self.dir.name,
                "model_id": m["model_id"], "layer": m["layer"],
                "dataset": m.get("corpus", "?"),
                "context_length": m.get("context_length"),
                "n_activations": m.get("n_activations"), "seed": m.get("seed")}

    def _shards(self):
        for name in self.manifest["shard_files"]:
            yield np.load(self.dir / name, allow_pickle=True)

    def pass1(self) -> Iterator[Chunk]:
        max_acts = self.cfg.n_prompts * (self.cfg.context_length - 1)
        seen = 0
        for data in self._shards():
            x, pid, pos = data["x"], data["prompt_ids"], data["position_ids"]
            base = len(self.prompts)
            self.prompts.extend(str(p) for p in data["prompts"])
            for s in range(0, len(x), self.cfg.chunk_size):
                if seen >= max_acts:
                    return
                sl = slice(s, s + self.cfg.chunk_size)
                yield (x[sl].astype(np.float32),
                       pid[sl].astype(np.int64) + base,
                       pos[sl].astype(np.int64))
                seen += len(x[sl])

    def pass2(self, gids: list[int]):
        """Yield ``(gid, X, positions)`` per winning prompt (see ForwardSource).

        ``extract_cache`` writes whole prompt batches per shard, so a prompt
        never spans shards; each shard's winners are yielded as it is read.
        """
        want = set(gids)
        seen: set[int] = set()
        base = 0
        for data in self._shards():
            x, pid, pos = data["x"], data["prompt_ids"], data["position_ids"]
            n_local = len(data["prompts"])
            g = pid.astype(np.int64) + base
            hit = np.isin(g, list(want & set(np.unique(g).tolist())))
            for gg in np.unique(g[hit]):
                gg = int(gg)
                if gg in seen:
                    logger.warning("pass2: gid %d spans shards — keeping the "
                                   "first occurrence only", gg)
                    continue
                seen.add(gg)
                m = g == gg
                o = np.argsort(pos[m])
                yield (gg, x[m].astype(np.float32)[o],
                       pos[m].astype(np.int64)[o])
            base += n_local
        for gg in want - seen:
            logger.warning("pass2: gid %d not found in cache", gg)


def make_source(cfg, layer: int):
    return CacheSource(cfg, layer) if cfg.cache_dir else ForwardSource(cfg, layer)
