"""Qwen3.5-2B adapter for Exemplar Partitioning.

The EP library (`ep`) was written against TransformerLens `HookedTransformer`
objects. Its discovery pipeline, however, only ever touches the model through a
single seam: an ``extract_fn(model, prompts, hook_name, **kwargs)`` that returns
an ``ExtractionResult`` with raw residual-stream activations of shape ``(N, D)``.
Everything downstream (centering, unit-normalisation, leader clustering) happens
inside `ep` on plain numpy arrays.

Qwen3.5-2B-Base is a vision-language model (`Qwen3_5ForConditionalGeneration`)
whose *text backbone* is a 24-layer hybrid of linear (Mamba-style) attention and
periodic full attention over a shared residual stream of width 2048. Rather than
wait for TransformerLens to support this brand-new architecture, we hook the raw
HuggingFace module tree directly: a forward hook on decoder layer ``L`` captures
its output, which is exactly the post-block residual stream (the analogue of
``blocks.L.hook_resid_post``).

This module provides:
  * ``QwenModel``      – loads the HF model + tokenizer, finds the decoder layer
                          stack, and exposes a residual-stream extractor.
  * ``make_extract_fn`` – builds an ``extract_fn`` bound to a chosen layer that
                          plugs straight into ``ep.calibrate_pipeline`` /
                          ``ep.discover``.
  * ``QwenModel.logit_lens`` – top-vocab tokens for an exemplar direction, for
                          interpreting partitions (tie_word_embeddings=True).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ep.discovery.extraction import ExtractionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B-Base"


def model_tag(model_id: str) -> str:
    """Filesystem-safe short tag for a HF model id.

    Used in run slugs, cache dir names, and the calibration cache key, all of
    which must distinguish models: the calibration center/threshold is only
    valid for the model it was measured on.
    """
    return model_id.split("/")[-1].replace(".", "_").lower()


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_decoder_layers(model) -> tuple[torch.nn.ModuleList, str]:
    """Locate the decoder-layer ``ModuleList`` in an HF model tree.

    Qwen VL models nest the text stack under ``model.model.language_model`` (or
    ``model.language_model.model``); plain causal-LMs put it at
    ``model.model``. Rather than hard-code one path, we try the known ones and
    then fall back to the longest ModuleList whose elements are named
    ``*DecoderLayer``.
    """
    candidates = [
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
        "transformer.h",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            logger.info("decoder layers found at model.%s (%d layers)", path, len(obj))
            return obj, path

    # Fallback: scan for the longest ModuleList of decoder-like blocks.
    best: tuple[torch.nn.ModuleList, str] | None = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            child = mod[0].__class__.__name__.lower()
            if "decoderlayer" in child or "block" in child:
                if best is None or len(mod) > len(best[0]):
                    best = (mod, name)
    if best is not None:
        logger.info("decoder layers found by scan at model.%s (%d layers)", best[1], len(best[0]))
        return best
    raise RuntimeError(
        "Could not locate decoder layers in the model tree. Inspect "
        "model.named_modules() and extend _find_decoder_layers()."
    )


def _find_final_norm(model):
    """Locate the final pre-unembed norm (RMSNorm) so the logit lens can apply
    it, matching how the model actually reads out the residual stream. Returns
    the module or None."""
    for path in ("model.language_model.norm", "model.model.language_model.norm",
                 "language_model.model.norm", "model.model.norm", "model.norm"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj
    return None


@dataclass
class _Cfg:
    """Minimal stand-in for TransformerLens `model.cfg` (only d_model is read
    by the EP extractors, but we keep n_layers for convenience)."""
    d_model: int
    n_layers: int


class QwenModel:
    """Thin wrapper around a HuggingFace Qwen model for EP activation extraction.

    Loads the model once, in eval mode, no grad. Extraction hooks the output of
    a chosen decoder layer (post-block residual stream).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        prepend_bos: bool = False,
        revision: str | None = None,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = _pick_device(device)
        self.prepend_bos = prepend_bos

        # ``tokenizer_id`` exists for checkpoint *pairs*. Two repos can carry the
        # same vocabulary and still tokenize differently: cais/Zephyr_RMU ships
        # only tokenizer.model, so transformers reconstructs a fast tokenizer
        # with add_prefix_space=True, while its base HuggingFaceH4/zephyr-7b-beta
        # ships a tokenizer.json with add_prefix_space=False. Identical text then
        # yields different ids, and any paired activation comparison is silently
        # comparing different token streams.
        tok_id = tokenizer_id or model_id
        tok_rev = tokenizer_revision if tokenizer_id else revision
        logger.info("loading tokenizer %s%s", tok_id,
                    "" if tok_id == model_id else f" (overriding {model_id})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_id, trust_remote_code=True, revision=tok_rev)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right-pad so content sits at positions [0, L) and pads trail.
        self.tokenizer.padding_side = "right"

        logger.info("loading model %s (dtype=%s, device=%s)", model_id, dtype, self.device)
        # low_cpu_mem_usage keeps peak RAM near model size during load.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            revision=revision,
        )
        self.model.to(self.device)
        self.model.eval()

        self.layers, self.layers_path = _find_decoder_layers(self.model)
        self.n_layers = len(self.layers)
        # Resolve d_model from config text branch (VL) or top level.
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.d_model = int(getattr(text_cfg, "hidden_size"))
        self.cfg = _Cfg(d_model=self.d_model, n_layers=self.n_layers)

        # Lazily resolved lm-head / embedding for the logit lens.
        self._lm_head_weight: torch.Tensor | None = None
        self._final_norm = _find_final_norm(self.model)

        logger.info(
            "QwenModel ready: d_model=%d n_layers=%d device=%s",
            self.d_model, self.n_layers, self.device,
        )

    # ------------------------------------------------------------------ tokens
    def to_tokens(self, prompts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=self.prepend_bos,
        )
        return enc["input_ids"]

    # -------------------------------------------------------------- extraction
    @torch.no_grad()
    def extract_per_position_multi(
        self,
        prompts: Iterable[str],
        layers: "list[int]",
        max_positions_per_prompt: int | None = None,
        batch_size: int = 16,
        skip_first: bool = True,
    ) -> "dict[int, ExtractionResult]":
        """Harvest several layers from **one** forward pass.

        A 27B forward is the same cost whether you read one residual stream or
        all 64, so extracting layer-by-layer pays for the model N times over.
        Returns ``{layer: ExtractionResult}``; every result shares identical
        ``prompt_ids`` / ``position_ids``, since they come from the same tokens.

        Peak host RAM is ``len(layers)`` × one sub-batch of activations, so the
        caller should keep ``batch_size`` modest when sweeping many layers.
        """
        prompts = list(prompts)
        empty = lambda: ExtractionResult(
            x=np.zeros((0, self.d_model), dtype=np.float32))
        if not prompts:
            return {l: empty() for l in layers}
        for l in layers:
            if not (0 <= l < self.n_layers):
                raise ValueError(f"layer {l} out of range [0, {self.n_layers})")

        per_layer_x: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        all_prompt_ids: list[np.ndarray] = []
        all_position_ids: list[np.ndarray] = []
        total_tokens = 0
        n_fwd = 0
        first_pos = 1 if skip_first else 0

        for start in range(0, len(prompts), batch_size):
            sub = prompts[start:start + batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=self.prepend_bos,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            lengths = attn.sum(dim=1)
            if max_positions_per_prompt is not None:
                lengths = torch.clamp(lengths, max=max_positions_per_prompt)
            total_tokens += int(lengths.sum().item())

            captured: dict[int, "torch.Tensor"] = {}

            def make_hook(idx: int):
                def hook(module, inputs, output):
                    captured[idx] = output[0] if isinstance(output, tuple) else output
                return hook

            handles = [self.layers[l].register_forward_hook(make_hook(l))
                       for l in layers]
            try:
                self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            n_fwd += 1

            # Mask is identical across layers — same tokens, same lengths.
            any_acts = captured[layers[0]]
            T = any_acts.shape[1]
            positions = torch.arange(T, device=any_acts.device)
            keep = ((positions[None, :] < lengths[:, None])
                    & (positions[None, :] >= first_pos))
            for l in layers:
                per_layer_x[l].append(
                    captured[l][keep].detach().to("cpu", dtype=torch.float32).numpy())

            lens_np = lengths.cpu().numpy()
            for i, L in enumerate(lens_np):
                L = int(L)
                if L <= first_pos:
                    continue
                n_kept = L - first_pos
                all_prompt_ids.append(np.full(n_kept, start + i, dtype=np.int64))
                all_position_ids.append(np.arange(first_pos, L, dtype=np.int64))

            captured.clear()
            del any_acts, input_ids, attn

        prompt_ids = (np.concatenate(all_prompt_ids) if all_prompt_ids
                      else np.array([], dtype=np.int64))
        position_ids = (np.concatenate(all_position_ids) if all_position_ids
                        else np.array([], dtype=np.int64))
        out = {}
        for l in layers:
            xs = per_layer_x[l]
            out[l] = ExtractionResult(
                x=np.concatenate(xs) if xs
                else np.zeros((0, self.d_model), np.float32),
                prompt_ids=prompt_ids, position_ids=position_ids,
                n_forward_passes=n_fwd, n_tokens=total_tokens,
            )
        return out

    def extract_per_position(
        self,
        prompts: Iterable[str],
        layer: int,
        max_positions_per_prompt: int | None = None,
        batch_size: int = 16,
        skip_first: bool = True,
    ) -> ExtractionResult:
        """Harvest post-block residual activations at ``layer`` for every kept
        position of every prompt.

        Mirrors ``ep.extract_per_position`` semantics: positions ``1..L-1`` are
        kept per prompt (skipping position 0, the attention-sink / BOS slot),
        one forward pass per padded sub-batch, activations gathered on-device
        then moved to CPU float32 once.
        """
        prompts = list(prompts)
        if not prompts:
            return ExtractionResult(x=np.zeros((0, self.d_model), dtype=np.float32))

        pad_id = self.tokenizer.pad_token_id or 0
        target = self.layers[layer]

        all_x: list[np.ndarray] = []
        all_prompt_ids: list[np.ndarray] = []
        all_position_ids: list[np.ndarray] = []
        n_fwd = 0
        total_tokens = 0

        first_pos = 1 if skip_first else 0

        for start in range(0, len(prompts), batch_size):
            sub = prompts[start:start + batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=self.prepend_bos,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            lengths = attn.sum(dim=1)  # (B,)
            if max_positions_per_prompt is not None:
                lengths = torch.clamp(lengths, max=max_positions_per_prompt)
            total_tokens += int(lengths.sum().item())

            captured: dict = {}

            def hook(module, inputs, output):
                # Decoder layers return a tuple (hidden_states, ...) or a bare
                # tensor. hidden_states is the post-block residual stream.
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = target.register_forward_hook(hook)
            try:
                self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                handle.remove()
            n_fwd += 1

            acts = captured["h"]  # (B, T, D)
            B, T, _ = acts.shape
            positions = torch.arange(T, device=acts.device)
            keep = (positions[None, :] < lengths[:, None]) & (positions[None, :] >= first_pos)
            flat = acts[keep].detach().to("cpu", dtype=torch.float32).numpy()
            all_x.append(flat)

            lens_np = lengths.cpu().numpy()
            for i, L in enumerate(lens_np):
                L = int(L)
                if L <= first_pos:
                    continue
                n_kept = L - first_pos
                all_prompt_ids.append(np.full(n_kept, start + i, dtype=np.int64))
                all_position_ids.append(np.arange(first_pos, L, dtype=np.int64))

            del captured, acts, input_ids, attn

        x = np.concatenate(all_x) if all_x else np.zeros((0, self.d_model), np.float32)
        prompt_ids = (np.concatenate(all_prompt_ids) if all_prompt_ids
                      else np.array([], dtype=np.int64))
        position_ids = (np.concatenate(all_position_ids) if all_position_ids
                        else np.array([], dtype=np.int64))
        return ExtractionResult(
            x=x, prompt_ids=prompt_ids, position_ids=position_ids,
            n_forward_passes=n_fwd, n_tokens=total_tokens,
        )

    @torch.no_grad()
    def extract_final_position(
        self,
        prompts: Iterable[str],
        layer: int,
        batch_size: int = 16,
    ) -> ExtractionResult:
        """Harvest the residual activation at the LAST non-pad position of each
        prompt — one activation per prompt, in prompt order."""
        prompts = list(prompts)
        if not prompts:
            return ExtractionResult(x=np.zeros((0, self.d_model), dtype=np.float32))

        target = self.layers[layer]
        all_x: list[np.ndarray] = []
        n_fwd = 0
        for start in range(0, len(prompts), batch_size):
            sub = prompts[start:start + batch_size]
            enc = self.tokenizer(
                sub, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=self.prepend_bos,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            lengths = attn.sum(dim=1)  # (B,)

            captured: dict = {}

            def hook(module, inputs, output):
                captured["h"] = output[0] if isinstance(output, tuple) else output

            handle = target.register_forward_hook(hook)
            try:
                self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                handle.remove()
            n_fwd += 1

            acts = captured["h"]  # (B, T, D)
            idx = (lengths - 1).clamp(min=0)
            final = acts[torch.arange(acts.shape[0], device=acts.device), idx]
            all_x.append(final.detach().to("cpu", dtype=torch.float32).numpy())
            del captured, acts, input_ids, attn

        x = np.concatenate(all_x) if all_x else np.zeros((0, self.d_model), np.float32)
        return ExtractionResult(
            x=x,
            prompt_ids=np.arange(len(prompts), dtype=np.int64),
            n_forward_passes=n_fwd, n_tokens=len(prompts),
        )

    # -------------------------------------------------------------- generation
    def _chat_messages(self, prompt: str, system: str | None = None) -> list[dict]:
        msgs = []
        if system:  # empty string == "no system message" (paper's default role)
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def format_chat(self, prompt: str, system: str | None = None) -> str:
        """User (+ optional system) prompt -> chat-template string ready to
        tokenize (generation prompt appended). Disables thinking mode where the
        template supports it, so max_new_tokens goes to the visible reply."""
        msgs = self._chat_messages(prompt, system)
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    def format_conversation(self, messages: list[dict],
                            add_generation_prompt: bool = True) -> str:
        """Render a full ``[{role, content}, …]`` chat to a template string."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt)

    @torch.no_grad()
    def _mean_span_activations(
        self,
        prompt_texts: list[str],
        full_texts: list[str],
        layers: list[int],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Mean post-block residual over the token span ``[len(prompt), len(full))``
        of each ``full_text``, for each layer. One forward pass per batch, one
        hook per layer. Returns ``(N, len(layers), d_model)`` float32."""
        n = len(full_texts)
        out = np.zeros((n, len(layers), self.d_model), dtype=np.float32)
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            for start in range(0, n, batch_size):
                sl = slice(start, min(start + batch_size, n))
                prompt_lens = [len(self.tokenizer(p, add_special_tokens=False)
                                   ["input_ids"]) for p in prompt_texts[sl]]
                enc = self.tokenizer(full_texts[sl], return_tensors="pt",
                                     padding=True, truncation=False,
                                     add_special_tokens=False)
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)

                captured: dict = {}
                handles = []
                for L in layers:
                    def mk(L):
                        def hook(module, inputs, output):
                            captured[L] = (output[0] if isinstance(output, tuple)
                                           else output)
                        return hook
                    handles.append(self.layers[L].register_forward_hook(mk(L)))
                try:
                    self.model(input_ids=input_ids, attention_mask=attn,
                               use_cache=False)
                finally:
                    for h in handles:
                        h.remove()

                lengths = attn.sum(dim=1).cpu().numpy()
                for bi in range(input_ids.shape[0]):
                    p0 = prompt_lens[bi]
                    p1 = int(lengths[bi])            # last non-pad (right-padded)
                    if p1 <= p0:                      # empty span -> last tok
                        p0 = max(0, p1 - 1)
                    for li, L in enumerate(layers):
                        h = captured[L][bi, p0:p1]
                        out[start + bi, li] = (
                            h.mean(0).detach().to("cpu", torch.float32).numpy())
        finally:
            self.tokenizer.padding_side = prev_side
        return out

    def mean_response_activations(
        self,
        systems: list[str | None],
        users: list[str],
        responses: list[str],
        layers: list[int],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Mean post-block residual over the assistant response tokens of each
        (system, user, response) chat — the paper's role-vector primitive.
        Returns ``(N, len(layers), d_model)``."""
        prompt_texts = [self.format_chat(u, system=s)
                        for s, u in zip(systems, users)]
        full_texts = [p + r for p, r in zip(prompt_texts, responses)]
        return self._mean_span_activations(prompt_texts, full_texts, layers,
                                           batch_size)

    @torch.no_grad()
    def response_token_activations(
        self,
        systems: list[str | None],
        users: list[str],
        responses: list[str],
        layer: int,
        batch_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-token post-block residual over the assistant response span of each
        (system, user, response) chat, at one layer. Returns
        ``(x (Ntok, d), rollout_ids (Ntok,), position_ids (Ntok,))`` — the raw
        material for building a persona-space EP dictionary in chat context."""
        prompt_texts = [self.format_chat(u, system=s)
                        for s, u in zip(systems, users)]
        full_texts = [p + r for p, r in zip(prompt_texts, responses)]
        xs, rids, pids = [], [], []
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            for start in range(0, len(full_texts), batch_size):
                sl = slice(start, min(start + batch_size, len(full_texts)))
                plens = [len(self.tokenizer(p, add_special_tokens=False)
                             ["input_ids"]) for p in prompt_texts[sl]]
                enc = self.tokenizer(full_texts[sl], return_tensors="pt",
                                     padding=True, truncation=False,
                                     add_special_tokens=False)
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                captured: dict = {}

                def hook(module, inputs, output):
                    captured["h"] = (output[0] if isinstance(output, tuple)
                                     else output)
                handle = self.layers[layer].register_forward_hook(hook)
                try:
                    self.model(input_ids=input_ids, attention_mask=attn,
                               use_cache=False)
                finally:
                    handle.remove()
                h = captured["h"]
                lengths = attn.sum(dim=1).cpu().numpy()
                for bi in range(input_ids.shape[0]):
                    p0, p1 = plens[bi], int(lengths[bi])
                    if p1 <= p0:
                        continue
                    seg = h[bi, p0:p1].detach().to("cpu", torch.float32).numpy()
                    xs.append(seg)
                    rid = start + bi
                    rids.append(np.full(seg.shape[0], rid, dtype=np.int64))
                    pids.append(np.arange(seg.shape[0], dtype=np.int64))
        finally:
            self.tokenizer.padding_side = prev_side
        if not xs:
            return (np.zeros((0, self.d_model), np.float32),
                    np.array([], np.int64), np.array([], np.int64))
        return (np.concatenate(xs), np.concatenate(rids), np.concatenate(pids))

    def mean_last_turn_activations(
        self,
        conversations: list[list[dict]],
        layers: list[int],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Mean post-block residual over the *final assistant turn* of each full
        conversation (message list ending in an assistant message). Returns
        ``(N, len(layers), d_model)`` — the per-turn drift primitive."""
        prompt_texts = [self.format_conversation(c[:-1], add_generation_prompt=True)
                        for c in conversations]
        full_texts = [self.format_conversation(c, add_generation_prompt=False)
                      for c in conversations]
        return self._mean_span_activations(prompt_texts, full_texts, layers,
                                           batch_size)

    @torch.no_grad()
    def generate(
        self,
        formatted_prompts: list[str],
        max_new_tokens: int = 60,
        batch_size: int = 8,
        layer_hook: tuple[int, callable] | None = None,
    ) -> list[str]:
        """Greedy-decode continuations for already-formatted prompts.

        ``layer_hook`` is ``(layer_idx, fn)`` where ``fn(hidden) -> hidden``
        edits the post-block residual stream (B, T, D); it stays installed for
        every forward pass of the generation loop, prefill and decode alike.
        Left-pads internally (restores padding_side afterwards).
        """
        handle = None
        if layer_hook is not None:
            layer_idx, fn = layer_hook

            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    return (fn(output[0]),) + tuple(output[1:])
                return fn(output)

            handle = self.layers[layer_idx].register_forward_hook(hook)

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        texts: list[str] = []
        try:
            for start in range(0, len(formatted_prompts), batch_size):
                sub = formatted_prompts[start:start + batch_size]
                enc = self.tokenizer(
                    sub, return_tensors="pt", padding=True, truncation=False,
                    add_special_tokens=False,
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                out = self.model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new = out[:, input_ids.shape[1]:]
                texts.extend(
                    self.tokenizer.decode(row, skip_special_tokens=True)
                    for row in new
                )
        finally:
            self.tokenizer.padding_side = prev_side
            if handle is not None:
                handle.remove()
        return texts

    # -------------------------------------------------------------- logit lens
    def _get_lm_head_weight(self) -> torch.Tensor:
        if self._lm_head_weight is not None:
            return self._lm_head_weight
        w = None
        if getattr(self.model, "get_output_embeddings", None) is not None:
            oe = self.model.get_output_embeddings()
            if oe is not None and hasattr(oe, "weight"):
                w = oe.weight
        if w is None:  # tied embeddings fallback
            ie = self.model.get_input_embeddings()
            w = ie.weight
        self._lm_head_weight = w.detach()
        return self._lm_head_weight

    @torch.no_grad()
    def logit_lens(self, direction: np.ndarray, k: int = 12) -> list[str]:
        """Top-k vocabulary tokens for a residual-stream direction.

        Projects the direction through the model's final RMSNorm (if found) and
        the unembedding — a proper logit lens. Falls back to a bare projection
        if the final norm can't be located.
        """
        w = self._get_lm_head_weight()
        v = torch.tensor(direction, dtype=w.dtype, device=w.device)
        if self._final_norm is not None:
            v = self._final_norm(v)
        logits = v @ w.T  # (vocab,)
        top = logits.topk(k).indices.cpu().tolist()
        return [self.tokenizer.decode([t]).strip() for t in top]


def make_final_extract_fn(qwen: QwenModel, layer: int, batch_size: int = 16):
    """Return an ``extract_fn`` that harvests only the LAST-position activation
    of each prompt — one activation per prompt. For behavioural dictionaries
    where the operative representation (e.g. the refusal decision) is only
    final-token-visible, so every activation is on the decision axis and a
    region's first-arrival exemplar is itself a genuine decision-axis point.
    """
    def extract_fn(model, prompts, hook_name, **kwargs):  # noqa: ARG001
        return qwen.extract_final_position(prompts, layer=layer,
                                           batch_size=batch_size)
    return extract_fn


def make_extract_fn(qwen: QwenModel, layer: int, batch_size: int = 16,
                    max_positions_per_prompt: int | None = None, skip_first: bool = True):
    """Return an ``extract_fn(model, prompts, hook_name, **kwargs)`` bound to a
    layer, matching the signature ``ep.calibrate_pipeline`` / ``ep.discover``
    expect. ``model`` and ``hook_name`` args are ignored (the closure already
    holds the QwenModel and layer); this keeps the EP pipeline untouched.
    """
    def extract_fn(model, prompts, hook_name, **kwargs):  # noqa: ARG001
        return qwen.extract_per_position(
            prompts, layer=layer, batch_size=batch_size,
            max_positions_per_prompt=max_positions_per_prompt, skip_first=skip_first,
        )
    return extract_fn
