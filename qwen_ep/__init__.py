"""Exemplar Partitioning on Qwen3.5-2B-Base.

A thin adaptation layer that lets the upstream `ep` library build EP
dictionaries on Qwen's HuggingFace model, without needing TransformerLens
support for the (brand-new, hybrid linear-attention) Qwen3.5 architecture.
"""

from .adapter import QwenModel, make_extract_fn

__all__ = ["QwenModel", "make_extract_fn"]
