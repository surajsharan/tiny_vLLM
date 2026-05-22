"""Tiny vLLM — a minimal continuous-batching engine.

Educational reimplementation of the core vLLM/SGLang ideas:
paged KV cache, prefix caching, continuous batching with chunked prefill,
and SSE streaming over a thin HTTP layer.
"""

# Lazy re-exports: importing this package should not pull in torch, so the
# lightweight block_manager/scheduler can be unit-tested without it.

from .config import EngineConfig, SamplingParams
from .request import Request, Sequence, SequenceStatus

__all__ = [
    "EngineConfig",
    "SamplingParams",
    "LLMEngine",
    "Request",
    "Sequence",
    "SequenceStatus",
]


def __getattr__(name: str):
    if name == "LLMEngine":
        from .engine import LLMEngine
        return LLMEngine
    raise AttributeError(name)
