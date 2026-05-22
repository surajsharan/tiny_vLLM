from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    # Model
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float32"  # "float32" on CPU; "float16"/"bfloat16" on GPU
    device: str = "cpu"     # "cpu" or "cuda"
    trust_remote_code: bool = False

    # Paged KV cache
    block_size: int = 16          # tokens per physical block
    num_blocks: int = 512         # total physical blocks in the pool
    enable_prefix_caching: bool = True

    # Scheduler
    max_num_seqs: int = 16                # max sequences in a batch
    max_num_batched_tokens: int = 512     # total tokens processed per step
    max_model_len: int = 2048             # upper bound on prompt + generated tokens

    # Logging / events
    emit_events: bool = True              # produce engine events for the UI
    event_buffer: int = 256

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError(
                "max_num_batched_tokens must be >= block_size "
                f"({self.max_num_batched_tokens} < {self.block_size})"
            )


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1                       # -1 disables top-k
    stop_token_ids: list[int] = field(default_factory=list)
    seed: Optional[int] = None
    ignore_eos: bool = False

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0
