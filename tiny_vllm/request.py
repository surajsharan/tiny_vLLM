from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"        # not yet started prefill
    PREFILLING = "prefilling"  # chunked prefill in progress
    RUNNING = "running"        # in decode loop
    FINISHED = "finished"
    PREEMPTED = "preempted"    # evicted; will restart prefill when capacity returns


_seq_counter = itertools.count()


def _next_seq_id() -> int:
    return next(_seq_counter)


@dataclass
class Sequence:
    """One in-flight request.

    The token sequence is `prompt_token_ids + output_token_ids`.
    `num_computed_tokens` tracks how many tokens already have their KV
    materialized in the paged cache.  Anything past that boundary is either
    waiting prefill (during PREFILLING) or the next token to sample (RUNNING).
    """

    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    request_id: str
    arrival_time: float = field(default_factory=time.monotonic)
    seq_id: int = field(default_factory=_next_seq_id)

    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # Paged KV bookkeeping (filled in by the BlockManager).
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0          # tokens with KV in the cache
    num_cached_prefix_tokens: int = 0     # tokens served from prefix cache hits

    # Outputs / streaming
    finish_reason: Optional[str] = None

    # ---- helpers --------------------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_token(self, position: int) -> int:
        if position < len(self.prompt_token_ids):
            return self.prompt_token_ids[position]
        return self.output_token_ids[position - len(self.prompt_token_ids)]

    @property
    def num_uncomputed_prompt_tokens(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)


@dataclass
class Request:
    """A user-submitted request before it becomes a Sequence."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
