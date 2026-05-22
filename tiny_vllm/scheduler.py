"""Continuous-batching scheduler with chunked prefill.

A scheduling step produces a SchedulerOutput listing which sequences run and
how many tokens each one advances.  Two phases each step:

  1. Decodes.  Every RUNNING sequence wants exactly one new token; we must
     ensure each has space for it.  If a sequence needs a new block and the
     pool is dry, we *preempt* the most recently admitted running sequence —
     free its KV blocks and push it back to the front of the waiting queue
     so it restarts prefill later (recompute-style preemption, as in vLLM).

  2. Prefill chunks.  With remaining token budget, pull from `waiting`.  A
     newly waiting sequence is admitted (prompt blocks allocated via the
     block manager, with prefix-cache hits taken).  Then we plan a chunk of
     up to `min(remaining_prefill, budget)` tokens.  Chunked prefill lets a
     long prompt share the budget with concurrent decodes instead of
     stalling them.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .config import EngineConfig
from .request import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int     # how many tokens to forward this step for this seq
    is_prefill: bool


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)  # seq_ids preempted
    newly_admitted: list[int] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        # Tracks order of admission so preemption picks the youngest first.
        self._admission_order: list[int] = []

    # ---- queue ops ------------------------------------------------------

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        for q in (self.waiting,):
            for s in list(q):
                if s.seq_id == seq_id:
                    q.remove(s)
                    s.status = SequenceStatus.FINISHED
                    s.finish_reason = "abort"
                    return True
        for s in list(self.running):
            if s.seq_id == seq_id:
                self.running.remove(s)
                self.block_manager.free(s)
                s.status = SequenceStatus.FINISHED
                s.finish_reason = "abort"
                return True
        return False

    @property
    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    # ---- scheduling -----------------------------------------------------

    def _preempt_one(self) -> Sequence | None:
        """Free the youngest running sequence and re-enqueue it for restart."""
        if not self.running:
            return None
        victim = self.running.pop()  # youngest by insertion order
        self.block_manager.free(victim)
        # Restart: forget computed-token progress; keep generated outputs so
        # the user-visible sequence is preserved.  (vLLM full-recompute: we'd
        # discard outputs too; we keep them so streaming makes sense.)
        victim.num_computed_tokens = 0
        victim.num_cached_prefix_tokens = 0
        victim.status = SequenceStatus.PREEMPTED
        self.waiting.appendleft(victim)
        return victim

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens

        # --- Phase 1: decodes for already-running sequences ---
        for seq in list(self.running):
            if seq.status != SequenceStatus.RUNNING:
                continue
            if budget <= 0:
                break
            # Ensure space for one more token.
            try:
                self.block_manager.append_slot(seq)
            except RuntimeError:
                # Out of blocks: try to free space by preempting the youngest
                # running sequence — which may be `seq` itself.
                victim = self._preempt_one()
                if victim is seq:
                    # We preempted ourselves; it's already off `running`.
                    out.preempted.append(seq.seq_id)
                    continue
                if victim is None:
                    # Nothing to preempt; preempt this seq manually.
                    self.running.remove(seq)
                    self.block_manager.free(seq)
                    seq.num_computed_tokens = 0
                    seq.num_cached_prefix_tokens = 0
                    seq.status = SequenceStatus.PREEMPTED
                    self.waiting.appendleft(seq)
                    out.preempted.append(seq.seq_id)
                    continue
                out.preempted.append(victim.seq_id)
                try:
                    self.block_manager.append_slot(seq)
                except RuntimeError:
                    # Still no room — give up on this seq this step.
                    continue
            out.scheduled.append(ScheduledSeq(seq=seq, num_tokens=1, is_prefill=False))
            budget -= 1
            out.total_tokens += 1

        # --- Phase 2: prefill chunks (admitting new sequences as needed) ---
        max_concurrent = self.config.max_num_seqs
        active_count = sum(1 for s in self.running if s.status != SequenceStatus.FINISHED)

        while self.waiting and budget > 0 and active_count < max_concurrent:
            seq = self.waiting[0]

            # Admit if needed.
            if not seq.block_table:
                ok, _ = self.block_manager.can_allocate_initial(seq)
                if not ok:
                    # Try to free up space by preempting the youngest running
                    # seq.  If nothing to preempt, we're stuck for this step.
                    if not self.running:
                        break
                    victim = self._preempt_one()
                    if victim is None:
                        break
                    out.preempted.append(victim.seq_id)
                    continue
                self.block_manager.admit(seq)
                out.newly_admitted.append(seq.seq_id)
                seq.status = SequenceStatus.PREFILLING

            # Plan a chunk.
            remaining = seq.num_uncomputed_prompt_tokens
            chunk = min(remaining, budget)
            if chunk <= 0:
                # Prompt already fully cached (shouldn't happen due to admit
                # capping, but defensive): move straight to RUNNING.
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                active_count += 1
                continue

            # Make sure block_table covers num_computed + chunk.
            try:
                self.block_manager.ensure_blocks_for_chunk(seq, chunk)
            except RuntimeError:
                # Couldn't expand.  Try preemption; otherwise give up.
                if self.running:
                    victim = self._preempt_one()
                    if victim is not None:
                        out.preempted.append(victim.seq_id)
                        continue
                break

            out.scheduled.append(ScheduledSeq(seq=seq, num_tokens=chunk, is_prefill=True))
            budget -= chunk
            out.total_tokens += chunk

            if chunk == remaining:
                # This step finishes prompt ingestion → seq becomes RUNNING.
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                active_count += 1
            else:
                # Still has more prompt to chew through; leave at head of
                # waiting queue with a partial block_table.
                break  # one prefill per step keeps things tidy

        return out

    # ---- post-step ------------------------------------------------------

    def finalize_step(self, scheduled: list[ScheduledSeq]) -> list[Sequence]:
        """Called after the model has produced new tokens.

        Returns the list of sequences that just finished this step (so the
        engine can free them and ship the final output to the caller).
        """
        finished: list[Sequence] = []
        for item in scheduled:
            seq = item.seq
            self.block_manager.register_filled_blocks(seq, prev_computed=0)
            if seq.status == SequenceStatus.FINISHED:
                if seq in self.running:
                    self.running.remove(seq)
                self.block_manager.free(seq)
                finished.append(seq)
        return finished
