"""LLMEngine: orchestrates scheduler + block manager + model runner + sampler.

Public surface:

  engine = LLMEngine(EngineConfig(...))
  await engine.startup()
  rid = engine.add_request(prompt_text, SamplingParams(...))
  async for delta in engine.stream(rid):
      ...

A single background task (`_run_loop`) drives the model.  Per-request output
goes through asyncio queues so the HTTP layer can stream incrementally.  A
second pub/sub channel emits engine-state snapshots for the visualization UI.
"""
from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from .block_manager import BlockManager
from .config import EngineConfig, SamplingParams
from .model_runner import ModelRunner
from .request import Sequence, SequenceStatus
from .sampler import Sampler
from .scheduler import Scheduler


@dataclass
class StreamItem:
    request_id: str
    new_text: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: Optional[str] = None
    cumulative_text: str = ""


@dataclass
class EngineEvent:
    step: int
    timestamp: float
    type: str
    payload: dict = field(default_factory=dict)


class LLMEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.model_runner: Optional[ModelRunner] = None
        self.block_manager: Optional[BlockManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.sampler: Optional[Sampler] = None

        # request_id → asyncio.Queue[StreamItem]
        self._output_queues: dict[str, asyncio.Queue[StreamItem]] = {}
        # request_id → Sequence (for inspection / abort)
        self._sequences: dict[str, Sequence] = {}
        # tracker for incremental detokenization
        self._prev_text_len: dict[str, int] = {}
        # event subscribers
        self._event_subscribers: list[asyncio.Queue[EngineEvent]] = []
        # control
        self._stop = asyncio.Event()
        self._step_idx = 0
        self._run_task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()

    # ---- lifecycle ------------------------------------------------------

    async def startup(self) -> None:
        # Heavy: model load happens in a worker thread so we don't block the loop.
        loop = asyncio.get_running_loop()

        def _build() -> ModelRunner:
            return ModelRunner(self.config)

        self.model_runner = await loop.run_in_executor(None, _build)
        self.block_manager = BlockManager(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        self.scheduler = Scheduler(self.config, self.block_manager)
        self.sampler = Sampler(self.model_runner.device)
        self._run_task = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._run_task is not None:
            try:
                await asyncio.wait_for(self._run_task, timeout=5)
            except asyncio.TimeoutError:
                self._run_task.cancel()

    # ---- request submission --------------------------------------------

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> str:
        if self.model_runner is None:
            raise RuntimeError("engine not started")
        if isinstance(prompt, str):
            token_ids = self.model_runner.encode(prompt)
        else:
            token_ids = list(prompt)
        if not token_ids:
            raise ValueError("empty prompt")
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(
                f"prompt length {len(token_ids)} >= max_model_len {self.config.max_model_len}"
            )
        rid = request_id or uuid.uuid4().hex
        seq = Sequence(
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            request_id=rid,
        )
        self._sequences[rid] = seq
        self._output_queues[rid] = asyncio.Queue()
        self._prev_text_len[rid] = 0
        assert self.scheduler is not None
        self.scheduler.add(seq)
        self._wake.set()
        return rid

    def abort(self, request_id: str) -> bool:
        seq = self._sequences.get(request_id)
        if seq is None:
            return False
        assert self.scheduler is not None
        ok = self.scheduler.abort(seq.seq_id)
        if ok:
            self._close_request(request_id, finish_reason="abort")
        return ok

    async def stream(self, request_id: str) -> AsyncIterator[StreamItem]:
        q = self._output_queues.get(request_id)
        if q is None:
            raise KeyError(request_id)
        while True:
            item = await q.get()
            yield item
            if item.finished:
                break

    # ---- event subscriptions -------------------------------------------

    def subscribe_events(self) -> asyncio.Queue[EngineEvent]:
        q: asyncio.Queue[EngineEvent] = asyncio.Queue(maxsize=self.config.event_buffer)
        self._event_subscribers.append(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue[EngineEvent]) -> None:
        try:
            self._event_subscribers.remove(q)
        except ValueError:
            pass

    def _emit(self, event_type: str, payload: dict) -> None:
        if not self.config.emit_events or not self._event_subscribers:
            return
        ev = EngineEvent(
            step=self._step_idx,
            timestamp=time.monotonic(),
            type=event_type,
            payload=payload,
        )
        for q in list(self._event_subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop oldest, push new.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass

    # ---- inspection ----------------------------------------------------

    def snapshot(self) -> dict:
        assert self.block_manager is not None and self.scheduler is not None
        def seq_view(s: Sequence) -> dict:
            return {
                "seq_id": s.seq_id,
                "request_id": s.request_id,
                "status": s.status.value,
                "prompt_len": s.prompt_len,
                "num_generated": len(s.output_token_ids),
                "num_computed_tokens": s.num_computed_tokens,
                "num_cached_prefix_tokens": s.num_cached_prefix_tokens,
                "block_table": list(s.block_table),
            }
        return {
            "step": self._step_idx,
            "block_pool": self.block_manager.snapshot(),
            "waiting": [seq_view(s) for s in self.scheduler.waiting],
            "running": [seq_view(s) for s in self.scheduler.running],
            "config": {
                "model": self.config.model,
                "block_size": self.config.block_size,
                "num_blocks": self.config.num_blocks,
                "max_num_seqs": self.config.max_num_seqs,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "prefix_caching": self.config.enable_prefix_caching,
            },
        }

    # ---- main loop -----------------------------------------------------

    async def _run_loop(self) -> None:
        assert self.scheduler is not None and self.model_runner is not None
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            if not self.scheduler.has_work:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            self._step_idx += 1
            t0 = time.monotonic()
            sched = self.scheduler.schedule()
            if sched.is_empty:
                # Nothing got through this step (probably starved on blocks).
                await asyncio.sleep(0.01)
                continue

            model_input = self.model_runner.prepare_input(sched.scheduled)
            # Run blocking model forward off-thread.
            logits = await loop.run_in_executor(None, self.model_runner.execute, model_input)

            # Update num_computed_tokens AFTER forward (the K/V is now stored).
            for item in sched.scheduled:
                item.seq.num_computed_tokens += item.num_tokens

            # Sample only for sequences that have finished prefill (i.e., the
            # last token in their chunk is the *final* prompt token).
            sampling_items = [item for item in sched.scheduled
                              if item.seq.num_computed_tokens >= item.seq.prompt_len]
            sampling_indices = [i for i, item in enumerate(sched.scheduled)
                                if item.seq.num_computed_tokens >= item.seq.prompt_len]

            new_tokens: dict[int, int] = {}
            if sampling_items:
                import torch  # local; cheap
                sampling_logits = logits.index_select(
                    0, torch.tensor(sampling_indices, device=logits.device)
                )
                params = [item.seq.sampling_params for item in sampling_items]
                generators = [
                    (torch.Generator(device=logits.device).manual_seed(item.seq.sampling_params.seed)
                     if item.seq.sampling_params.seed is not None else None)
                    for item in sampling_items
                ]
                token_ids = self.sampler.sample(sampling_logits, params, generators)
                for item, tok in zip(sampling_items, token_ids):
                    new_tokens[item.seq.seq_id] = tok

            # Apply new tokens, check stopping, register filled blocks.
            assert self.block_manager is not None
            finished_now: list[Sequence] = []
            for item in sched.scheduled:
                seq = item.seq
                if seq.seq_id in new_tokens:
                    tok = new_tokens[seq.seq_id]
                    seq.append_output_token(tok)
                    # The just-produced token's KV will be written on the NEXT
                    # step (when this token is the input).  But the new token
                    # may complete a block once its KV lands; we hash blocks
                    # only after their KV exists, so post-forward in the next
                    # step is the right time.  Here we register newly-filled
                    # blocks based on the just-finalized num_computed_tokens.
                    self.block_manager.register_filled_blocks(seq, prev_computed=0)

                    if self._should_stop(seq, tok):
                        seq.status = SequenceStatus.FINISHED
                        seq.finish_reason = self._stop_reason(seq, tok)
                        finished_now.append(seq)
                else:
                    # Still in prefill; just register newly filled prompt blocks.
                    self.block_manager.register_filled_blocks(seq, prev_computed=0)

            # Free finished sequences.
            for seq in finished_now:
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)
                self.block_manager.free(seq)

            # Emit outputs to per-request queues.
            for item in sched.scheduled:
                seq = item.seq
                rid = seq.request_id
                if seq.seq_id in new_tokens or seq in finished_now:
                    new_text, new_text_len = self.model_runner.detokenize_incremental(
                        seq.all_token_ids(), self._prev_text_len.get(rid, 0)
                    )
                    self._prev_text_len[rid] = new_text_len
                    is_done = seq.status == SequenceStatus.FINISHED
                    new_toks = [new_tokens[seq.seq_id]] if seq.seq_id in new_tokens else []
                    si = StreamItem(
                        request_id=rid,
                        new_text=new_text,
                        new_token_ids=new_toks,
                        finished=is_done,
                        finish_reason=seq.finish_reason,
                        cumulative_text=self.model_runner.tokenizer.decode(
                            seq.output_token_ids, skip_special_tokens=True
                        ),
                    )
                    q = self._output_queues.get(rid)
                    if q is not None:
                        await q.put(si)
                    if is_done:
                        # Clean up.
                        self._sequences.pop(rid, None)
                        self._prev_text_len.pop(rid, None)

            # Emit engine events for the UI.
            self._emit("step", {
                "duration_ms": (time.monotonic() - t0) * 1000,
                "num_seqs": len(sched.scheduled),
                "num_tokens": sched.total_tokens,
                "num_prefill_seqs": sum(1 for it in sched.scheduled if it.is_prefill),
                "num_decode_seqs": sum(1 for it in sched.scheduled if not it.is_prefill),
                "preempted": sched.preempted,
                "newly_admitted": sched.newly_admitted,
                "finished": [s.request_id for s in finished_now],
                "snapshot": self.snapshot(),
            })

            # Yield control between steps so the HTTP layer can ship bytes.
            await asyncio.sleep(0)

    # ---- helpers -------------------------------------------------------

    def _should_stop(self, seq: Sequence, last_token: int) -> bool:
        sp = seq.sampling_params
        if len(seq.output_token_ids) >= sp.max_tokens:
            return True
        if not sp.ignore_eos:
            eos = self.model_runner.eos_token_id if self.model_runner else None
            if eos is not None and last_token == eos:
                return True
        if last_token in sp.stop_token_ids:
            return True
        if seq.total_len >= self.config.max_model_len:
            return True
        return False

    def _stop_reason(self, seq: Sequence, last_token: int) -> str:
        sp = seq.sampling_params
        if len(seq.output_token_ids) >= sp.max_tokens:
            return "length"
        if seq.total_len >= self.config.max_model_len:
            return "length"
        return "stop"

    def _close_request(self, request_id: str, finish_reason: str) -> None:
        q = self._output_queues.get(request_id)
        if q is None:
            return
        q.put_nowait(StreamItem(
            request_id=request_id,
            new_text="",
            new_token_ids=[],
            finished=True,
            finish_reason=finish_reason,
        ))
        self._sequences.pop(request_id, None)
        self._prev_text_len.pop(request_id, None)
