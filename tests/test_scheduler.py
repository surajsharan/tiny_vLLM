"""Scheduler logic tests, model-free."""
from __future__ import annotations

from tiny_vllm.block_manager import BlockManager
from tiny_vllm.config import EngineConfig, SamplingParams
from tiny_vllm.request import Sequence, SequenceStatus
from tiny_vllm.scheduler import Scheduler


def _engine_cfg(**kw) -> EngineConfig:
    cfg = EngineConfig(
        model="ignored", block_size=4, num_blocks=8,
        max_num_seqs=4, max_num_batched_tokens=8, max_model_len=128,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _seq(ids: list[int]) -> Sequence:
    return Sequence(prompt_token_ids=list(ids),
                    sampling_params=SamplingParams(max_tokens=4),
                    request_id=f"r{ids[0]}")


def test_short_prompt_fully_prefilled_in_one_step():
    cfg = _engine_cfg()
    bm = BlockManager(cfg.num_blocks, cfg.block_size)
    sch = Scheduler(cfg, bm)
    s = _seq([1, 2, 3, 4, 5])  # 5 tokens, fits in budget=8
    sch.add(s)
    out = sch.schedule()
    assert len(out.scheduled) == 1
    assert out.scheduled[0].num_tokens == 5
    assert out.scheduled[0].is_prefill
    assert s in sch.running


def test_chunked_prefill_splits_long_prompt_across_steps():
    cfg = _engine_cfg(max_num_batched_tokens=4)
    bm = BlockManager(cfg.num_blocks, cfg.block_size)
    sch = Scheduler(cfg, bm)
    s = _seq([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 9 tokens vs budget=4
    sch.add(s)
    out1 = sch.schedule()
    assert out1.scheduled[0].num_tokens == 4
    assert s.status == SequenceStatus.PREFILLING
    # Engine would update num_computed_tokens after model fwd; simulate:
    s.num_computed_tokens += 4
    out2 = sch.schedule()
    assert out2.scheduled[0].num_tokens == 4
    s.num_computed_tokens += 4
    out3 = sch.schedule()
    # Last chunk: 1 token left → fills, transitions to RUNNING.
    assert out3.scheduled[0].num_tokens == 1
    s.num_computed_tokens += 1
    assert s in sch.running


def test_decodes_interleave_with_prefills():
    cfg = _engine_cfg(max_num_batched_tokens=6)
    bm = BlockManager(cfg.num_blocks, cfg.block_size)
    sch = Scheduler(cfg, bm)

    # Get a sequence fully into RUNNING state.
    runner = _seq([1, 2, 3, 4, 5])
    sch.add(runner)
    out0 = sch.schedule()
    assert out0.scheduled and out0.scheduled[0].num_tokens == 5
    # Simulate model forward.
    runner.num_computed_tokens = runner.prompt_len
    assert runner.status == SequenceStatus.RUNNING

    # New waiting seq.
    waiter = _seq([100, 101, 102])
    sch.add(waiter)

    out = sch.schedule()
    kinds = [(it.is_prefill, it.num_tokens, it.seq.seq_id) for it in out.scheduled]
    # runner decodes 1 token, waiter prefills 3 — all fit in budget=6.
    assert any(not it.is_prefill and it.num_tokens == 1 and it.seq is runner for it in out.scheduled)
    assert any(it.is_prefill and it.num_tokens == 3 and it.seq is waiter for it in out.scheduled)


def test_preemption_triggers_when_blocks_exhaust():
    """When a decoding sequence needs a new block but the pool is dry, the
    scheduler preempts the youngest running seq (here, only itself) and
    re-enqueues it.  schedule() must not crash."""
    cfg = _engine_cfg(num_blocks=2, block_size=4, max_num_batched_tokens=16)
    bm = BlockManager(cfg.num_blocks, cfg.block_size)
    sch = Scheduler(cfg, bm)
    s1 = _seq([1, 2, 3, 4, 5, 6, 7])  # 2 blocks consumed exactly on prompt
    sch.add(s1)
    sch.schedule()
    s1.num_computed_tokens = s1.prompt_len

    # Push s1 to the brink: pretend it has decoded enough to fill block 2.
    s1.output_token_ids.extend([99] * (8 - 7))  # total_len = 8, fits in 2 blocks
    # Next decode (position 8) would require a 3rd block; only 2 exist.
    out = sch.schedule()
    # s1 should have been preempted (and may then be re-admitted in the same
    # step via prefix cache — what matters is the preempt event fired).
    assert s1.seq_id in out.preempted
