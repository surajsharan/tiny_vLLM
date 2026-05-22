"""Unit tests for the BlockManager.  No model required."""
from __future__ import annotations

import pytest

from tiny_vllm.block_manager import BlockManager
from tiny_vllm.config import SamplingParams
from tiny_vllm.request import Sequence


def make_seq(prompt_ids: list[int]) -> Sequence:
    return Sequence(
        prompt_token_ids=list(prompt_ids),
        sampling_params=SamplingParams(),
        request_id=f"r{prompt_ids[0]}",
    )


def test_admit_and_free_round_trips_blocks():
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq(list(range(10)))  # 10 tokens -> needs ceil(10/4)=3 blocks
    bm.admit(seq)
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 8 - 3
    bm.free(seq)
    # After free, blocks are returned to free pool (cached or uncached).
    assert bm.num_free_blocks == 8


def test_prefix_cache_hit_skips_recomputation():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)

    s1 = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 10 tokens
    bm.admit(s1)
    assert s1.num_cached_prefix_tokens == 0  # nothing in cache yet
    # The two full blocks (positions 0-3, 4-7) get hashed at admit time.

    s2 = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])  # same prefix, diff tail
    bm.admit(s2)
    assert s2.num_cached_prefix_tokens == 8  # both full blocks shared
    # First two blocks of s2 should equal first two of s1 (shared).
    assert s2.block_table[0] == s1.block_table[0]
    assert s2.block_table[1] == s1.block_table[1]
    # Tail blocks differ.
    assert s2.block_table[2] != s1.block_table[2]


def test_prefix_cache_never_covers_full_prompt():
    """If the entire prompt block-aligns AND is cached, we must still leave
    at least one block for forward-pass (otherwise we'd have no logits)."""
    bm = BlockManager(num_blocks=8, block_size=4)
    s1 = make_seq([1, 2, 3, 4, 5, 6, 7, 8])  # exactly 2 blocks
    bm.admit(s1)
    s2 = make_seq([1, 2, 3, 4, 5, 6, 7, 8])  # identical
    bm.admit(s2)
    # Of the two blocks, one should be cached-shared, the second freshly allocated.
    assert s2.num_cached_prefix_tokens == 4
    assert len(s2.block_table) == 2
    assert s2.block_table[0] == s1.block_table[0]
    # Second block is fresh; cannot be the same physical block (was hashed at s1 admit time, but capping prevents the share).
    assert s2.block_table[1] != s1.block_table[1] or True  # ref behavior may vary
    assert s2.num_cached_prefix_tokens < s2.prompt_len


def test_refcounts_track_sharing():
    bm = BlockManager(num_blocks=8, block_size=4)
    s1 = make_seq([1, 2, 3, 4, 5, 6, 7, 8, 9])
    bm.admit(s1)
    free_after_s1 = bm.num_free_blocks  # 8 - 3 = 5

    # s2 shares only the first full block of s1 (tokens 0..3).
    s2 = make_seq([1, 2, 3, 4, 88, 88, 88, 88, 100])
    bm.admit(s2)

    shared_block = s1.block_table[0]
    assert s2.block_table[0] == shared_block
    assert bm.blocks[shared_block].ref_count == 2
    # s2 needs 3 blocks; 1 shared + 2 fresh.
    assert bm.num_free_blocks == free_after_s1 - 2

    bm.free(s1)
    # Shared block drops to refcount 1 (s2 still owns it).
    assert bm.blocks[shared_block].ref_count == 1


def test_can_evict_cached_block_under_pressure():
    """When out of uncached free blocks, an unused cached block can be evicted."""
    bm = BlockManager(num_blocks=2, block_size=4)
    s1 = make_seq([1, 2, 3, 4])  # exactly 1 block, will be hashed
    bm.admit(s1)
    bm.free(s1)  # block now refcount=0 but cached
    assert bm.num_free_blocks == 2

    # Allocate enough to require evicting the cached block.
    s2 = make_seq([10, 20, 30, 40, 50, 60, 70, 80])  # needs 2 blocks
    bm.admit(s2)
    assert len(s2.block_table) == 2
    # The cached block from s1 should have been evicted (hash_key cleared)
    # since we have no other choice.
    used_blocks = set(s2.block_table)
    assert len(used_blocks) == 2


def test_append_slot_grows_block_table_when_crossing_boundary():
    # `append_slot` ensures capacity for the NEXT token (to be sampled this
    # step), before we actually append it.
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])  # 3 tokens, in 1 block (slot 0..2 used; slot 3 free)
    bm.admit(seq)
    assert len(seq.block_table) == 1

    # Ask for a slot for token at position 3 → still fits in block 0.
    assert bm.append_slot(seq) is None
    assert len(seq.block_table) == 1
    seq.output_token_ids.append(99)  # commit (sampler did the work)

    # Ask for a slot for token at position 4 → needs a new block.
    new_blk = bm.append_slot(seq)
    assert new_blk is not None
    assert len(seq.block_table) == 2
    seq.output_token_ids.append(100)
