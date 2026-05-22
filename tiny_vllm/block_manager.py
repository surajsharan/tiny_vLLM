"""Paged KV-cache block manager with hash-based automatic prefix caching.

Concepts (matching vLLM / SGLang terminology):

  Physical block:  a fixed-size slot in the KV-cache pool that holds the K and V
                   tensors for ``block_size`` consecutive tokens of one sequence.

  Block table:     per-sequence list of physical block ids that holds the
                   sequence's KV in logical order.  Position ``p`` of the
                   sequence lives in physical block ``block_table[p // B]`` at
                   slot ``p % B``.

  Prefix cache:    a content-addressed lookup from
                       hash(prev_block_hash, tuple_of_token_ids_in_block)
                   to a physical block id.  When two sequences share a prefix
                   that aligns to a block boundary, the second sequence can
                   point its block_table at the cached blocks instead of
                   recomputing KV, and the scheduler can skip those tokens.

The "chained" hash means two prefixes match iff they are identical from
position 0 — exactly the property we need for prefix sharing.

This manager is allocation-only: it does NOT store the KV tensors.  The
ModelRunner owns the actual ``[num_blocks, ...]`` tensors and consults the
block tables here to know where to write/read KV.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .request import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash_key: Optional[int] = None  # set when the block is full and registered


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # Two-tier free list: ephemeral (no hash) reused first, then cached
        # (preserved as long as we have ephemeral capacity).
        self._free_uncached: deque[int] = deque(range(num_blocks))
        self._free_cached: deque[int] = deque()
        self._cache: dict[int, int] = {}  # hash → block_id

        # Stats (visible via events).
        self.prefix_cache_hits = 0
        self.prefix_cache_lookups = 0

    # ---- introspection --------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_uncached) + len(self._free_cached)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def snapshot(self) -> dict:
        """Cheap dict for the event stream / UI."""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "num_free_blocks": self.num_free_blocks,
            "num_cached_entries": len(self._cache),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_lookups": self.prefix_cache_lookups,
            "ref_counts": [b.ref_count for b in self.blocks],
            "hashed": [b.hash_key is not None for b in self.blocks],
        }

    # ---- low-level pool ops --------------------------------------------

    def _block_hash(self, prev_hash: Optional[int], token_ids: tuple[int, ...]) -> int:
        # Python's hash() is randomized per process but that's fine: the cache
        # only lives for the engine's lifetime.
        return hash((prev_hash, token_ids))

    def _take_free_block(self) -> int:
        if self._free_uncached:
            bid = self._free_uncached.popleft()
        elif self._free_cached:
            bid = self._free_cached.popleft()
            # Evict its cache entry — we're about to repurpose it.
            blk = self.blocks[bid]
            if blk.hash_key is not None:
                self._cache.pop(blk.hash_key, None)
                blk.hash_key = None
        else:
            raise RuntimeError("BlockManager out of free blocks")
        blk = self.blocks[bid]
        blk.ref_count = 1
        return bid

    def _share(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        if blk.ref_count == 0:
            # Was sitting in the cached free list; pull it out.
            try:
                self._free_cached.remove(block_id)
            except ValueError:
                pass
        blk.ref_count += 1

    def _release(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        blk.ref_count -= 1
        assert blk.ref_count >= 0, f"block {block_id} refcount went negative"
        if blk.ref_count == 0:
            if blk.hash_key is not None and self.enable_prefix_caching:
                self._free_cached.append(block_id)
            else:
                self._free_uncached.append(block_id)

    def _register(self, block_id: int, hash_key: int) -> None:
        if not self.enable_prefix_caching:
            return
        if hash_key in self._cache:
            # Two sequences independently produced the same content for
            # different physical blocks. Keep the older one; this one becomes
            # ephemeral so it gets reclaimed first.
            return
        self.blocks[block_id].hash_key = hash_key
        self._cache[hash_key] = block_id

    # ---- per-sequence allocation ---------------------------------------

    def num_blocks_needed_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate_initial(self, seq: Sequence) -> tuple[bool, int]:
        """Worst-case allocation check for the prompt of `seq`, ignoring prefix
        cache hits.  Returns (ok, num_new_blocks_needed)."""
        need = self.num_blocks_needed_for(seq.prompt_len)
        return self.num_free_blocks >= need, need

    def admit(self, seq: Sequence) -> None:
        """Set up `seq` in the cache.

        Walks the prompt block-by-block.  For each full block of *prompt* tokens
        we already know, check the prefix cache: hit → share; miss → allocate
        fresh and register the hash now (we know the tokens already).

        The trailing partial block (if any) is allocated fresh and left
        un-hashed; it will be hashed by `finalize_step` once it fills up.
        """
        assert not seq.block_table, "admit called on an already-admitted sequence"
        prev_hash: Optional[int] = None
        cached_tokens = 0
        prompt = seq.prompt_token_ids
        B = self.block_size
        num_full = seq.prompt_len // B

        # IMPORTANT: never let prefix cache cover the entire prompt — we need
        # at least one token to forward through the model to get logits for
        # the first sampled token.  If the full prompt block-aligns AND every
        # block is cached, drop the last cached block.
        cap_full = num_full
        if seq.prompt_len % B == 0:
            cap_full = max(0, num_full - 1)

        for i in range(num_full):
            tokens = tuple(prompt[i * B : (i + 1) * B])
            h = self._block_hash(prev_hash, tokens)
            self.prefix_cache_lookups += 1
            if self.enable_prefix_caching and h in self._cache and i < cap_full:
                # Cache hit.
                self.prefix_cache_hits += 1
                bid = self._cache[h]
                self._share(bid)
                seq.block_table.append(bid)
                cached_tokens += B
                prev_hash = h
            else:
                # Miss: allocate, and since the block content is fully known
                # (prompt tokens), register its hash right away so the next
                # request with this prefix can hit.
                bid = self._take_free_block()
                self._register(bid, h)
                seq.block_table.append(bid)
                prev_hash = h

        # Trailing partial block, if any.
        if seq.prompt_len % B != 0:
            bid = self._take_free_block()
            seq.block_table.append(bid)

        seq.num_computed_tokens = cached_tokens
        seq.num_cached_prefix_tokens = cached_tokens

    def append_slot(self, seq: Sequence) -> Optional[int]:
        """Ensure `seq` has a slot for one more token (decode path).

        Returns the block_id that was newly allocated, or None if existing
        capacity already covered the new token.  Raises if no block available.
        """
        new_position = seq.total_len  # 0-indexed slot we are about to write
        needed_blocks = self.num_blocks_needed_for(new_position + 1)
        if needed_blocks <= len(seq.block_table):
            return None
        if self.num_free_blocks == 0:
            raise RuntimeError("out of blocks")
        bid = self._take_free_block()
        seq.block_table.append(bid)
        return bid

    def ensure_blocks_for_chunk(self, seq: Sequence, chunk_tokens: int) -> int:
        """Prefill path: make sure `seq.block_table` covers
        `seq.num_computed_tokens + chunk_tokens` tokens.

        Returns number of newly-allocated blocks.
        """
        target = seq.num_computed_tokens + chunk_tokens
        needed = self.num_blocks_needed_for(target)
        new_alloc = 0
        while len(seq.block_table) < needed:
            bid = self._take_free_block()
            seq.block_table.append(bid)
            new_alloc += 1
        return new_alloc

    def free(self, seq: Sequence) -> None:
        for bid in seq.block_table:
            self._release(bid)
        seq.block_table.clear()

    # ---- post-step bookkeeping -----------------------------------------

    def register_filled_blocks(self, seq: Sequence, prev_computed: int) -> None:
        """After a forward pass, hash & register any blocks that just became
        full so future requests can prefix-cache them."""
        if not self.enable_prefix_caching:
            return
        B = self.block_size
        # Re-chain hashes from the start so we always have prev_hash correct.
        prev_hash: Optional[int] = None
        for i in range(seq.num_computed_tokens // B):
            bid = seq.block_table[i]
            blk = self.blocks[bid]
            if blk.hash_key is not None:
                prev_hash = blk.hash_key
                continue
            # This block became full in this step (or earlier but unhashed).
            if (i + 1) * B > seq.num_computed_tokens:
                break  # not actually full yet — defensive
            tokens = tuple(seq.get_token(i * B + j) for j in range(B))
            h = self._block_hash(prev_hash, tokens)
            self._register(bid, h)
            prev_hash = h
