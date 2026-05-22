"""The actual KV tensor pool that the BlockManager indexes into.

We store one ``[num_blocks, block_size, num_kv_heads, head_dim]`` tensor per
layer for K and V.  The block_manager owns the *allocation* of block ids; this
class owns the *bytes*.  Reads and writes happen by (block_id, offset).
"""
from __future__ import annotations

import torch


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]

    def write(
        self,
        layer_id: int,
        k: torch.Tensor,           # [T, num_kv_heads, head_dim]
        v: torch.Tensor,           # [T, num_kv_heads, head_dim]
        slot_mapping: torch.Tensor # [T] int64, slot_id = block_id*block_size + offset
    ) -> None:
        block_ids = (slot_mapping // self.block_size).long()
        offsets = (slot_mapping % self.block_size).long()
        self.k_cache[layer_id][block_ids, offsets] = k.to(self.dtype)
        self.v_cache[layer_id][block_ids, offsets] = v.to(self.dtype)

    def gather(
        self,
        layer_id: int,
        block_table: list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous [num_tokens, num_kv_heads, head_dim] K and V
        for one sequence, by walking its block table."""
        if num_tokens == 0:
            empty = torch.zeros(
                0, self.num_kv_heads, self.head_dim,
                dtype=self.dtype, device=self.device,
            )
            return empty, empty.clone()
        num_full = num_tokens // self.block_size
        tail = num_tokens % self.block_size
        idxs = block_table[:num_full + (1 if tail else 0)]
        idx_tensor = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
        # [P, block_size, H, D]
        k_blocks = self.k_cache[layer_id].index_select(0, idx_tensor)
        v_blocks = self.v_cache[layer_id].index_select(0, idx_tensor)
        # Flatten the first two dims then trim.
        k_flat = k_blocks.reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_blocks.reshape(-1, self.num_kv_heads, self.head_dim)
        return k_flat[:num_tokens], v_flat[:num_tokens]
