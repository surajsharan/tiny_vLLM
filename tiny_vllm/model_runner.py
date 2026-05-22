"""Minimal Qwen2 forward pass that consumes a paged KV cache.

We deliberately re-implement Qwen2 from scratch (rather than using the HF
forward) so the path of K/V tensors through the cache is fully visible.
Weights are loaded from a HuggingFace checkpoint by matching parameter names.

Layout of inputs per step ("varlen" packing):

  input_ids       [T_total]               concatenated tokens for all seqs
  positions       [T_total]               position-in-sequence of each token
  slot_mapping    [T_total]               where to write new K/V in the cache
  segments        list of (q_start, q_end, block_table, k_len, seq_id)

For attention, we loop over `segments`: gather each sequence's full K/V from
its block table, run SDPA, scatter the result back into a flat buffer.  All
other ops (norms, MLP, projections) run on the full packed tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EngineConfig
from .paged_kv import PagedKVCache
from .request import Sequence


# ---------------------------------------------------------------------------
# Qwen2 building blocks
# ---------------------------------------------------------------------------


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden]
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D], cos/sin: [T, D] → returns [T, H, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class AttnSegment:
    """One sequence's slice of the packed batch."""
    q_start: int           # start index in the packed tensor
    q_end: int             # exclusive
    block_table: list[int] # KV blocks for this sequence
    k_len: int             # total K length (= num_computed_tokens + q_len)


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,        # [T, hidden]
        positions: torch.Tensor,            # [T] long
        slot_mapping: torch.Tensor,         # [T] long
        cos_table: torch.Tensor,            # [max_pos, head_dim]
        sin_table: torch.Tensor,            # [max_pos, head_dim]
        segments: list[AttnSegment],
        kv_cache: PagedKVCache,
    ) -> torch.Tensor:
        T = hidden_states.size(0)
        q = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)

        cos = cos_table.index_select(0, positions)  # [T, head_dim]
        sin = sin_table.index_select(0, positions)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Write the NEW K/V into the paged cache before reading it back.
        kv_cache.write(self.layer_idx, k, v, slot_mapping)

        out = torch.empty_like(q)  # [T, num_heads, head_dim]
        rep = self.num_heads // self.num_kv_heads  # GQA fan-out

        for seg in segments:
            q_slice = q[seg.q_start:seg.q_end]                  # [q_len, H_q, D]
            k_full, v_full = kv_cache.gather(self.layer_idx, seg.block_table, seg.k_len)
            # GQA: expand K/V heads to match Q heads.
            if rep > 1:
                k_full = k_full.repeat_interleave(rep, dim=1)
                v_full = v_full.repeat_interleave(rep, dim=1)

            q_len = q_slice.size(0)
            k_len = seg.k_len
            num_past = k_len - q_len

            # Causal mask: Q at logical position (num_past + i) attends to K at
            # positions [0, num_past + i].  True = participate (SDPA convention).
            idx_q = torch.arange(q_len, device=q.device).unsqueeze(1) + num_past
            idx_k = torch.arange(k_len, device=q.device).unsqueeze(0)
            attn_mask = idx_k <= idx_q  # [q_len, k_len]

            # SDPA wants [..., heads, q_len, head_dim].  Reshape and run.
            q_h = q_slice.transpose(0, 1).unsqueeze(0)   # [1, H, q_len, D]
            k_h = k_full.transpose(0, 1).unsqueeze(0)    # [1, H, k_len, D]
            v_h = v_full.transpose(0, 1).unsqueeze(0)
            attn = F.scaled_dot_product_attention(
                q_h, k_h, v_h,
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),  # [1,1,q_len,k_len]
                scale=self.scale,
            )                                            # [1, H, q_len, D]
            out[seg.q_start:seg.q_end] = attn.squeeze(0).transpose(0, 1)

        return self.o_proj(out.reshape(T, self.num_heads * self.head_dim))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.self_attn = Qwen2Attention(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.mlp = Qwen2MLP(cfg["hidden_size"], cfg["intermediate_size"])

    def forward(self, hidden_states, positions, slot_mapping, cos_table, sin_table, segments, kv_cache):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, positions, slot_mapping, cos_table, sin_table, segments, kv_cache)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        return residual + h


class Qwen2Model(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        )
        self.norm = Qwen2RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, input_ids, positions, slot_mapping, cos_table, sin_table, segments, kv_cache):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, positions, slot_mapping, cos_table, sin_table, segments, kv_cache)
        return self.norm(h)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.cfg = cfg

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# ModelRunner: prepares inputs, runs forward, extracts last-token logits.
# ---------------------------------------------------------------------------


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    segments: list[AttnSegment]
    # Index in the packed batch of the LAST token of each scheduled seq —
    # that's where we'll read logits from for sampling.
    last_token_indices: torch.Tensor


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.dtype]

        hf_cfg = AutoConfig.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )

        model_type = getattr(hf_cfg, "model_type", "?")
        if model_type not in ("qwen2", "qwen2_moe", "llama"):
            # Llama-style works too because the math is identical; we issue a
            # warning rather than a hard fail.
            print(f"[tiny_vllm] WARNING: model_type={model_type!r}; expected qwen2-like. "
                  "Continuing — assuming Llama-compatible config.")

        head_dim = getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // hf_cfg.num_attention_heads)
        cfg = {
            "vocab_size": hf_cfg.vocab_size,
            "hidden_size": hf_cfg.hidden_size,
            "intermediate_size": hf_cfg.intermediate_size,
            "num_hidden_layers": hf_cfg.num_hidden_layers,
            "num_attention_heads": hf_cfg.num_attention_heads,
            "num_key_value_heads": getattr(hf_cfg, "num_key_value_heads",
                                            hf_cfg.num_attention_heads),
            "head_dim": head_dim,
            "rms_norm_eps": getattr(hf_cfg, "rms_norm_eps", 1e-6),
            "rope_theta": getattr(hf_cfg, "rope_theta", 10000.0),
            "max_position_embeddings": getattr(hf_cfg, "max_position_embeddings", 4096),
            "tie_word_embeddings": getattr(hf_cfg, "tie_word_embeddings", False),
        }
        self.model_cfg = cfg

        # Build our own model, then copy HF weights into it.
        model = Qwen2ForCausalLM(cfg).to(self.device, self.dtype)
        hf_model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype=self.dtype,
            trust_remote_code=config.trust_remote_code,
        )
        missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=False)
        if cfg["tie_word_embeddings"] and "lm_head.weight" in (missing or []):
            model.tie_weights()
        del hf_model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        # Precompute RoPE tables.
        max_pos = min(cfg["max_position_embeddings"], config.max_model_len)
        inv_freq = 1.0 / (
            cfg["rope_theta"]
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)        # [max_pos, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, head_dim]
        self.cos_table = emb.cos().to(self.device, self.dtype)
        self.sin_table = emb.sin().to(self.device, self.dtype)

        # Paged KV cache pool.
        self.kv_cache = PagedKVCache(
            num_layers=cfg["num_hidden_layers"],
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    # ---- input building ------------------------------------------------

    def prepare_input(self, scheduled) -> ModelInput:
        """`scheduled` is a list of (Sequence, num_tokens, is_prefill) triples
        from the scheduler."""
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        segments: list[AttnSegment] = []
        last_indices: list[int] = []

        cursor = 0
        B = self.config.block_size
        for item in scheduled:
            seq = item.seq
            n = item.num_tokens
            # Logical token positions this step processes.
            start_pos = seq.num_computed_tokens
            for off in range(n):
                pos = start_pos + off
                input_ids.append(seq.get_token(pos))
                positions.append(pos)
                block_id = seq.block_table[pos // B]
                slot_mapping.append(block_id * B + (pos % B))

            q_end = cursor + n
            segments.append(AttnSegment(
                q_start=cursor,
                q_end=q_end,
                block_table=list(seq.block_table),
                k_len=start_pos + n,
            ))
            last_indices.append(q_end - 1)
            cursor = q_end

        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=self.device),
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
            segments=segments,
            last_token_indices=torch.tensor(last_indices, dtype=torch.long, device=self.device),
        )

    # ---- forward -------------------------------------------------------

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> torch.Tensor:
        """Run one forward pass.  Returns logits for the LAST token of each
        scheduled sequence: shape [num_seqs, vocab_size]."""
        hidden = self.model.model(
            input_ids=model_input.input_ids,
            positions=model_input.positions,
            slot_mapping=model_input.slot_mapping,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
            segments=model_input.segments,
            kv_cache=self.kv_cache,
        )                                                # [T, hidden]
        last_hidden = hidden.index_select(0, model_input.last_token_indices)
        logits = self.model.lm_head(last_hidden)         # [num_seqs, vocab]
        return logits

    # ---- helpers -------------------------------------------------------

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def detokenize_incremental(self, full_ids: list[int], prev_text_len: int) -> tuple[str, int]:
        """Detokenize the full list, return the new text added since last call
        and the new total length."""
        text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        return text[prev_text_len:], len(text)
