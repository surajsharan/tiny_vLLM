"""Per-request sampling.  Temperature, top-p, top-k, greedy."""
from __future__ import annotations

from typing import Optional

import torch

from .config import SamplingParams


class Sampler:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def sample(
        self,
        logits: torch.Tensor,                 # [num_seqs, vocab]
        params: list[SamplingParams],
        generators: Optional[list[Optional[torch.Generator]]] = None,
    ) -> list[int]:
        out: list[int] = []
        for i, p in enumerate(params):
            row = logits[i]
            if p.is_greedy:
                out.append(int(row.argmax().item()))
                continue

            # Temperature.
            row = row / max(p.temperature, 1e-5)
            # Top-k.
            if p.top_k > 0 and p.top_k < row.size(-1):
                topk_vals, _ = torch.topk(row, p.top_k)
                row = torch.where(row < topk_vals[-1], torch.full_like(row, float("-inf")), row)
            # Top-p (nucleus).
            if 0.0 < p.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                # Drop tokens whose CUMULATIVE prob (including themselves) exceeds top_p,
                # but always keep the highest-probability one.
                drop = cumprobs > p.top_p
                drop[0] = False
                drop = drop.roll(shifts=1, dims=0)  # so the boundary token stays
                drop[0] = False
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                row = torch.full_like(row, float("-inf"))
                row.scatter_(0, sorted_idx, sorted_logits)

            probs = torch.softmax(row, dim=-1)
            gen = generators[i] if generators else None
            token = torch.multinomial(probs, num_samples=1, generator=gen)
            out.append(int(token.item()))
        return out
