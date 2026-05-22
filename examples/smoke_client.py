"""Fire concurrent prompts at a running tiny_vllm server.

Run the server first:
    python -m tiny_vllm.server --model Qwen/Qwen2.5-0.5B-Instruct

Then in another shell:
    python examples/smoke_client.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


PROMPTS = [
    "Write a haiku about paged attention.",
    "Explain GQA in one paragraph.",
    "What is continuous batching, briefly?",
    "List three uses of prefix caching.",
]


async def one(client: httpx.AsyncClient, prompt: str, idx: int) -> tuple[str, float]:
    t0 = time.monotonic()
    print(f"[{idx}] >> {prompt!r}")
    text_parts: list[str] = []
    async with client.stream(
        "POST", "/generate",
        json={"prompt": prompt, "max_tokens": 48, "temperature": 0.7, "top_p": 0.9, "stream": True},
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: "):
                continue
            data = raw[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("text"):
                text_parts.append(chunk["text"])
            if chunk.get("finished"):
                break
    dt = time.monotonic() - t0
    text = "".join(text_parts)
    print(f"[{idx}] << ({dt:.2f}s) {text}")
    return text, dt


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--prefix-demo", action="store_true",
                   help="send same prompt 3x to show prefix cache speedup")
    args = p.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        if args.prefix_demo:
            prompt = PROMPTS[0]
            for i in range(3):
                await one(client, prompt, i)
            return
        for r in range(args.rounds):
            tasks = [one(client, p, i + r * len(PROMPTS)) for i, p in enumerate(PROMPTS)]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
