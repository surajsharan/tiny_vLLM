# tiny_vllm

A **minimal continuous-batching LLM engine** built to be read end-to-end.  It
re-implements the load-bearing ideas of vLLM / SGLang in ~1.5k lines of
Python:

- **Paged KV cache** with logical block tables — physical blocks are a flat
  pool; per-sequence block tables map logical positions → physical slots.
- **Automatic prefix caching** via content-addressed hashes — two requests
  with the same prompt prefix share KV blocks.
- **Continuous batching with chunked prefill** — each scheduling step packs a
  budget of tokens from any mix of new prefills and ongoing decodes; long
  prompts are sliced so they don't starve the decoders.
- **Recompute-style preemption** — when the pool runs dry, the youngest
  running sequence is evicted and re-enqueued.
- **SSE streaming** over a thin FastAPI layer — both token deltas
  (`/generate`, OpenAI-compatible `/v1/completions`) and a parallel engine
  event stream (`/engine/events`) the demo page subscribes to.
- A **visualization demo page** that renders the block pool, scheduler
  queues, per-sequence block tables, and live tokens as the engine runs.

It is **not** vLLM.  Attention runs in plain PyTorch SDPA (per-sequence loop),
there are no fused or paged-attention kernels, and CPU is the default device.
This is a learning artifact, not a serving stack.

## Quick start

```bash
pip install -r requirements.txt
# or: pip install -e .

python -m tiny_vllm.server --model Qwen/Qwen2.5-0.5B-Instruct --device cpu
```

Open [http://localhost:8000](http://localhost:8000) for the live
visualization, or hit the API directly:

```bash
# OpenAI-style streaming
curl -N http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"In two sentences, what is paged attention?","max_tokens":80,"stream":true}'

# A simpler endpoint
curl -N http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"haiku about KV caches","max_tokens":48,"stream":true}'
```

Smoke test with concurrent requests:

```bash
python examples/smoke_client.py            # 4 prompts in parallel
python examples/smoke_client.py --prefix-demo   # show prefix-cache speedup
```

## The pieces

| File | What |
|---|---|
| `tiny_vllm/config.py` | `EngineConfig`, `SamplingParams` |
| `tiny_vllm/request.py` | `Sequence`, status enum, KV bookkeeping fields |
| `tiny_vllm/block_manager.py` | Physical block pool, refcounts, prefix-cache (hash-chain) |
| `tiny_vllm/scheduler.py` | Continuous batching + chunked prefill + preemption |
| `tiny_vllm/paged_kv.py` | The actual KV tensors that block ids point into |
| `tiny_vllm/model_runner.py` | Minimal Qwen2 forward (RoPE, RMSNorm, GQA) using the paged cache |
| `tiny_vllm/sampler.py` | Greedy / top-k / top-p |
| `tiny_vllm/engine.py` | Orchestrator: scheduler ⟶ model ⟶ sampler ⟶ outputs + events |
| `tiny_vllm/server.py` | FastAPI: `/generate`, `/v1/completions`, `/engine/events`, `/` |
| `web/` | Static demo page (vanilla HTML/CSS/JS, no framework) |

The model-free parts (block manager, scheduler) have unit tests:

```bash
pip install pytest
python -m pytest tests/
```

## GitHub Pages demo (replay mode)

The visualization can run as a **static page** on GitHub Pages with no
backend.  It plays back a recorded session from `web/events.jsonl`:

1. The repo ships a fabricated `web/events.jsonl` so the page works on first
   deploy (run `python scripts/make_demo_recording.py > web/events.jsonl` to
   regenerate).
2. To use a **real** recording instead, run the server with `--record`:
   ```bash
   python -m tiny_vllm.server --record web/events.jsonl
   # …submit some prompts via the UI or smoke_client…
   # Ctrl-C the server.  events.jsonl now contains the full session.
   git add web/events.jsonl && git commit -m "fresh demo recording" && git push
   ```
3. Enable Pages once: **repo → Settings → Pages → Source: "GitHub Actions"**.
   The workflow in `.github/workflows/deploy-pages.yml` then publishes
   `web/` on every push to `main` that touches it.

The page auto-detects mode:
- Tries `/engine/events` SSE first; if it responds within 2s it's **live**.
- Otherwise falls back to **replay**, fetching `events.jsonl` from the same
  directory and playing it back with original timing (speed control / pause
  / restart in the controls row).
- Force a mode with `?mode=replay` or `?mode=live`; point at a different
  recording with `?session=URL`.

## What the demo page shows

| Panel | What you're looking at |
|---|---|
| **Block pool** | One cell per physical block.  Color = state (free / cached-evictable / in-use / shared).  Orange border = the block has been hashed and is discoverable in the prefix cache. |
| **Scheduler** | Live stats: tokens this step, prefill-vs-decode split, step latency, prefix-cache hit-rate, preemption count.  Step log scrolls below. |
| **Sequences** | Every active sequence's block table (cell per block, blue = prefix-cache hit, purple = shared), status, generated text. |

Click **Send ×2** to fire the same prompt twice — the second send should
prefix-cache the entire prompt and start decoding almost immediately.

## Reading order

If you want to learn the system:

1. `request.py` — what a request becomes.
2. `block_manager.py` — read `admit()` and `_take_free_block()`; the prefix
   cache lives here.
3. `scheduler.py` — read `schedule()`; the two-phase loop is the heart of
   continuous batching.
4. `model_runner.py` → `Qwen2Attention.forward` — see how Q/K/V get written
   into and read out of the paged cache.
5. `engine.py::_run_loop` — how everything is wired step-by-step.
6. `server.py` — the SSE surface.

## Known limitations

- CPU-friendly defaults; no custom CUDA / Triton kernels.
- Per-sequence attention loop inside each layer (not packed/varlen-fused).
- Only Llama/Qwen2-style decoder architectures (RMSNorm + RoPE + GQA + SwiGLU MLP).
- Single-prompt completions (`n=1`); no beam search.
- No tensor parallel, no quantization.
- Prefix-cache eviction is LRU on the free list — not the full
  reference-counted radix tree vLLM ships.

## License

MIT.
