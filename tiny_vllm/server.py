"""FastAPI front-end.

Two SSE streams live behind this server:

  POST /generate         — submit a prompt, stream back token deltas
  POST /v1/completions   — OpenAI-compatible streaming completions
  GET  /engine/events    — stream of engine-state snapshots (one per step)
                            — what the demo page subscribes to
  GET  /engine/snapshot  — one-shot current state (JSON)
  GET  /                 — static demo page

The demo page subscribes to /engine/events and renders the block pool,
scheduler queues, and live token streams.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EngineConfig, SamplingParams
from .engine import LLMEngine


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    ignore_eos: bool = False
    stream: bool = True


class CompletionsRequest(BaseModel):
    model: Optional[str] = None
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _sse(data: dict | str) -> bytes:
    if isinstance(data, dict):
        data = json.dumps(data, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


def build_app(config: EngineConfig, cors_allow_origins: Optional[list[str]] = None) -> FastAPI:
    app = FastAPI(title="tiny_vllm", version="0.1.0")
    # CORS so the GH-Pages frontend (or any external page) can call this
    # backend when it's deployed on HF Spaces.  Read-only inference, so the
    # blast radius of opening this up is small.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    engine = LLMEngine(config)

    @app.on_event("startup")
    async def _on_startup() -> None:
        await engine.startup()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await engine.shutdown()

    # ---- root + static -------------------------------------------------

    static_dir = Path(__file__).parent.parent / "web"
    if static_dir.exists():
        # Keep the legacy /static prefix working for anyone bookmarking it,
        # but also serve assets at root so the SAME HTML works on GH Pages
        # (which has no Python backend and just serves files from web/).
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def root() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/style.css")
        async def _css() -> FileResponse:
            return FileResponse(str(static_dir / "style.css"))

        @app.get("/app.js")
        async def _js() -> FileResponse:
            return FileResponse(str(static_dir / "app.js"))

        @app.get("/events.jsonl")
        async def _jsonl() -> FileResponse:
            return FileResponse(str(static_dir / "events.jsonl"))
    else:
        @app.get("/")
        async def root() -> dict:
            return {"name": "tiny_vllm", "status": "ok",
                    "hint": "demo page not found; POST to /generate"}

    # ---- introspection -------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok" if engine.model_runner is not None else "starting",
            "model": config.model,
            "device": config.device,
        }

    @app.get("/engine/snapshot")
    async def snapshot() -> dict:
        return engine.snapshot()

    @app.get("/engine/events")
    async def events(request: Request) -> StreamingResponse:
        q = engine.subscribe_events()

        async def gen() -> AsyncIterator[bytes]:
            # Push initial snapshot so a freshly-connected client has state.
            yield _sse({"type": "snapshot", "payload": engine.snapshot()})
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    yield _sse({
                        "type": ev.type,
                        "step": ev.step,
                        "timestamp": ev.timestamp,
                        "payload": ev.payload,
                    })
            finally:
                engine.unsubscribe_events(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- generation ----------------------------------------------------

    def _params(req: GenerateRequest) -> SamplingParams:
        return SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
            ignore_eos=req.ignore_eos,
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest, request: Request) -> StreamingResponse | JSONResponse:
        try:
            rid = engine.add_request(req.prompt, _params(req))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not req.stream:
            text_parts: list[str] = []
            finish_reason: Optional[str] = None
            async for item in engine.stream(rid):
                text_parts.append(item.new_text)
                if item.finished:
                    finish_reason = item.finish_reason
                    break
            return JSONResponse({
                "request_id": rid,
                "text": "".join(text_parts),
                "finish_reason": finish_reason,
            })

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for item in engine.stream(rid):
                    if await request.is_disconnected():
                        engine.abort(rid)
                        break
                    yield _sse({
                        "request_id": rid,
                        "text": item.new_text,
                        "finished": item.finished,
                        "finish_reason": item.finish_reason,
                    })
                    if item.finished:
                        yield b"data: [DONE]\n\n"
                        break
            except asyncio.CancelledError:
                engine.abort(rid)
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionsRequest, request: Request):
        # Single-prompt only (n=1) for the minimal impl.
        if isinstance(req.prompt, list):
            if len(req.prompt) != 1:
                raise HTTPException(400, "tiny_vllm only supports a single prompt per call")
            prompt = req.prompt[0]
        else:
            prompt = req.prompt
        try:
            rid = engine.add_request(
                prompt,
                SamplingParams(
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    seed=req.seed,
                ),
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

        created = int(time.time())
        model_id = req.model or config.model

        if not req.stream:
            text_parts: list[str] = []
            finish_reason: Optional[str] = None
            async for item in engine.stream(rid):
                text_parts.append(item.new_text)
                if item.finished:
                    finish_reason = item.finish_reason
                    break
            return JSONResponse({
                "id": f"cmpl-{rid}",
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": "".join(text_parts),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
            })

        async def gen() -> AsyncIterator[bytes]:
            try:
                async for item in engine.stream(rid):
                    if await request.is_disconnected():
                        engine.abort(rid)
                        break
                    chunk = {
                        "id": f"cmpl-{rid}",
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "text": item.new_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": item.finish_reason if item.finished else None,
                        }],
                    }
                    yield _sse(chunk)
                    if item.finished:
                        yield b"data: [DONE]\n\n"
                        break
            except asyncio.CancelledError:
                engine.abort(rid)
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/abort/{request_id}")
    async def abort(request_id: str) -> dict:
        ok = engine.abort(request_id)
        return {"aborted": ok}

    return app


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny_vllm server")
    parser.add_argument("--model", default=os.environ.get("TINY_VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--device", default=os.environ.get("TINY_VLLM_DEVICE", "cpu"))
    parser.add_argument("--dtype", default=os.environ.get("TINY_VLLM_DTYPE", "float32"))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--record", default=None,
                        help="Append every engine event to this JSONL file "
                             "(e.g. web/events.jsonl) to power the static replay demo.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--cors-origins", default=os.environ.get("TINY_VLLM_CORS_ORIGINS", "*"),
        help="Comma-separated allowed origins for CORS (default '*' — fine "
             "for the demo since this server is read-only inference).",
    )
    args = parser.parse_args()

    cfg = EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        enable_prefix_caching=not args.disable_prefix_caching,
        record_path=args.record,
    )

    cors_origins: list[str] | None
    if args.cors_origins.strip() in ("*", ""):
        cors_origins = None  # defaults to ["*"]
    else:
        cors_origins = [o.strip() for o in args.cors_origins.split(",") if o.strip()]

    import uvicorn
    app = build_app(cfg, cors_allow_origins=cors_origins)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
