# this_file: bananamendy/src/bananamendy/server.py
"""A persistent OpenAI-compatible server over local BananaMind-2 checkpoints.

Implements the subset of the API that clients actually need to talk to a local
model: `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/completions`,
each with optional SSE streaming. Sampling parameters absent from a request fall
back to the TOML config rather than to OpenAI's defaults, so the server and the
CLI behave the same.

Generation is single-flight by construction (see `engine.Engine`): requests are
served one at a time. That is a property of CPU inference in one process, not an
oversight — do not add a worker pool expecting parallel decodes.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import Config, load_config
from .engine import Engine
from .models import REGISTRY, ModelError


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = None
    seed: int | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = None
    stream: bool = False


def _sampling(config: Config, request: ChatRequest | CompletionRequest) -> dict[str, Any]:
    """Request parameters over config defaults, in bananamendr's vocabulary."""
    max_new = request.max_tokens
    if isinstance(request, ChatRequest) and request.max_completion_tokens is not None:
        max_new = request.max_completion_tokens
    penalty = None
    if isinstance(request, ChatRequest) and request.frequency_penalty is not None:
        # OpenAI's additive frequency penalty has no direct equivalent; the
        # nearest local knob is the multiplicative repetition penalty.
        penalty = 1.0 + max(request.frequency_penalty, 0.0)
    merged = config.merged(
        max_new_tokens=max_new,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
        repetition_penalty=penalty,
    )
    return merged.sampling()


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def create_app(config: Config | None = None, engine: Engine | None = None) -> FastAPI:
    """Build the ASGI app. Injectable engine so tests need no real weights."""
    settings = config or load_config()
    runtime = engine or Engine()
    app = FastAPI(title="bananamendy", version="1", docs_url="/docs")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded": runtime.loaded_names()}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        names = dict.fromkeys([settings.model, *REGISTRY, *runtime.loaded_names()])
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "bananamendy"}
                for name in names
            ],
        }

    def _resolve_model(requested: str | None) -> str:
        return requested or settings.model

    def _stream_response(
        model: str,
        *,
        chunks: Iterator[str],
        completion_id: str,
        created: int,
        chat: bool,
    ) -> StreamingResponse:
        object_name = "chat.completion.chunk" if chat else "text_completion"

        def body() -> Iterator[str]:
            if chat:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": object_name,
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                )
            for delta in chunks:
                choice: dict[str, Any] = (
                    {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                    if chat
                    else {"index": 0, "text": delta, "finish_reason": None}
                )
                yield _sse(
                    {
                        "id": completion_id,
                        "object": object_name,
                        "created": created,
                        "model": model,
                        "choices": [choice],
                    }
                )
            final: dict[str, Any] = (
                {"index": 0, "delta": {}, "finish_reason": "stop"}
                if chat
                else {"index": 0, "text": "", "finish_reason": "stop"}
            )
            yield _sse(
                {
                    "id": completion_id,
                    "object": object_name,
                    "created": created,
                    "model": model,
                    "choices": [final],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> Any:
        model = _resolve_model(request.model)
        sampling = _sampling(settings, request)
        messages = [m.model_dump() for m in request.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            if request.stream:
                chunks = runtime.stream(model, messages=messages, **sampling)
                return _stream_response(
                    model,
                    chunks=chunks,
                    completion_id=completion_id,
                    created=created,
                    chat=True,
                )
            generation = runtime.chat(model, messages, **sampling)
        except ModelError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generation.text},
                    "finish_reason": "stop" if generation.finished_by_eos else "length",
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": len(generation.tokens),
                "total_tokens": generation.prompt_tokens + len(generation.tokens),
            },
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> Any:
        model = _resolve_model(request.model)
        sampling = _sampling(settings, request)
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            if request.stream:
                chunks = runtime.stream(model, prompt=request.prompt, **sampling)
                return _stream_response(
                    model,
                    chunks=chunks,
                    completion_id=completion_id,
                    created=created,
                    chat=False,
                )
            generation = runtime.generate(model, request.prompt, **sampling)
        except ModelError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": generation.text,
                    "finish_reason": "stop" if generation.finished_by_eos else "length",
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": len(generation.tokens),
                "total_tokens": generation.prompt_tokens + len(generation.tokens),
            },
        }

    return app


def serve(
    *,
    model: str | None = None,
    host: str | None = None,
    port: int | None = None,
    preload: bool = True,
) -> None:
    """Run the server until interrupted."""
    import uvicorn

    settings = load_config().merged(model=model, host=host, port=port)
    engine = Engine()
    if preload:
        engine.load(settings.model)
    uvicorn.run(create_app(settings, engine), host=settings.host, port=settings.port)
