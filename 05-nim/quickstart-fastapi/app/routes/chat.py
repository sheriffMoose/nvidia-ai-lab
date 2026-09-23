"""Chat routes: one streaming (SSE), one buffered — so you can feel the difference."""

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..nim_client import NimClient
from ..schemas import ChatRequest, ChatResponse, Timings

router = APIRouter(prefix="/chat", tags=["chat"])

# Without these, a proxy (nginx, a cloud LB) will happily buffer the whole
# response and hand the browser one blob — the exact failure this day is about.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _nim(request: Request) -> NimClient:
    client: NimClient | None = getattr(request.app.state, "nim", None)
    if client is None:
        raise HTTPException(status_code=503, detail="NIM client unavailable — check NVIDIA_API_KEY")
    return client


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    """Server-sent events. Each `token` event is flushed as it arrives from the NIM."""
    client = _nim(request)

    async def event_stream() -> AsyncIterator[str]:
        # Flush something immediately so the client can timestamp the connection
        # separately from the model's first token.
        yield _sse({"type": "open", "model": payload.model or client.default_model})
        async for event in client.stream_chat(
            payload.messages,
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        ):
            if await request.is_disconnected():
                break
            yield _sse(event)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("", response_model=ChatResponse)
async def buffered_chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Same upstream call, collected before responding. Useful as the latency baseline."""
    client = _nim(request)
    started = time.perf_counter()
    parts: list[str] = []
    ttft_ms: float | None = None
    chunks = 0

    async for event in client.stream_chat(
        payload.messages,
        model=payload.model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    ):
        if event["type"] == "token":
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - started) * 1000, 1)
            parts.append(event["text"])
        elif event["type"] == "error":
            raise HTTPException(status_code=event.get("status_code", 502), detail=event["message"])
        elif event["type"] == "done":
            chunks = event["chunks"]

    return ChatResponse(
        model=payload.model or client.default_model,
        content="".join(parts),
        timings=Timings(
            ttft_ms=ttft_ms,
            total_ms=round((time.perf_counter() - started) * 1000, 1),
            chunks=chunks,
        ),
    )
