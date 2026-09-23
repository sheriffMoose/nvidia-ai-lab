"""Request/response shapes for the quickstart API."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=8192)


class Timings(BaseModel):
    """Server-side view. The client measures its own TTFT to catch buffering in between."""

    ttft_ms: float | None = None
    total_ms: float
    chunks: int


class ChatResponse(BaseModel):
    model: str
    content: str
    timings: Timings
