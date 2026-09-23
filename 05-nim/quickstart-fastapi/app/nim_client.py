"""Thin wrapper over the NIM's OpenAI-compatible endpoint.

The only NVIDIA-specific part is `base_url` + the NGC key; everything else is the
OpenAI SDK. That is the whole point of NIM, and the reason Week 8's self-hosted
swap should be a one-line config change.
"""

import logging
import time
from collections.abc import AsyncIterator

from openai import APIStatusError, AsyncOpenAI, OpenAIError

from .config import Settings
from .schemas import Message

logger = logging.getLogger(__name__)


class NimError(RuntimeError):
    """Upstream NIM failure, already reduced to something safe to show a client."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class NimClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.nim_base_url,
            timeout=settings.request_timeout_s,
        )

    @property
    def default_model(self) -> str:
        return self._settings.nim_model

    async def aclose(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[str]:
        """What this key can actually reach. The hosted catalogue changes under you."""
        try:
            page = await self._client.models.list()
        except OpenAIError as exc:
            raise NimError(f"Could not list models: {type(exc).__name__}") from exc
        return sorted(model.id for model in page.data)

    async def stream_chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[dict]:
        """Yield `{"type": ...}` events: `token`/`reasoning` per delta, then one `done` or `error`.

        Timings are measured here, upstream of our own SSE layer, so comparing them
        with the client's numbers tells you whether *we* are the ones buffering.

        Reasoning models (the nemotron family) stream `reasoning_content` deltas first
        and only then visible `content`. Those are emitted as `reasoning` events so
        "first byte" and "first *visible* token" stay separate numbers.
        """
        started = time.perf_counter()
        first_delta_at: float | None = None
        first_token_at: float | None = None
        chunks = 0
        reasoning_chunks = 0
        resolved_model = model or self._settings.nim_model

        try:
            stream = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[m.model_dump() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                thought = getattr(delta, "reasoning_content", None)
                if not text and not thought:
                    continue
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                if thought:
                    reasoning_chunks += 1
                    yield {"type": "reasoning", "text": thought}
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks += 1
                    yield {"type": "token", "text": text}
        except APIStatusError as exc:
            # 410 usually means the model was retired — GET /models for live ids.
            detail = str(getattr(exc, "message", "") or "").strip()[:300]
            logger.warning(
                "NIM returned %s for model %s: %s", exc.status_code, resolved_model, detail
            )
            yield {
                "type": "error",
                "message": f"NIM upstream error {exc.status_code} for '{resolved_model}'"
                + (f": {detail}" if detail else ""),
                "status_code": exc.status_code,
            }
            return
        except OpenAIError as exc:
            logger.exception("NIM request failed")
            yield {"type": "error", "message": f"NIM request failed: {type(exc).__name__}"}
            return

        yield {
            "type": "done",
            "model": resolved_model,
            "chunks": chunks,
            "reasoning_chunks": reasoning_chunks,
            # First visible token vs. first anything on the wire. They differ on
            # reasoning models, and only the first one is what a user perceives.
            "ttft_ms": None
            if first_token_at is None
            else round((first_token_at - started) * 1000, 1),
            "first_byte_ms": None
            if first_delta_at is None
            else round((first_delta_at - started) * 1000, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
