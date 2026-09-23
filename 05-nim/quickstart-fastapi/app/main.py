"""FastAPI app: a thin, streaming-safe front door to a hosted NIM.

Run: uv run uvicorn app.main:app --reload --port 8080
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .nim_client import NimClient, NimError
from .routes import chat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.nim = NimClient(settings)
        logger.info("NIM client ready — %s @ %s", settings.nim_model, settings.nim_base_url)
    except RuntimeError as exc:
        # Boot anyway so /healthz can explain what's missing instead of crash-looping.
        app.state.nim = None
        logger.error("%s", exc)
    yield
    if app.state.nim is not None:
        await app.state.nim.aclose()


app = FastAPI(title="NIM quickstart", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
if _settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

app.include_router(chat.router)


@app.get("/healthz")
async def healthz() -> dict:
    settings = get_settings()
    return {
        "ok": app.state.nim is not None,
        "model": settings.nim_model,
        "base_url": settings.nim_base_url,
        "key_loaded": bool(settings.nvidia_api_key or settings.ngc_api_key),
    }


@app.get("/models")
async def models() -> dict:
    """Live model ids for this key. Start here when a request comes back 410."""
    if app.state.nim is None:
        raise HTTPException(status_code=503, detail="NIM client unavailable — check NVIDIA_API_KEY")
    try:
        return {"models": await app.state.nim.list_models()}
    except NimError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
