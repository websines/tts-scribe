from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tts_server.config import settings
from tts_server.engine import TTSEngine
from tts_server.logging import setup_logging
from tts_server.routes import health, speech, stream, voices

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting TTS server ...")

    engine = TTSEngine(settings)
    app.state.engine = engine
    await engine.start()

    yield

    await engine.stop()
    logger.info("TTS server shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Qwen3-TTS Streaming Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(health.router)
    app.include_router(voices.router)
    app.include_router(speech.router)
    app.include_router(stream.router)

    return app


app = create_app()


def cli() -> None:
    import uvicorn

    uvicorn.run(
        "tts_server.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    cli()
