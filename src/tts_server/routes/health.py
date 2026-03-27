from __future__ import annotations

from fastapi import APIRouter, Request, Response

from tts_server.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    engine = request.app.state.engine
    return HealthResponse(
        status="ok" if engine.is_ready else "loading",
        model=engine.config.model_name,
        queue_depth=engine.queue_depth,
        uptime_seconds=round(engine.uptime, 1),
    )


@router.get("/health/ready")
async def ready(request: Request) -> Response:
    engine = request.app.state.engine
    if engine.is_ready:
        return Response(status_code=200, content="ok")
    return Response(status_code=503, content="not ready")
