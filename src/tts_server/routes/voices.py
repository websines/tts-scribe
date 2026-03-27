from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from tts_server.auth import check_api_key
from tts_server.schemas import VoiceInfo, VoicesResponse

router = APIRouter(tags=["voices"], dependencies=[Depends(check_api_key)])


@router.get("/v1/voices")
async def list_voices(request: Request) -> VoicesResponse:
    engine = request.app.state.engine
    return VoicesResponse(
        voices=[VoiceInfo(id=s, name=s) for s in engine.speakers],
        languages=engine.languages,
    )
