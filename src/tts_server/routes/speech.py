from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from tts_server.audio import CONTENT_TYPES, encode_audio, encode_pcm
from tts_server.auth import check_api_key
from tts_server.config import settings
from tts_server.schemas import SpeechRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["speech"], dependencies=[Depends(check_api_key)])


@router.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    engine = request.app.state.engine
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    params = {
        "text": body.input,
        "voice": body.voice or settings.default_voice,
        "language": body.language or settings.default_language,
        "instruct": body.instruct,
        "temperature": body.generation.temperature,
        "top_k": body.generation.top_k,
        "top_p": body.generation.top_p,
        "repetition_penalty": body.generation.repetition_penalty,
        "max_new_tokens": body.generation.max_new_tokens,
    }

    content_type = CONTENT_TYPES.get(body.response_format, "audio/mpeg")

    # Streaming REST response
    if body.stream:
        async def audio_stream():
            async for chunk, sr in engine.synthesize_stream(params):
                if body.response_format == "pcm":
                    yield encode_pcm(chunk)
                else:
                    yield encode_audio(chunk, sr, body.response_format)

        return StreamingResponse(audio_stream(), media_type=content_type)

    # Batch response
    try:
        audio, sr = await engine.synthesize(params)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

    audio_bytes = encode_audio(audio, sr, body.response_format)
    return Response(content=audio_bytes, media_type=content_type)
