from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tts_server.audio import encode_pcm, encode_audio
from tts_server.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

WS_IDLE_TIMEOUT = 300  # seconds


@router.websocket("/v1/audio/speech/stream")
async def stream_speech(ws: WebSocket):
    # Auth check for WebSocket
    if settings.api_key:
        token = ws.query_params.get("api_key", "")
        if token != settings.api_key:
            await ws.close(code=4001, reason="Invalid API key")
            return

    await ws.accept()
    session_id = str(uuid.uuid4())[:8]
    engine = ws.app.state.engine
    logger.info("[%s] WebSocket connected", session_id)

    try:
        while True:
            # Wait for a request message
            raw = await ws.receive_text()
            msg = json.loads(raw)

            text = msg.get("text", "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "Empty text"})
                continue

            voice = msg.get("voice") or settings.default_voice
            language = msg.get("language") or settings.default_language
            instruct = msg.get("instruct", "")
            fmt = msg.get("format", "pcm")
            gen = msg.get("generation", {})

            params = {
                "text": text,
                "voice": voice,
                "language": language,
                "instruct": instruct,
                **gen,
            }

            if not engine.is_ready:
                await ws.send_json({"type": "error", "message": "Model not ready"})
                continue

            start = time.monotonic()

            # Send start signal
            await ws.send_json({
                "type": "start",
                "session_id": session_id,
                "sample_rate": 24000,
                "format": fmt,
            })

            chunk_count = 0
            total_samples = 0

            try:
                async for chunk, sr in engine.synthesize_stream(params):
                    if fmt == "pcm":
                        audio_bytes = encode_pcm(chunk)
                    else:
                        audio_bytes = encode_audio(chunk, sr, fmt)
                    await ws.send_bytes(audio_bytes)
                    chunk_count += 1
                    total_samples += len(chunk)
            except RuntimeError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            except TimeoutError:
                await ws.send_json({"type": "error", "message": "Request timed out"})
                continue

            elapsed_ms = round((time.monotonic() - start) * 1000)
            duration_ms = round(total_samples / 24000 * 1000) if total_samples else 0

            await ws.send_json({
                "type": "done",
                "chunks": chunk_count,
                "duration_ms": duration_ms,
                "elapsed_ms": elapsed_ms,
            })

            logger.info(
                "[%s] Generated %d chunks, %dms audio in %dms",
                session_id, chunk_count, duration_ms, elapsed_ms,
            )

    except WebSocketDisconnect:
        logger.info("[%s] WebSocket disconnected", session_id)
    except Exception:
        logger.exception("[%s] WebSocket error", session_id)
        try:
            await ws.close(code=1011, reason="Internal error")
        except Exception:
            pass
