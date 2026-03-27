from __future__ import annotations

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from tts_server.config import settings

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def check_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not settings.api_key:
        return
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    # Accept "Bearer <key>" or raw key
    token = key.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
