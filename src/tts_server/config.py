from __future__ import annotations

import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Model
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    device: str = "cuda:0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Auth
    api_key: str = ""

    # Queue / Limits
    max_queue_size: int = 50
    request_timeout: int = 120

    # Defaults
    default_voice: str = "Aiden"
    default_language: str = "English"

    # CORS
    cors_origins: list[str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",")]
        return v

    # Logging
    log_level: str = "INFO"

    # Streaming
    emit_every_frames: int = 4
    decode_window_frames: int = 80

    # Optimizations
    enable_compile: bool = True
    warmup_on_start: bool = True


settings = Settings()
