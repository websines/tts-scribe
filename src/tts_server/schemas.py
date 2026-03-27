from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048


class SpeechRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request with extensions."""

    model: str = "qwen3-tts"
    input: str
    voice: str = ""
    response_format: str = Field(default="mp3", pattern=r"^(mp3|wav|pcm)$")
    language: str = ""
    instruct: str = ""
    stream: bool = False
    generation: GenerationParams = GenerationParams()


class StreamRequest(BaseModel):
    """WebSocket streaming request."""

    text: str
    voice: str = ""
    language: str = ""
    instruct: str = ""
    format: str = Field(default="pcm", pattern=r"^(pcm|wav|mp3)$")
    generation: GenerationParams = GenerationParams()


class VoiceInfo(BaseModel):
    id: str
    name: str


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    languages: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    queue_depth: int
    uptime_seconds: float
