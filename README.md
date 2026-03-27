# Qwen3-TTS Streaming Server

Production-grade, multi-user streaming TTS API server powered by [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) with true token-level streaming via [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming).

## Features

- **True streaming** — audio chunks yielded as they're generated (~333ms per chunk)
- **WebSocket + REST** — OpenAI-compatible REST endpoint and persistent WebSocket connections
- **Multi-user** — async GPU queue with FIFO serialization; multiple clients served concurrently
- **torch.compile optimizations** — ~6x faster inference vs upstream qwen-tts
- **9 built-in voices** across 10 languages
- **Optional API key auth**, CORS, structured logging

## Requirements

- Python 3.12+
- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- CUDA toolkit + cuDNN
- Sox: `sudo apt install sox libsox-fmt-all`
- FFmpeg (for MP3 encoding): `sudo apt install ffmpeg`

## Setup

```bash
git clone <repo-url> && cd tts
cp .env.example .env   # edit as needed

uv sync
uv pip install flash-attn --no-build-isolation
```

## Run

```bash
uv run tts-server
```

First boot downloads the model (~3.5GB) and runs torch.compile warmup. Subsequent starts are faster.

Server listens on `http://0.0.0.0:8000` by default (configure via `.env`).

## API

### REST: `POST /v1/audio/speech`

OpenAI-compatible. Returns audio bytes.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "Aiden", "response_format": "mp3"}' \
  --output test.mp3
```

Request body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *required* | Text to synthesize |
| `voice` | string | `Aiden` | Speaker name |
| `response_format` | string | `mp3` | `mp3`, `wav`, or `pcm` |
| `language` | string | `English` | Target language |
| `instruct` | string | `""` | Emotion/style instruction |
| `stream` | bool | `false` | If true, returns chunked streaming response |
| `generation` | object | `{}` | `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens` |

### WebSocket: `WS /v1/audio/speech/stream`

Persistent connection. Send JSON, receive binary audio chunks.

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://localhost:8000/v1/audio/speech/stream") as ws:
        await ws.send(json.dumps({
            "text": "Hello, streaming TTS!",
            "voice": "Aiden",
            "format": "pcm",
        }))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print(f"Audio chunk: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                print(data)
                if data["type"] in ("done", "error"):
                    break

asyncio.run(main())
```

**Protocol:**

| Direction | Type | Format | Description |
|-----------|------|--------|-------------|
| Client -> Server | JSON | `{"text": "...", "voice": "...", ...}` | Synthesis request |
| Server -> Client | JSON | `{"type": "start", "sample_rate": 24000}` | Stream begins |
| Server -> Client | Binary | Raw audio bytes | Audio chunk (~333ms) |
| Server -> Client | JSON | `{"type": "done", "duration_ms": ...}` | Stream complete |
| Server -> Client | JSON | `{"type": "error", "message": "..."}` | Error |

The connection stays open — send multiple requests sequentially.

### `GET /v1/voices`

Returns available voices and supported languages.

### `GET /health` / `GET /health/ready`

Health check. `/ready` returns 200 when the model is loaded, 503 otherwise.

## Available Voices

| Voice | Description | Language |
|-------|-------------|----------|
| Aiden | Sunny American male | English |
| Ryan | Dynamic male | English |
| Vivian | Bright young female | Chinese |
| Serena | Warm, gentle female | Chinese |
| Uncle_Fu | Seasoned male, low timbre | Chinese |
| Dylan | Youthful Beijing male | Chinese |
| Eric | Lively Chengdu male | Chinese |
| Ono_Anna | Playful Japanese female | Japanese |
| Sohee | Warm Korean female | Korean |

## Configuration

All settings via environment variables (prefix `TTS_`) or `.env` file. See [`.env.example`](.env.example) for the full list.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace model ID |
| `TTS_DEVICE` | `cuda:0` | GPU device |
| `TTS_PORT` | `8000` | Server port |
| `TTS_API_KEY` | *(empty)* | Set to require auth on all endpoints |
| `TTS_MAX_QUEUE_SIZE` | `50` | Max queued requests before 503 |
| `TTS_EMIT_EVERY_FRAMES` | `4` | Streaming chunk frequency (lower = faster first chunk) |
| `TTS_ENABLE_COMPILE` | `true` | torch.compile optimizations |

## Architecture

```
Clients (WS/REST) ──> FastAPI (uvicorn, 1 worker)
                          │
                    asyncio.Queue (FIFO, max 50)
                          │
                    GPU Worker (single consumer)
                          │
                    Qwen3-TTS-streaming model
                          │
                    Audio chunks ──> back to clients
```

Single process, single GPU worker. The asyncio queue serializes GPU access while the event loop keeps all connections responsive.
# tts-scribe
