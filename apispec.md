# TTS Server API Specification

**Base URL**: `http://100.127.90.120:8153`

---

## Endpoints

### `GET /health`

Server health and status.

**Response** `200 OK`
```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "queue_depth": 0,
  "uptime_seconds": 123.4
}
```

---

### `GET /health/ready`

Readiness probe. Returns `200` when model is loaded, `503` otherwise.

---

### `GET /v1/voices`

List available voices and languages.

**Headers**: `Authorization: Bearer <api_key>` *(if auth enabled)*

**Response** `200 OK`
```json
{
  "voices": [
    { "id": "Aiden", "name": "Aiden" },
    { "id": "Ryan", "name": "Ryan" },
    { "id": "Vivian", "name": "Vivian" },
    { "id": "Serena", "name": "Serena" },
    { "id": "Uncle_Fu", "name": "Uncle_Fu" },
    { "id": "Dylan", "name": "Dylan" },
    { "id": "Eric", "name": "Eric" },
    { "id": "Ono_Anna", "name": "Ono_Anna" },
    { "id": "Sohee", "name": "Sohee" }
  ],
  "languages": [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian"
  ]
}
```

---

### `POST /v1/audio/speech`

Generate speech from text. OpenAI-compatible with extensions.

**Headers**:
- `Content-Type: application/json`
- `Authorization: Bearer <api_key>` *(if auth enabled)*

**Request Body**:

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `input` | string | | yes | Text to synthesize |
| `voice` | string | `"Aiden"` | no | Speaker name |
| `response_format` | string | `"mp3"` | no | `mp3`, `wav`, or `pcm` |
| `language` | string | `"English"` | no | Target language |
| `instruct` | string | `""` | no | Emotion/style instruction |
| `stream` | bool | `false` | no | Stream chunked audio response |
| `model` | string | `"qwen3-tts"` | no | Ignored (single model) |
| `generation` | object | `{}` | no | See generation params below |

**Generation Params** (all optional):

| Field | Type | Default |
|-------|------|---------|
| `temperature` | float | `0.9` |
| `top_k` | int | `50` |
| `top_p` | float | `1.0` |
| `repetition_penalty` | float | `1.05` |
| `max_new_tokens` | int | `2048` |

**Response** (`stream: false`):

Returns raw audio bytes.

| Format | Content-Type |
|--------|-------------|
| `mp3` | `audio/mpeg` |
| `wav` | `audio/wav` |
| `pcm` | `audio/pcm` |

PCM format: 16-bit signed integer, 24kHz, mono, little-endian.

**Response** (`stream: true`):

Chunked transfer encoding. Same content types. Each chunk is a segment of audio. Concatenate all chunks for the full audio.

**Example**:
```bash
curl -X POST http://100.127.90.120:8153/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "Aiden", "response_format": "wav"}' \
  --output test.wav
```

**Errors**:

| Status | Reason |
|--------|--------|
| `401` | Missing/invalid API key |
| `503` | Model not ready or queue full |
| `504` | Request timed out |

---

### `WS /v1/audio/speech/stream`

WebSocket endpoint for real-time streaming TTS.

**Connection**: `ws://100.127.90.120:8153/v1/audio/speech/stream`

Auth via query param: `?api_key=<key>` *(if auth enabled)*

The connection is persistent — send multiple requests sequentially on one connection.

#### Client → Server (JSON text frame)

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `text` | string | | yes | Text to synthesize |
| `voice` | string | `"Aiden"` | no | Speaker name |
| `language` | string | `"English"` | no | Target language |
| `instruct` | string | `""` | no | Emotion/style instruction |
| `format` | string | `"pcm"` | no | `pcm`, `wav`, or `mp3` |
| `generation` | object | `{}` | no | Same generation params as REST |

```json
{
  "text": "Hello, this is streaming TTS!",
  "voice": "Aiden",
  "language": "English",
  "format": "pcm"
}
```

#### Server → Client

Messages arrive in this order per request:

1. **Start** (JSON text frame):
```json
{
  "type": "start",
  "session_id": "a1b2c3d4",
  "sample_rate": 24000,
  "format": "pcm"
}
```

2. **Audio chunks** (binary frames):
   Raw audio bytes. For PCM: 16-bit signed int, 24kHz, mono.
   Each chunk is ~333ms of audio.

3. **Done** (JSON text frame):
```json
{
  "type": "done",
  "chunks": 12,
  "duration_ms": 4000,
  "elapsed_ms": 2100
}
```

**Error** (JSON text frame, can arrive at any point):
```json
{
  "type": "error",
  "message": "Server overloaded — try again later"
}
```

#### Protocol Flow

```
Client                              Server
  |── connect ──────────────────────→ |
  |←── (accept) ────────────────────  |
  |                                   |
  |── {"text":"Hello", ...} ────────→ |
  |←── {"type":"start", ...} ───────  |
  |←── [binary: audio chunk 1] ─────  |
  |←── [binary: audio chunk 2] ─────  |
  |←── [binary: audio chunk N] ─────  |
  |←── {"type":"done", ...} ────────  |
  |                                   |
  |── {"text":"Next one", ...} ─────→ |  (reuse connection)
  |←── ...                            |
  |                                   |
  |── close ────────────────────────→ |
```
