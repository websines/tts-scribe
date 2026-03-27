from __future__ import annotations

import io
import struct

import numpy as np


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def encode_pcm(audio: np.ndarray) -> bytes:
    return float32_to_int16(audio).tobytes()


def encode_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    pcm = float32_to_int16(audio)
    data = pcm.tobytes()
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


def encode_mp3(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    from pydub import AudioSegment

    pcm = float32_to_int16(audio)
    segment = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3", bitrate="128k")
    return buf.getvalue()


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return encode_pcm(audio)
    elif fmt == "wav":
        return encode_wav(audio, sample_rate)
    elif fmt == "mp3":
        return encode_mp3(audio, sample_rate)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


CONTENT_TYPES = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}
