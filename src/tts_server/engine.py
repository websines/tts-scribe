from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

from tts_server.config import Settings

logger = logging.getLogger(__name__)

_SENTINEL = object()


class InferenceMode(Enum):
    BATCH = auto()
    STREAM = auto()


@dataclass
class InferenceJob:
    mode: InferenceMode
    params: dict[str, Any]
    # For BATCH: caller awaits this future for the final (audio, sr) result
    future: asyncio.Future | None = None
    # For STREAM: engine pushes chunks here; caller reads from it
    chunk_queue: asyncio.Queue | None = None
    created_at: float = field(default_factory=time.monotonic)


class TTSEngine:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.model: Any = None
        self._queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=config.max_queue_size)
        self._running = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._started_at: float = 0.0
        self._speakers: list[str] = []
        self._languages: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Lifecycle --

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Loading model %s on %s ...", self.config.model_name, self.config.device)

        self.model = await asyncio.to_thread(self._load_model)
        self._speakers = list(self.model.get_supported_speakers())
        self._languages = list(self.model.get_supported_languages())
        logger.info("Speakers: %s", self._speakers)
        logger.info("Languages: %s", self._languages)

        if self.config.enable_compile:
            logger.info("Enabling streaming optimizations (torch.compile) ...")
            await asyncio.to_thread(self._enable_optimizations)

        if self.config.warmup_on_start:
            logger.info("Running warmup ...")
            await asyncio.to_thread(self._warmup)

        self._running.set()
        self._started_at = time.monotonic()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="tts-gpu-worker")
        logger.info("Engine ready.")

    async def stop(self) -> None:
        logger.info("Shutting down engine ...")
        self._running.clear()
        if self._worker_task:
            # Push a sentinel so the worker wakes up and exits
            try:
                self._queue.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass
            await self._worker_task
        # Drain remaining jobs and cancel them
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if job and job.future and not job.future.done():
                job.future.cancel()
        logger.info("Engine stopped.")

    def _load_model(self) -> Any:
        from qwen_tts import Qwen3TTSModel

        torch.set_float32_matmul_precision("high")

        # Use flash_attention_2 if available, fall back to sdpa (built into PyTorch)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("Using flash_attention_2")
        except ImportError:
            attn_impl = "sdpa"
            logger.info("flash-attn not installed, using PyTorch SDPA")

        return Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )

    def _enable_optimizations(self) -> None:
        if not hasattr(self.model, "enable_streaming_optimizations"):
            logger.warning("Model does not support enable_streaming_optimizations — skipping")
            return
        self.model.enable_streaming_optimizations(
            decode_window_frames=self.config.decode_window_frames,
            use_compile=True,
            use_cuda_graphs=False,
            compile_mode="reduce-overhead",
        )

    def _warmup(self) -> None:
        warmup_texts = [
            "Warmup one two three.",
            "Second warmup pass for compilation.",
            "Third warmup to stabilize performance.",
        ]
        for text in warmup_texts:
            if hasattr(self.model, "stream_generate_pcm"):
                for _chunk, _sr in self.model.stream_generate_pcm(
                    text=text,
                    language="English",
                    speaker=self.config.default_voice,
                    emit_every_frames=self.config.emit_every_frames,
                    decode_window_frames=self.config.decode_window_frames,
                ):
                    pass
            else:
                self.model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker=self.config.default_voice,
                )
        logger.info("Warmup complete.")

    # -- Properties --

    @property
    def speakers(self) -> list[str]:
        return self._speakers

    @property
    def languages(self) -> list[str]:
        return self._languages

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._started_at if self._started_at else 0.0

    @property
    def is_ready(self) -> bool:
        return self._running.is_set() and self.model is not None

    # -- Public API --

    async def synthesize(self, params: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Submit a batch (non-streaming) job. Returns (audio_array, sample_rate)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, int]] = loop.create_future()
        job = InferenceJob(mode=InferenceMode.BATCH, params=params, future=future)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise RuntimeError("Server overloaded — try again later")
        return await asyncio.wait_for(future, timeout=self.config.request_timeout)

    async def synthesize_stream(self, params: dict[str, Any]):
        """Submit a streaming job. Yields (chunk_bytes, sample_rate) tuples."""
        chunk_queue: asyncio.Queue = asyncio.Queue()
        job = InferenceJob(mode=InferenceMode.STREAM, params=params, chunk_queue=chunk_queue)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise RuntimeError("Server overloaded — try again later")

        while True:
            item = await asyncio.wait_for(
                chunk_queue.get(), timeout=self.config.request_timeout
            )
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # -- Worker --

    async def _worker_loop(self) -> None:
        logger.info("GPU worker started.")
        while self._running.is_set():
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if job is None:
                break

            try:
                if job.mode == InferenceMode.STREAM:
                    await asyncio.to_thread(self._run_streaming, job)
                else:
                    await asyncio.to_thread(self._run_batch, job)
            except Exception as exc:
                logger.exception("Inference failed")
                if job.future and not job.future.done():
                    self._loop.call_soon_threadsafe(job.future.set_exception, exc)
                if job.chunk_queue:
                    self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, exc)
                    self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, _SENTINEL)
            finally:
                self._queue.task_done()

        logger.info("GPU worker stopped.")

    def _run_batch(self, job: InferenceJob) -> None:
        p = job.params
        wavs, sr = self.model.generate_custom_voice(
            text=p["text"],
            language=p.get("language", self.config.default_language),
            speaker=p.get("voice", self.config.default_voice),
            instruct=p.get("instruct", ""),
            do_sample=True,
            top_k=p.get("top_k", 50),
            top_p=p.get("top_p", 1.0),
            temperature=p.get("temperature", 0.9),
            repetition_penalty=p.get("repetition_penalty", 1.05),
            max_new_tokens=p.get("max_new_tokens", 2048),
        )
        result = (wavs[0], sr)
        self._loop.call_soon_threadsafe(job.future.set_result, result)

    def _run_streaming(self, job: InferenceJob) -> None:
        p = job.params
        voice = p.get("voice", self.config.default_voice)
        language = p.get("language", self.config.default_language)
        instruct = p.get("instruct", "")

        gen_kwargs = dict(
            text=p["text"],
            language=language,
            emit_every_frames=self.config.emit_every_frames,
            decode_window_frames=self.config.decode_window_frames,
        )

        # Use stream_generate_pcm for custom voice if available
        if hasattr(self.model, "stream_generate_pcm"):
            gen_kwargs["speaker"] = voice
            if instruct:
                gen_kwargs["instruct"] = instruct
            stream = self.model.stream_generate_pcm(**gen_kwargs)
        elif hasattr(self.model, "stream_generate_voice_clone"):
            # Fallback to voice clone streaming (requires prompt)
            stream = self.model.stream_generate_voice_clone(**gen_kwargs)
        else:
            # No streaming support — fall back to batch and send as single chunk
            wavs, sr = self.model.generate_custom_voice(
                text=p["text"],
                language=language,
                speaker=voice,
                instruct=instruct,
            )
            self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, (wavs[0], sr))
            self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, _SENTINEL)
            return

        for chunk, sr in stream:
            self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, (chunk, sr))

        self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, _SENTINEL)
