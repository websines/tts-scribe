from __future__ import annotations

import asyncio
import logging
import re
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
    future: asyncio.Future | None = None
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
        self._voice_clone_prompt: Any = None
        self._is_base_model: bool = False

    # -- Lifecycle --

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Loading model %s on %s ...", self.config.model_name, self.config.device)

        self.model = await asyncio.to_thread(self._load_model)

        # Detect model type
        model_type = getattr(self.model, "tts_model_type", "")
        self._is_base_model = "base" in str(model_type).lower() or "Base" in self.config.model_name
        logger.info("Model type: %s (is_base=%s)", model_type, self._is_base_model)

        self._speakers = list(self.model.get_supported_speakers())
        self._languages = list(self.model.get_supported_languages())
        logger.info("Speakers: %s", self._speakers)
        logger.info("Languages: %s", self._languages)

        # Pre-compute voice clone prompt for Base model
        if self._is_base_model and self.config.ref_audio_path:
            logger.info("Creating voice clone prompt from %s ...", self.config.ref_audio_path)
            self._voice_clone_prompt = await asyncio.to_thread(self._create_clone_prompt)
            logger.info("Voice clone prompt ready.")

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
            try:
                self._queue.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass
            await self._worker_task
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if job and job.future and not job.future.done():
                job.future.cancel()
        logger.info("Engine stopped.")

    def _load_model(self) -> Any:
        from qwen_tts import Qwen3TTSModel

        torch.set_float32_matmul_precision("high")

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

    def _create_clone_prompt(self) -> Any:
        return self.model.create_voice_clone_prompt(
            ref_audio=self.config.ref_audio_path,
            ref_text=self.config.ref_text or "",
            x_vector_only_mode=True,
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
            use_fast_codebook=True,
            compile_codebook_predictor=True,
            compile_talker=True,
        )

    def _warmup(self) -> None:
        warmup_texts = [
            "Warmup one two three.",
            "Second warmup pass for compilation.",
            "Third warmup to stabilize performance.",
        ]
        for text in warmup_texts:
            if self._is_base_model and self._voice_clone_prompt:
                # Warmup with streaming voice clone
                if hasattr(self.model, "stream_generate_voice_clone"):
                    for _chunk, _sr in self.model.stream_generate_voice_clone(
                        text=text,
                        language="English",
                        voice_clone_prompt=self._voice_clone_prompt,
                        emit_every_frames=self.config.emit_every_frames,
                        decode_window_frames=self.config.decode_window_frames,
                    ):
                        pass
                else:
                    self.model.generate_voice_clone(
                        text=text,
                        language="English",
                        voice_clone_prompt=self._voice_clone_prompt,
                    )
            else:
                # Warmup with custom voice
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
    def is_base_model(self) -> bool:
        return self._is_base_model

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
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, int]] = loop.create_future()
        job = InferenceJob(mode=InferenceMode.BATCH, params=params, future=future)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            raise RuntimeError("Server overloaded — try again later")
        return await asyncio.wait_for(future, timeout=self.config.request_timeout)

    async def synthesize_stream(self, params: dict[str, Any]):
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

    # -- Batch inference --

    def _run_batch(self, job: InferenceJob) -> None:
        p = job.params
        language = p.get("language", self.config.default_language)
        gen_kwargs = dict(
            do_sample=True,
            top_k=p.get("top_k", 50),
            top_p=p.get("top_p", 1.0),
            temperature=p.get("temperature", 0.9),
            repetition_penalty=p.get("repetition_penalty", 1.05),
            max_new_tokens=p.get("max_new_tokens", 2048),
        )

        if self._is_base_model and self._voice_clone_prompt:
            wavs, sr = self.model.generate_voice_clone(
                text=p["text"],
                language=language,
                voice_clone_prompt=self._voice_clone_prompt,
                **gen_kwargs,
            )
        else:
            wavs, sr = self.model.generate_custom_voice(
                text=p["text"],
                language=language,
                speaker=p.get("voice", self.config.default_voice),
                instruct=p.get("instruct", ""),
                **gen_kwargs,
            )

        self._loop.call_soon_threadsafe(job.future.set_result, (wavs[0], sr))

    # -- Streaming inference --

    def _run_streaming(self, job: InferenceJob) -> None:
        p = job.params
        language = p.get("language", self.config.default_language)
        text = p["text"]

        # Base model: true token-level streaming via stream_generate_voice_clone
        if self._is_base_model and self._voice_clone_prompt and hasattr(self.model, "stream_generate_voice_clone"):
            for chunk, sr in self.model.stream_generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self._voice_clone_prompt,
                emit_every_frames=self.config.emit_every_frames,
                decode_window_frames=self.config.decode_window_frames,
            ):
                self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, (chunk, sr))
            self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, _SENTINEL)
            return

        # CustomVoice fallback: sentence-level streaming
        voice = p.get("voice", self.config.default_voice)
        instruct = p.get("instruct", "")
        sentences = _split_sentences(text)

        for sentence in sentences:
            wavs, sr = self.model.generate_custom_voice(
                text=sentence,
                language=language,
                speaker=voice,
                instruct=instruct,
                do_sample=True,
                top_k=p.get("top_k", 50),
                top_p=p.get("top_p", 1.0),
                temperature=p.get("temperature", 0.9),
                repetition_penalty=p.get("repetition_penalty", 1.05),
                max_new_tokens=p.get("max_new_tokens", 2048),
            )
            self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, (wavs[0], sr))

        self._loop.call_soon_threadsafe(job.chunk_queue.put_nowait, _SENTINEL)


_SENTENCE_RE = re.compile(r'(?<=[.!?;:。！？；：…])\s+|(?<=\.\.\.)\s+')


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    if not parts:
        return [text.strip()]
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(part) < 15:
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    return sentences if sentences else [text.strip()]
