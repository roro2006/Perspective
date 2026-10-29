"""Streaming speech-to-text. Whisper isn't a streaming model, so while you
talk the growing buffer is re-transcribed every `partial_every` seconds (one
request in flight at a time; a slow GPU just gives fewer partials), and once
more at the end of speech for the final transcript."""

from __future__ import annotations

import io
import json
import logging
import threading
import urllib.request
import uuid
import wave
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from curio_robot.events import Heard
from curio_robot.grounding.anchor import parse_words
from curio_robot.voice.audio import BLOCK, SAMPLE_RATE, EnergyVAD

log = logging.getLogger(__name__)

PRE_ROLL_S = 0.3  # audio kept from before the VAD fired, so the first word isn't clipped


class Transcriber(Protocol):
    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str: ...


class WordTranscriber(Transcriber, Protocol):
    """Optional: a transcriber that can also give word timings (target 2). The
    recognizer uses it for the final when present; plain Transcribers still work."""

    def transcribe_words(self, pcm: np.ndarray, sample_rate: int) -> dict:
        """An OpenAI-style verbose_json dict: "text" plus "words" or "segments"."""
        ...


def wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class WhisperCppServer:
    """whisper.cpp's `whisper-server`, e.g.
    `whisper-server -m ggml-base.en.bin --port 8178 -t 8 -ac 512`. The prebuilt
    Windows CPU build is enough for short commands (docs/bringup-7900xt.md)."""

    def __init__(self, url: str = "http://127.0.0.1:8178", timeout_s: float = 5.0) -> None:
        self._url = url.rstrip("/") + "/inference"
        self._timeout = timeout_s

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        return self._post(pcm, sample_rate, {"response_format": "json"})["text"].strip()

    def transcribe_words(self, pcm: np.ndarray, sample_rate: int) -> dict:
        """verbose_json with word timestamps: {"text", "segments": [{"words": [{"word",
        "start", "end", ...}]}]}, as anchor.parse_words reads it (checked in T2-10)."""
        fields = {"response_format": "verbose_json", "timestamp_granularities[]": "word"}
        return self._post(pcm, sample_rate, fields)

    def _post(self, pcm: np.ndarray, sample_rate: int, extra: dict[str, str]) -> dict:
        boundary = uuid.uuid4().hex
        fields = {**extra, "temperature": "0"}
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()
        ]
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + wav_bytes(pcm, sample_rate)
            + b"\r\n"
        )
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self._url, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.load(resp)


class StreamingRecognizer:
    """VoiceIn backed by a microphone (or any iterable of (block, t))."""

    def __init__(
        self,
        transcriber: Transcriber,
        source: Iterable[tuple[np.ndarray, float]],
        *,
        partial_every: float = 0.3,
        vad: EnergyVAD | None = None,
    ) -> None:
        self._stt, self._source = transcriber, source
        self._partial_every = partial_every
        self._vad = vad or EnergyVAD()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._partial_busy = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self, emit: Callable[[Heard], None]) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(emit,), name="voice-in", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        close = getattr(self._source, "close", None)
        if close:
            close()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self, emit: Callable[[Heard], None]) -> None:
        pre_roll: deque[np.ndarray] = deque(maxlen=max(1, round(PRE_ROLL_S * SAMPLE_RATE / BLOCK)))
        pre_times: deque[float] = deque(maxlen=pre_roll.maxlen)
        speech: list[np.ndarray] = []
        speech_t0 = 0.0  # runtime time of the buffer's first block: the word-timing offset
        last_partial = 0.0
        for block, t in self._source:
            if not self._running:
                break
            event = self._vad.push(block, t)
            if self._vad.speaking or (event and event.kind == "end"):
                speech.append(block)
            else:
                pre_roll.append(block)
                pre_times.append(t)
            if event and event.kind == "start":
                speech_t0 = pre_times[0] if pre_times else t
                speech = [*pre_roll, *speech]
                pre_roll.clear()
                pre_times.clear()
                last_partial = t
                emit(Heard("start", "", event.t))
            elif event and event.kind == "end":
                emit(Heard("end", "", event.t))
                self._pool.submit(self._final, np.concatenate(speech), event.t, emit, speech_t0)
                speech = []
            elif self._vad.speaking and t - last_partial >= self._partial_every:
                if not self._partial_busy.is_set():
                    self._partial_busy.set()
                    last_partial = t
                    self._pool.submit(self._partial, np.concatenate(speech), t, emit)

    def _partial(self, pcm: np.ndarray, t: float, emit: Callable[[Heard], None]) -> None:
        try:
            text = self._stt.transcribe(pcm, SAMPLE_RATE)
            if text:
                emit(Heard("partial", text, t))
        except Exception as e:  # noqa: BLE001 - a lost partial is fine; the final still comes
            log.debug("partial transcription failed: %s", e)
        finally:
            self._partial_busy.clear()

    def _final(
        self, pcm: np.ndarray, t: float, emit: Callable[[Heard], None], t0: float = 0.0
    ) -> None:
        words: tuple = ()
        try:
            with_words = getattr(self._stt, "transcribe_words", None)
            if with_words is None:
                text = self._stt.transcribe(pcm, SAMPLE_RATE)
            else:
                data = with_words(pcm, SAMPLE_RATE)
                text = str(data.get("text", "")).strip()
                words = tuple(parse_words(data, offset=t0))
        except Exception as e:  # noqa: BLE001
            log.warning("final transcription failed: %s", e)
            text = ""
        emit(Heard("final", text, t, words))
