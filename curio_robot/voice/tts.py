"""Text-to-speech out, on its own thread so speaking never blocks the arm.
Sentences are synthesized and played in order; interrupt() cuts the current
one and drops the rest, which is how the human talks over the robot."""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import urllib.request
import wave
from collections.abc import Callable
from typing import Protocol

import numpy as np

from curio_robot.voice.audio import Speaker

log = logging.getLogger(__name__)


class Synthesizer(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


class OpenAISpeech:
    """Any OpenAI-compatible /v1/audio/speech endpoint. Default: Kokoro-FastAPI
    on port 8880 (its ONNX build runs on the 7900 XT through DirectML)."""

    def __init__(
        self, url: str = "http://127.0.0.1:8880/v1", model: str = "kokoro",
        voice: str = "af_heart", timeout_s: float = 5.0,
    ) -> None:
        self._url = url.rstrip("/") + "/audio/speech"
        self._model, self._voice, self._timeout = model, voice, timeout_s

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        body = {"model": self._model, "input": text, "voice": self._voice, "response_format": "wav"}
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = resp.read()
        with wave.open(io.BytesIO(data)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
            return pcm, w.getframerate()


class SpokenOutput:
    """VoiceOut: a synthesizer plus a speaker, with barge-in."""

    def __init__(
        self,
        synth: Synthesizer,
        speaker: Speaker | None = None,
        on_start: Callable[[str], None] | None = None,
    ) -> None:
        self._synth, self._speaker = synth, speaker or Speaker()
        self._on_start = on_start
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._generation = 0  # bumped by interrupt(); stale lines are dropped
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="voice-out", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        self._lines.put(text)

    def interrupt(self) -> None:
        with self._lock:
            self._generation += 1
        while not self._lines.empty():
            try:
                self._lines.get_nowait()
            except queue.Empty:
                break
        self._speaker.stop()

    def close(self) -> None:
        self.interrupt()
        self._lines.put(None)
        self._thread.join(timeout=2)

    def _loop(self) -> None:
        while (text := self._lines.get()) is not None:
            with self._lock:
                gen = self._generation
            try:
                pcm, sr = self._synth.synthesize(text)
            except Exception as e:  # noqa: BLE001 - a missed confirmation must not stop the robot
                log.warning("tts failed: %s", e)
                continue
            with self._lock:
                if gen != self._generation:
                    continue
            if self._on_start:
                self._on_start(text)
            self._speaker.play(pcm, sr)
