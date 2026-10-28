"""Audio plumbing: microphone blocks, a voice-activity detector, playback.
sounddevice (PortAudio) is imported lazily, so the sim and tests run without it."""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480  # 30 ms at 16 kHz


@dataclass(frozen=True, slots=True)
class VadEvent:
    kind: str  # "start" or "end"
    t: float  # perf_counter time of the first / last voiced block


class EnergyVAD:
    """RMS against an adaptive noise floor. Speech starts after `start_blocks`
    voiced blocks in a row and ends after `hangover_s` of silence."""

    def __init__(
        self,
        *,
        ratio: float = 3.0,
        min_rms: float = 0.004,
        start_blocks: int = 3,
        hangover_s: float = 0.45,
        block_s: float = BLOCK / SAMPLE_RATE,
    ) -> None:
        self._ratio, self._min_rms = ratio, min_rms
        self._start_blocks = start_blocks
        self._hangover = max(1, round(hangover_s / block_s))
        self._floor = min_rms / ratio
        self._run = 0  # consecutive voiced blocks while silent
        self._quiet = 0  # consecutive silent blocks while speaking
        self._first_voiced = 0.0
        self._last_voiced = 0.0
        self.speaking = False

    def push(self, block: np.ndarray, t: float) -> VadEvent | None:
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        voiced = rms > max(self._min_rms, self._floor * self._ratio)
        if not voiced:
            self._floor = 0.95 * self._floor + 0.05 * rms
        if not self.speaking:
            if voiced:
                self._run += 1
                if self._run == 1:
                    self._first_voiced = t
                if self._run >= self._start_blocks:
                    self.speaking, self._quiet, self._last_voiced = True, 0, t
                    return VadEvent("start", self._first_voiced)
            else:
                self._run = 0
            return None
        if voiced:
            self._quiet, self._last_voiced = 0, t
            return None
        self._quiet += 1
        if self._quiet >= self._hangover:
            self.speaking, self._run = False, 0
            return VadEvent("end", self._last_voiced)
        return None


class Microphone:
    """Yields (block, perf_counter time) from the default input device."""

    def __init__(self, device: int | str | None = None) -> None:
        self._device = device
        self._blocks: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._stream = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            self._blocks.put((indata[:, 0].copy(), time.perf_counter()))

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
            device=self._device, callback=callback,
        )
        self._stream.start()
        while self._stream is not None:
            try:
                yield self._blocks.get(timeout=0.5)
            except queue.Empty:
                continue

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Speaker:
    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Blocks until done or until stop() is called from another thread."""
        import sounddevice as sd

        sd.play(pcm, sample_rate)
        sd.wait()

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
