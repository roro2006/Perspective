"""Stand-ins for the sim and tests: a speaker who says a line at a natural
pace, with Whisper-like partials, and a voice out that prints instead of talking."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from curio_robot.events import Heard


class ScriptedVoice:
    """VoiceIn that "says" each (delay_s, text) line: speech start, one partial
    per word (lagging the audio by `stt_lag_s`), speech end, then the final
    transcript `final_lag_s` later."""

    def __init__(
        self,
        lines: list[tuple[float, str]],
        *,
        words_per_s: float = 2.8,
        stt_lag_s: float = 0.15,
        final_lag_s: float = 0.2,
    ) -> None:
        self._lines = lines
        self._word_s = 1.0 / words_per_s
        self._stt_lag, self._final_lag = stt_lag_s, final_lag_s
        self._stop = threading.Event()

    def start(self, emit: Callable[[Heard], None]) -> None:
        threading.Thread(target=self._run, args=(emit,), name="voice-in", daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def _run(self, emit: Callable[[Heard], None]) -> None:
        for delay, text in self._lines:
            if self._stop.wait(delay):
                return
            words = text.split()
            t_start = time.perf_counter()
            emit(Heard("start", "", t_start))
            for i in range(1, len(words) + 1):
                if self._stop.wait(self._word_s if i > 1 else self._stt_lag):
                    return
                emit(Heard("partial", " ".join(words[:i]), time.perf_counter() - self._stt_lag))
            t_end = time.perf_counter() - self._stt_lag + self._word_s / 2
            emit(Heard("end", "", t_end))
            if self._stop.wait(self._final_lag):
                return
            emit(Heard("final", text, t_end))


class PrintedVoice:
    """VoiceOut that records and prints what the robot would say."""

    def __init__(self, on_start: Callable[[str], None] | None = None, quiet: bool = False) -> None:
        self.lines: list[str] = []
        self._on_start, self._quiet = on_start, quiet

    def say(self, text: str) -> None:
        self.lines.append(text)
        if self._on_start:
            self._on_start(text)
        if not self._quiet:
            print(f"  robot: {text}")

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        pass
