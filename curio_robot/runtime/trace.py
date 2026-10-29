"""Latency tracer: one timestamp per stage per utterance, read back as the
waterfall on the results slide. Times are perf_counter seconds; the waterfall
is in ms relative to the end of speech, so speculative motion shows up as a
negative "motion_start"."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import numpy as np

# The order stages appear in a waterfall, earliest first.
STAGES = (
    "speech_start",
    "first_partial",
    "approach",
    "speculative_target",
    "motion_start",
    "speech_end",
    "final_transcript",
    "rules_target",
    "vlm_sent",
    "vlm_reply",
    "redirected",
    "arrived",
    "tts_start",
)


class Tracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._marks: dict[int, dict[str, float]] = defaultdict(dict)
        self._frame_ms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=200))

    def mark(self, utterance: int, stage: str, t: float | None = None) -> None:
        """First mark wins: a stage is when it *first* happened."""
        with self._lock:
            self._marks[utterance].setdefault(stage, time.perf_counter() if t is None else t)

    def has(self, utterance: int, stage: str) -> bool:
        with self._lock:
            return stage in self._marks.get(utterance, {})

    def marks(self, utterance: int) -> dict[str, float]:
        """Every stage of one utterance, absolute perf_counter seconds (target 3 sessions)."""
        with self._lock:
            return dict(self._marks.get(utterance, {}))

    def waterfall(self, utterance: int) -> dict[str, float]:
        with self._lock:
            marks = dict(self._marks.get(utterance, {}))
        zero = marks.get("speech_end", min(marks.values(), default=0.0))
        order = {s: i for i, s in enumerate(STAGES)}
        return {
            s: round((t - zero) * 1000, 1)
            for s, t in sorted(marks.items(), key=lambda kv: (kv[1], order.get(kv[0], 99)))
        }

    def frame(self, stage: str, ms: float) -> None:
        """Per-frame stage cost (detect, fuse, ...), kept as a rolling window."""
        with self._lock:
            self._frame_ms[stage].append(ms)

    def frame_stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            windows = {k: list(v) for k, v in self._frame_ms.items()}
        return {
            k: {"p50": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                "n": len(v)}
            for k, v in windows.items()
            if v
        }
