"""Which moment of the utterance to look at. "Hand me *that* one": the frame
when "that" was said. Task T2-02."""

from __future__ import annotations

from dataclasses import dataclass

DEICTIC = ("this", "that", "these", "those", "here", "there")


@dataclass(frozen=True, slots=True)
class Word:
    text: str  # as transcribed, may include punctuation / leading space
    start: float  # runtime clock (perf_counter) seconds
    end: float


def parse_words(data: dict, offset: float = 0.0) -> list[Word]:
    """Words from an OpenAI-style `verbose_json` transcription, times shifted
    by `offset` (the runtime time at which the audio buffer started).

    Accepts either a top-level "words": [{"word", "start", "end"}, ...] or
    "segments": [{"words": [...]}, ...] (concatenated in order). Missing both
    -> []. Words with empty text after stripping are skipped. The real
    whisper-server format is confirmed during bring-up (T2-10)."""
    raise NotImplementedError("T2-02")


def anchor_time(words: list[Word]) -> float | None:
    """When to look. Normalize each word: lowercase, strip non-letters.
    1. The start of the LAST deictic word (DEICTIC), if any.
    2. Else the start of the last colour word (resolver.COLORS).
    3. Else the end of the last word.
    4. No words -> None."""
    raise NotImplementedError("T2-02")
