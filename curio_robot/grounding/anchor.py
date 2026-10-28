"""Which moment of the utterance to look at. "Hand me *that* one": the frame
when "that" was said. Task T2-02."""

from __future__ import annotations

import re
from dataclasses import dataclass

from curio_robot.resolver import COLORS

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
    raw_words = []
    if "words" in data:
        raw_words = data["words"]
    elif "segments" in data:
        for seg in data["segments"]:
            if "words" in seg:
                raw_words.extend(seg["words"])

    words = []
    for item in raw_words:
        text = item["word"]
        if not text.strip():
            continue
        words.append(Word(
            text=text,
            start=offset + item["start"],
            end=offset + item["end"]
        ))
    return words


def anchor_time(words: list[Word]) -> float | None:
    """When to look. Normalize each word: lowercase, strip non-letters.
    1. The start of the LAST deictic word (DEICTIC), if any.
    2. Else the start of the last colour word (resolver.COLORS).
    3. Else the end of the last word.
    4. No words -> None."""
    if not words:
        return None

    last_deictic = None
    last_colour = None

    for w in words:
        norm = re.sub(r"[^a-z]", "", w.text.lower())
        if norm in DEICTIC:
            last_deictic = w
        elif norm in COLORS:
            last_colour = w

    if last_deictic is not None:
        return last_deictic.start
    if last_colour is not None:
        return last_colour.start
    return words[-1].end
