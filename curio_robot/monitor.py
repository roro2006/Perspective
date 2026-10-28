"""Reaction monitor: notices when the human objects, so the robot stops and
re-asks. Every interrupt is logged as a labelled correction."""

from __future__ import annotations

import re
from dataclasses import dataclass

_STOP_WORDS = {"no", "stop", "wait", "wrong", "other"}


@dataclass(frozen=True, slots=True)
class Interrupt:
    source: str  # "speech" or "hand"
    detail: str


def check_speech(transcript: str) -> Interrupt | None:
    words = set(re.findall(r"[a-z]+", transcript.lower()))
    hit = words & _STOP_WORDS
    return Interrupt("speech", min(hit)) if hit else None


def check_hand(hand_in_workspace: bool) -> Interrupt | None:
    """TODO(B): feed from MediaPipe Hands on the human camera; True when a hand
    enters the arm's workspace polygon on the table."""
    return Interrupt("hand", "hand entered workspace") if hand_in_workspace else None
