"""Voice in and out, owned by this project.

In:  microphone -> energy VAD -> streaming Whisper (partials while you talk,
     a final at the end) -> Heard events. Partials are what let the arm start
     moving before you finish the sentence.
Out: Kokoro TTS, spoken on a queue that the human can talk over (barge-in).

On the RX 7900 XT both models run as local servers: whisper.cpp's
whisper-server (Vulkan build) and Kokoro-FastAPI (or any OpenAI-compatible
/v1/audio/speech). Use a headset mic: the VAD has no echo cancellation, so
the robot's own voice on speakers would count as you talking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from curio_robot.events import Heard


class VoiceIn(Protocol):
    def start(self, emit: Callable[[Heard], None]) -> None: ...
    def close(self) -> None: ...


class VoiceOut(Protocol):
    def say(self, text: str) -> None:
        """Queue a sentence; returns immediately."""
        ...

    def interrupt(self) -> None:
        """Stop talking now and drop anything queued (the human barged in)."""
        ...

    def close(self) -> None: ...
