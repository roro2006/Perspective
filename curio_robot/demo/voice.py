"""Voice control for the demo (T5-33): "Curio, hand me the red one on my left".

Windows' built-in speech recognizer (System.Speech, offline, nothing to download) listens for
a fixed list of phrases, each starting with the wake word "Curio". The list is exactly what the
rule resolver understands (a colour plus my/your left/right, closest/furthest) and the demo's
commands, so the recognizer only ever hears sentences the demo can act on. That is far more
reliable in a noisy room than open dictation, and it never passes on words Curio would have to
guess at. Replies are spoken with Windows' own synthesizer.

The recognizer runs as a PowerShell child process (listen.ps1) and prints one line per
recognition; `Listener` turns those into the same command strings the terminal accepts.
Recognitions that arrive while Curio is speaking are dropped, so it doesn't hear itself.
Windows only; elsewhere `available()` is False and the demo keeps the terminal."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from curio_robot.resolver import COLORS

WAKE = "curio"
SCRIPT = Path(__file__).resolve().parent / "listen.ps1"
MIN_CONFIDENCE = 0.55
ECHO_GUARD_S = 0.6  # ignore what is heard this long after Curio stops talking

VERBS = ("hand me", "give me", "pass me", "pick up", "get", "get me", "grab", "go to")
NOUNS = ("one", "block")
RELATIONS = ("", "on my left", "on my right", "on your left", "on your right",
             "closest to me", "furthest from me", "nearest me", "closest to you",
             "furthest from you")

# spoken (after the wake word) -> terminal command
COMMANDS = {
    "on": "on", "turn on": "on", "switch on": "on",
    "off": "off", "turn off": "off", "switch off": "off",
    "compare": "c", "compare both": "c", "show the difference": "c",
    "next scene": "n", "next": "n",
    "reset": "r", "reset the scene": "r",
    "walk around": "seat 180", "go to the other side": "seat 180",
    "stand beside the robot": "seat 180", "stand next to the robot": "seat 180",
    "sit across": "seat 0", "go back": "seat 0", "sit across the table": "seat 0",
    "walk": "walk", "keep walking": "walk",
    "go to the end": "seat 90", "go to the other end": "seat -90",
}

# The dog demo (T5-34): a firefighter sends the Go2 to a room. Its requests are what the rule
# resolver reads there (left / right, mine or yours); "room" is not a colour, so it ranks both.
DOG_OPENERS = ("the survivor's in the room", "survivor's in the room", "search the room",
               "go to the room", "check the room", "go check the room")
DOG_SIDES = ("on my left", "on my right", "on your left", "on your right")
# T5-35's buildings mark rooms by door colour: "the red room on my left", "... closest to me".
DOG_COLOR_VERBS = ("the", "search the", "go to the", "check the")
DOG_RELATIONS = (*DOG_SIDES, "closest to me", "furthest from me", "nearest me")
DOG_COMMANDS = {
    "on": "on", "turn on": "on", "switch on": "on",
    "off": "off", "turn off": "off", "switch off": "off",
    "compare": "c", "compare both": "c", "show the difference": "c",
    "reset": "r", "reset the scene": "r",
    "turn around": "turn", "face the other way": "turn", "about face": "turn",
    "next scene": "n", "next": "n", "next building": "n",
}


def requests(robot: str = "arm") -> list[str]:
    if robot == "dog":
        out = [f"{o} {side}" for o in DOG_OPENERS for side in DOG_SIDES]
        out += [f"{v} {c} room {rel}" for v in DOG_COLOR_VERBS for c in COLORS
                for rel in DOG_RELATIONS]
        return out
    out = []
    for verb in VERBS:
        for color in COLORS:
            for noun in NOUNS:
                for rel in RELATIONS:
                    out.append(" ".join(p for p in (verb, "the", color, noun, rel) if p))
    return out


def commands(robot: str = "arm") -> dict[str, str]:
    return DOG_COMMANDS if robot == "dog" else COMMANDS


def phrases(robot: str = "arm") -> list[str]:
    return [f"{WAKE} {p}" for p in (*requests(robot), *commands(robot))]


def to_command(text: str, robot: str = "arm") -> str | None:
    """A recognized phrase -> the terminal command it means (a request is its own words)."""
    words = text.lower().strip().removeprefix(WAKE).strip(" ,")
    if not words:
        return None
    return commands(robot).get(words, words)


def available() -> bool:
    return sys.platform == "win32" and shutil.which("powershell") is not None and SCRIPT.exists()


def _powershell(*args: str) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *args]


class Listener:
    """Runs the recognizer; calls `on_command(command, heard_text, confidence)`."""

    def __init__(self, on_command: Callable[[str, str, float], None],
                 on_status: Callable[[str], None] | None = None,
                 wave_file: Path | None = None, robot: str = "arm") -> None:
        self._on_command = on_command
        self._on_status = on_status or (lambda s: None)
        self._wave = wave_file
        self._robot = robot
        self._proc: subprocess.Popen | None = None
        self._speaking_until = 0.0
        self._say_lock = threading.Lock()
        self._dir = tempfile.TemporaryDirectory(prefix="curio-voice-")
        self._phrases = Path(self._dir.name) / "phrases.txt"
        self._phrases.write_text("\n".join(phrases(robot)), encoding="utf-8")
        self.ready = threading.Event()
        self.lines: list[str] = []

    def start(self) -> None:
        args = ["-File", str(SCRIPT), "-PhraseFile", str(self._phrases)]
        if self._wave is not None:
            args += ["-WaveFile", str(self._wave)]
        self._proc = subprocess.Popen(_powershell(*args), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read, name="voice-in", daemon=True).start()

    def wait(self, timeout: float = 30.0) -> None:
        """Wait for the recognizer process to end (wave-file mode)."""
        if self._proc is not None:
            self._proc.wait(timeout)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._dir.cleanup()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip("\r\n")
            self.lines.append(line)
            if line == "READY":
                self.ready.set()
                self._on_status("listening")
                continue
            parts = line.split("\t")
            if len(parts) != 3 or parts[0] != "HEARD":
                continue
            conf, text = float(parts[1]), parts[2]
            if time.perf_counter() < self._speaking_until:
                continue  # Curio hearing itself
            if conf < MIN_CONFIDENCE:
                self._on_status(f'not sure I heard "{text}" ({conf:.2f})')
                continue
            cmd = to_command(text, self._robot)
            if cmd is not None:
                self._on_command(cmd, text, conf)
        self._on_status("voice stopped")

    def say(self, text: str) -> None:
        """Speak `text` on Windows' synthesizer without blocking the caller."""
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def _say(self, text: str) -> None:
        safe = text.replace("'", "''")
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Speak('{safe}')")
        with self._say_lock:
            self._speaking_until = float("inf")
            try:
                subprocess.run(_powershell("-Command", script), check=False, timeout=30,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            finally:
                self._speaking_until = time.perf_counter() + ECHO_GUARD_S


def synthesize(text: str, wave: Path) -> None:
    """Write `text` as speech to a wave file (for the self-check and tests)."""
    safe, path = text.replace("'", "''"), str(wave).replace("'", "''")
    script = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{safe}'); $s.Dispose()")
    subprocess.run(_powershell("-Command", script), check=True, timeout=60,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
