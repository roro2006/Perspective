"""The whole live runtime against the sim: a scripted speaker, the real
pipelining (partials, speculative motion, VLM confirm, interrupts) and the
latency waterfall, with no hardware and no GPU.

    uv run curio-sim live perspective --seed 3
    uv run curio-sim live cluttered --runs 20 --vlm-latency 0.8
    uv run curio-sim live perspective --llm-url http://127.0.0.1:8080/v1   # real local VLM"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from curio_robot.events import Heard
from curio_robot.perception import SegmentationDetector
from curio_robot.resolver import resolve
from curio_robot.runtime import Runtime, RuntimeConfig
from curio_robot.sim.ab import SUCCESS_RADIUS
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.world import SimWorld
from curio_robot.types import ObjectMap
from curio_robot.vlm import ChatEndpoint, VlmAnswer, VlmResolver
from curio_robot.voice.scripted import PrintedVoice, ScriptedVoice


class MockVlm:
    """Stands in for the VLM: the rule resolver's answer after `latency_s`."""

    def __init__(self, latency_s: float) -> None:
        self._latency = latency_s

    def resolve(self, text: str, object_map: ObjectMap) -> VlmAnswer:
        time.sleep(self._latency)
        target = resolve(text, object_map)
        return VlmAnswer(target.object_id, "", "mock", self._latency * 1000)


class ReactiveVoice:
    """VoiceIn for a simulated person who reacts to the robot, which a fixed
    script can't: say `request` after `delay_s`, then keep asking `react()`
    (which watches the arm and what the robot said) for the next line, e.g.
    "no, the other one" or "yours". Same pacing and partials as ScriptedVoice."""

    def __init__(
        self,
        request: str,
        react: Callable[[], str | None],
        *,
        delay_s: float = 0.8,
        poll_s: float = 0.05,
        words_per_s: float = 2.8,
        stt_lag_s: float = 0.15,
        final_lag_s: float = 0.2,
        on_say: Callable[[str], None] | None = None,
    ) -> None:
        self._request, self._react, self._on_say = request, react, on_say
        self._delay, self._poll = delay_s, poll_s
        self._word_s = 1.0 / words_per_s
        self._stt_lag, self._final_lag = stt_lag_s, final_lag_s
        self._stop = threading.Event()
        self._speaking = False
        self._last_final: float | None = None  # when the last line was finished (perf_counter)

    def start(self, emit: Callable[[Heard], None]) -> None:
        threading.Thread(target=self._run, args=(emit,), name="voice-in", daemon=True).start()

    def close(self) -> None:
        self._stop.set()

    def idle_for(self, seconds: float) -> bool:
        """The request has been said and nothing has been said for `seconds`."""
        last = self._last_final
        return not self._speaking and last is not None and time.perf_counter() - last >= seconds

    def _run(self, emit: Callable[[Heard], None]) -> None:
        if self._stop.wait(self._delay) or not self._say(emit, self._request):
            return
        while not self._stop.wait(self._poll):
            line = self._react()
            if line and not self._say(emit, line):
                return

    def _say(self, emit: Callable[[Heard], None], text: str) -> bool:
        """Speak one line; False if closed mid-line."""
        self._speaking = True
        try:
            if self._on_say:
                self._on_say(text)
            words = text.split()
            emit(Heard("start", "", time.perf_counter()))
            for i in range(1, len(words) + 1):
                if self._stop.wait(self._word_s if i > 1 else self._stt_lag):
                    return False
                emit(Heard("partial", " ".join(words[:i]), time.perf_counter() - self._stt_lag))
            t_end = time.perf_counter() - self._stt_lag + self._word_s / 2
            emit(Heard("end", "", t_end))
            if self._stop.wait(self._final_lag):
                return False
            emit(Heard("final", text, t_end))
            self._last_final = time.perf_counter()
            return True
        finally:
            self._speaking = False


def run_live(
    scenario: str,
    seed: int = 0,
    *,
    vlm_latency: float = 0.6,
    llm_url: str | None = None,
    use_human_view: bool = True,
    say_after: float = 0.8,
    timeout: float = 15.0,
    quiet: bool = False,
    world: SimWorld | None = None,
    person: str = "anonymous",
    profiles_dir: Path | None = None,
    decisions_path: Path | None = None,
) -> dict:
    world = world or SimWorld(320, 240)
    world.home()
    sc = SCENARIOS[scenario](world, np.random.default_rng(seed))
    if llm_url:
        vlm = VlmResolver([ChatEndpoint("local", llm_url)])
    else:
        vlm = MockVlm(vlm_latency)
    mark_tts: list = []
    voice_out = PrintedVoice(on_start=lambda _: mark_tts and mark_tts[0](), quiet=quiet)
    rt = Runtime(
        world,
        SegmentationDetector(world.block_colors),
        voice_in=ScriptedVoice([(say_after, sc.request)]),
        voice_out=voice_out,
        vlm=vlm,
        config=RuntimeConfig(use_human_view=use_human_view, person=person,
                             profiles_dir=profiles_dir, decisions_path=decisions_path),
    )
    mark_tts.append(lambda: rt.tracer.mark(rt.brain.utterance, "tts_start"))
    if not quiet:
        print(f"  human: {sc.request}")
    rt.start()
    try:
        rt.run(timeout, until=rt.brain.done.is_set)
        rt.run(0.2)  # let the last events land
    finally:
        rt.close()
    error = float(np.linalg.norm(world.gripper_position()[:2] - sc.expected_xy))
    return {
        "scenario": scenario,
        "seed": seed,
        "request": sc.request,
        "curio": "on" if use_human_view else "off",
        "success": error < SUCCESS_RADIUS,
        "error_m": round(error, 4),
        "waterfall_ms": rt.tracer.waterfall(1),
        "frame_ms": rt.tracer.frame_stats(),
        "said": voice_out.lines,
    }
