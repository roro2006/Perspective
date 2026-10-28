"""The whole live runtime against the sim: a scripted speaker, the real
pipelining (partials, speculative motion, VLM confirm, interrupts) and the
latency waterfall, with no hardware and no GPU.

    uv run curio-sim live perspective --seed 3
    uv run curio-sim live cluttered --runs 20 --vlm-latency 0.8
    uv run curio-sim live perspective --llm-url http://127.0.0.1:8080/v1   # real local VLM"""

from __future__ import annotations

import time

import numpy as np

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
        config=RuntimeConfig(use_human_view=use_human_view),
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
