"""The VLM is one vote against the learned belief, not an override: it can tip
a close call, but not a learned habit, an explicit "my left", or a pick
outside what the request could mean."""

from __future__ import annotations

import queue
import time

import numpy as np

from curio_robot.conventions.belief import ConventionBelief
from curio_robot.conventions.readings import enumerate_readings
from curio_robot.events import Heard
from curio_robot.resolver import frame_habit, vlm_prompt
from curio_robot.runtime.orchestrator import Orchestrator
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import ObjectMap
from curio_robot.voice.scripted import PrintedVoice
from tests.test_runtime import HUMAN, REDS, ROBOT, FakeMotion, FixedVlm, _obj

SPEAKER_LEFT = (0.25, -0.08)  # red_2
ROBOT_LEFT = (0.25, 0.08)  # red_1
BLUE = _obj("blue_1", "blue", (0.3, 0.0))


def _brain(vlm_pick: str, belief: ConventionBelief):
    maps: Latest[ObjectMap] = Latest()
    maps.put(ObjectMap([*REDS, BLUE], ROBOT, HUMAN))
    bus: queue.Queue = queue.Queue()
    motion, voice, tracer = FakeMotion(), PrintedVoice(quiet=True), Tracer()
    vlm = FixedVlm(vlm_pick)
    brain = Orchestrator(bus, maps, motion, voice, tracer, vlm=vlm, belief=belief)
    return brain, bus, motion, voice, tracer, vlm


def _say(brain, text):
    t = time.perf_counter()
    brain.handle(Heard("start", "", t))
    for i in range(1, len(text.split()) + 1):
        brain.handle(Heard("partial", " ".join(text.split()[:i]), t))
    brain.handle(Heard("end", "", t))
    brain.handle(Heard("final", text, t))


def test_learned_habit_beats_a_viewpoint_biased_vlm():
    robot_frame_person = ConventionBelief(own_frame_a=1.6, own_frame_b=4.4)  # p_own ~0.27
    brain, bus, motion, voice, tracer, vlm = _brain("red_2", robot_frame_person)
    _say(brain, "the red one on the left")
    assert np.allclose(motion.goals[-1][0][:2], ROBOT_LEFT)
    n = len(motion.goals)
    brain.handle(bus.get(timeout=2))
    assert len(motion.goals) == n and brain.last_target == "red_1"
    assert tracer.has(1, "vlm_overruled")
    assert vlm.p_own_frame < 0.3  # and the VLM was told about the habit
    assert voice.lines == ["Got it, the red one on my left."]


def test_vlm_tips_only_when_the_belief_is_unsure():
    readings = enumerate_readings("the red one on the left", ObjectMap(REDS, ROBOT, HUMAN))
    unsure, _, _, _, _, _ = _brain("red_1", ConventionBelief(own_frame_a=0.6, own_frame_b=0.4))
    assert unsure._vlm_tips(readings, "red_1")  # 0.4 x 2 beats 0.6
    sure, _, _, _, _, _ = _brain("red_1", ConventionBelief(own_frame_a=0.72, own_frame_b=0.28))
    assert not sure._vlm_tips(readings, "red_1")  # 0.28 x 2 loses to 0.72
    tie, _, _, _, _, _ = _brain("red_1", ConventionBelief(own_frame_a=2.0, own_frame_b=1.0))
    assert not tie._vlm_tips(readings, "red_1")  # 1/3 x 2 ties 2/3: the belief keeps it


def test_vlm_cannot_pick_outside_the_request():
    brain, bus, motion, _, tracer, _ = _brain("blue_1", ConventionBelief())
    _say(brain, "the red one on the left")
    n = len(motion.goals)
    brain.handle(bus.get(timeout=2))
    assert len(motion.goals) == n and brain.last_target == "red_2"
    assert tracer.has(1, "vlm_overruled")


def test_prompt_labels_both_viewpoints_and_the_habit():
    prompt = vlm_prompt("the red one on the left", ObjectMap(REDS, ROBOT, HUMAN), 0.25)
    assert "red_1: red at x=0.25 y=0.08 (speaker's right, robot's left)" in prompt
    assert "red_2: red at x=0.25 y=-0.08 (speaker's left, robot's right)" in prompt
    assert "usually mean the robot's left/right (confidence 0.75)" in prompt
    no_human = vlm_prompt("the red one", ObjectMap(REDS, ROBOT, None))
    assert "(robot's left)" in no_human and "speaker's" not in no_human
    assert "usually" not in no_human


def test_frame_habit_wording():
    assert "their own" in frame_habit(0.8)
    assert "robot's" in frame_habit(0.3)
    assert "don't yet know" in frame_habit(0.5)
