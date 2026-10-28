"""T1-08: the runtime decides from the belief, learns from outcomes, keeps a
profile per person and records every decision. Same FakeMotion / _brain
pattern as test_runtime.py, driven by brain.handle() so nothing is timed."""

from __future__ import annotations

import queue
import time

import numpy as np

from curio_robot.conventions.belief import ConventionBelief
from curio_robot.conventions.profiles import ProfileStore
from curio_robot.events import ArmEvent, Heard
from curio_robot.recorder import DecisionRecorder, load_records
from curio_robot.runtime.orchestrator import Orchestrator
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import ObjectMap
from curio_robot.voice.scripted import PrintedVoice
from tests.test_runtime import HUMAN, REDS, ROBOT, FakeMotion

SPEAKER_LEFT = (0.25, -0.08)  # red_2: the person faces -x, so their left is -y
ROBOT_LEFT = (0.25, 0.08)  # red_1


def _brain(**kwargs):
    maps: Latest[ObjectMap] = Latest()
    maps.put(ObjectMap(REDS, ROBOT, HUMAN))
    motion, voice = FakeMotion(), PrintedVoice(quiet=True)
    brain = Orchestrator(queue.Queue(), maps, motion, voice, Tracer(), **kwargs)
    return brain, motion, voice


def _say(brain, text):
    t = time.perf_counter()
    brain.handle(Heard("start", "", t))
    words = text.split()
    for i in range(1, len(words) + 1):
        brain.handle(Heard("partial", " ".join(words[:i]), t))
    brain.handle(Heard("end", "", t))
    brain.handle(Heard("final", text, t))


def test_fresh_belief_acts_on_the_speakers_left():
    brain, motion, voice = _brain()
    _say(brain, "the red one on the left")
    assert motion.goals[-1][1] == "speculative"  # decided before the final arrived
    assert np.allclose(motion.goals[-1][0][:2], SPEAKER_LEFT)
    assert brain.last_target == "red_2"
    assert voice.lines == ["Got it, the red one on your left."]


def test_correction_lowers_own_frame_and_saves_the_profile(tmp_path):
    store = ProfileStore(tmp_path)
    brain, motion, voice = _brain(person="ana", profiles=store)
    before = brain.belief.p_own_frame()
    _say(brain, "the red one on the left")
    _say(brain, "no, the other one")
    assert np.allclose(motion.goals[-1][0][:2], ROBOT_LEFT)
    assert voice.lines[-1] == "Sorry, the other red one."
    assert brain.belief.p_own_frame() < before
    saved = store.load("ana")
    assert saved.p_own_frame() == brain.belief.p_own_frame()


def test_even_belief_asks_and_the_answer_teaches():
    brain, motion, voice = _brain(belief=ConventionBelief(1.0, 1.0))
    _say(brain, "the red one on the left")
    assert voice.lines == ["Your left or mine?"]
    assert motion.goals[-1] == (None, "stop")
    _say(brain, "yours")
    assert np.allclose(motion.goals[-1][0][:2], ROBOT_LEFT)
    assert brain.last_target == "red_1"
    assert brain.belief.p_own_frame() < 0.5
    assert voice.lines[-1] == "Got it, the red one on my left."


def test_decisions_are_recorded_with_outcomes(tmp_path):
    path = tmp_path / "decisions.jsonl"
    recorder = DecisionRecorder(path)
    brain, motion, _ = _brain(recorder=recorder)
    _say(brain, "the red one on the left")
    brain.handle(ArmEvent("arrived", motion.goals[-1][0], time.perf_counter(), "speculative"))
    _say(brain, "the red one on my right")  # the next request closes the accept window
    recorder.close()
    records = load_records(path)
    first = [r for r in records if r["utterance"] == 1]
    assert [r["stage"] for r in first if r["stage"] == "final"] == ["final"]
    assert any(r["stage"] == "partial" for r in first)
    assert all(r["outcome"] == "accepted" for r in first)
    final = next(r for r in first if r["stage"] == "final")
    assert final["decision"]["kind"] == "act" and final["decision"]["object_id"] == "red_2"
    assert final["person"] == "anonymous" and final["versions"]["belief"] == "1"
    second = [r for r in records if r["utterance"] == 2]
    assert second and all(r["outcome"] is None for r in second)
