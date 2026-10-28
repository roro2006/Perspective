"""T2-11: the grounder in the live runtime. A fake grounder returns fixed
probabilities; fusion with the belief decides whether it may move the arm.
Same FakeMotion / _brain pattern as test_t1_runtime.py, driven by handle()."""

from __future__ import annotations

import queue
import time

import numpy as np

from curio_robot.conventions.belief import ConventionBelief
from curio_robot.events import GroundingReply, Heard
from curio_robot.grounding.anchor import Word
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.history import FrameHistory
from curio_robot.runtime.orchestrator import Orchestrator
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import HUMAN_CAM, CameraFrame, Intrinsics, ObjectMap
from curio_robot.voice.scripted import PrintedVoice
from curio_robot.voice.stt import StreamingRecognizer
from tests.test_runtime import HUMAN, REDS, ROBOT, FakeMotion, _audio, _obj

SPEAKER_LEFT = (0.25, -0.08)  # red_2
ROBOT_LEFT = (0.25, 0.08)  # red_1
BLUE = _obj("blue_1", "blue", (0.3, 0.0))
K = Intrinsics(250.0, 250.0, 160.0, 120.0, 320, 240)


class FakeGrounder:
    def __init__(self, probs: dict[str, float]):
        self.probs = probs
        self.requests = []

    def score(self, request):
        self.requests.append(request)
        return GroundingAnswer(dict(self.probs), None, 1.0, "fake")


class SpyHistory(FrameHistory):
    def __init__(self):
        super().__init__()
        self.asked: list[float] = []

    def at(self, t):
        self.asked.append(t)
        return super().at(t)


def _frame() -> CameraFrame:
    return CameraFrame(HUMAN_CAM, np.zeros((240, 320, 3), np.uint8), K, HUMAN, 0.0)


def _brain(probs, belief=None, history=None):
    maps: Latest[ObjectMap] = Latest()
    maps.put(ObjectMap([*REDS, BLUE], ROBOT, HUMAN))
    bus: queue.Queue = queue.Queue()
    history = FrameHistory() if history is None else history  # empty is falsy (len 0)
    history.push(_frame(), time.perf_counter())
    motion, voice, tracer = FakeMotion(), PrintedVoice(quiet=True), Tracer()
    grounder = FakeGrounder(probs)
    brain = Orchestrator(bus, maps, motion, voice, tracer, belief=belief,
                         grounder=grounder, history=history)
    return brain, bus, motion, voice, tracer, grounder


def _say(brain, text, words=()):
    t = time.perf_counter()
    brain.handle(Heard("start", "", t))
    for i in range(1, len(text.split()) + 1):
        brain.handle(Heard("partial", " ".join(text.split()[:i]), t))
    brain.handle(Heard("end", "", t))
    brain.handle(Heard("final", text, t, tuple(words)))


def _reply(brain, bus) -> GroundingReply:
    reply = bus.get(timeout=2)
    assert isinstance(reply, GroundingReply)
    brain.handle(reply)
    return reply


def test_confident_grounder_cannot_move_an_explicit_my_left():
    brain, bus, motion, voice, tracer, grounder = _brain({"red_1": 0.99, "red_2": 0.01})
    _say(brain, "the red one on my left")
    assert brain.last_target == "red_2" and tracer.has(1, "ground_sent")
    n = len(motion.goals)
    _reply(brain, bus)
    assert len(motion.goals) == n and brain.last_target == "red_2"
    assert tracer.has(1, "ground_reply") and not tracer.has(1, "ground_redirected")
    assert grounder.requests and voice.lines == ["Got it, the red one on your left."]


def test_no_readings_lets_the_grounder_act():
    brain, bus, motion, voice, _, grounder = _brain({"blue_1": 0.9, "red_1": 0.05,
                                                     "red_2": 0.05})
    _say(brain, "hand me that one")
    assert voice.lines == []  # no "which one?" while the grounder looks
    _reply(brain, bus)
    assert brain.last_target == "blue_1" and motion.goals[-1][1] == "vlm"
    assert np.allclose(motion.goals[-1][0][:2], (0.3, 0.0))
    assert set(grounder.requests[0].letters.values()) == {"blue_1", "red_1", "red_2"}
    assert voice.lines == ["Got it, the blue one."]


def test_close_belief_and_confident_grounder_redirects():
    close = ConventionBelief(own_frame_a=0.72, own_frame_b=0.28)  # acts on red_2 at 0.72
    brain, bus, motion, _, tracer, _ = _brain({"red_1": 0.95, "red_2": 0.05}, belief=close)
    _say(brain, "the red one on the left")
    assert brain.last_target == "red_2"
    _reply(brain, bus)
    assert brain.last_target == "red_1" and tracer.has(1, "ground_redirected")
    assert np.allclose(motion.goals[-1][0][:2], ROBOT_LEFT)
    assert brain._committed.chosen_id == "red_1"  # the outcome judges where the arm went


def test_history_is_asked_for_the_anchor_word_time():
    spy = SpyHistory()
    brain, bus, *_ = _brain({"blue_1": 1.0}, history=spy)
    t0 = time.perf_counter() - 1.0
    words = [Word("hand", t0, t0 + 0.2), Word(" me", t0 + 0.2, t0 + 0.3),
             Word(" that", t0 + 0.4, t0 + 0.6), Word(" one.", t0 + 0.6, t0 + 0.8)]
    _say(brain, "hand me that one", words)
    assert spy.asked == [t0 + 0.4]  # "that", not the latest frame or speech end
    _reply(brain, bus)


def test_stale_grounding_reply_is_ignored():
    brain, bus, motion, _, tracer, _ = _brain({"red_1": 0.99, "red_2": 0.01},
                                              belief=ConventionBelief(0.72, 0.28))
    _say(brain, "the red one on the left")
    reply = bus.get(timeout=2)
    brain.handle(Heard("start", "", time.perf_counter()))  # the person moved on
    n = len(motion.goals)
    brain.handle(reply)
    assert len(motion.goals) == n and not tracer.has(1, "ground_reply")


class WordStt:
    def transcribe(self, pcm, sample_rate):
        return "hand me"

    def transcribe_words(self, pcm, sample_rate):
        return {"text": " hand me that", "words": [{"word": " that", "start": 0.5, "end": 0.7}]}


def test_recognizer_puts_word_timings_on_the_final():
    heard: list[Heard] = []
    rec = StreamingRecognizer(WordStt(), iter(_audio()), partial_every=0.2)
    rec.start(heard.append)
    deadline = time.time() + 3
    while time.time() < deadline and not any(h.kind == "final" for h in heard):
        time.sleep(0.01)
    rec.close()
    final = next(h for h in heard if h.kind == "final")
    assert final.text == "hand me that" and len(final.words) == 1
    assert final.words[0].text == " that" and final.words[0].start >= 0.5  # shifted by offset
    assert all(h.words == () for h in heard if h.kind == "partial")
