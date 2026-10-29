"""Target 2, doc section 3: a partial that names the object is grounded early,
and the final reuses that answer when it says the same words about a recent
enough look. Same _brain harness as test_t2_runtime.py, driven by handle()."""

from __future__ import annotations

import time

from curio_robot.events import Heard
from curio_robot.grounding.anchor import Word
from curio_robot.grounding.history import FrameHistory
from tests.test_t2_runtime import _brain, _frame, _reply


def _partial(brain, text):
    brain.handle(Heard("partial", text, time.perf_counter()))
    pre = brain._prefetch
    if pre is not None:
        pre.future.result(timeout=2)  # let it finish, as a slow speaker would


def _final(brain, text, words=()):
    t = time.perf_counter()
    brain.handle(Heard("end", "", t))
    brain.handle(Heard("final", text, t, tuple(words)))


def test_final_reuses_the_partial_answer():
    brain, bus, _, voice, tracer, grounder = _brain({"blue_1": 0.9, "red_1": 0.05,
                                                     "red_2": 0.05})
    brain.handle(Heard("start", "", time.perf_counter()))
    _partial(brain, "hand me that one")
    assert len(grounder.requests) == 1 and tracer.has(1, "ground_prefetch")
    _final(brain, "Hand me that one.")
    _reply(brain, bus)
    assert len(grounder.requests) == 1 and tracer.has(1, "ground_reused")
    assert brain.last_target == "blue_1" and voice.lines == ["Got it, the blue one."]


def test_different_final_words_send_a_new_request():
    brain, bus, _, _, tracer, grounder = _brain({"blue_1": 0.9, "red_1": 0.05, "red_2": 0.05})
    brain.handle(Heard("start", "", time.perf_counter()))
    _partial(brain, "hand me that")
    _final(brain, "hand me that one")
    _reply(brain, bus)
    assert len(grounder.requests) == 2 and not tracer.has(1, "ground_reused")
    assert bus.empty()  # the unused prefetch never posts a reply


def test_an_old_key_word_is_looked_at_again():
    history = FrameHistory()
    now = time.perf_counter()
    history.push(_frame(), now - 2.0)  # what the person saw at "that"
    brain, bus, _, _, tracer, grounder = _brain({"blue_1": 1.0}, history=history)
    brain.handle(Heard("start", "", now - 2.5))
    _partial(brain, "hand me that one")  # scores the latest frame, pushed just now
    words = [Word(" that", now - 2.0, now - 1.8), Word(" one", now - 1.8, now - 1.6)]
    _final(brain, "hand me that one", words)
    _reply(brain, bus)
    assert len(grounder.requests) == 2 and not tracer.has(1, "ground_reused")


def test_partial_without_an_object_word_sends_nothing():
    brain, _, _, _, tracer, grounder = _brain({"blue_1": 1.0})
    brain.handle(Heard("start", "", time.perf_counter()))
    _partial(brain, "hand me")
    assert brain._prefetch is None and not grounder.requests
    assert not tracer.has(1, "ground_prefetch")
