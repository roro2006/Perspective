"""Acceptance tests for T1-10: the live learning demo in the full runtime."""

from __future__ import annotations

import time

from curio_robot.conventions.belief import ConventionBelief
from curio_robot.sim.learn import live_learning
from curio_robot.sim.live import ReactiveVoice
from curio_robot.sim.users import SimUser

ROBOT_FRAME = SimUser("robot-frame", p_own_frame=0.05)


def test_reactive_voice_says_request_then_reacts_once():
    replies = iter(["no, the other one"])
    heard = []
    voice = ReactiveVoice("hand me the red one", lambda: next(replies, None), delay_s=0.0,
                          words_per_s=50, stt_lag_s=0.0, final_lag_s=0.0, poll_s=0.01)
    assert not voice.idle_for(0.0)
    voice.start(heard.append)
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and not voice.idle_for(0.05):
        time.sleep(0.01)
    voice.close()
    finals = [h.text for h in heard if h.kind == "final"]
    assert finals == ["hand me the red one", "no, the other one"]
    assert [h.kind for h in heard[:2]] == ["start", "partial"]


def test_robot_frame_person_is_corrected_and_learned_live():
    # Seeded: request 1's two frames disagree, so the prior's "own frame" guess is wrong.
    r = live_learning(ROBOT_FRAME, requests=3, seed=1, quiet=True)
    assert len(r["outcomes"]) == len(r["p_own_frame"]) == 3
    assert "corrected" in r["outcomes"]
    assert r["final_p_own_frame"] < ConventionBelief().p_own_frame()
