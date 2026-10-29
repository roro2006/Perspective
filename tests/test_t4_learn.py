"""T4-10: phase 2a, feedback sources driving a target-choice learner."""

from __future__ import annotations

import pytest
from critic_util import make_ep

from curio_robot.critic.learn import (
    FrameBelief,
    interaction,
    label_for,
    run,
    run_person,
    summarize,
)
from curio_robot.critic.reactions import Style

STYLE = Style("attentive", 0.3, 0.9, 0.2, 0.6, 2.0, 0.1, 0.3)


def test_belief_starts_on_the_speaker_frame_and_follows_evidence():
    b = FrameBelief()
    assert b.choose() == "speaker"
    b.update("robot")
    assert b.choose() == "robot"


def test_interaction_reads_the_words_in_both_frames():
    for k in range(10):
        it = interaction(0, 1, k, 0.5, STYLE, "robot")
        ep = it.episode
        assert ep.arm_target == it.picks["robot"]
        assert ep.intended == it.picks[it.said]  # the meant frame picks the wanted block
        again = interaction(0, 1, k, 0.5, STYLE, "speaker")
        assert again.episode.intended == ep.intended and again.said == it.said


def test_labels_per_source():
    wrong_said = make_ep(intended=1, arm_target=0, speech=((1.2, "no"),), t_no=1.0)
    wrong_silent = make_ep(intended=1, arm_target=0)
    assert label_for("none", wrong_said, None, 0.85) is None
    assert label_for("oracle", wrong_silent, None, 0.85) == 1
    assert label_for("spoken", wrong_said, None, 0.85) == 1
    assert label_for("spoken", wrong_silent, None, 0.85) == 0  # silence reads as yes
    assert label_for("critic+spoken", wrong_said, None, 0.85) == 1  # no critic needed
    with pytest.raises(ValueError):
        label_for("critic", wrong_silent, None, 0.85)


def test_spoken_feedback_cannot_fix_a_silent_robot_frame_person():
    oracle = run_person(0, 3, "robot-frame", "silent", "oracle", 10, None, 0.85)
    spoken = run_person(0, 3, "robot-frame", "silent", "spoken", 10, None, 0.85)
    late = slice(5, None)
    assert sum(s.wrong for s in oracle[late]) < sum(s.wrong for s in spoken[late])


def test_run_and_summarize():
    steps = run(1, 4, 0, None, 0.85, sources=("none", "spoken", "oracle"))
    assert len(steps) == 3 * 4 * 1 * 4 * 3  # types x families x people x interactions x sources
    s = summarize(steps, last=2)
    assert set(s) == {"none", "spoken", "oracle"}
    assert s["none"]["overall"]["coverage"] == 0.0
    assert s["oracle"]["overall"]["label_acc"] == 1.0
    assert len(s["oracle"]["curve"]) == 4
    assert set(s["spoken"]["by_family"]) == {"attentive", "slow", "distracted", "silent"}
