"""T3-07: a steady look at another reading of the request re-targets the arm
before the person says "no". Orchestrator tests use the _brain pattern of
test_t1_runtime.py (driven by handle(), nothing timed); one live task checks
the whole loop."""

from __future__ import annotations

import time

import numpy as np

from curio_robot.collab.run import run_task
from curio_robot.perception import SegmentationDetector
from curio_robot.runtime import Runtime, RuntimeConfig
from curio_robot.runtime.orchestrator import GazeCue
from curio_robot.sim.users import SimUser
from curio_robot.sim.world import SimWorld
from tests.test_runtime import REDS, _obj
from tests.test_t1_runtime import ROBOT_LEFT, _brain, _say

BLUE = _obj("blue_1", "blue", (0.3, 0.0))


def _committed_to_speakers_left(objects=None):
    brain, motion, voice = _brain()
    if objects is not None:
        brain._maps.put(type(brain._maps.get()[1])(objects, *_poses(brain)))
    _say(brain, "the red one on the left")
    assert brain.last_target == "red_2" and brain.arm_active
    return brain, motion, voice


def _poses(brain):
    m = brain._maps.get()[1]
    return m.robot_pose, m.human_pose


def test_look_at_the_other_reading_retargets_and_learns():
    brain, motion, voice = _committed_to_speakers_left()
    before = brain.belief.p_own_frame()
    brain.handle(GazeCue("red_1", time.perf_counter()))
    assert brain.last_target == "red_1"
    assert motion.goals[-1][1] == "correction"
    assert np.allclose(motion.goals[-1][0][:2], ROBOT_LEFT)
    assert brain._tracer.has(brain.utterance, "gaze_redirected")
    assert brain.belief.p_own_frame() < before  # learned like a spoken "no"
    assert voice.lines[-1] == "Oh, the red one you're looking at."


def test_look_at_something_unrelated_changes_nothing():
    brain, motion, _ = _committed_to_speakers_left([*REDS, BLUE])
    n = len(motion.goals)
    brain.handle(GazeCue("blue_1", time.perf_counter()))
    assert brain.last_target == "red_2" and len(motion.goals) == n


def test_look_at_the_goal_or_with_the_arm_idle_changes_nothing():
    brain, motion, _ = _committed_to_speakers_left()
    n = len(motion.goals)
    brain.handle(GazeCue("red_2", time.perf_counter()))  # where it's already going
    assert len(motion.goals) == n
    brain._halt()
    brain.handle(GazeCue("red_1", time.perf_counter()))
    assert brain.last_target == "red_2"


def test_off_by_default():
    with SimWorld(64, 48) as world:
        rt = Runtime(world, SegmentationDetector(world.block_colors), config=RuntimeConfig())
        assert rt._gaze is None
        on = Runtime(world, SegmentationDetector(world.block_colors),
                     config=RuntimeConfig(implicit_feedback=True))
        assert on._gaze is not None


def test_live_task_with_gaze_needs_no_spoken_correction():
    # A robot-frame person: with the view on, the first "left" goes to their side
    # (seed 3 has a pair goal), and only gaze can fix it before they speak.
    robot_frame = SimUser("robot-frame", p_own_frame=0.0)
    with SimWorld(320, 240) as world:
        s = run_task(robot_frame, "view-on+gaze", 3, world=world)
    assert any("gaze_redirected" in st.marks for st in s.steps)
    assert not any(st.kind == "correction" for st in s.steps)
    assert s.done == s.goals
