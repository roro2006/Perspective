"""T3-06: a whole fetch task through the live runtime comes back as a Session.
Short tasks (2 goals) to keep the suite quick; each runs in real time."""

from __future__ import annotations

import pytest

from curio_robot.collab.run import CONDITIONS, run_task
from curio_robot.sim.users import SimUser
from curio_robot.sim.world import SimWorld

OWN = SimUser("own", p_own_frame=1.0)


@pytest.fixture(scope="module")
def world() -> SimWorld:
    with SimWorld(320, 240) as w:
        yield w


def test_view_on_task_is_recorded(world):
    s = run_task(OWN, "view-on", 0, n_goals=2, world=world)
    assert s.session_id == "own-view-on-0" and s.condition == "view-on" and s.task == "fetch"
    assert len(s.goals) == 2 and len(set(s.goals)) == 2
    assert s.t_end > s.t_start
    assert [st.utterance for st in s.steps] == list(range(1, len(s.steps) + 1))
    assert s.steps[0].kind == "request" and s.steps[0].goal_id == s.goals[0]
    assert sum(st.kind == "request" for st in s.steps) == 2
    assert "speech_end" in s.steps[0].marks and "final_transcript" in s.steps[0].marks
    assert set(s.done) <= set(s.goals)
    assert s.done == s.goals  # someone who means their own left, with the view on


def test_view_off_runs_and_is_labelled(world):
    s = run_task(OWN, "view-off", 1, n_goals=2, world=world)
    assert s.condition == "view-off" and len(s.goals) == 2
    assert all(st.goal_id in s.goals for st in s.steps)


def test_bad_conditions():
    assert CONDITIONS == ("view-off", "view-on", "view-on+gaze")
    with pytest.raises(ValueError):
        run_task(OWN, "curio-on", 0)
