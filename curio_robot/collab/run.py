"""One simulated person does a whole fetch task with the live runtime; the run
comes back as a Session. Task T3-06 (the "view-on+gaze" condition: T3-07)."""

from __future__ import annotations

from pathlib import Path

from curio_robot.collab.session import Session
from curio_robot.sim.users import SimUser
from curio_robot.sim.world import SimWorld

CONDITIONS = ("view-off", "view-on", "view-on+gaze")


def run_task(
    user: SimUser,
    condition: str,
    seed: int,
    *,
    n_goals: int = 4,
    p_deictic: float = 0.0,
    occlude: bool = False,
    world: SimWorld | None = None,
    profiles_dir: Path | None = None,
    grounder_url: str | None = None,
    quiet: bool = True,
) -> Session:
    """Behaviour: tasks/T3-06-task-runner.md. ValueError for a condition not in
    CONDITIONS."""
    raise NotImplementedError("T3-06")
