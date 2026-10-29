"""The simulated person's side of a task: which objects they want, in what
order, and how they say each one, with their frame habit from sim.users.
A test of the loop, not a model of real people. Task T3-03."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot.conventions.readings import Frame
from curio_robot.sim.users import SimUser
from curio_robot.types import ObjectMap


@dataclass(frozen=True, slots=True)
class Goal:
    object_id: str
    text: str  # what the person says for it
    frame: Frame | None  # whose left/right the text means; None: no relation word
    deictic: bool  # "that one": only where they look says which


def describe(
    object_id: str,
    object_map: ObjectMap,
    user: SimUser,
    rng: np.random.Generator,
    deictic: bool = False,
) -> Goal:
    """How `user` asks for `object_id`. object_map is the true scene and must
    have a human_pose (ValueError otherwise); unknown object_id -> KeyError.

    - deictic: Goal(object_id, "Hand me that one", None, True), no rng draws.
    - The only object of its colour: f"Hand me the {color} one", frame None,
      no rng draws.
    - Otherwise draw the frame ("speaker" if rng.random() < user.p_own_frame,
      else "robot"), then the marker (rng.random() < user.p_marker), in that
      order. The viewer is human_pose for "speaker", robot_pose for "robot".
      side = "left" if perspective.pick(same_colour, "left", viewer) is the
      object, "right" if pick(..., "right", viewer) is, else ValueError (a
      middle object can't be named by side). Text as SimUser.utter:
      f"Hand me the {color} one on {m} {side}" with m "my" (speaker) or
      "your" (robot) when marked, else f"Hand me the {color} one on the {side}"."""
    raise NotImplementedError("T3-03")


def plan_task(
    object_map: ObjectMap,
    user: SimUser,
    n_goals: int,
    rng: np.random.Generator,
    p_deictic: float = 0.0,
) -> list[Goal]:
    """n_goals distinct objects in a random order, each described.

    ids = sorted object ids; order = rng.permutation(len(ids)); take the first
    n_goals in that order. For each, in order: deictic = rng.random() <
    p_deictic (always drawn, even when p_deictic is 0), then describe(...).
    ValueError if n_goals > len(ids) or n_goals < 1."""
    raise NotImplementedError("T3-03")
