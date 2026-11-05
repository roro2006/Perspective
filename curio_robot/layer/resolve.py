"""resolve(): from what the person said to a Goal any robot policy can use. Task T5-27.

It wraps the pieces built so far and changes none of them:
- grounding: the production adapter on the served VLM, asked exactly as the live pipeline and
  the bench "geometry" condition ask it (letters on the person's head-camera view, each
  object's side for both viewers). Without a head frame it is asked on the robot's view.
- belief (T5-21): when the speaker's Belief is given, belief.resolve routes memory wording
  ("the one I left on my right") to the view they last saw and adds the moved-it note.
- clarify (T5-22): when the top probability is under TAU_ACT, the most informative yes/no
  question is attached as `ask`, so the caller can ask instead of acting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from curio_robot import belief as belief_mod
from curio_robot import clarify
from curio_robot.grounding.client import GroundingAnswer, VllmGrounder
from curio_robot.grounding.marks import place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, build_request
from curio_robot.layer.goal import Base, Goal, base_from_camera
from curio_robot.perspective import horizontal_distance, lateral
from curio_robot.types import ROBOT_CAM, CameraFrame, CameraPose, ObjectMap

LAYER_URL = "http://127.0.0.1:8081/v1"  # the shared llama-server; the layer never starts one
REGISTRY = Path(__file__).resolve().parents[1] / "flywheel" / "registry.json"
TAU_ACT = 0.95  # T5-22's act threshold (tuned on the bench5c dev half)
TIMEOUT_S = 30.0


class GroundingUnavailable(RuntimeError):
    """The grounder gave no answer (server down, timeout, or the target not in the map)."""


@lru_cache(maxsize=1)
def production_adapter(registry: Path = REGISTRY) -> str:
    """The adapter name the registry marks as production (e.g. "r16-v4")."""
    return json.loads(registry.read_text(encoding="utf-8"))["production"]


@lru_cache(maxsize=4)
def default_grounder(url: str = LAYER_URL) -> VllmGrounder:
    return VllmGrounder(url, "qwen-vl", name=production_adapter(), timeout_s=TIMEOUT_S)


def geometry_request(text: str, om: ObjectMap, frame: CameraFrame,
                     p_own_frame: float | None = None) -> GroundingRequest:
    """The live pipeline's request: bench.conditions "geometry" for any map and frame."""
    marks = place_marks(om, frame)
    image, _ = render_marks(frame, marks)
    return build_request(text, om, marks, image, p_own_frame)


def robot_view(robot_frames: Mapping[str, CameraFrame]) -> CameraFrame:
    """The robot camera the layer grounds on when there is no head frame."""
    if ROBOT_CAM in robot_frames:
        return robot_frames[ROBOT_CAM]
    if not robot_frames:
        raise ValueError("no robot frame")
    return next(iter(robot_frames.values()))


def resolve(
    utterance: str,
    head_frame: CameraFrame | None,
    robot_frames: Mapping[str, CameraFrame],
    object_map: ObjectMap,
    speaker: belief_mod.Belief | None,
    *,
    grounder: belief_mod.Grounder | None = None,
    base: Base | None = None,
    p_own_frame: float | None = None,
    movers: Mapping[str, str] | None = None,
    tau: float = TAU_ACT,
) -> Goal:
    """Ground `utterance` and return the Goal.

    head_frame: the speaker's head-camera frame when they spoke (None: robot view only, and
    the map's human pose is ignored). speaker: their Belief (T5-21) or None. base: the robot
    base pose in the world (default: below the robot camera, base_from_camera). p_own_frame:
    the speaker's learned habit, shown to the grounder as the live pipeline does. Raises
    GroundingUnavailable when the grounder gives no answer."""
    grounder = grounder or default_grounder()
    om = object_map if head_frame is not None else replace(object_map, human_pose=None)
    base = base or base_from_camera(om.robot_pose)
    probs, note = _ground(utterance, head_frame, robot_frames, om, speaker, grounder,
                          p_own_frame, movers)
    chosen = max(probs, key=probs.get)
    obj = next((o for o in om.objects if o.object_id == chosen), None)
    if obj is None:
        raise GroundingUnavailable(f"{chosen} is not in the object map")
    xyz = tuple(float(v) for v in base.to_base(obj.position))
    others = sorted(((k, float(p)) for k, p in probs.items() if k != chosen),
                    key=lambda kp: (-kp[1], kp[0]))
    return Goal(
        target_id=chosen,
        target_xyz_robot=xyz,
        frame="speaker" if head_frame is not None else "robot",
        confidence=float(probs[chosen]),
        alternatives=tuple(others),
        ask=ask_back(probs, om, tau),
        note=note,
        adapter=production_adapter(),
    )


def _ground(text, head_frame, robot_frames, om, speaker, grounder, p_own_frame, movers
            ) -> tuple[dict[str, float], str | None]:
    """(probs over object ids, moved-it note) from the wrapped module for this case."""
    if head_frame is not None and speaker is not None:
        res = belief_mod.resolve(text, grounder, head_frame, om, speaker, movers)
        if not res.probs:
            raise GroundingUnavailable(res.error or "no answer")
        return dict(res.probs), res.note
    frame = head_frame if head_frame is not None else robot_view(robot_frames)
    answer: GroundingAnswer = grounder.score(geometry_request(text, om, frame, p_own_frame))
    if not answer.probs:
        raise GroundingUnavailable(answer.error or "no answer")
    return dict(answer.probs), None


def ask_back(probs: dict[str, float], om: ObjectMap, tau: float = TAU_ACT) -> str | None:
    """T5-22's first question when the top probability is under tau and some question is
    worth asking, in the speaker's frame (the robot's when the map has no human pose)."""
    post = clarify.normalize(probs)
    if not post or max(post.values()) >= tau:
        return None
    q = clarify.best_question(candidates(om), post)
    return None if q is None else q.text


def candidates(om: ObjectMap) -> list[clarify.Candidate]:
    speaker: CameraPose = om.human_pose if om.human_pose is not None else om.robot_pose
    return [clarify.Candidate(o.object_id, o.color, lateral(o, speaker),
                              horizontal_distance(o, speaker)) for o in om.objects]
