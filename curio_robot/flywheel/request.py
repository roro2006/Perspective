"""The ahead/behind request layout the PoC sent for "behind me" (scripts/poc/scene_base.py,
T5-18), here in the product package so deployment, training rows and the gate all build the
same text. The shipped prompt names left/right only, and an object behind the speaker can't
be in their camera, so every object gets a letter and only the ones in view are drawn."""

from __future__ import annotations

import numpy as np

from curio_robot.grounding.marks import LETTERS, Mark, place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, _encode_png
from curio_robot.types import CameraFrame, CameraPose, ObjectMap

INTRO = "You control a robot arm sharing a table with a person."
PREAMBLE = ("The image is what the person saw when they spoke. Objects are marked "
            "with letters; some are out of view.")
ANSWER = "Answer with the single letter of the object they mean."


def head_coords(point, pose: CameraPose) -> np.ndarray:
    """A world point in the viewer's frame: (right, up, forward)."""
    d = np.asarray(point, float) - pose.position
    return np.array([d @ pose.rotation[:, 0], d @ pose.rotation[:, 1], -d @ pose.rotation[:, 2]])


def side_of(point, pose: CameraPose) -> str:
    return "left" if head_coords(point, pose)[0] < 0 else "right"


def depth_of(point, pose: CameraPose) -> str:
    return "ahead" if head_coords(point, pose)[2] > 0 else "behind"


def depth_request(text: str, omap: ObjectMap, frame: CameraFrame) -> GroundingRequest:
    """scene_base.depth_request, line for line: letters over all objects sorted by id, marks
    drawn for those in view, one line per object with the speaker's and the robot's
    front/behind and left/right."""
    visible = {m.object_id: m for m in place_marks(omap, frame)}
    marks = [visible.get(o.object_id, Mark(LETTERS[i], o.object_id, (-1.0, -1.0)))
             for i, o in enumerate(sorted(omap.objects, key=lambda o: o.object_id))]
    marks = [Mark(LETTERS[i], m.object_id, m.pixel) for i, m in enumerate(marks)]
    image, _ = render_marks(frame, [m for m in marks if m.pixel[0] >= 0])
    lines = [INTRO, PREAMBLE]
    for m in marks:
        o = next(o for o in omap.objects if o.object_id == m.object_id)
        rob = (f"{depth_of(o.position, omap.robot_pose)} the robot, robot's "
               f"{side_of(o.position, omap.robot_pose)}")
        if omap.human_pose is not None:
            sp = (f"{depth_of(o.position, omap.human_pose)} the speaker, speaker's "
                  f"{side_of(o.position, omap.human_pose)}; ")
        else:
            sp = ""
        lines.append(f"{m.letter}: {o.color} ({sp}{rob})".replace("ahead the", "in front of the"))
    lines += [f'The person said: "{text}"', ANSWER]
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png(image)}"}},
        {"type": "text", "text": "\n".join(lines)}]}]
    return GroundingRequest(messages=msg, letters={m.letter: m.object_id for m in marks})
