"""Three ways to ask a VLM for the target: with Curio geometry from the
person view, with both views and no hints, or with the robot view only.
Task T5-02."""

from __future__ import annotations

import numpy as np

from curio_robot.bench.items import BenchItem
from curio_robot.grounding.marks import LETTERS, Mark, place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, _encode_png, build_request
from curio_robot.resolver import frame_habit

CONDITIONS = ("geometry", "paired", "robot-only")
INTRO = "You control a robot arm sharing a table with a person who sits across from it."
ANSWER = "Answer with the single letter of the object they mean."
_PAIRED_PREAMBLE = (
    "Image 1 is what the person sees. Image 2 is what the robot sees. "
    "The same letter marks the same object in both."
)
_ROBOT_PREAMBLE = "The image is what the robot sees. Objects are marked with letters."


def build(condition: str, item: BenchItem) -> GroundingRequest:
    """The request for one item (ValueError for an unknown condition).

    - geometry: exactly the live pipeline. marks = place_marks(object_map,
      item.human); image, _ = render_marks(item.human, marks); return
      build_request(item.text, object_map, marks, image, item.p_own_frame).
    - paired: one letter per object over all objects sorted by object_id (A,
      B, ...), the same letter in both views. Marks per view = place_marks on
      that view with letters replaced by the global ones; robot-view marks
      only for ids in item.robot_visible. Images: render_marks of each view.
      Text lines joined by "\\n": INTRO; "Image 1 is what the person sees.
      Image 2 is what the robot sees. The same letter marks the same object
      in both."; one line per object in letter order "<letter>: <colour>";
      resolver.frame_habit(p) if p_own_frame is not None; the line
      The person said: "<text>" ; then ANSWER. Content: image 1, image 2,
      then the text, each image as {"type": "image_url", "image_url": {"url":
      "data:image/png;base64,..."}} (prompt._encode_png). letters: every
      global letter -> object_id.
    - robot-only: marks = place_marks on item.robot, kept only for ids in
      item.robot_visible, letters re-assigned A, B, ... in that order; image =
      render_marks(item.robot, marks)[0]. Text: INTRO; "The image is what the
      robot sees. Objects are marked with letters."; "<letter>: <colour>" per
      mark; the habit line if any; the said line; ANSWER. letters from marks."""
    if condition == "geometry":
        return _build_geometry(item)
    if condition == "paired":
        return _build_paired(item)
    if condition == "robot-only":
        return _build_robot_only(item)
    raise ValueError(f"unknown condition: {condition}")


def _build_geometry(item: BenchItem) -> GroundingRequest:
    marks = place_marks(item.object_map, item.human)
    image, _ = render_marks(item.human, marks)
    return build_request(item.text, item.object_map, marks, image, item.p_own_frame)


def _build_paired(item: BenchItem) -> GroundingRequest:
    colours = {o.object_id: o.color for o in item.object_map.objects}
    all_ids = sorted(colours)
    global_letter = {object_id: LETTERS[i] for i, object_id in enumerate(all_ids)}

    human_marks = [
        Mark(global_letter[m.object_id], m.object_id, m.pixel)
        for m in place_marks(item.object_map, item.human)
    ]
    robot_marks = [
        Mark(global_letter[m.object_id], m.object_id, m.pixel)
        for m in place_marks(item.object_map, item.robot)
        if m.object_id in item.robot_visible
    ]
    human_image, _ = render_marks(item.human, human_marks)
    robot_image, _ = render_marks(item.robot, robot_marks)

    lines = [INTRO, _PAIRED_PREAMBLE]
    lines += [f"{global_letter[object_id]}: {colours[object_id]}" for object_id in all_ids]
    lines += _tail_lines(item)
    letters = {global_letter[object_id]: object_id for object_id in all_ids}
    messages = _message([human_image, robot_image], "\n".join(lines))
    return GroundingRequest(messages=messages, letters=letters)


def _build_robot_only(item: BenchItem) -> GroundingRequest:
    colours = {o.object_id: o.color for o in item.object_map.objects}
    shown = [m for m in place_marks(item.object_map, item.robot) if m.object_id in item.robot_visible]
    marks = [Mark(LETTERS[i], m.object_id, m.pixel) for i, m in enumerate(shown)]
    image, _ = render_marks(item.robot, marks)

    lines = [INTRO, _ROBOT_PREAMBLE]
    lines += [f"{m.letter}: {colours[m.object_id]}" for m in marks]
    lines += _tail_lines(item)
    letters = {m.letter: m.object_id for m in marks}
    messages = _message([image], "\n".join(lines))
    return GroundingRequest(messages=messages, letters=letters)


def _tail_lines(item: BenchItem) -> list[str]:
    """The habit line (if any), the words, and the answer instruction."""
    tail: list[str] = []
    if item.p_own_frame is not None:
        tail.append(frame_habit(item.p_own_frame))
    tail.append(f'The person said: "{item.text}"')
    tail.append(ANSWER)
    return tail


def _message(images: list[np.ndarray], text: str) -> list[dict]:
    """One OpenAI-style user message: the images in order, then the text."""
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_png(img)}"}}
        for img in images
    ]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]
