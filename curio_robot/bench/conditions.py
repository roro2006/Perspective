"""Three ways to ask a VLM for the target: with Curio geometry from the
person view, with both views and no hints, or with the robot view only.
Task T5-02."""

from __future__ import annotations

from curio_robot.bench.items import BenchItem
from curio_robot.grounding.prompt import GroundingRequest

CONDITIONS = ("geometry", "paired", "robot-only")
INTRO = "You control a robot arm sharing a table with a person who sits across from it."
ANSWER = "Answer with the single letter of the object they mean."


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
    raise NotImplementedError("T5-02")
