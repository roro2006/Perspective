"""Acceptance tests for T2-04 (grounding/prompt.py)."""

import base64
import io

import numpy as np
from PIL import Image

from curio_robot.grounding.marks import Mark
from curio_robot.grounding.prompt import PROMPT_VERSION, build_request
from curio_robot.types import ObjectMap

MARKS = [Mark("A", "blue_1", (10.0, 10.0)), Mark("B", "red_1", (20.0, 10.0)),
         Mark("C", "red_2", (30.0, 10.0))]
IMAGE = np.zeros((40, 60, 3), np.uint8)


def _text(req) -> str:
    content = req.messages[0]["content"]
    return next(c["text"] for c in content if c["type"] == "text")


def test_message_shape_and_image(pair_map):
    req = build_request("hand me the red one on the left", pair_map, MARKS, IMAGE)
    assert len(req.messages) == 1 and req.messages[0]["role"] == "user"
    content = req.messages[0]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    png = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert png.format == "PNG" and png.size == (60, 40)
    assert req.letters == {"A": "blue_1", "B": "red_1", "C": "red_2"}
    assert req.prompt_version == PROMPT_VERSION


def test_prompt_text(pair_map):
    text = _text(build_request("hand me the red one on the left", pair_map, MARKS, IMAGE, 0.25))
    assert text.split("\n") == [
        "You control a robot arm sharing a table with a person.",
        "The image is what the person saw when they spoke. Objects are marked with letters.",
        "A: blue (speaker's right, robot's right)",
        "B: red (speaker's right, robot's left)",
        "C: red (speaker's left, robot's right)",
        (
            "When this person says left or right without saying whose, they usually mean "
            "the robot's left/right (confidence 0.75)."
        ),
        'The person said: "hand me the red one on the left"',
        "Answer with the single letter of the object they mean.",
    ]


def test_no_habit_line_without_a_belief(pair_map):
    text = _text(build_request("that one", pair_map, MARKS, IMAGE))
    assert "usually" not in text and "know" not in text


def test_without_the_human_view(pair_map):
    m = ObjectMap(pair_map.objects, pair_map.robot_pose, None)
    text = _text(build_request("that one", m, MARKS[:1], IMAGE))
    assert "A: blue (robot's right)" in text.split("\n")
