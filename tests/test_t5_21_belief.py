"""T5-21: the belief tracker, false-belief scenes and belief-aware grounding (CPU only;
the grounder is a stand-in that reads the prompt's side labels)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from curio_robot import belief as belief_mod
from curio_robot import belief_bench as bb
from curio_robot import perspective
from curio_robot.belief import (
    Belief,
    moved_note,
    present_reading,
    refers_to_memory,
    resolve,
    visible_ids,
)
from curio_robot.geometry import intrinsics_from_fovy, look_at
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.types import HUMAN_CAM, CameraFrame, CameraPose, ObjectMap, WorldObject

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import belief_eval

_LINE = re.compile(r"^([A-Z]): (\w+) \(speaker's (left|right), robot's (left|right)\)$")


class LabelGrounder:
    """Answers from the prompt's own labels: colour and side from the words, frame from
    "my"/"your". A perfect reader of the request it is given."""

    def __init__(self) -> None:
        self.requests: list[GroundingRequest] = []

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        self.requests.append(request)
        text = request.messages[0]["content"][1]["text"]
        said = re.search(r'The person said: "(.*)"', text).group(1)
        side = re.findall(r"\b(left|right)\b", said)[-1]
        col = 4 if " your " in f" {said} " else 3
        for line in text.splitlines():
            m = _LINE.match(line)
            if m and m.group(2) in said and m.group(col) == side:
                return GroundingAnswer({request.letters[m.group(1)]: 1.0}, m.group(1), 1.0, "fake")
        return GroundingAnswer({}, None, 1.0, "unavailable", "no match")


def _pose(eye, look) -> CameraPose:
    eye = np.asarray(eye, float)
    return CameraPose(eye, look_at(eye, np.asarray(look, float)))


def _frame(pose: CameraPose, t: float) -> CameraFrame:
    k = intrinsics_from_fovy(70, 320, 240)
    return CameraFrame(HUMAN_CAM, np.zeros((240, 320, 3), np.uint8), k, pose, t, None)


def _map(positions: dict[str, tuple[float, float]], human: CameraPose) -> ObjectMap:
    objects = [
        WorldObject(oid, oid.split("_")[0], np.array([x, y, 0.0125])) for oid, (x, y) in positions.items()
    ]
    robot = _pose([-0.2, 0, 0.55], [0.3, 0, 0])
    return ObjectMap(objects, robot, human)


SEAT = _pose([0.7, 0.0, 0.4], [0.25, 0.0, 0.0])
AWAY = _pose([0.7, 0.0, 0.4], [1.5, 0.0, 0.4])


def test_belief_updates_only_from_what_the_person_sees():
    before = _map({"red_1": (0.2, -0.1), "red_2": (0.2, 0.1)}, SEAT)
    seen = _frame(SEAT, 0.0)
    belief = Belief("p").observe(seen, before, visible_ids(seen, before))
    assert set(belief.sightings) == {"red_1", "red_2"}

    after = _map({"red_1": (0.3, 0.2), "red_2": (0.2, 0.1)}, SEAT)
    away = _frame(AWAY, 1.0)
    assert visible_ids(away, after) == ()
    belief2 = belief.observe(away, after, visible_ids(away, after))
    assert belief2 is belief  # nothing seen, nothing changes
    assert belief2.false_beliefs(after) == {"red_1"}
    assert belief2.last_view.timestamp == 0.0
    np.testing.assert_allclose(belief2.sightings["red_1"].position[:2], (0.2, -0.1))

    back = _frame(SEAT, 2.0)
    belief3 = belief2.observe(back, after, visible_ids(back, after))
    assert belief3.false_beliefs(after) == set()
    assert belief3.last_view.timestamp == 2.0
    assert belief2.false_beliefs(after) == {"red_1"}  # the old belief is untouched


def test_believed_map_is_seen_from_the_last_view():
    before = _map({"red_1": (0.2, -0.1)}, SEAT)
    assert Belief("p").believed_map(before.robot_pose) is None
    belief = Belief("p").observe(_frame(SEAT, 0.0), before, ["red_1"])
    believed = belief.believed_map(before.robot_pose)
    assert believed.human_pose is SEAT
    assert [o.object_id for o in believed.objects] == ["red_1"]


def test_visible_ids_with_segmentation_counts_pixels():
    omap = _map({"red_1": (0.2, -0.1), "red_2": (0.2, 0.1)}, SEAT)
    frame = _frame(SEAT, 0.0)
    seg = np.full((240, 320), -1)
    seg[:5, :5] = 7  # 25 px of red_1
    seg[10, :10] = 8  # 10 px of red_2: too few
    frame.segmentation = seg
    assert visible_ids(frame, omap, {"red_1": 7, "red_2": 8}) == ("red_1",)


def test_memory_wording_is_routed_to_the_belief():
    for _, template in bb.MEMORY_TEMPLATES:
        assert refers_to_memory(template.format(color="red", noun="one", side="left")), template
    for _, template in bb.PRESENT_TEMPLATES:
        assert not refers_to_memory(template.format(color="red", noun="one", side="left"))
    assert not refers_to_memory("Hand me the red one on my left")


def test_present_reading_drops_the_memory_clause():
    assert present_reading("Get the red one I left on my right") == "Get the red one on my right"
    assert present_reading("Pass me the red cube that was on your left") == (
        "Pass me the red cube on your left"
    )
    assert present_reading("Could you get the red one I left to my left earlier") == (
        "Could you get the red one to my left"
    )
    assert present_reading("Bring me the blue block I set down on my left") == (
        "Bring me the blue block on my left"
    )
    for _, template in bb.MEMORY_TEMPLATES:
        words = present_reading(template.format(color="red", noun="one", side="right"))
        assert "left" not in words and not refers_to_memory(words), words


def test_moved_note_speaks_in_the_persons_frame():
    # The person at x=0.7 faces -x, so their right is +y.
    obj = WorldObject("red_1", "red", np.array([0.2, 0.15, 0.0125]))
    note = moved_note(obj, np.array([0.2, -0.15, 0.0125]), SEAT, "robot")
    assert note.startswith("I moved it; it's now on your right")
    nearer = WorldObject("red_1", "red", np.array([0.5, 0.15, 0.0125]))
    assert "nearer you" in moved_note(nearer, np.array([0.2, -0.15, 0.0125]), SEAT, "person")
    assert moved_note(nearer, np.zeros(3), SEAT, "person").startswith("Someone moved it")


def test_resolve_grounds_memory_on_the_belief_and_goes_to_the_true_position():
    before = _map({"red_1": (0.2, -0.1), "red_2": (0.2, 0.1)}, SEAT)
    belief = Belief("p").observe(_frame(SEAT, 0.0), before, ["red_1", "red_2"])
    # Their right is +y. The robot swaps the sides: red_2 to their left, red_1 to their right.
    after = _map({"red_1": (0.2, 0.2), "red_2": (0.1, -0.25)}, SEAT)
    g = LabelGrounder()
    res = resolve("Get the red one I left on my right", g, _frame(SEAT, 3.0), after, belief,
                  {"red_2": "robot"})
    assert res.used_belief and res.object_id == "red_2"
    np.testing.assert_allclose(res.goal[:2], (0.1, -0.25))
    assert res.note is not None and res.note.startswith("I moved it")
    now = resolve("Get the red one on my right", g, _frame(SEAT, 3.0), after, belief)
    assert not now.used_belief and now.object_id == "red_1"
    base = resolve("Get the red one I left on my right", g, _frame(SEAT, 3.0), after, belief,
                   use_belief=False)
    assert not base.used_belief and base.object_id == "red_1" and base.note is None


@pytest.fixture(scope="module")
def items():
    from curio_robot.sim.world import SimWorld

    with SimWorld(320, 240) as world:
        return bb.generate(world, {"false-belief": 8, "control": 4, "present": 4})


def test_generated_items_follow_the_timeline(items):
    for item in items:
        seen, gone, away_after = item.observations
        assert len(seen.visible) == len(seen.truth.objects)
        assert gone.visible == () and away_after.visible == ()
        belief = bb.tracked(item)
        assert belief.last_view.timestamp == 0.0
        flagged = belief.false_beliefs(item.truth)
        if item.kind == "control":
            assert item.moved_id is None and flagged == set()
        else:
            assert flagged == {item.moved_id}
    fb = [i for i in items if i.kind == "false-belief"]
    assert sum(i.target_moved for i in fb) == len(fb) // 2


class GeometryGrounder:
    """A perfect grounder: it sees the map and frame each request was built from (through
    a wrapped ground_request) and answers with perspective.pick from the words."""

    def __init__(self) -> None:
        self.built: dict[int, tuple[str, ObjectMap]] = {}

    def wrap(self, build):
        def ground_request(text, omap, frame):
            request = build(text, omap, frame)
            self.built[id(request)] = (text, omap)
            return request

        return ground_request

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        text, omap = self.built[id(request)]
        side = re.findall(r"\b(left|right)\b", text)[-1]
        viewer = omap.robot_pose if " your " in f" {text} " else omap.human_pose
        colour = next(c for c in ("red", "blue", "green", "yellow") if c in text)
        chosen = perspective.pick(omap.by_color(colour), side, viewer).object_id
        letter = next(k for k, v in request.letters.items() if v == chosen)
        return GroundingAnswer({chosen: 1.0}, letter, 1.0, "fake")


def test_perfect_grounder_separates_the_systems(items, monkeypatch):
    g = GeometryGrounder()
    monkeypatch.setattr(belief_mod, "ground_request", g.wrap(belief_mod.ground_request))
    rows = belief_eval.evaluate(items, g)
    fb = [r for r in rows if r["kind"] == "false-belief"]
    assert all(r["belief"]["ok"] for r in fb)
    assert not any(r["current"]["ok"] for r in fb)
    assert all(r["flag_ok"] for r in rows if r["kind"] != "present")
    present = [r for r in rows if r["kind"] == "present"]
    assert all(r["belief"]["ok"] and not r["belief"]["used_belief"] for r in present)
    s = belief_eval.summarise(rows)
    assert s["gate"]["pass"]
    assert "false-belief" in belief_eval.table(s)


def test_generation_is_deterministic_and_seeds_avoid_training():
    assert not set(bb.SEEDS.values()) & ({0, 7} | set(range(11, 20)))
    assert all(s < 70000 for s in bb.SEEDS.values())


def test_wilson_and_paired_bootstrap():
    lo, hi = bb.wilson(85, 100)
    assert lo < 0.85 < hi and 0.76 < lo < 0.78
    d, lo, hi = bb.paired_diff_ci([True] * 9 + [False], [False] * 10)
    assert d == pytest.approx(0.9) and lo <= 0.9 <= hi
