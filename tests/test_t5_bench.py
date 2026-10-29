"""T5-01..03 and T5-10: the perspective benchmark, grounding conditions and scoring.
Test names start with items_, conditions_ or score_ so each card can run its own
(-k items, -k conditions, -k score)."""

from __future__ import annotations

import pytest

from curio_robot import perspective
from curio_robot.bench import conditions, items, score
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.marks import place_marks, render_marks
from curio_robot.grounding.prompt import build_request


@pytest.fixture(scope="module")
def world():
    from curio_robot.sim.world import SimWorld

    with SimWorld(320, 240) as w:
        yield w


@pytest.fixture(scope="module")
def bench(world):
    return items.generate(world, 3, seed=0)


def _side(text: str) -> str:
    return "left" if text.endswith("left") else "right"


def _viewer(item, frame):
    m = item.object_map
    return m.human_pose if frame == "speaker" else m.robot_pose


# ---- T5-01 items ----


def test_items_are_deterministic_and_ordered(world, bench):
    again = items.generate(world, 3, seed=0)
    assert [(b.id, b.text, b.target_id) for b in bench] == [
        (b.id, b.text, b.target_id) for b in again
    ]
    assert [b.kind for b in bench] == [k for k in items.KINDS for _ in range(3)]
    assert bench[0].id == "color-0000"


def test_items_marked_and_habit_targets_follow_the_words(bench):
    for b in bench:
        if b.kind not in ("marked", "habit"):
            continue
        ids = [o.object_id for o in b.object_map.objects]
        color = b.object_map.objects[ids.index(b.target_id)].color
        cands = b.object_map.by_color(color)
        side = _side(b.text)
        assert perspective.pick(cands, side, _viewer(b, b.frame)).object_id == b.target_id
        other = "robot" if b.frame == "speaker" else "speaker"
        assert perspective.pick(cands, side, _viewer(b, other)).object_id != b.target_id
        if b.kind == "marked":
            assert (" my " in b.text) == (b.frame == "speaker") and b.p_own_frame is None
        else:
            assert b.p_own_frame in items.HABITS and " my " not in b.text


def test_items_hidden_targets_are_hidden_from_the_robot(bench):
    for b in bench:
        if b.kind == "hidden":
            assert b.target_id not in b.robot_visible
        if b.kind == "color":
            assert b.target_id in b.robot_visible
            u, v = b.target_pixel_robot
            assert 0 <= u < b.robot.intrinsics.width and 0 <= v < b.robot.intrinsics.height
        assert b.human.segmentation is None and b.robot.segmentation is None


def test_items_new_kinds_leave_the_old_items_unchanged(world, bench):
    old = items.generate(world, 3, seed=0, kinds=items.KINDS[:4])
    assert items.KINDS[:4] == ("color", "marked", "habit", "hidden")
    assert len(old) == 12

    def key(b):
        return (b.id, b.text, b.target_id, b.frame, b.p_own_frame, b.robot_visible)

    assert [key(b) for b in old] == [key(b) for b in bench[:12]]
    for a, b in zip(old, bench[:12], strict=True):
        assert all(
            (x.position == y.position).all()
            for x, y in zip(a.object_map.objects, b.object_map.objects, strict=True)
        )


def _partner(b):
    target = next(o for o in b.object_map.objects if o.object_id == b.target_id)
    return next(o for o in b.object_map.by_color(target.color) if o is not target)


def test_items_hidden_pair_target_is_hidden_and_follows_the_words(bench):
    hidden_pairs = [b for b in bench if b.kind == "hidden-pair"]
    assert hidden_pairs
    for b in hidden_pairs:
        assert b.target_id not in b.robot_visible
        assert _partner(b).object_id in b.robot_visible
        assert b.p_own_frame is None
        assert (" my " in b.text) == (b.frame == "speaker")
        assert (" your " in b.text) == (b.frame == "robot")
        cands = b.object_map.by_color(_partner(b).color)
        side = _side(b.text)
        assert perspective.pick(cands, side, _viewer(b, b.frame)).object_id == b.target_id
        other = "robot" if b.frame == "speaker" else "speaker"
        assert perspective.pick(cands, side, _viewer(b, other)).object_id != b.target_id


def _free_relation(text: str) -> str:
    if any(w in text for w in ("closest", "nearest", "nearer", "end of the table")):
        return "nearest"
    if any(w in text for w in ("farthest", "further", "far side")):
        return "farthest"
    return "left" if " left" in text else "right"


@pytest.fixture(scope="module")
def free_items(world):
    return items.generate(world, 20, seed=1, kinds=("free",))


def test_items_free_targets_follow_the_words_and_frame(free_items):
    texts = {b.text for b in free_items}
    assert len(texts) >= len(free_items) // 2  # varied wording
    for b in free_items:
        assert b.p_own_frame is None and b.frame in ("speaker", "robot")
        assert "that one" not in b.text and "look" not in b.text  # no deictic or gaze cues
        cands = b.object_map.by_color(_partner(b).color)
        assert len(cands) == 2
        relation = _free_relation(b.text)
        assert perspective.pick(cands, relation, _viewer(b, b.frame)).object_id == b.target_id
        other = "robot" if b.frame == "speaker" else "speaker"
        assert perspective.pick(cands, relation, _viewer(b, other)).object_id != b.target_id
    relations = {_free_relation(b.text) for b in free_items}
    assert {"nearest", "farthest"} <= relations and relations & {"left", "right"}


def test_items_save_and_load(tmp_path, bench):
    path = items.save(bench[:4], tmp_path / "b")
    assert path.name == "items.jsonl"
    back = items.load(tmp_path / "b")
    assert [b.id for b in back] == [b.id for b in bench[:4]]
    a, b = bench[1], back[1]
    assert b.text == a.text and b.target_id == a.target_id and b.frame == a.frame
    assert b.robot_visible == a.robot_visible
    assert (b.human.rgb == a.human.rgb).all() and b.robot.rgb.shape == a.robot.rgb.shape
    assert abs(b.object_map.human_pose.position - a.object_map.human_pose.position).max() < 1e-9
    assert b.robot.intrinsics == a.robot.intrinsics


# ---- T5-02 conditions ----


def _text(req) -> str:
    return next(c["text"] for c in req.messages[0]["content"] if c["type"] == "text")


def _n_images(req) -> int:
    return sum(c["type"] == "image_url" for c in req.messages[0]["content"])


def test_conditions_geometry_is_the_live_pipeline(bench):
    b = bench[4]
    marks = place_marks(b.object_map, b.human)
    image, _ = render_marks(b.human, marks)
    live = build_request(b.text, b.object_map, marks, image, b.p_own_frame)
    req = conditions.build("geometry", b)
    assert req.letters == live.letters and _text(req) == _text(live)


def test_conditions_paired_gives_both_views_and_no_geometry(bench):
    for b in bench:
        req = conditions.build("paired", b)
        assert _n_images(req) == 2
        assert sorted(req.letters.values()) == sorted(o.object_id for o in b.object_map.objects)
        text = _text(req)
        assert "(speaker's" not in text and "(robot's" not in text  # no per-object geometry
        assert b.text in text and text.startswith(conditions.INTRO)


def test_conditions_robot_only_hides_what_the_robot_cannot_see(bench):
    for b in bench:
        req = conditions.build("robot-only", b)
        assert _n_images(req) == 1
        assert set(req.letters.values()) <= set(b.robot_visible)
        if b.kind in ("hidden", "hidden-pair"):
            assert b.target_id not in req.letters.values()
        if b.kind == "hidden-pair":
            assert _partner(b).object_id in req.letters.values()
        if b.kind == "habit":
            assert "left/right" in _text(req)


def test_conditions_build_for_every_kind(bench, free_items):
    for b in [*bench, *free_items[:3]]:
        for condition in conditions.CONDITIONS:
            req = conditions.build(condition, b)
            assert b.text in _text(req) and req.letters
        geometry = conditions.build("geometry", b)
        assert b.target_id in geometry.letters.values()


def test_conditions_unknown_fails(bench):
    with pytest.raises(ValueError):
        conditions.build("telepathy", bench[0])


# ---- T5-03 score ----


def test_score_rule_is_exact_without_habit_guessing(bench):
    for b in bench:
        if b.kind in ("color", "marked", "hidden", "hidden-pair"):
            assert score.rule_predict(b) == b.target_id
        if b.kind == "habit":
            majority = "speaker" if b.p_own_frame >= 0.5 else "robot"
            assert (score.rule_predict(b) == b.target_id) == (b.frame == majority)


def test_score_rule_misses_most_free_wording(free_items):
    answers = [score.rule_predict(b) for b in free_items]  # must not crash
    right = sum(a == b.target_id for a, b in zip(answers, free_items, strict=True))
    assert right < len(free_items) / 2
    assert None in answers  # distance wording has no side word for the rule


def test_score_counts_by_kind(bench):
    perfect = score.score(bench, lambda b: b.target_id)
    assert perfect["n"] == len(bench) and perfect["acc"] == 1.0
    assert list(perfect["by_kind"]) == list(items.KINDS)
    assert score.score(bench, lambda b: None)["acc"] == 0.0


def test_score_vlm_predict_takes_the_most_likely_object(bench):
    b = bench[0]

    class Fake:
        def score(self, request):
            ids = list(request.letters.values())
            probs = {i: (0.9 if i == b.target_id else 0.1 / len(ids)) for i in ids}
            return GroundingAnswer(probs, "A", 1.0, "fake")

    assert score.vlm_predict(Fake(), "geometry")(b) == b.target_id

    class Empty:
        def score(self, request):
            return GroundingAnswer({}, None, 1.0, "unavailable")

    assert score.vlm_predict(Empty(), "paired")(b) is None
