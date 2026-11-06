"""T5-31: the LIBERO-Spatial words adapter, the person's seats, and the gate harness with a
fake policy. CPU only; no LIBERO, LeRobot or grounder needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import libero_eval as ev
import libero_scenes as sc

from curio_robot.grounding.marks import place_marks
from curio_robot.layer import libero_words as lw

# P-style positions (world, m) like LIBERO-Spatial task 0: bowl 1 between the plate and the
# ramekin, bowl 2 next to the ramekin (further to the robot's left and nearer the robot).
TASK0 = {
    "akita_black_bowl_1": np.array([-0.063, 0.202, 0.898]),
    "akita_black_bowl_2": np.array([-0.189, 0.320, 0.898]),
    "plate_1": np.array([0.053, 0.205, 0.898]),
    "glazed_rim_porcelain_ramekin_1": np.array([-0.197, 0.189, 0.898]),
    "cookies_1": np.array([0.058, 0.026, 0.898]),
    "wooden_cabinet_1": np.array([0.035, -0.278, 0.905]),
    "flat_stove_1": np.array([-0.404, -0.146, 0.905]),
}

BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:init
    (On akita_black_bowl_1 main_table_between_plate_ramekin_region)
    (In akita_black_bowl_2 wooden_cabinet_1_top_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)"""


# --- libero_words -----------------------------------------------------------------------------


def test_target_bowl_gets_the_scenes_own_instruction():
    for task in range(len(lw.TASKS)):
        assert lw.libero_words(task, lw.TARGET) == lw.TASKS[task]


def test_look_alike_gets_the_instruction_that_names_its_region():
    assert lw.libero_words(0, "akita_black_bowl_2") == lw.TASKS[1]  # next to the ramekin
    assert lw.libero_words(5, "akita_black_bowl_2") == lw.TASKS[3]  # on the cookie box
    assert lw.libero_words(3, "akita_black_bowl_2") == lw.TASKS[9]  # on the wooden cabinet


def test_the_two_bowls_never_share_an_instruction():
    for task in range(len(lw.TASKS)):
        assert lw.bowl_task(task, lw.BOWLS[0]) != lw.bowl_task(task, lw.BOWLS[1])


def test_every_region_names_exactly_one_task():
    assert sorted(lw.REGION_TASK.values()) == list(range(len(lw.TASKS)))


def test_mapping_is_deterministic_and_rejects_unknowns():
    assert lw.libero_words(7, "akita_black_bowl_2") == lw.libero_words(7, "akita_black_bowl_2")
    with pytest.raises(ValueError):
        lw.libero_words(10, lw.TARGET)
    with pytest.raises(ValueError):
        lw.libero_words(0, "plate_1")


def test_placements_from_bddl_reads_on_and_in():
    assert lw.placements_from_bddl(BDDL) == ("main_table_between_plate_ramekin_region",
                                             "wooden_cabinet_1_top_region")
    with pytest.raises(ValueError):
        lw.placements_from_bddl(BDDL.replace("akita_black_bowl_2", "plate_2"))


# --- seats and the object map -----------------------------------------------------------------


def test_robot_frame_left_is_plus_y():
    robot = sc.robot_pose()
    assert np.allclose(robot.right, [0.0, -1.0, 0.0])


def test_across_seat_flips_left_and_right():
    pose = sc.head_pose("across")
    a, b = TASK0["akita_black_bowl_1"], TASK0["akita_black_bowl_2"]
    # bowl 2 is the robot's left (larger y); from across the table it is the person's right
    assert sc.margin(b, a, sc.robot_pose(), "on my left") > 0
    assert sc.margin(b, a, pose, "on my right") > 0


def test_every_seat_sees_the_task0_bowls():
    for seat in sc.SEATS:
        pose = sc.head_pose(seat)
        assert sc.in_view(TASK0["akita_black_bowl_1"], pose), seat
        assert sc.in_view(TASK0["akita_black_bowl_2"], pose), seat


def test_choose_request_reverses_under_a_robot_frame_reading():
    a, b = TASK0["akita_black_bowl_1"], TASK0["akita_black_bowl_2"]
    for rotate in range(len(sc.SEATS)):
        req = sc.choose_request(a, b, rotate)
        assert req is not None
        pose = sc.head_pose(req.seat)
        assert sc.margin(a, b, pose, req.phrase) >= 0.02  # the person means the target
        assert sc.margin(b, a, sc.robot_pose(), req.phrase) >= 0.02  # robot reads the other
        assert req.words == sc.TEMPLATE.format(phrase=req.phrase)
        assert ("my " in req.words or " me " in req.words) and "robot" not in req.words


def test_choose_request_rotates_seats():
    a, b = TASK0["akita_black_bowl_1"], TASK0["akita_black_bowl_2"]
    seats = {sc.choose_request(a, b, r).seat for r in range(len(sc.SEATS))}
    assert len(seats) > 1


def test_choose_request_none_when_bowls_coincide():
    a = TASK0["akita_black_bowl_1"]
    assert sc.choose_request(a, a + np.array([0.005, 0.005, 0.0]), 0) is None


def test_bowl_ids_follow_position_not_the_target():
    ids = sc.bowl_ids(TASK0)
    assert ids == {"akita_black_bowl_2": "bowl_1", "akita_black_bowl_1": "bowl_2"}
    swapped = {**TASK0, "akita_black_bowl_1": TASK0["akita_black_bowl_2"],
               "akita_black_bowl_2": TASK0["akita_black_bowl_1"]}
    assert sc.bowl_ids(swapped) == {"akita_black_bowl_1": "bowl_1", "akita_black_bowl_2": "bowl_2"}


def test_object_map_and_marks_on_the_head_frame():
    from curio_robot.types import CameraFrame

    pose = sc.head_pose("across")
    om, back = sc.object_map(TASK0, pose)
    assert len(om.objects) == 7
    assert [o.color for o in om.objects if o.object_id.startswith("bowl")] == ["black", "black"]
    assert set(back.values()) == set(TASK0)
    frame = CameraFrame("human", np.zeros((sc.HEAD_H, sc.HEAD_W, 3), np.uint8),
                        sc.head_intrinsics(), pose, 0.0)
    marked = {m.object_id for m in place_marks(om, frame)}
    assert {"bowl_1", "bowl_2"} <= marked


# --- the harness with a fake policy -----------------------------------------------------------


def _trials(n: int = 4) -> list[dict]:
    rows = [(task, 10, ev.trial_row(task, 10, TASK0)) for task in range(n)]
    kept, skipped = ev.select_trials(rows, n)
    assert not skipped
    return kept


class FakePolicy:
    """Succeeds exactly when it is told the scene's own LIBERO-Spatial instruction."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, str]] = []

    def __call__(self, task: int, init: int, seed: int, text: str) -> dict:
        self.calls.append((task, init, seed, text))
        return {"success": text == lw.TASKS[task], "steps": 100}


def test_visit_order_and_selection_skip_and_stop():
    order = list(ev.visit_order(10, 11))
    assert order[:2] == [(0, 10), (1, 10)] and order[10] == (0, 11) and len(order) == 20
    rows = [(0, 10, None), (1, 10, {"id": "a"}), (2, 10, {"id": "b"}), (3, 10, {"id": "c"})]
    kept, skipped = ev.select_trials(rows, 2)
    assert [r["id"] for r in kept] == ["a", "b"] and skipped == ["t0_i10"]


def test_trial_rows_use_init_states_from_ten_and_lerobot_seeds():
    t = ev.trial_row(3, 12, TASK0)
    assert t["init"] == 12 and t["seed"] == ev.BASE_SEED + 12 and t["id"] == "t3_i12"
    assert ev.CURIO_FIRST_INIT == 10 and min(ev.SMOKE_INITS) >= ev.P_EPISODES
    assert max(ev.SMOKE_INITS) < ev.CURIO_FIRST_INIT


def test_task_text_per_condition():
    t = _trials(1)[0]
    right = {"goal_id": "bowl_2", "bowl": lw.TARGET}
    wrong = {"goal_id": "bowl_1", "bowl": "akita_black_bowl_2"}
    none = {"goal_id": None, "bowl": None}
    assert ev.task_text("raw", t, right) == t["words"]
    assert ev.task_text("oracle", t, wrong) == lw.TASKS[0]
    assert ev.task_text("curio", t, right) == lw.TASKS[0]
    assert ev.task_text("curio", t, wrong) == lw.TASKS[1]
    assert ev.task_text("curio", t, none) == t["words"]


def test_curio_bowl_maps_ids_back():
    back = {"bowl_1": "akita_black_bowl_2", "bowl_2": "akita_black_bowl_1", "plate": "plate_1"}
    assert ev.curio_bowl("bowl_2", back) == lw.TARGET
    assert ev.curio_bowl("plate", back) is None and ev.curio_bowl(None, back) is None


def _goals(trials: list[dict], wrong: set[str]) -> dict[str, dict]:
    return {t["id"]: ({"goal_id": "bowl_1", "bowl": "akita_black_bowl_2", "ask": None}
                      if t["id"] in wrong else
                      {"goal_id": "bowl_2", "bowl": lw.TARGET, "ask": None}) for t in trials}


def test_run_trials_with_fake_policy_and_resume():
    trials = _trials(4)
    goals = _goals(trials, {trials[0]["id"]})
    rows: list[dict] = []
    fake = FakePolicy()
    ev.run_trials("fake", trials, goals, fake, set(), rows.append)
    assert len(rows) == 12 and len(fake.calls) == 12
    by = {(r["condition"], r["id"]): r["success"] for r in rows}
    assert all(by[("oracle", t["id"])] for t in trials)
    assert not any(by[("raw", t["id"])] for t in trials)
    assert [by[("curio", t["id"])] for t in trials] == [False, True, True, True]
    assert {c[2] for c in fake.calls} == {ev.BASE_SEED + 10}
    done = {(r["policy"], r["condition"], r["id"]) for r in rows}
    more: list[dict] = []
    ev.run_trials("fake", trials, goals, fake, done, more.append)
    assert more == []


def test_gate_lines_pass_and_fail():
    trials = _trials(10)
    goals = _goals(trials, set())
    rows: list[dict] = []
    ev.run_trials("fake", trials, goals, FakePolicy(), set(), rows.append)
    res = ev.gate({"fake": {"k": 40, "n": 50}}, rows, trials, goals)
    e = res["policies"]["fake"]
    assert res["grounding"]["rate"] == 1.0
    assert e["rates"]["A"]["rate"] == 0.0 and e["rates"]["B"]["rate"] == 1.0
    assert [x["pass"] for x in e["lines"]] == [True, True, True] and e["verdict"] == "PASS"
    assert e["lines"][0]["ci95"] == [1.0, 1.0]
    # two wrong groundings out of ten: B = 0.8 < 0.85 x O and grounding 0.8 < 0.9
    goals2 = _goals(trials, {trials[0]["id"], trials[1]["id"]})
    rows2: list[dict] = []
    ev.run_trials("fake", trials, goals2, FakePolicy(), set(), rows2.append)
    e2 = ev.gate({"fake": {"k": 40, "n": 50}}, rows2, trials, goals2)["policies"]["fake"]
    assert e2["verdict"] == "FAIL (line 2, 3)" and e2["rates"]["B"]["rate"] == 0.8
    md = ev.markdown({**ev.gate({"fake": {"k": 40, "n": 50}}, rows2, trials, goals2),
                      "adapter": "r16-v4", "seats": {}, "phrases": {}})
    assert "FAIL (line 2, 3)" in md and "| B curio | 8/10" in md


def test_gate_one_wrong_grounding_in_ten_still_passes():
    trials = _trials(10)
    goals = _goals(trials, {trials[3]["id"]})
    rows: list[dict] = []
    ev.run_trials("fake", trials, goals, FakePolicy(), set(), rows.append)
    e = ev.gate({"fake": {"k": 30, "n": 50}}, rows, trials, goals)["policies"]["fake"]
    assert e["verdict"] == "PASS" and e["lines"][2]["value"] == 0.9


def test_gate_skips_policy_that_fails_p():
    trials = _trials(2)
    goals = _goals(trials, set())
    res = ev.gate({"fake": {"k": 29, "n": 50, "diagnosis": "zeros"}}, [], trials, goals)
    e = res["policies"]["fake"]
    assert e["P"]["pass"] is False and e["verdict"] == "NOT RUN (P failed)"
    assert "rates" not in e and e["diagnosis"] == "zeros"
    assert ev.gate({"fake": {"k": 30, "n": 50}}, [], trials, goals)["policies"]["fake"][
        "P"]["pass"]


def test_hits_needs_every_trial():
    trials = _trials(2)
    rows = [{"policy": "fake", "condition": "raw", "id": trials[0]["id"], "success": True}]
    with pytest.raises(ValueError):
        ev.hits(rows, "fake", "raw", [t["id"] for t in trials])
