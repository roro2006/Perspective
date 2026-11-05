"""T5-26: v4's data is v2's (sha-checked) plus the flywheel rows plus deictic replay to at
least a quarter of all rows, never touching a test seed or a held-out id; the gate's new
line 1 (remove half of v2's errors, at least 0.90) sits beside T5-24's unchanged rule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from curio_robot.flywheel import gate
from curio_robot.flywheel.deploy import run_episode
from curio_robot.flywheel.log import EventStore
from curio_robot.flywheel.scenes import DEPLOY_SEEDS, HELDOUT_SEEDS, FBWorld, make_item
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.sim.grounding_data import load_dataset, write_dataset

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def data():
    return _load("flywheel_data")


@pytest.fixture(scope="module")
def world():
    with FBWorld() as w:
        yield w


def _wrong(item):
    """A grounder that always picks the target's look-alike (so the event is a correction)."""
    def score(request):
        label = next(o.color for o in item.object_map.objects if o.object_id == item.target_id)
        pick = next(o.object_id for o in item.object_map.objects
                    if o.color == label and o.object_id != item.target_id)
        return GroundingAnswer({v: float(v == pick) for v in request.letters.values()},
                               None, 1.0, "fake")
    return score


@pytest.fixture(scope="module")
def sources(tmp_path_factory, world):
    """A small v2 set, an event store, a target-2 set and its held-out ids."""
    root = tmp_path_factory.mktemp("v4src")
    v2 = write_dataset(root / "v2", 8, seed=5, kinds=("color",))
    store = EventStore(root / "events")
    for i in range(2):
        item = make_item(world, DEPLOY_SEEDS[0] + i, f"t{i}")
        run_episode(item, i, _wrong(item), "fake", np.random.default_rng(i), store)
    t2 = write_dataset(root / "t2", 6, seed=0, kinds=("deictic", "color"))
    deictic = [r["id"] for r in load_dataset(t2) if r["kind"] == "deictic"]
    eval_ids = root / "eval_ids.json"
    eval_ids.write_text(json.dumps([deictic[0]]), encoding="utf-8")
    sha = hashlib.sha256(v2.read_bytes()).hexdigest()
    return {"root": root, "v2": v2.parent, "events": root / "events", "t2": t2,
            "eval_ids": eval_ids, "held": deictic[0], "sha": sha}


def _build(data, s, out: str, **kw):
    return data.build_v4(s["v2"], s["events"], s["root"] / f"{out}-fresh", s["root"] / out,
                         kw.pop("sha", s["sha"]), target2=s["t2"], eval_ids=s["eval_ids"], **kw)


def test_v4_data_is_v2_plus_flywheel_plus_deictic_replay(data, sources):
    meta = _build(data, sources, "v4")
    rows = load_dataset(sources["root"] / "v4" / "data.jsonl")
    v2_rows = load_dataset(sources["v2"] / "data.jsonl")
    assert rows[:len(v2_rows)] == v2_rows  # v2's rows first, unchanged
    assert meta["flywheel_rows"] == 2 and meta["replay_rows"] == 2  # 3 deictic, 1 held out
    ids = [r["id"] for r in rows]
    assert sources["held"] not in ids and len(ids) == len(set(ids))
    n_deictic = sum(r["kind"] == "deictic" for r in rows)
    assert n_deictic / len(rows) >= 0.25 and meta["deictic_share"] >= 0.25
    # as many fresh rows as needed and no more: one fewer would fall under a quarter
    assert (n_deictic - 1) / (len(rows) - 1) < 0.25
    fresh = [i for i in ids if i.startswith(f"{gate.DEICTIC_TRAIN_SEED}-")]
    assert len(fresh) == meta["fresh_rows"] > 0
    for r in rows:
        assert (sources["root"] / "v4" / r["image"]).is_file()


def test_v4_builder_checks_v2_sha(data, sources):
    with pytest.raises(SystemExit):
        _build(data, sources, "badsha", sha="0" * 64)


def test_v4_builder_refuses_the_test_seeds(data, sources):
    with pytest.raises(ValueError):
        _build(data, sources, "seed41", seed=gate.DEICTIC_TEST_SEED)
    for seed in (gate.DEICTIC_TEST_SEED, HELDOUT_SEEDS[0], 624999, gate.FB2_SEEDS[0], 625999):
        with pytest.raises(ValueError):
            data.check_deictic_seed(seed)
    data.check_deictic_seed(gate.DEICTIC_TRAIN_SEED)
    event = EventStore(sources["events"]).load()[0]
    data.check_events([event])
    for seed in (624000, 624999, 625000, 625999):
        with pytest.raises(ValueError):
            data.check_events([replace(event, seed=seed)])


def test_v4_row_check_refuses_leaks(data):
    ok = [{"id": f"r{i}", "kind": "deictic" if i < 3 else "color"} for i in range(8)]
    data.check_v4_rows(ok, {"held"}, set(), 0.25)
    with pytest.raises(SystemExit):  # a target-2 held-out id that is not v2's
        data.check_v4_rows([*ok, {"id": "held", "kind": "deictic"}], {"held"}, set(), 0.25)
    with pytest.raises(SystemExit):  # a row of the deictic test seed
        data.check_v4_rows([*ok, {"id": "41-00000", "kind": "deictic"}], set(), set(), 0.25)
    with pytest.raises(SystemExit):  # too little deictic
        data.check_v4_rows(ok[2:], set(), set(), 0.25)
    with pytest.raises(SystemExit):
        data.check_v4_rows([*ok, ok[0]], set(), set(), 0.25)


def test_fresh_rows_needed():
    data = _load("flywheel_data")
    assert data.fresh_needed(5173, 969) == 433  # the real v4 counts
    assert data.fresh_needed(100, 0) == 34  # 34/134 >= 0.25 > 33/133
    assert data.fresh_needed(8, 2) == 0 and data.fresh_needed(8, 5) == 0


def test_fresh_front_behind_set_is_built_like_t524s(world):
    fg = _load("flywheel_gate")
    assert not set(gate.FB2_SEEDS) & set(HELDOUT_SEEDS)
    assert not set(gate.FB2_SEEDS) & set(DEPLOY_SEEDS)
    items = fg.fb2_set(world, 8)
    assert [it.seed for it in items] == list(range(625000, 625008))
    assert [it.pool for it in items] == ["train"] * 4 + ["heldout"] * 4
    assert [it.frame for it in items] == ["speaker"] * 3 + ["robot"] + ["speaker"] * 3 + ["robot"]


@pytest.mark.parametrize("old,new,ok", [
    (0.80, 0.90, True), (0.80, 0.895, False),  # half of 0.20 errors = 0.90 exactly
    (160 / 200, 180 / 200, True), (160 / 200, 179 / 200, False),
    (0.90, 0.95, True), (0.90, 0.945, False),  # above the floor, the half rule binds
    (0.60, 0.90, True), (0.60, 0.85, False),  # below it, the 0.90 floor binds
    (0.78, 0.89, False), (0.78, 0.90, True),
    (1.00, 1.00, True), (1.00, 0.995, False),
])
def test_line_1_removes_half_of_v2s_errors(old, new, ok):
    assert gate.removes_half_the_errors(old, new) is ok


def _suite(kind_hits: dict[str, list[bool]]) -> list[dict]:
    return [{"id": f"{k}-{i}", "kind": k, "hit": h}
            for k, hits in kind_hits.items() for i, h in enumerate(hits)]


def _res(fb2: int, bench: dict[str, int], d1: int, d2: int) -> dict:
    """Hits out of 100 on b2, 20 per bench kind, 20 on d1 and d2."""
    def hits(k, n):
        return [True] * k + [False] * (n - k)
    return {"front-behind-2": _suite({"front-behind": hits(fb2, 100)}),
            "bench5c": _suite({k: hits(v, 20) for k, v in bench.items()}),
            "deictic": _suite({"deictic": hits(d1, 20)}),
            "deictic-2": _suite({"deictic": hits(d2, 20)})}


def test_decide_v4_applies_all_three_lines_against_v2():
    v2 = _res(80, {"a": 20, "b": 18}, 10, 12)
    v3 = _res(99, {"a": 20, "b": 18}, 2, 2)  # v3 is reported only
    good = _res(90, {"a": 20, "b": 18}, 9, 11)  # d1, d2 fall 0.05: allowed
    verdict = gate.decide_v4(v2, v3, good)
    assert verdict["promote"], verdict["reasons"]
    assert verdict["rows"][0]["acc"] == {"v2": 0.8, "v3": 0.99, "v4": 0.9}
    assert not gate.decide_v4(v2, v3, _res(89, {"a": 20, "b": 18}, 10, 12))["promote"]
    overall = gate.decide_v4(v2, v3, _res(95, {"a": 19, "b": 18}, 10, 12))  # -0.025 overall
    assert not overall["promote"] and overall["reasons"][1].startswith("FAIL")
    d2_drop = gate.decide_v4(v2, v3, _res(95, {"a": 20, "b": 18}, 10, 10))  # d2 -0.10
    assert not d2_drop["promote"] and d2_drop["reasons"][2].startswith("FAIL")
    d1_drop = gate.decide_v4(v2, v3, _res(95, {"a": 20, "b": 18}, 8, 12))
    assert not d1_drop["promote"]
    table = gate.format_table3(verdict["rows"])
    assert table.count("\n") == 1 + len(verdict["rows"]) and "deictic d2" in table


def test_t524_rule_is_still_available_and_unchanged():
    assert (gate.FB_MIN, gate.FB_GAIN, gate.OVERALL_DROP, gate.KIND_DROP) == (
        0.85, 0.25, 0.01, 0.05)
    v2 = {"front-behind": _suite({"fb": [True] * 78 + [False] * 22}),
          "bench5c": _suite({"a": [True] * 20}), "deictic": _suite({"d": [True] * 10})}
    v3 = {**v2, "front-behind": _suite({"fb": [True] * 99 + [False]})}
    assert not gate.decide(v2, v3)["promote"]  # +0.21 < +0.25, as T5-24 found
