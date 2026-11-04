"""T5-24: corrections become training rows in train_lora's format, the held-out front/behind
test can never leak into them, and the promotion gate applies the card's rule."""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from curio_robot.flywheel import gate, serve
from curio_robot.flywheel.deploy import run_episode
from curio_robot.flywheel.log import EventStore, to_rows
from curio_robot.flywheel.request import depth_of, depth_request
from curio_robot.flywheel.scenes import (
    DEPLOY_SEEDS,
    HELDOUT_SEEDS,
    HELDOUT_TEMPLATES,
    TRAIN_LEADS,
    TRAIN_TEMPLATES,
    FBWorld,
    heldout_phrasing,
    heldout_set,
    load_items,
    make_item,
    request_for,
    save_items,
)
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.prompt import PROMPT_VERSION
from curio_robot.types import ObjectMap

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world():
    with FBWorld() as w:
        yield w


@pytest.fixture(scope="module")
def items(world):
    return [make_item(world, DEPLOY_SEEDS[0] + i, f"t{i}") for i in range(6)]


def _fake(pick_target: bool):
    """A grounder that is always right (or always picks the target's look-alike)."""
    def make(item):
        def score(request):
            label = next(o.color for o in item.object_map.objects
                         if o.object_id == item.target_id)
            other = next(o.object_id for o in item.object_map.objects
                         if o.color == label and o.object_id != item.target_id)
            pick = item.target_id if pick_target else other
            probs = {v: (0.9 if v == pick else 0.1 / (len(request.letters) - 1))
                     for v in request.letters.values()}
            return GroundingAnswer(probs, None, 1.0, "fake")
        return score
    return make


def test_seed_ranges_are_disjoint_from_each_other_and_the_old_sets():
    assert not set(DEPLOY_SEEDS) & set(HELDOUT_SEEDS)
    for r in (DEPLOY_SEEDS, HELDOUT_SEEDS):
        assert min(r) > 70000 + 10000 and 0 not in r and not set(range(11, 20)) & set(r)


def test_request_is_the_poc_depth_layout(items):
    sys.path.insert(0, str(ROOT / "scripts" / "poc"))
    scene_base = pytest.importorskip("scene_base")
    for it in items:
        ours = request_for(it)
        theirs = scene_base.depth_request(it.text, it.object_map, it.view)
        assert ours.messages == theirs.messages and ours.letters == theirs.letters


def test_items_are_deterministic_and_the_frame_matters(world, items):
    again = make_item(world, DEPLOY_SEEDS[0], "t0")
    assert again.text == items[0].text and again.target_id == items[0].target_id
    assert np.array_equal(again.view.rgb, items[0].view.rgb)
    for it in items:
        om = it.object_map
        label = next(o.color for o in om.objects if o.object_id == it.target_id)
        pair = [o for o in om.objects if o.color == label]
        assert len(pair) == 2
        for o in pair:  # the speaker's and the robot's front/behind disagree on both
            assert depth_of(o.position, om.human_pose) != depth_of(o.position, om.robot_pose)
        assert it.target_id in request_for(it).letters.values()


def test_heldout_set_split_and_save_roundtrip(world, tmp_path):
    test = heldout_set(world, 8)
    assert [t.pool for t in test] == ["train"] * 4 + ["heldout"] * 4
    assert [t.frame for t in test[:4]] == ["speaker"] * 3 + ["robot"]
    assert all(t.seed in HELDOUT_SEEDS for t in test)
    assert all(heldout_phrasing(t.text) for t in test if t.pool == "heldout")
    save_items(test, tmp_path)
    back = load_items(tmp_path)
    assert [request_for(a).messages for a in back] == [request_for(a).messages for a in test]


def test_heldout_phrasing_detects_only_test_wording():
    for _, _, t in HELDOUT_TEMPLATES:
        assert heldout_phrasing(t.format(lead="Grab", noun="tote") + ".")
    for _, _, t in TRAIN_TEMPLATES:
        for lead in TRAIN_LEADS:
            assert not heldout_phrasing(t.format(lead=lead, noun="bin"))


def test_wrong_pick_logs_a_correction_that_trains_on_the_true_target(items, tmp_path):
    train_lora = _load("train_lora")
    store = EventStore(tmp_path)
    rng = np.random.default_rng(0)
    for k, it in enumerate(items):
        out = run_episode(it, k, _fake(False)(it), "fake", rng, store)
        assert not out.correct and out.event.outcome == "corrected"
        assert out.event.correction and out.event.resolved_by == "retry"
        assert out.event.resolved_target == it.target_id != out.event.pick
    events = store.load()
    assert len(events) == len(items)
    rows = to_rows(events)
    train_lora.check_rows(rows, tmp_path)
    for row, it in zip(rows, items):
        chat, answer = train_lora.to_chat(row)
        assert row["letters"][answer] == it.target_id
        assert row["prompt_version"] == PROMPT_VERSION
        text = chat[0]["content"][1]["text"]
        assert it.text in text and "behind the speaker" in text
        png = (tmp_path / row["image"]).read_bytes()
        url = request_for(it).messages[0]["content"][0]["image_url"]["url"]
        assert png == base64.b64decode(url.split(",", 1)[1])
        Image.open(io.BytesIO(png)).verify()


def test_correct_answers_are_logged_only_at_the_confirm_rate(items, tmp_path):
    rng = np.random.default_rng(0)
    none = run_episode(items[0], 0, _fake(True)(items[0]), "fake", rng, None, 0.0)
    assert none.correct and none.event is None
    some = run_episode(items[0], 1, _fake(True)(items[0]), "fake", rng, None, 1.0)
    assert some.event.outcome == "confirmed" and some.event.correction is None


def test_data_builder_refuses_heldout_seeds_and_phrasing(items, tmp_path):
    data = _load("flywheel_data")
    store = EventStore(tmp_path)
    run_episode(items[0], 0, _fake(False)(items[0]), "fake", np.random.default_rng(0), store)
    event = store.load()[0]
    data.check_events([event])
    with pytest.raises(ValueError):
        data.check_events([replace(event, seed=HELDOUT_SEEDS[0])])
    with pytest.raises(ValueError):
        data.check_events([replace(event, utterance="Grab the tote I have my back to")])


def test_trainer_refuses_to_write_the_served_adapters(tmp_path):
    data = _load("flywheel_data")
    with pytest.raises(SystemExit):
        data.train(tmp_path / "d.jsonl", tmp_path / "o", serve.LORA_DIR / "x.gguf")
    with pytest.raises(SystemExit):
        data.train(tmp_path / "d.jsonl", tmp_path / "o", tmp_path / "curio-ground-r16-v2-f16.gguf")


def _rows(kind_hits: dict[str, list[bool]], prefix: str = "") -> list[dict]:
    return [{"id": f"{prefix}{k}-{i}", "kind": k, "hit": h}
            for k, hits in kind_hits.items() for i, h in enumerate(hits)]


def _results(fb: list[bool], bench: dict[str, list[bool]], deictic: list[bool]) -> dict:
    return {"front-behind": _rows({"front-behind": fb}), "bench5c": _rows(bench),
            "deictic": _rows({"deictic": deictic})}


def test_gate_promotes_only_when_all_three_criteria_hold():
    v2 = _results([True] * 50 + [False] * 50, {"a": [True] * 20, "b": [True] * 20},
                  [True] * 10 + [False] * 10)
    good = _results([True] * 90 + [False] * 10, {"a": [True] * 20, "b": [True] * 20},
                    [True] * 10 + [False] * 10)
    assert gate.decide(v2, good)["promote"]
    weak = _results([True] * 80 + [False] * 20, {"a": [True] * 20, "b": [True] * 20},
                    [True] * 10 + [False] * 10)
    assert not gate.decide(v2, weak)["promote"]  # 0.80 < 0.85
    kind_drop = _results([True] * 90 + [False] * 10, {"a": [True] * 18 + [False] * 2,
                         "b": [True] * 20}, [True] * 10 + [False] * 10)
    verdict = gate.decide(v2, kind_drop)  # a: 1.00 -> 0.90, overall 1.00 -> 0.95
    assert not verdict["promote"] and verdict["reasons"][2].startswith("FAIL")
    deictic_drop = _results([True] * 90 + [False] * 10, {"a": [True] * 20, "b": [True] * 20},
                            [True] * 8 + [False] * 12)
    assert not gate.decide(v2, deictic_drop)["promote"]


def test_gate_intervals():
    lo, hi = gate.wilson(85, 100)
    assert lo == pytest.approx(0.767, abs=2e-3) and hi == pytest.approx(0.907, abs=2e-3)
    assert gate.wilson(0, 0) == (0.0, 1.0)
    lo, hi = gate.paired_bootstrap([False] * 50 + [True] * 50, [True] * 100)
    assert lo > 0.35 and hi <= 0.6
    with pytest.raises(ValueError):
        gate.paired([{"id": "x", "hit": True}], [{"id": "y", "hit": True}])


def test_gpu_lock_is_exclusive_and_releases_only_its_own(tmp_path):
    lock = tmp_path / "gpu.lock"
    with serve.gpu_lock("test", lock, poll_s=0.01):
        assert lock.read_text().startswith("T5-24 test")
    assert not lock.exists()
    with serve.gpu_lock("test", lock, poll_s=0.01):
        lock.write_text("T5-21 someone else")
    assert lock.read_text() == "T5-21 someone else"


def test_depth_request_without_speaker_pose_lists_robot_only(items):
    om = items[0].object_map
    req = depth_request(items[0].text, ObjectMap(om.objects, om.robot_pose, None), items[0].view)
    lines = req.messages[0]["content"][1]["text"].splitlines()
    object_lines = [ln for ln in lines if ln[:2] in {f"{k}:" for k in req.letters}]
    assert object_lines and all("speaker" not in ln and "robot" in ln for ln in object_lines)
