"""The end-to-end harness's ground stage with the fake grounder, and its scoring (T5-09).
The rollout stage needs torch and lerobot and is smoke-tested by hand (tasks/notes/T5-09.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from curio_robot import perspective
from curio_robot.collab.tasks import fetch_scene, true_map

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e2e():
    spec = importlib.util.spec_from_file_location("e2e_eval", ROOT / "scripts" / "e2e_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2e_eval"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def grounded(e2e, tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    result = e2e.ground(out, "base", e2e.FirstLetterGrounder(), n=8, seed=70000)
    return out, result


def test_ground_writes_trials_and_every_setup(e2e, grounded):
    out, result = grounded
    trials = e2e.read_jsonl(out / "trials.jsonl")
    assert [t["kind"] for t in trials] == list(e2e.KINDS) * 2
    assert len({t["seed"] for t in trials}) == 8
    assert all(t["seed"] >= 70000 for t in trials)
    assert result["n"] == 8 and result["errors"] == 0
    choices = e2e.load_choices(out, None)
    assert list(choices) == ["words", "base", "oracle"]
    for t in trials:
        assert choices["oracle"][t["id"]]["choice"] == t["target"]
        assert choices["oracle"][t["id"]]["correct"] is True
        assert choices["words"][t["id"]]["choice"] is None
        base = choices["base"][t["id"]]
        assert base["object_id"] == t["object_ids"][base["choice"]]
        assert base["pixel_ground"] == t["pixels_ground"][base["choice"]]
        assert base["correct"] == (base["choice"] == t["target"])


def test_trials_match_their_words_and_need_the_frame(e2e, grounded):
    out, _ = grounded
    for t in e2e.read_jsonl(out / "trials.jsonl"):
        omap = true_map(e2e.scene_for(t))
        target = omap.objects[t["target"]]
        assert target.color in t["words"]
        assert t["pixels_policy"][t["target"]] is not None
        if t["kind"] == "color":
            assert t["partner"] is None and len(omap.by_color(target.color)) == 1
            continue
        side = "left" if "left" in t["words"] else "right"
        viewer = omap.human_pose if t["frame"] == "speaker" else omap.robot_pose
        cands = omap.by_color(target.color)
        assert perspective.pick(cands, side, viewer).object_id == target.object_id
        other = omap.robot_pose if t["frame"] == "speaker" else omap.human_pose
        assert perspective.pick(cands, side, other).object_id == t["object_ids"][t["partner"]]
        assert ({"my": " my ", "your": " your ", "habit": " the "}[t["kind"]]) in t["words"]
        assert (t["p_own_frame"] is not None) == (t["kind"] == "habit")


def test_second_grounder_reuses_trials_and_rejects_others(e2e, grounded):
    out, _ = grounded
    before = (out / "trials.jsonl").read_text(encoding="utf-8")
    e2e.ground(out, "lora", e2e.FirstLetterGrounder(), n=8, seed=70000)
    assert (out / "trials.jsonl").read_text(encoding="utf-8") == before
    assert list(e2e.load_choices(out, None)) == ["words", "base", "lora", "oracle"]
    with pytest.raises(SystemExit):
        e2e.ground(out, "other", e2e.FirstLetterGrounder(), n=4, seed=71000)
    with pytest.raises(ValueError):
        e2e.ground(out, "oracle", e2e.FirstLetterGrounder(), n=8, seed=70000)


def test_scene_for_rebuilds_the_scene(e2e, grounded):
    out, _ = grounded
    t = e2e.read_jsonl(out / "trials.jsonl")[0]
    scene = fetch_scene(np.random.default_rng(t["seed"]))
    assert e2e.scene_for(t).blocks == scene.blocks
    with pytest.raises(RuntimeError):
        e2e.scene_for({**t, "seed": t["seed"] + 1, "object_ids": ["nope"]})


def test_summarize_rates_and_outcomes(e2e):
    rec = {"target": 0, "partner": 1}
    assert [e2e.outcome(p, rec) for p in (0, 1, 3, None)] == list(e2e.OUTCOMES)
    rows = [
        {"setup": "base", "kind": "my", "grounded": True, "outcome": "target"},
        {"setup": "base", "kind": "my", "grounded": False, "outcome": "partner"},
        {"setup": "base", "kind": "color", "grounded": True, "outcome": "nothing"},
        {"setup": "words", "kind": "my", "grounded": None, "outcome": "other"},
    ]
    s = e2e.summarize(rows, ["words", "base"])
    assert s["base"]["n"] == 3 and s["base"]["grounding_acc"] == pytest.approx(0.667)
    assert s["base"]["by_kind"]["my"]["partner"] == 0.5
    assert s["words"]["grounding_acc"] is None and s["words"]["other"] == 1.0
    table = e2e.format_table(s)
    assert "| words | all | 1 | - |" in table and "| base | color |" in table
