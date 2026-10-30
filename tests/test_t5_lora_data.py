"""T5-06: benchmark-style LoRA data is the geometry request, in the format
train_lora.py reads, and never comes from the benchmark or e2e seeds."""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import pytest

from curio_robot.bench import items as bench_items
from curio_robot.bench.conditions import build
from curio_robot.grounding.prompt import PROMPT_VERSION
from curio_robot.sim.grounding_data import load_dataset
from curio_robot.sim.world import SimWorld

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_seed_spec_refuses_benchmark_and_e2e_seeds():
    data = _load("bench_lora_data")
    assert data.parse_seeds("11-13,15") == [11, 12, 13, 15]
    with pytest.raises(ValueError):
        data.parse_seeds("0-3")
    with pytest.raises(ValueError):
        data.parse_seeds("70000")


def test_rows_are_the_geometry_request_and_pass_the_trainer_checks(tmp_path):
    data = _load("bench_lora_data")
    train = _load("train_lora")
    mix = {"free": 1, "hidden-pair": 1, "color": 1}
    images = tmp_path / "images"
    images.mkdir()
    rows, skipped = data.bench_rows([11], mix, images)
    assert skipped == 0 and [r["kind"] for r in rows] == ["free", "hidden-pair", "color"]
    train.check_rows(rows, tmp_path)  # images exist, prompt version current

    # The same draw, rebuilt: text, letters and image bytes match the served request.
    import numpy as np

    with SimWorld(320, 240) as world:
        item = bench_items.make_item(world, np.random.default_rng(11), "free", "free-0000")
    request = build("geometry", item)
    row = rows[0]
    assert row["text"] == item.text and row["letters"] == request.letters
    assert row["letters"][row["answer"]] == item.target_id
    assert row["prompt_version"] == PROMPT_VERSION
    url = request.messages[0]["content"][0]["image_url"]["url"]
    png = base64.b64decode(url.split(",", 1)[1])
    assert (tmp_path / row["image"]).read_bytes() == png
    chat, answer = train.to_chat(row)
    assert answer == row["answer"] and chat[0]["content"][1]["text"] == (
        request.messages[0]["content"][1]["text"]
    )


def test_main_writes_a_loadable_dataset(tmp_path):
    data = _load("bench_lora_data")
    assert data.main([str(tmp_path), "--seeds", "12", "--scale", "0.01"]) == 0
    rows = load_dataset(tmp_path / "data.jsonl")
    assert len(rows) == len(data.MIX)  # one per kind at this scale
    assert len({r["id"] for r in rows}) == len(rows)
