"""T2-09: the LoRA script's dry run and its pure helpers, all without torch.
The training loop itself only runs on the GPU machine (T2-10)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from curio_robot.grounding.prompt import PROMPT_VERSION
from curio_robot.sim.grounding_data import load_dataset, write_dataset

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_lora.py"


def _train_module():
    spec = importlib.util.spec_from_file_location("train_lora", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_lora"] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("grounding"), n=6)


def _run(data: Path, out: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--data", str(data), "--model", "dummy",
           "--out", str(out), "--dry-run"]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)


def test_dry_run_prints_counts(dataset, tmp_path):
    proc = _run(dataset, tmp_path / "lora")
    assert proc.returncode == 0, proc.stderr
    rows = load_dataset(dataset)
    assert "examples: 6" in proc.stdout
    assert "kinds: color=2, deictic=2, relation=2" in proc.stdout
    answers = sorted({r["answer"] for r in rows})
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("answers: "))
    assert sum(int(part.split("=")[1]) for part in line[9:].split(", ")) == 6
    for letter in answers:
        assert f"{letter}=" in line
    assert not (tmp_path / "lora").exists()  # dry run writes nothing


def test_dry_run_fails_on_missing_image(dataset, tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(dataset.read_text())  # same rows, but images/ is not next to it
    proc = _run(data, tmp_path / "lora")
    assert proc.returncode != 0
    assert "missing image" in proc.stderr
    assert "images/" in proc.stderr


def test_dry_run_does_not_import_torch(dataset, tmp_path):
    tl = _train_module()
    assert tl.main(["--data", str(dataset), "--model", "x", "--out", str(tmp_path / "o"),
                    "--dry-run"]) == 0
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    assert "peft" not in sys.modules


def test_split_is_stable_and_about_ten_percent():
    tl = _train_module()
    ids = [f"0-{i:05d}" for i in range(2000)]
    held = [i for i in ids if tl.is_eval_id(i)]
    assert 140 <= len(held) <= 260
    assert held == [i for i in reversed(ids) if tl.is_eval_id(i)][::-1]
    rows = [{"id": i} for i in ids]
    train, ev = tl.split_rows(rows)
    assert len(train) + len(ev) == len(rows)
    assert {r["id"] for r in ev} == set(held)


def test_to_chat_and_answer_mask(dataset):
    tl = _train_module()
    row = load_dataset(dataset)[0]
    chat, answer = tl.to_chat(row)
    assert answer == row["answer"]
    assert [m["role"] for m in chat] == ["user"]
    assert chat[0]["content"][0] == {"type": "image"}
    assert chat[0]["content"][1]["text"].endswith("single letter of the object they mean.")
    labels = tl.answer_labels(4, [42])
    assert labels == [tl.IGNORE_INDEX] * 4 + [42]
    with pytest.raises(ValueError):
        tl.answer_labels(4, [1, 2])


def test_train_config_attributes_the_run(dataset):
    tl = _train_module()
    cfg = tl.build_config("qwen-vl", dataset, tl.TrainConfig(), 5, 1)
    assert cfg["prompt_version"] == PROMPT_VERSION
    assert len(cfg["data_sha256"]) == 64
    hp = cfg["hyperparameters"]
    assert hp["lora_r"] == 16 and hp["batch_size"] == 1 and hp["bf16"]
    assert set(hp["lora_targets"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    json.dumps(cfg)
