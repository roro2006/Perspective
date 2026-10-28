"""Tests for the ground-eval feature: the eval runs offline with a fake grounder."""

from __future__ import annotations

import json
import pathlib

import pytest

from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.sim import cli
from curio_robot.sim.ground_eval import run_ground_eval
from curio_robot.sim.grounding_data import load_dataset, write_dataset


def wrong_letter(row: dict) -> str:
    """The first sorted letter in row["letters"] that is not the answer."""
    for letter in sorted(row["letters"]):
        if letter != row["answer"]:
            return letter
    raise ValueError("row has no wrong letter to give")


class FakeGrounder:
    """Answers in call order (ground_eval scores rows in dataset order).

    Rows whose kind is in wrong_kinds get a wrong letter; the rest are correct."""

    def __init__(self, wrong_kinds: set[str], rows: list[dict]) -> None:
        self._wrong_kinds = wrong_kinds
        self._rows = rows
        self.requests: list[GroundingRequest] = []

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        # The stored assistant answer must be dropped and the image embedded.
        assert request.messages[-1]["role"] == "user"
        image_url = request.messages[0]["content"][0]["image_url"]["url"]
        assert image_url.startswith("data:image/png;base64,")

        row = self._rows[len(self.requests)]
        self.requests.append(request)
        if row["kind"] in self._wrong_kinds:
            letter = wrong_letter(row)
        else:
            letter = row["answer"]
        return GroundingAnswer(probs={}, letter=letter, latency_ms=10.0, source="fake")


def test_rebuilds_requests_with_data_urls(tmp_path: pathlib.Path):
    # Arrange
    data_path = write_dataset(tmp_path / "grounding", n=6)
    rows = load_dataset(data_path)
    grounder = FakeGrounder(set(), rows)

    # Act
    run_ground_eval(data_path, grounder)

    # Assert
    assert len(grounder.requests) == 6
    for row, request in zip(rows, grounder.requests):
        assert request.letters == row["letters"]


def test_scores_per_kind_when_color_is_wrong(tmp_path: pathlib.Path):
    # Arrange
    data_path = write_dataset(tmp_path / "grounding", n=6)
    rows = load_dataset(data_path)
    grounder = FakeGrounder({"color"}, rows)

    # Act
    result, verdict = run_ground_eval(data_path, grounder)

    # Assert
    assert result["n"] == 6
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["by_kind"]["relation"] == {"n": 2, "accuracy": 1.0}
    assert result["by_kind"]["deictic"] == {"n": 2, "accuracy": 1.0}
    assert result["by_kind"]["color"] == {"n": 2, "accuracy": 0.0}
    color_ids = [row["id"] for row in rows if row["kind"] == "color"]
    assert result["failures"] == color_ids
    assert verdict is None


def test_ids_filter_keeps_dataset_order_and_rejects_unknown_ids(tmp_path: pathlib.Path):
    # Arrange
    data_path = write_dataset(tmp_path / "grounding", n=6)
    rows = load_dataset(data_path)
    ids = [rows[3]["id"], rows[0]["id"]]
    scored_rows = [rows[0], rows[3]]  # dataset order, which is the order they are scored in

    # Act
    result, _ = run_ground_eval(data_path, FakeGrounder(set(), scored_rows), ids=ids)

    # Assert
    assert result["n"] == 2
    assert result["failures"] == []
    with pytest.raises(ValueError, match="ids not in dataset"):
        run_ground_eval(data_path, FakeGrounder(set(), rows), ids=["no-such-id"])


def test_worse_candidate_fails_gate_against_better_current(tmp_path: pathlib.Path):
    # Arrange
    data_path = write_dataset(tmp_path / "grounding", n=6)
    rows = load_dataset(data_path)
    current, _ = run_ground_eval(data_path, FakeGrounder(set(), rows))

    # Act
    _, worse_verdict = run_ground_eval(
        data_path, FakeGrounder({"color"}, rows), current=current, max_p95_ms=800.0
    )
    _, same_verdict = run_ground_eval(
        data_path, FakeGrounder(set(), rows), current=current, max_p95_ms=800.0
    )

    # Assert
    assert worse_verdict == (False, "worse on color: 0.000 vs 1.000")
    assert same_verdict == (True, "accuracy 1.000 vs 1.000")


def test_cli_ground_eval_writes_json_and_gates(tmp_path: pathlib.Path, monkeypatch):
    # Arrange
    data_path = write_dataset(tmp_path / "grounding", n=6)
    rows = load_dataset(data_path)
    good_out = tmp_path / "good.json"
    bad_out = tmp_path / "bad.json"
    base_argv = ["ground-eval", "--data", str(data_path), "--url", "http://unused.invalid/v1"]

    # Act: the good model runs first and writes the current result
    monkeypatch.setattr(cli, "VllmGrounder", lambda *a, **k: FakeGrounder(set(), rows))
    cli.main(base_argv + ["--out", str(good_out)])

    # Act: the color-wrong model is gated against it and must exit 1
    monkeypatch.setattr(cli, "VllmGrounder", lambda *a, **k: FakeGrounder({"color"}, rows))
    with pytest.raises(SystemExit) as exc:
        cli.main(base_argv + ["--out", str(bad_out), "--current", str(good_out)])

    # Assert
    assert exc.value.code == 1
    good = json.loads(good_out.read_text())
    assert good["n"] == 6
    assert good["accuracy"] == 1.0
    bad = json.loads(bad_out.read_text())
    assert bad["accuracy"] == pytest.approx(4 / 6)
    assert bad["by_kind"]["color"]["accuracy"] == 0.0
