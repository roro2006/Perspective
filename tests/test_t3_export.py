"""T3-08: sessions as episodes on disk."""

from __future__ import annotations

import json

import pytest
from test_t3_session import make_session

from curio_robot.collab.export import FORMAT, export_episodes


def lines(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_layout_and_counts(tmp_path):
    out = tmp_path / "ds"
    result = export_episodes([make_session("a"), make_session("b", "view-off")], out)
    assert result == {"episodes": 2, "steps": 6}
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info == {"format": FORMAT, "total_episodes": 2, "total_steps": 6,
                    "conditions": ["view-off", "view-on"]}
    eps = lines(out / "meta" / "episodes.jsonl")
    assert [e["episode_index"] for e in eps] == [0, 1]
    assert eps[0]["session_id"] == "a" and eps[1]["condition"] == "view-off"
    assert eps[0]["length"] == 3 and eps[0]["goals"] == ["blue_1", "red_2"]
    assert eps[0]["done"] == ["blue_1"] and eps[0]["success"] == pytest.approx(0.5)
    assert eps[0]["person"] == "mixed" and eps[0]["task"] == "fetch" and eps[0]["seed"] == 7
    assert (out / "data" / "episode_000001.jsonl").exists()


def test_steps_are_relative_and_scored(tmp_path):
    export_episodes([make_session()], tmp_path)
    steps = lines(tmp_path / "data" / "episode_000000.jsonl")
    assert [s["step_index"] for s in steps] == [0, 1, 2]
    assert steps[0]["episode_index"] == 0 and steps[0]["utterance"] == 1
    assert steps[0]["t"] == {"speech_start": 0.0, "speech_end": 1.5, "arrived": 3.25}
    assert steps[0]["correct"] is True and steps[0]["kind"] == "request"
    assert steps[1]["correct"] is True and steps[1]["t"] == {"speech_start": 4.0}
    assert steps[2]["correct"] is False and steps[2]["chosen_id"] is None
    assert steps[2]["t"] == {} and steps[2]["asked"] is False


def test_refuses_to_overwrite_unless_asked(tmp_path):
    export_episodes([make_session()], tmp_path)
    with pytest.raises(FileExistsError):
        export_episodes([make_session()], tmp_path)
    assert export_episodes([make_session(), make_session("b")], tmp_path, overwrite=True) == {
        "episodes": 2, "steps": 6}
