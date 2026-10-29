"""T3-01: sessions survive a round trip through dicts and JSON lines."""

from __future__ import annotations

import json

from curio_robot.collab.session import (
    Session,
    Step,
    load_sessions,
    save_sessions,
    session_from_dict,
    session_to_dict,
)


def make_session(session_id: str = "s1", condition: str = "view-on") -> Session:
    steps = (
        Step(1, "request", "Hand me the blue one", "blue_1", "blue_1", False,
             {"speech_start": 10.0, "speech_end": 11.5, "arrived": 13.25}),
        Step(2, "correction", "no, the other one", "red_2", "red_2", False,
             {"speech_start": 14.0}),
        Step(3, "answer", "mine", None, None, False, {}),
    )
    return Session(session_id, "mixed", condition, "fetch", 7, 10.0, 20.5,
                   ("blue_1", "red_2"), ("blue_1",), steps)


def test_round_trip():
    s = make_session()
    assert session_from_dict(session_to_dict(s)) == s


def test_dict_is_json_safe_with_lists():
    d = session_to_dict(make_session())
    json.dumps(d)
    assert d["session_id"] == "s1" and d["condition"] == "view-on" and d["seed"] == 7
    assert d["goals"] == ["blue_1", "red_2"] and d["done"] == ["blue_1"]
    assert isinstance(d["steps"], list) and d["steps"][1]["kind"] == "correction"
    assert d["steps"][0]["marks"] == {"speech_start": 10.0, "speech_end": 11.5, "arrived": 13.25}
    assert d["steps"][2]["goal_id"] is None and d["steps"][2]["asked"] is False


def test_from_dict_gives_tuples_steps_and_floats():
    d = session_to_dict(make_session())
    d["steps"][0]["marks"]["arrived"] = 13  # an int from someone else's JSON
    s = session_from_dict(d)
    assert isinstance(s.goals, tuple) and isinstance(s.done, tuple)
    assert isinstance(s.steps, tuple) and all(isinstance(st, Step) for st in s.steps)
    assert isinstance(s.steps[0].marks["arrived"], float)


def test_save_and_load_replaces_the_file(tmp_path):
    path = tmp_path / "deep" / "sessions.jsonl"
    sessions = [make_session("a"), make_session("b", "view-off")]
    save_sessions(sessions, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert load_sessions(path) == sessions
    save_sessions(sessions[:1], path)
    assert load_sessions(path) == sessions[:1]


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    save_sessions([make_session()], path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_sessions(path)) == 1
