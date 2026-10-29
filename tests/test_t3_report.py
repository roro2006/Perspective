"""T3-09: the scoreboard table on a hand-made summary, and the collab subcommand's parsing."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from curio_robot.collab.metrics import Fluency
from curio_robot.collab.report import format_table
from curio_robot.sim import cli

SUMMARY = {
    "view-off": {
        "completion_s": 12.5,
        "success": 0.5,
        "turns": 3.0,
        "corrections": 1.0,
        "questions": 0.0,
        "human_idle_s": 2.25,
        "robot_idle_frac": 0.25,
        "concurrent_frac": 0.1,
        "functional_delay_s": None,
        "reply_s": 0.5,
        "n": 2.0,
    },
    "view-on": {
        "completion_s": 10.0,
        "success": 0.8,
        "turns": 2.0,
        "corrections": 0.5,
        "questions": 0.2,
        "human_idle_s": 1.5,
        "robot_idle_frac": 0.1,
        "concurrent_frac": 0.3,
        "functional_delay_s": 0.4,
        "reply_s": 0.3,
        "n": 3.0,
    },
}


def _lines() -> list[str]:
    return format_table(SUMMARY).split("\n")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def test_header_line():
    # Arrange
    lines = _lines()

    # Act
    header = lines[0]

    # Assert
    assert header == "| metric | view-off | view-on | better |"


def test_separator_line():
    # Arrange
    lines = _lines()

    # Act
    separator = lines[1]

    # Assert
    assert separator.startswith("| ---")


def test_metric_order():
    # Arrange
    lines = _lines()

    # Act
    metric_names = [line.split("|")[1].strip() for line in lines[2:]]

    # Assert
    assert metric_names == [f.name for f in fields(Fluency)] + ["n"]


def test_value_formatting_two_decimals():
    # Arrange
    lines = _lines()

    # Act
    completion_row = lines[2]  # completion_s is the first field

    # Assert
    assert "| 12.50 |" in completion_row


def test_none_renders_as_dash():
    # Arrange
    lines = _lines()

    # Act
    # functional_delay_s is the 9th field: row index 2 + 8
    cells = _cells(lines[2 + 8])

    # Assert
    assert cells[1] == "functional_delay_s"
    assert cells[2] == "-"
    assert cells[3] == "0.40"


def test_n_renders_as_integer_with_no_better_direction():
    # Arrange
    lines = _lines()

    # Act
    cells = _cells(lines[-1])

    # Assert
    assert cells[1] == "n"
    assert cells[2] == "2"
    assert cells[3] == "3"
    assert cells[4] == ""


def test_better_column_directions():
    # Arrange
    lines = _lines()

    # Act
    better = {_cells(line)[1]: _cells(line)[-2] for line in lines[2:-1]}

    # Assert
    assert better["success"] == "higher"
    assert better["concurrent_frac"] == "higher"
    assert better["completion_s"] == "lower"
    assert better["reply_s"] == "lower"


def test_empty_summary_says_no_sessions():
    # Arrange
    empty: dict = {}

    # Act
    result = format_table(empty)

    # Assert
    assert result == "(no sessions)"


def test_collab_subcommand_parses_with_defaults(monkeypatch):
    # Arrange
    calls = []
    monkeypatch.setattr(cli, "_collab", calls.append)

    # Act
    cli.main(["collab", "--tasks", "1"])

    # Assert
    assert len(calls) == 1
    args = calls[0]
    assert args.tasks == 1
    assert args.goals == 4
    assert args.user == "mixed"
    assert args.conditions == "view-off,view-on"
    assert args.deictic == 0.0
    assert args.occlude is False
    assert args.out == Path("out/collab")
