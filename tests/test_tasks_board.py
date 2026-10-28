"""The task board stays consistent: no file owned by two cards that can run in
parallel, no missing reads, every open card's context pack within budget."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tasks_module():
    spec = importlib.util.spec_from_file_location("tasks_board", ROOT / "scripts" / "tasks.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tasks_board"] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def test_board_checks_clean(capsys):
    tasks = _tasks_module()
    assert tasks.cmd_check(tasks.load_cards()) == 0, capsys.readouterr().out
    assert "0 warnings" in capsys.readouterr().out


def test_every_open_card_has_an_accept_test_or_command():
    tasks = _tasks_module()
    for card in tasks.load_cards():
        assert card.accept, card.id
        for word in card.accept.split():
            if word.startswith("tests/") and card.status != "done":
                assert (ROOT / word).exists() or word in card.owns, (card.id, word)
