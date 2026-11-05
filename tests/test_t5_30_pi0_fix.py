"""T5-30: the pi0 train/inference mismatch. CPU only, no torch.

T5-28 trained on batches from smolvla_train's loader, which pins host memory. On this machine
the preprocessor's non-blocking copy of a pinned batch sometimes handed the model a corrupted
action chunk, so pi0 learned targets that inference never sees. These tests fail on the T5-28
code: pi0's loaders must not pin memory, and the T5-30 check must judge precondition P as the
card wrote it."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import pi0_check as C
import pi0_data as D


def test_pi0_loaders_never_pin_host_memory():
    for workers, shuffle in ((4, True), (0, False)):
        opts = D.loader_options(16, workers, shuffle)
        assert opts["pin_memory"] is False
        assert opts["num_workers"] == workers and opts["shuffle"] is shuffle
        assert opts["drop_last"] is shuffle  # training drops the ragged batch, eval keeps it


def _tree(name: str) -> ast.Module:
    return ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))


def test_pi0_train_builds_its_loaders_from_the_unpinned_options():
    tree = _tree("pi0_train.py")
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and n.module == "smolvla_train" for a in n.names}
    assert "_loader" not in imported, "smolvla_train._loader pins memory"
    loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_loader")
    calls = {n.func.id for n in ast.walk(loader) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)}
    assert "loader_options" in calls
    keywords = {k.arg for n in ast.walk(loader) if isinstance(n, ast.Call) for k in n.keywords}
    assert "pin_memory" not in keywords


def test_pi0_check_scores_with_pi0_trains_loader():
    tree = _tree("pi0_check.py")
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             and n.module == "pi0_train" for a in n.names}
    assert "_loader" in names


def test_precondition_passes_inside_both_tolerances():
    res = C.precondition(1.05, 0.968, np.array([[0.1, -0.05], [0.12, 0.0]]))
    assert res["loss_ok"] and res["joint_ok"] and res["pass"]
    assert res["relative_gap"] == pytest.approx(0.0847, abs=1e-4)


@pytest.mark.parametrize("loss, errors", [
    (1.12, [0.01, 0.02]),  # 15.7% above the logged loss
    (0.80, [0.01, 0.02]),  # 17.4% below it: "within 15%" cuts both ways
    (0.968, [1.4, 1.5, 0.1]),  # T5-28's failure: joint targets near 0 where demos say 1.5 rad
])
def test_precondition_fails_outside_either_tolerance(loss, errors):
    assert not C.precondition(loss, 0.968, np.array(errors))["pass"]


def test_logged_eval_loss_reads_the_step(tmp_path):
    log = tmp_path / "train_log.jsonl"
    rows = [{"step": 22000, "loss": 0.002}, {"step": 20000, "eval_loss": 1.028},
            {"step": 22000, "eval_loss": 0.9678}]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert C.logged_eval_loss(log, 22000) == 0.9678
    with pytest.raises(KeyError):
        C.logged_eval_loss(log, 4000)
