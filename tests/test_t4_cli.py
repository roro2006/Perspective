"""T4-07: curio-sim critic."""

from __future__ import annotations

import json

from curio_robot.critic.cli import format_table, phase1
from curio_robot.sim.cli import main


def test_phase1_writes_the_model_and_summary(tmp_path):
    summary = phase1(40, 20, 20, epochs=2, out=tmp_path)
    assert (tmp_path / "model.npz").exists()
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    for key in ("critic", "critic_slow", "baseline", "gate"):
        assert key in saved
    assert set(saved["critic"]["by_family"]) == {"attentive", "slow", "distracted", "silent"}
    table = format_table(summary)
    assert "| precision |" in table and "Gate:" in table


def test_gen_train_eval_from_the_command_line(tmp_path, capsys):
    data, model = tmp_path / "eps.jsonl", tmp_path / "m.npz"
    main(["critic", "gen", "--n", "12", "--seed", "4", "--out", str(data)])
    main(["critic", "train", "--data", str(data), "--out", str(model), "--epochs", "1"])
    main(["critic", "eval", "--data", str(data), "--model", str(model),
          "--json", str(tmp_path / "r.json")])
    main(["critic", "eval", "--data", str(data), "--baseline"])
    out = capsys.readouterr().out
    assert "12 episodes" in out and "loss" in out
    assert "overall" in json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))


def test_phase2_runs_every_source_and_gates(tmp_path):
    from curio_robot.critic.cli import format_phase2, phase2
    from curio_robot.critic.model import Critic

    summary = phase2(1, 3, 0, Critic(seed=0), 0.85, tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert set(summary["sources"]) == {"none", "spoken", "critic", "critic+spoken", "oracle"}
    assert set(summary["gate"]) == {"fewer_errors", "no_type_worse", "labels_accurate"}
    assert "Gate:" in format_phase2(summary)
