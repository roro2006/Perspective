"""Sessions as a dataset: episodes in a LeRobot-style folder layout, JSON
instead of parquet (no new dependencies). Frames come later. Task T3-08."""

from __future__ import annotations

import json
from pathlib import Path

from curio_robot.collab.session import Session

FORMAT = "curio-collab-0.1"


def export_episodes(sessions: list[Session], out_dir: Path, overwrite: bool = False) -> dict:
    """Write
      out_dir/meta/info.json: {"format": FORMAT, "total_episodes", "total_steps",
          "conditions": sorted distinct conditions}
      out_dir/meta/episodes.jsonl, one line per session in order:
          {"episode_index", "session_id", "person", "condition", "task", "seed",
           "length": number of steps, "goals": list, "done": list,
           "success": len(done) / len(goals) (0.0 with no goals)}
      out_dir/data/episode_{index:06d}.jsonl, one line per step in order:
          {"episode_index", "step_index", "utterance", "kind", "text", "goal_id",
           "chosen_id", "asked", "correct": chosen_id is not None and
           chosen_id == goal_id, "t": {stage: round(mark - t_start, 4)}}
    FileExistsError if meta/info.json already exists and not overwrite.
    Returns {"episodes": n, "steps": total steps}."""
    info_path = out_dir / "meta" / "info.json"
    if info_path.exists() and not overwrite:
        raise FileExistsError(str(info_path))

    meta_dir = out_dir / "meta"
    data_dir = out_dir / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    total_steps = sum(len(s.steps) for s in sessions)
    conditions = sorted({s.condition for s in sessions})

    info = {
        "format": FORMAT,
        "total_episodes": len(sessions),
        "total_steps": total_steps,
        "conditions": conditions,
    }
    with info_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(info))

    episodes_lines = []
    for idx, session in enumerate(sessions):
        goals = list(session.goals)
        done = list(session.done)
        success = len(done) / len(goals) if goals else 0.0
        ep = {
            "episode_index": idx,
            "session_id": session.session_id,
            "person": session.person,
            "condition": session.condition,
            "task": session.task,
            "seed": session.seed,
            "length": len(session.steps),
            "goals": goals,
            "done": done,
            "success": success,
        }
        episodes_lines.append(json.dumps(ep))

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for line in episodes_lines:
            f.write(line + "\n")

    for idx, session in enumerate(sessions):
        ep_path = data_dir / f"episode_{idx:06d}.jsonl"
        with ep_path.open("w", encoding="utf-8") as f:
            for step_idx, step in enumerate(session.steps):
                t = {stage: round(mark - session.t_start, 4) for stage, mark in step.marks.items()}
                correct = step.chosen_id is not None and step.chosen_id == step.goal_id
                rec = {
                    "episode_index": idx,
                    "step_index": step_idx,
                    "utterance": step.utterance,
                    "kind": step.kind,
                    "text": step.text,
                    "goal_id": step.goal_id,
                    "chosen_id": step.chosen_id,
                    "asked": step.asked,
                    "correct": correct,
                    "t": t,
                }
                f.write(json.dumps(rec) + "\n")

    return {"episodes": len(sessions), "steps": total_steps}
