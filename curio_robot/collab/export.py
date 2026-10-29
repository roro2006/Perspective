"""Sessions as a dataset: episodes in a LeRobot-style folder layout, JSON
instead of parquet (no new dependencies). Frames come later. Task T3-08."""

from __future__ import annotations

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
    raise NotImplementedError("T3-08")
