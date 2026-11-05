"""T5-30: does the pi0 checkpoint, loaded the way pi0_eval loads it, behave like the trained one?

Uses only the training and held-out demo frames (never an eval seed). Three numbers:
- the held-out loss on the same 256 held-out frames, with the same flow-matching seed, that
  pi0_train.evaluate logged (0.968 at step 22,000);
- the training-path loss (pi0_train.expert_loss) on evenly spaced training frames;
- open loop: pi0_eval's Pi0Runner predicts the first action of each held-out frame's chunk
  (the action select_action executes) and it is compared with the demo's action, in radians.

Precondition P of the T5-30 gate: held-out loss within 15% of the logged value, and a median
absolute joint error <= 0.15 rad. Runs in the lerobot venv, offline:

    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/pi0_check.py \\
        --checkpoint C:/Users/Rohan/dev/Perspective/out/t5_28/pi0/checkpoints/022000 \\
        --out C:/Users/Rohan/dev/Perspective/out/t5_30/check.json"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEMOS = Path("C:/Users/Rohan/dev/Perspective/out/demos800b")
RELABEL = Path("C:/Users/Rohan/dev/Perspective/out/t5_28/data/relabel.jsonl")
LOSS_TOLERANCE = 0.15  # P: held-out loss within 15% of the logged value
JOINT_TOLERANCE = 0.15  # P: median absolute joint error, rad
TRAIN_SAMPLES = 256
OPEN_LOOP_SAMPLES = 256


def logged_eval_loss(train_log: Path, step: int) -> float:
    """The eval_loss pi0_train logged at this step."""
    with train_log.open(encoding="utf-8") as f:
        for row in map(json.loads, f):
            if row.get("step") == step and "eval_loss" in row:
                return float(row["eval_loss"])
    raise KeyError(f"no eval_loss at step {step} in {train_log}")


def precondition(loss: float, logged: float, joint_errors: np.ndarray) -> dict:
    """Gate precondition P from the loss and the per-joint absolute errors (rad)."""
    rel = abs(loss - logged) / logged
    median = float(np.median(np.abs(joint_errors)))
    return {"heldout_loss": round(loss, 5), "logged_eval_loss": logged,
            "relative_gap": round(rel, 4), "loss_ok": bool(rel <= LOSS_TOLERANCE),
            "median_abs_joint_error_rad": round(median, 4),
            "joint_ok": bool(median <= JOINT_TOLERANCE),
            "pass": bool(rel <= LOSS_TOLERANCE and median <= JOINT_TOLERANCE)}


def evenly(n: int, k: int) -> list[int]:
    return list(range(0, n, max(1, n // k)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoints/<step>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demos", type=Path, default=DEMOS)
    parser.add_argument("--relabel", type=Path, default=RELABEL)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from pi0_data import ACTION, Pi0Frames, load_tasks, split
    from pi0_eval import Pi0Runner
    from pi0_train import EVAL_SAMPLES, evaluate
    from smolvla_train import _loader

    from curio_robot.sim import demos

    t0 = time.perf_counter()
    info = demos.load_info(args.demos)
    tasks = load_tasks(args.relabel)
    train_eps, eval_eps = split(info["total_episodes"])
    step = int(args.checkpoint.name)
    logged = logged_eval_loss(args.checkpoint.parents[1] / "train_log.jsonl", step)

    runner = Pi0Runner(args.checkpoint / "pretrained_model", args.device)
    chunk = runner.policy.config.chunk_size
    train_set = Pi0Frames(args.demos, train_eps, chunk, tasks)
    eval_set = Pi0Frames(args.demos, eval_eps, chunk, tasks)
    # pi0_train's eval subset and loader, so the flow-matching noise matches the logged run.
    eval_subset = torch.utils.data.Subset(eval_set, evenly(len(eval_set), EVAL_SAMPLES))
    train_subset = torch.utils.data.Subset(train_set, evenly(len(train_set), TRAIN_SAMPLES))
    heldout = evaluate(runner.policy, runner.pre, _loader(eval_subset, 16, 0, False, 0))
    train_loss = evaluate(runner.policy, runner.pre, _loader(train_subset, 16, 0, False, 0))
    runner.policy.eval()
    print(f"held-out loss {heldout:.4f} (logged {logged}), train loss {train_loss:.4f}")

    errors, rows = [], []
    for k in evenly(len(eval_set), OPEN_LOOP_SAMPLES):
        sample = eval_set[k]
        e, f = eval_set.index[k]
        frame = eval_set.episodes[e]["frames"][f]
        images = {cam: np.asarray(_png(args.demos / frame["images"][cam]))
                  for cam in runner.cameras}
        runner.reset(k)
        pred = runner.act(images, np.asarray(frame["state"], np.float32), sample["task"])
        err = pred - sample[ACTION][0]
        errors.append(err)
        if len(rows) < 8:
            rows.append({"frame": k, "pred": np.round(pred, 3).tolist(),
                         "demo": np.round(sample[ACTION][0], 3).tolist()})
    errors = np.asarray(errors)
    res = precondition(heldout, logged, errors)
    res.update(checkpoint=str(args.checkpoint), train_loss=round(train_loss, 5),
               open_loop_frames=len(errors),
               median_abs_error_per_joint=np.round(np.median(np.abs(errors), 0), 4).tolist(),
               examples=rows, seconds=round(time.perf_counter() - t0, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def _png(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


if __name__ == "__main__":
    main()
