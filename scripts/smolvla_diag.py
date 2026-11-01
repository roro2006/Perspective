"""Where does a SmolVLA policy close its jaws relative to the target block? (T5-15, T5-16)

Replays the T5-08 ring-on-target rollouts (same scenes, same seeds as smolvla_eval) and logs,
at the first close command, the block centre relative to the grasp point in the gripper frame
(expert.grasp_error; x is the closing axis), whether the jaws straddled it (expert.straddles),
what touches the block, and how many times the policy closed. Runs in the lerobot venv:

    set HF_HUB_OFFLINE=1
    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/smolvla_diag.py \\
        --checkpoint out/smolvla/run4/checkpoints/012000/pretrained_model --n 30 \\
        --out out/smolvla/diag_run4.json"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import mujoco
import smolvla_eval as ev

from curio_robot.sim import demos
from curio_robot.sim.expert import SUBSTEPS, grasp_error, straddles
from curio_robot.sim.physics import PhysWorld
from curio_robot.types import ROBOT_CAM

CLOSE_CMD = 0.15  # rad: a jaw command below this counts as closing
OPEN_CMD = 0.3  # rad: above this the jaws count as open again
FRAMES = 60


def _touching(world: PhysWorld, i: int) -> list[str]:
    """Names of the geoms (or their bodies) in contact with block i."""
    m, d = world.model, world.data
    block = world.block_ids()[i]
    out = []
    for c in d.contact[: d.ncon]:
        if block in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == block else c.geom1
            out.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other)
                       or m.body(m.geom_bodyid[other]).name)
    return out


def _rollout(world: PhysWorld, policy, pre, post, cams: list[str], trial) -> dict:
    import torch

    world.reset(trial.scene)
    ring = demos.target_pixel(world, trial.target, world.render_camera(ROBOT_CAM))
    policy.reset()
    torch.manual_seed(trial.seed)
    lifts, t0 = ev._Lifts(len(trial.scene.blocks)), world.time
    prev = world.ctrl()
    close, closes, was_open = None, 0, True
    for f in range(FRAMES):
        batch, _ = ev._observe(world, ring, trial.words, cams)
        with torch.inference_mode():
            a = post(policy.select_action(pre(batch))).squeeze(0).numpy().astype(float)
        for k in range(1, 6):
            world.step(prev + (a - prev) * k / 5, SUBSTEPS)
            lifts.update(world, world.time - t0)
            cmd = prev[5] + (a[5] - prev[5]) * k / 5
            if was_open and cmd < CLOSE_CMD:
                closes += 1
                was_open = False
                if close is None:
                    err = grasp_error(world, trial.target)
                    close = {"frame": f, "err_m": err.round(4).tolist(),
                             "straddles": straddles(world, trial.target),
                             "touch": _touching(world, trial.target)}
            elif cmd > OPEN_CMD:
                was_open = True
        prev = a
        if lifts.picked is not None:
            break
    return {"seed": trial.seed, "outcome": ev._outcome(lifts.picked, trial), "close": close,
            "closes": closes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-action-steps", type=int, default=None)
    args = parser.parse_args()
    import torch

    policy, pre, post, (h, w) = ev.load_policy(args.checkpoint, "cuda", args.n_action_steps)
    cams = ev.policy_cameras(policy)
    world = PhysWorld(w, h)
    rows = []
    try:
        for trial in ev.make_trials(args.n, ev.EVAL_SEED, world):
            rows.append(_rollout(world, policy, pre, post, cams, trial))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        world.close()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
