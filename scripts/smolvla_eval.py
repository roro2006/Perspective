"""Closed-loop evaluation of a fine-tuned SmolVLA in the physics sim. Task T5-08.

Does the policy pick the block the ring points at? Runs in the lerobot venv, offline:

    set HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/smolvla_eval.py \\
        --checkpoint out/smolvla/run1/checkpoints/002500/pretrained_model \\
        --n 50 --device cuda --out out/smolvla/eval_run1

Scenes are collab.tasks.fetch_scene without the box, from seeds disjoint from the
demonstrations (they start at 1000 and used 301; evaluation starts at 50000). The target is
always one block of a same-colour pair, so the words need "left"/"right" (demos.words_for).
Each scene and target is rolled out under every condition:

- marker: ring on the target, words name it (the intended use);
- words: no ring, words only (can the policy resolve the person's "left" itself?);
- swapped: ring on the pair partner, words name the target (ring or words?).

Timing follows the recordings: observe at 10 Hz (robot image + ring, person image, 6 joints),
ask the policy, and let each action cover the 0.1 s a demo action covered (5 control ticks of
20 ms). Within those ticks the ctrl ramps linearly from the previous action, as the expert's
ctrl did (--ctrl hold: constant instead). Actions come from policy.select_action, so a chunk of
n_action_steps runs open loop before the next model call, as lerobot's eval does.

A block counts as picked when it stays lifted >= 3 cm for 0.5 s (expert.LIFT_HEIGHT,
HOLD_TIME), checked every control tick; the rollout stops there or at the horizon."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a plain script from the lerobot venv
    sys.path.insert(0, str(REPO))

from curio_robot.collab.tasks import TaskScene, fetch_scene, true_map
from curio_robot.sim import demos
from curio_robot.sim.expert import HOLD_TIME, LIFT_HEIGHT, SUBSTEPS
from curio_robot.sim.physics import PhysWorld
from curio_robot.sim.world import BLOCK_HALF
from curio_robot.types import HUMAN_CAM, ROBOT_CAM

EVAL_SEED = 50000  # demos300 used seeds 1000-1300; keep well clear of any re-recording
HORIZON_S = 6.0  # demo episodes are 4.1 s (42 frames)
CONDITIONS = {"marker": "target", "words": None, "swapped": "partner"}  # where the ring goes
FRAME_EVERY = 10  # save one composite PNG per this many frames (first rollout per condition)
MAX_SEED_TRIES = 20  # per trial, for the rare scene whose ring pixel is off the image
# Feature names as in smolvla_dataset (which imports torch, so it loads only after the GPU
# env vars are set; load_policy checks the two agree).
STATE = "observation.state"
CAMERA_KEYS = {"robot": "observation.images.camera1", "human": "observation.images.camera2"}


@dataclass(frozen=True, slots=True)
class Trial:
    """One scene and target, shared by every condition."""

    seed: int
    scene: TaskScene
    target: int
    partner: int
    words: str


def make_trials(n: int, seed: int, world: PhysWorld) -> list[Trial]:
    """n scenes from seeds seed, seed+1, ...; the target is a random pair block. Seeds where
    the target or partner projects outside the robot image are skipped (no ring possible)."""
    trials: list[Trial] = []
    s = seed
    while len(trials) < n:
        if s - seed > MAX_SEED_TRIES * n:
            raise RuntimeError("too many scenes with the pair outside the robot image")
        rng = np.random.default_rng(s)
        scene = fetch_scene(rng)
        objects = true_map(scene).objects
        pairs = [i for i, o in enumerate(objects)
                 if sum(p.color == o.color for p in objects) == 2]
        target = int(rng.choice(pairs))
        partner = next(i for i in pairs if i != target and objects[i].color == objects[target].color)
        world.reset(scene)
        robot = world.render_camera(ROBOT_CAM)
        if all(demos.target_pixel(world, i, robot) for i in (target, partner)):
            trials.append(Trial(s, scene, target, partner, demos.words_for(scene, target)[0]))
        s += 1
    return trials


def load_policy(checkpoint: Path, device: str, n_action_steps: int | None):
    """Policy and processors from a training checkpoint, configured as in training."""
    import smolvla_dataset
    from lerobot.policies.factory import make_pre_post_processors
    from smolvla_train import build_policy

    if (smolvla_dataset.STATE, smolvla_dataset.CAMERA_KEYS) != (STATE, CAMERA_KEYS):
        raise RuntimeError("feature names differ from smolvla_dataset")

    cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    image_hw = tuple(cfg["input_features"][CAMERA_KEYS["robot"]]["shape"][1:])
    policy = build_policy("", image_hw, device, checkpoint)
    if n_action_steps:
        policy.config.n_action_steps = n_action_steps
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post, image_hw


def _observe(world: PhysWorld, ring: tuple[float, float] | None, words: str):
    """The policy input at this instant, and the camera1 image it saw (for PNGs)."""
    import torch

    robot = demos.draw_marker(world.render_camera(ROBOT_CAM).rgb, ring)
    human = world.render_camera(HUMAN_CAM).rgb
    batch = {
        STATE: torch.tensor(world.arm_qpos(), dtype=torch.float32)[None],
        CAMERA_KEYS["robot"]: _chw(robot)[None],
        CAMERA_KEYS["human"]: _chw(human)[None],
        "task": [words],
    }
    return batch, robot, human


class _Lifts:
    """Which block, if any, has stayed lifted for HOLD_TIME (the expert's success)."""

    def __init__(self, n_blocks: int) -> None:
        self.since: list[float | None] = [None] * n_blocks
        self.picked: int | None = None
        self.t_lift: float | None = None

    def update(self, world: PhysWorld, t: float) -> None:
        for i, since in enumerate(self.since):
            if not world.lifted(i, LIFT_HEIGHT):
                self.since[i] = None
            elif since is None:
                self.since[i] = t
            elif t - since >= HOLD_TIME - 1e-9 and self.picked is None:
                self.picked, self.t_lift = i, since


def rollout(world: PhysWorld, model, trial: Trial, condition: str, horizon: float,
            ctrl_mode: str, frames_dir: Path | None) -> dict:
    """Run one closed-loop episode; return its outcome and timings."""
    import torch

    policy, pre, post = model
    world.reset(trial.scene)
    ring_block = {"target": trial.target, "partner": trial.partner}.get(CONDITIONS[condition])
    ring = None if ring_block is None else demos.target_pixel(
        world, ring_block, world.render_camera(ROBOT_CAM))
    policy.reset()
    torch.manual_seed(trial.seed)  # same flow-matching noise for every condition
    lifts, t0, start = _Lifts(len(trial.scene.blocks)), world.time, time.perf_counter()
    query_s: list[float] = []
    prev = world.ctrl()
    max_target_z = BLOCK_HALF
    n_frames = round(horizon * demos.FPS)
    for f in range(n_frames):
        batch, robot, human = _observe(world, ring, trial.words)
        if frames_dir is not None and f % FRAME_EVERY == 0:
            _save_frame(frames_dir / f"{condition}_{f:03d}.png", world, robot, human)
        needs_model = len(policy._queues["action"]) == 0  # lerobot's chunk queue
        tq = time.perf_counter()
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch)))
        if needs_model:
            query_s.append(time.perf_counter() - tq)
        action = action.squeeze(0).numpy().astype(float)
        for k in range(1, demos.TICKS_PER_FRAME + 1):
            a = k / demos.TICKS_PER_FRAME if ctrl_mode == "ramp" else 1.0
            world.step(prev + (action - prev) * a, SUBSTEPS)
            lifts.update(world, world.time - t0)
            max_target_z = max(max_target_z, float(world.block_position(trial.target)[2]))
        prev = action
        if lifts.picked is not None:
            break
    if frames_dir is not None:
        _save_frame(frames_dir / f"{condition}_{f:03d}_end.png", world, robot, human)
    return {
        "seed": trial.seed, "condition": condition, "words": trial.words,
        "target": trial.target, "partner": trial.partner,
        "ring": None if ring is None else [round(x, 2) for x in ring],
        "outcome": _outcome(lifts.picked, trial),
        "picked_block": lifts.picked,
        "t_lift": None if lifts.t_lift is None else round(lifts.t_lift, 3),
        "target_max_lift_m": round(max_target_z - BLOCK_HALF, 4),
        "frames": f + 1,
        "queries": len(query_s),
        "query_s": round(float(np.mean(query_s)), 3) if query_s else None,
        "rollout_s": round(time.perf_counter() - start, 2),
    }


def _outcome(picked: int | None, trial: Trial) -> str:
    if picked is None:
        return "nothing"
    if picked == trial.target:
        return "target"
    return "partner" if picked == trial.partner else "other"


def _chw(rgb: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0


def _save_frame(path: Path, world: PhysWorld, robot: np.ndarray, human: np.ndarray) -> None:
    """camera1 as the policy saw it | person camera | overview, side by side."""
    overview = world.render_camera("overview").rgb
    Image.fromarray(np.concatenate([robot, human, overview], axis=1)).save(path)


def summarize(results: list[dict], conditions: list[str]) -> dict:
    """Rates per condition, mean time to lift for target picks, mean timings."""
    out = {}
    for c in conditions:
        rows = [r for r in results if r["condition"] == c]
        n = len(rows)
        rates = {k: round(sum(r["outcome"] == k for r in rows) / n, 3)
                 for k in ("target", "partner", "other", "nothing")}
        lifts = [r["t_lift"] for r in rows if r["outcome"] == "target"]
        queries = [r["query_s"] for r in rows if r["query_s"] is not None]
        out[c] = {
            "n": n, **rates,
            "t_lift_target_mean": round(float(np.mean(lifts)), 3) if lifts else None,
            "query_s_mean": round(float(np.mean(queries)), 3) if queries else None,
            "rollout_s_mean": round(float(np.mean([r["rollout_s"] for r in rows])), 2),
        }
    return out


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    t_load = time.perf_counter()
    policy, pre, post, (h, w) = load_policy(args.checkpoint, args.device, args.n_action_steps)
    load_s = time.perf_counter() - t_load
    world = PhysWorld(w, h)
    results: list[dict] = []
    try:
        trials = make_trials(args.n, args.seed, world)
        for c in args.conditions:
            for k, trial in enumerate(trials):
                frames = out / "frames" if k == 0 else None
                r = rollout(world, (policy, pre, post), trial, c, args.horizon, args.ctrl, frames)
                results.append(r)
                print(json.dumps(r), flush=True)
    finally:
        world.close()
    report = {
        "checkpoint": str(args.checkpoint), "device": args.device, "seed": args.seed,
        "n": args.n, "horizon_s": args.horizon, "ctrl": args.ctrl,
        "n_action_steps": policy.config.n_action_steps, "chunk_size": policy.config.chunk_size,
        "torch_threads": torch.get_num_threads(), "load_s": round(load_s, 1),
        "summary": summarize(results, args.conditions), "rollouts": results,
    }
    (out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _hide_gpus_for_cpu(device: str) -> None:
    """A CPU eval must not touch a GPU another job is training on; this has to run before
    torch is imported, hence the lazy torch imports in this script."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help=".../checkpoints/<step>/pretrained_model")
    parser.add_argument("--n", type=int, default=50, help="scenes (each run per condition)")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS),
                        default=list(CONDITIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=HORIZON_S, help="seconds per rollout")
    parser.add_argument("--ctrl", choices=("ramp", "hold"), default="ramp")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="actions run per model call (default: the checkpoint's)")
    args = parser.parse_args()
    _hide_gpus_for_cpu(args.device)
    report = evaluate(args)
    print(json.dumps({"load_s": report["load_s"], "summary": report["summary"]}, indent=1))


if __name__ == "__main__":
    main()
