"""Recorded pick demonstrations for a target-conditioned policy (task T5-05).

The scripted expert (sim.expert) picks a block in the physics world; this
records what a learned policy would see and do: both camera images, the arm
joint state, the actuator targets, the target's pixel in the robot camera and
the words that asked for it. The folder follows collab/export.py's
LeRobot-style layout with JSON lines and PNGs instead of parquet and video, so
it needs no new dependencies and converts to a LeRobot dataset later.

Scenes are collab.tasks.fetch_scene without the box: the expert picks a hidden
block only about half the time (tasks/notes/T5-04.md), so "hidden" demos wait
for a better expert.

Frames are 10 Hz, one per CONTROL_DT * TICKS_PER_FRAME. Frame f is observed at
t = f / FPS; its action is the ctrl in force when frame f + 1 is observed (the
expert's ramp sampled at 10 Hz, so replaying actions at 10 Hz follows the
demonstration). The last frame's action is the final ctrl (hold)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from curio_robot import perspective
from curio_robot.collab.tasks import TaskScene, fetch_scene, true_map
from curio_robot.geometry import project
from curio_robot.sim.expert import CONTROL_DT, pick
from curio_robot.sim.physics import ARM_JOINTS, PhysWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame

FORMAT = "curio-demos-0.1"
FPS = 10
TICKS_PER_FRAME = round(1.0 / (FPS * CONTROL_DT))  # 5 control ticks of 20 ms
MAX_ATTEMPTS_PER_EPISODE = 5  # when failed picks are skipped, give up after this many tries


@dataclass(frozen=True, slots=True)
class _Frame:
    state: np.ndarray  # (6,) joint positions
    target_pixel: tuple[float, float] | None
    robot_rgb: np.ndarray
    human_rgb: np.ndarray


def words_for(scene: TaskScene, i: int) -> tuple[str, str | None]:
    """The request for block i and whose frame its side word uses.

    A unique colour: "Hand me the blue one" (frame None). One of a same-colour
    pair: "Hand me the red one on my left" with the side from the person's view
    (perspective.pick with the human camera pose; frame "speaker")."""
    object_map = true_map(scene)
    target = object_map.objects[i]
    same = object_map.by_color(target.color)
    if len(same) == 1:
        return f"Hand me the {target.color} one", None
    left = perspective.pick(same, "left", object_map.human_pose)
    side = "left" if left.object_id == target.object_id else "right"
    return f"Hand me the {target.color} one on my {side}", "speaker"


def target_pixel(world: PhysWorld, i: int, robot: CameraFrame) -> tuple[float, float] | None:
    """Pixel (u, v) of block i's centre in the robot image; None if outside it."""
    uv = project(world.block_position(i), robot.intrinsics, robot.pose)
    if uv is None:
        return None
    u, v = uv
    if not (0 <= u < robot.intrinsics.width and 0 <= v < robot.intrinsics.height):
        return None
    return (round(float(u), 2), round(float(v), 2))


def _observe(world: PhysWorld, i: int) -> _Frame:
    robot = world.render_camera(ROBOT_CAM)
    human = world.render_camera(HUMAN_CAM)
    return _Frame(world.arm_qpos(), target_pixel(world, i, robot), robot.rgb, human.rgb)


def _run_episode(
    world: PhysWorld, scene: TaskScene, i: int
) -> tuple[bool, str, list[_Frame], list[np.ndarray]]:
    """Reset to scene, pick block i, return (success, reason, frames, actions)."""
    world.reset(scene)
    frames = [_observe(world, i)]
    actions: list[np.ndarray] = []
    ticks = 0

    def on_tick(w: PhysWorld, ctrl: np.ndarray) -> None:
        nonlocal ticks
        ticks += 1
        if ticks % TICKS_PER_FRAME == 0:
            actions.append(ctrl.copy())
            frames.append(_observe(w, i))

    result = pick(world, i, on_tick)
    actions.append(world.ctrl())  # the last frame holds
    return result.success, result.reason, frames, actions


def _episode_seed_and_target(seed: int, attempt: int) -> tuple[int, TaskScene, int]:
    episode_seed = seed + attempt
    rng = np.random.default_rng(episode_seed)
    scene = fetch_scene(rng)
    return episode_seed, scene, int(rng.integers(0, len(scene.blocks)))


def _prepare(out_dir: Path, overwrite: bool) -> None:
    info_path = out_dir / "meta" / "info.json"
    if info_path.exists() and not overwrite:
        raise FileExistsError(str(info_path))
    for stale in ("meta", "data", "images"):  # a smaller re-recording must not keep old ones
        shutil.rmtree(out_dir / stale, ignore_errors=True)
    for sub in ("meta", "data", "images"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def _write_episode(
    out_dir: Path, index: int, words: str, frames: list[_Frame], actions: list[np.ndarray]
) -> None:
    images = out_dir / "images" / f"episode_{index:06d}"
    images.mkdir(parents=True, exist_ok=True)
    with (out_dir / "data" / f"episode_{index:06d}.jsonl").open("w", encoding="utf-8") as f:
        for k, (frame, action) in enumerate(zip(frames, actions, strict=True)):
            paths = {}
            for cam, rgb in ((ROBOT_CAM, frame.robot_rgb), (HUMAN_CAM, frame.human_rgb)):
                rel = f"images/episode_{index:06d}/{cam}_{k:04d}.png"
                Image.fromarray(rgb).save(out_dir / rel)
                paths[cam] = rel
            rec = {
                "episode_index": index,
                "frame_index": k,
                "timestamp": round(k / FPS, 4),
                "state": [round(float(x), 5) for x in frame.state],
                "action": [round(float(x), 5) for x in action],
                "target_pixel": None if frame.target_pixel is None else list(frame.target_pixel),
                "words": words,
                "images": paths,
            }
            f.write(json.dumps(rec) + "\n")


def record(
    n_episodes: int,
    seed: int,
    out_dir: Path,
    width: int = 320,
    height: int = 240,
    include_failed: bool = False,
    overwrite: bool = False,
) -> dict:
    """Record n_episodes expert picks into out_dir.

    Attempt a uses scene fetch_scene(default_rng(seed + a)) and target block
    rng.integers(0, n_blocks) from the same generator. Failed picks are skipped
    (the next attempt takes their place, up to MAX_ATTEMPTS_PER_EPISODE *
    n_episodes attempts) unless include_failed, when they are kept with
    "success": false. FileExistsError if meta/info.json exists and not
    overwrite. Returns {"episodes", "frames", "attempts", "successes", "failed"}."""
    out_dir = Path(out_dir)
    _prepare(out_dir, overwrite)
    world = PhysWorld(width, height)
    episodes: list[dict] = []
    attempts = successes = total_frames = 0
    try:
        while len(episodes) < n_episodes and attempts < MAX_ATTEMPTS_PER_EPISODE * n_episodes:
            episode_seed, scene, i = _episode_seed_and_target(seed, attempts)
            attempts += 1
            success, reason, frames, actions = _run_episode(world, scene, i)
            successes += success
            if not success and not include_failed:
                continue
            index = len(episodes)
            words, frame = words_for(scene, i)
            _write_episode(out_dir, index, words, frames, actions)
            total_frames += len(frames)
            episodes.append({
                "episode_index": index,
                "seed": episode_seed,
                "target_index": i,
                "target_id": true_map(scene).objects[i].object_id,
                "words": words,
                "frame": frame,
                "target_pixel": None if frames[0].target_pixel is None
                else list(frames[0].target_pixel),
                "success": success,
                "reason": reason,
                "length": len(frames),
            })
    finally:
        world.close()

    with (out_dir / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    info = {
        "format": FORMAT,
        "fps": FPS,
        "image_size": {"width": width, "height": height},
        "cameras": [ROBOT_CAM, HUMAN_CAM],
        "state_names": list(ARM_JOINTS),
        "action_names": list(ARM_JOINTS),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "seed": seed,
        "include_failed": include_failed,
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {
        "episodes": len(episodes),
        "frames": total_frames,
        "attempts": attempts,
        "successes": successes,
        "failed": attempts - successes,
    }


def load_info(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "meta" / "info.json").read_text(encoding="utf-8"))


def load_episode(out_dir: Path, i: int, images: bool = False) -> dict:
    """{"episode": its episodes.jsonl line, "frames": its data lines}; with
    images, each frame also gets "robot_rgb" and "human_rgb" (uint8 arrays)."""
    out_dir = Path(out_dir)
    with (out_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        episode = next(ep for ep in map(json.loads, f) if ep["episode_index"] == i)
    with (out_dir / "data" / f"episode_{i:06d}.jsonl").open(encoding="utf-8") as f:
        frames = [json.loads(line) for line in f]
    if images:
        for frame in frames:
            for cam, rel in frame["images"].items():
                frame[f"{cam}_rgb"] = np.asarray(Image.open(out_dir / rel).convert("RGB"))
    return {"episode": episode, "frames": frames}
