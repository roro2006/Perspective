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
demonstration). The last frame's action is the final ctrl (hold).

Target marker (T5-07): a policy learns *which* block from a ring drawn on the
robot image at the target pixel, the visual prompt the grounding layer will
give it. Recordings stay raw; the ring is drawn at conversion time
(draw_marker, marker_uv), so its look and mode can change without
re-recording. Mode "start" keeps the ring at the target's start pixel for the
whole episode, as at run time, where grounding points once before the pick;
"track" follows the block.

Wrist camera (T5-12): the SO-101's wrist_cam (so101.xml, on the camera mount)
sees the jaws and the block close up for the last centimetres, which the robot
camera resolves at about 8 px per block. Recordings store it as a third image
per frame unless wrist=False; folders recorded before it have two cameras and
still load. Its near clip plane is lowered for that render only (render_wrist):
the scene's default (1% of an 8 m extent, 8 cm) cuts away the jaws and a
grasped block."""

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

# The target ring, sized for 256x256 robot images, where a block is about 8 px across.
MARKER_RADIUS = 9  # px from the target pixel to the middle of the ring
MARKER_WIDTH = 2  # px, ring thickness
MARKER_COLOUR = (255, 0, 255)  # magenta: no block, table or arm has it
MARKER_MODES = ("start", "track")

WRIST_CAM = "wrist_cam"  # camera name in so101.xml; images are stored under "wrist"
WRIST = "wrist"
WRIST_ZNEAR_M = 0.005  # near clip for the wrist view: the fixed jaw is about 3 cm away


@dataclass(frozen=True, slots=True)
class _Frame:
    state: np.ndarray  # (6,) joint positions
    target_pixel: tuple[float, float] | None
    robot_rgb: np.ndarray
    human_rgb: np.ndarray
    wrist_rgb: np.ndarray | None = None


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


PAIR_WORDINGS = (("my", 0.3), ("your", 0.3), ("the", 0.15), ("free", 0.25))
FREE_TRIES = 20  # template draws before falling back to "my" wording


def words_varied(scene: TaskScene, i: int, rng: np.random.Generator) -> tuple[str, str | None]:
    """Like words_for, but with the wording the policy meets at run time: for a pair
    block "my"/"your"/"the" side words or a free-wording template (bench.items), drawn
    with PAIR_WORDINGS weights; for a unique colour, half plain and half a free lead and
    noun. The target is always block i; the marker carries the answer, the words vary."""
    from curio_robot.bench.items import (
        FREE_LEADS,
        FREE_NOUNS,
        FREE_TEMPLATES,
        MIN_DISTANCE_GAP,
    )

    object_map = true_map(scene)
    target = object_map.objects[i]
    same = object_map.by_color(target.color)
    viewers = {"speaker": object_map.human_pose, "robot": object_map.robot_pose}
    lead = FREE_LEADS[int(rng.integers(0, len(FREE_LEADS)))]
    noun = FREE_NOUNS[int(rng.integers(0, len(FREE_NOUNS)))]
    if len(same) == 1:
        if rng.random() < 0.5:
            return f"Hand me the {target.color} one", None
        return f"{lead} the {target.color} {noun}", None

    def side_from(frame: str) -> str:
        left = perspective.pick(same, "left", viewers[frame])
        return "left" if left.object_id == target.object_id else "right"

    names, weights = zip(*PAIR_WORDINGS, strict=True)
    wording = names[int(rng.choice(len(names), p=np.array(weights) / sum(weights)))]
    if wording == "free":
        for _ in range(FREE_TRIES):
            relation, frame, template = FREE_TEMPLATES[int(rng.integers(0, len(FREE_TEMPLATES)))]
            viewer = viewers[frame]
            if relation == "side":
                side = side_from(frame)
            else:
                a, b = (perspective.horizontal_distance(o, viewer) for o in same)
                if abs(a - b) < MIN_DISTANCE_GAP:
                    continue
                if perspective.pick(same, relation, viewer).object_id != target.object_id:
                    continue
                side = ""
            return template.format(lead=lead, color=target.color, noun=noun, side=side), frame
        wording = "my"
    if wording == "the":
        frame = "speaker" if rng.random() < 0.5 else "robot"
        return f"Hand me the {target.color} one on the {side_from(frame)}", frame
    frame = "speaker" if wording == "my" else "robot"
    return f"Hand me the {target.color} one on {wording} {side_from(frame)}", frame


def target_pixel(world: PhysWorld, i: int, robot: CameraFrame) -> tuple[float, float] | None:
    """Pixel (u, v) of block i's centre in the robot image; None if outside it."""
    uv = project(world.block_position(i), robot.intrinsics, robot.pose)
    if uv is None:
        return None
    u, v = uv
    if not (0 <= u < robot.intrinsics.width and 0 <= v < robot.intrinsics.height):
        return None
    return (round(float(u), 2), round(float(v), 2))


def draw_marker(
    rgb: np.ndarray,
    uv: tuple[float, float] | list[float] | None,
    radius: float = MARKER_RADIUS,
    width: float = MARKER_WIDTH,
    colour: tuple[int, int, int] = MARKER_COLOUR,
) -> np.ndarray:
    """A copy of rgb with a ring of the given radius centred on pixel uv.

    uv None (target outside the image) leaves the copy unmarked. The ring may
    run off the image edge; only the part inside is drawn."""
    out = np.array(rgb, copy=True)
    if uv is None:
        return out
    h, w = out.shape[:2]
    vv, uu = np.mgrid[0:h, 0:w]
    # Pixel centres are at integer + 0.5 in the projection's convention.
    dist = np.hypot(uu + 0.5 - uv[0], vv + 0.5 - uv[1])
    out[np.abs(dist - radius) <= width / 2] = colour
    return out


def marker_uv(episode: dict, frame: dict, mode: str = "start") -> list[float] | None:
    """Where the ring goes on this frame: the target's start pixel ("start",
    episodes.jsonl) or its current pixel ("track", the frame's line)."""
    if mode == "start":
        return episode["target_pixel"]
    if mode == "track":
        return frame["target_pixel"]
    raise ValueError(f"marker mode {mode!r} not in {MARKER_MODES}")


def render_wrist(world: PhysWorld) -> np.ndarray:
    """The wrist camera image (RGB uint8), rendered with a near plane of WRIST_ZNEAR_M.
    The scene's own near plane is restored afterwards, so the other cameras render
    exactly as before."""
    vis_map = world.model.vis.map
    znear = float(vis_map.znear)
    vis_map.znear = WRIST_ZNEAR_M / float(world.model.stat.extent)
    try:
        return world.render_camera(WRIST_CAM).rgb
    finally:
        vis_map.znear = znear


def _observe(world: PhysWorld, i: int, wrist: bool) -> _Frame:
    robot = world.render_camera(ROBOT_CAM)
    human = world.render_camera(HUMAN_CAM)
    return _Frame(world.arm_qpos(), target_pixel(world, i, robot), robot.rgb, human.rgb,
                  render_wrist(world) if wrist else None)


def _run_episode(
    world: PhysWorld, scene: TaskScene, i: int, wrist: bool = True
) -> tuple[bool, str, list[_Frame], list[np.ndarray]]:
    """Reset to scene, pick block i, return (success, reason, frames, actions)."""
    world.reset(scene)
    frames = [_observe(world, i, wrist)]
    actions: list[np.ndarray] = []
    ticks = 0

    def on_tick(w: PhysWorld, ctrl: np.ndarray) -> None:
        nonlocal ticks
        ticks += 1
        if ticks % TICKS_PER_FRAME == 0:
            actions.append(ctrl.copy())
            frames.append(_observe(w, i, wrist))

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
            views = [(ROBOT_CAM, frame.robot_rgb), (HUMAN_CAM, frame.human_rgb)]
            if frame.wrist_rgb is not None:
                views.append((WRIST, frame.wrist_rgb))
            for cam, rgb in views:
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
    words: str = "fixed",
    wrist: bool = True,
) -> dict:
    """Record n_episodes expert picks into out_dir.

    Attempt a uses scene fetch_scene(default_rng(seed + a)) and target block
    rng.integers(0, n_blocks) from the same generator. Failed picks are skipped
    (the next attempt takes their place, up to MAX_ATTEMPTS_PER_EPISODE *
    n_episodes attempts) unless include_failed, when they are kept with
    "success": false. FileExistsError if meta/info.json exists and not
    overwrite. With wrist, every frame also stores the wrist camera image ("wrist").
    Returns {"episodes", "frames", "attempts", "successes", "failed"}."""
    out_dir = Path(out_dir)
    _prepare(out_dir, overwrite)
    world = PhysWorld(width, height)
    episodes: list[dict] = []
    attempts = successes = total_frames = 0
    try:
        while len(episodes) < n_episodes and attempts < MAX_ATTEMPTS_PER_EPISODE * n_episodes:
            episode_seed, scene, i = _episode_seed_and_target(seed, attempts)
            attempts += 1
            success, reason, frames, actions = _run_episode(world, scene, i, wrist)
            successes += success
            if not success and not include_failed:
                continue
            index = len(episodes)
            if words == "varied":  # its own generator, so scenes match a "fixed" recording
                text, frame = words_varied(scene, i, np.random.default_rng([episode_seed, 7]))
            else:
                text, frame = words_for(scene, i)
            _write_episode(out_dir, index, text, frames, actions)
            total_frames += len(frames)
            episodes.append({
                "episode_index": index,
                "seed": episode_seed,
                "target_index": i,
                "target_id": true_map(scene).objects[i].object_id,
                "words": text,
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
        "cameras": [ROBOT_CAM, HUMAN_CAM, *([WRIST] if wrist else [])],
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
    images, each frame also gets "<camera>_rgb" (uint8 arrays) for every camera it
    stored: "robot" and "human", and "wrist" when recorded with it."""
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
