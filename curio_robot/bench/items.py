"""Benchmark items: one rendered scene with the person view and the robot view,
the words, and the exact answer. Task T5-01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curio_robot.conventions.readings import Frame
from curio_robot.geometry import project
from curio_robot.sim.world import SimWorld
from curio_robot.types import CameraFrame, ObjectMap

KINDS = ("color", "marked", "habit", "hidden")
HABITS = (0.15, 0.85)  # p_own_frame shown for "habit" items, as in sim/grounding_data.py
MIN_VISIBLE_PX = 20  # a block with fewer robot-camera pixels counts as hidden from the robot
MAX_ATTEMPTS = 200  # redraws allowed while looking for an ambiguous pair


@dataclass(frozen=True, slots=True)
class BenchItem:
    id: str
    kind: str
    text: str
    frame: Frame | None  # whose left/right the words mean; None: no side word
    p_own_frame: float | None  # the habit shown to grounders ("habit" items only)
    target_id: str
    object_map: ObjectMap  # exact block positions; robot_pose and human_pose from the renders
    robot_visible: tuple[str, ...]  # object ids with >= MIN_VISIBLE_PX robot-camera pixels
    human: CameraFrame  # rendered person view, segmentation None
    robot: CameraFrame  # rendered robot view, segmentation None

    @property
    def target_xyz(self) -> np.ndarray:
        return next(o.position for o in self.object_map.objects if o.object_id == self.target_id)

    @property
    def target_pixel_robot(self) -> tuple[float, float] | None:
        """The target projected into the robot image (what a policy is told)."""
        return project(self.target_xyz, self.robot.intrinsics, self.robot.pose)


def make_item(world: SimWorld, rng: np.random.Generator, kind: str, item_id: str) -> BenchItem:
    """One item of `kind` (ValueError for an unknown kind).

    Scene: scene = collab.tasks.fetch_scene(rng, occlude=(kind == "hidden"));
    apply_scene(world, scene); human = world.render_camera(HUMAN_CAM); robot =
    world.render_camera(ROBOT_CAM, segmentation=True). object_map =
    ObjectMap(true_map(scene).objects, robot.pose, human.pose). Block i of the
    scene is object i and geom world._block_geoms[i]; robot_visible = ids of
    objects whose geom fills >= MIN_VISIBLE_PX pixels of robot.segmentation.
    Store both frames with segmentation=None.

    - color, hidden: target = object 4 (the single block, a unique colour;
      hidden by the box for "hidden"); text f"Hand me the {color} one";
      frame None, p_own_frame None.
    - marked, habit: after the scene, draw in this order: idx =
      int(rng.integers(0, 4)) (a pair block); for "habit" p =
      HABITS[int(rng.integers(0, 2))], else p = 0.5; frame = "speaker" if
      rng.random() < p else "robot". The viewer is human.pose for "speaker",
      robot.pose for "robot"; side = "left" if perspective.pick(same-colour
      objects, "left", viewer) is the target, else "right". If both viewers
      pick the same block for that side, the item is not ambiguous: draw a new
      scene and repeat (at most MAX_ATTEMPTS, then RuntimeError).
      marked: text f"Hand me the {color} one on {m} {side}", m "my" for
      speaker, "your" for robot; p_own_frame None.
      habit: text f"Hand me the {color} one on the {side}"; p_own_frame = p."""
    raise NotImplementedError("T5-01")


def generate(
    world: SimWorld, n_per_kind: int, seed: int, kinds: tuple[str, ...] = KINDS
) -> list[BenchItem]:
    """rng = np.random.default_rng(seed); for kind in kinds, for i in
    range(n_per_kind): make_item(world, rng, kind, f"{kind}-{i:04d}").
    Same arguments -> same items."""
    raise NotImplementedError("T5-01")


def save(items: list[BenchItem], out_dir: Path) -> Path:
    """Write out_dir/images/{id}_human.png and {id}_robot.png (the rgb arrays)
    and out_dir/items.jsonl, one JSON object per item:
      {"id", "kind", "text", "frame", "p_own_frame", "target_id",
       "robot_visible": [...],
       "objects": [{"id", "color", "xyz": [x, y, z]}, ...] in map order,
       "cameras": {"human": cam, "robot": cam}} with
      cam = {"position": [3], "rotation": [[3] x 3], "fx", "fy", "cx", "cy",
             "width", "height", "timestamp"}.
    Creates folders. Returns the items.jsonl path."""
    raise NotImplementedError("T5-01")


def load(out_dir: Path) -> list[BenchItem]:
    """Inverse of save: CameraFrame(camera name, rgb from the PNG as uint8
    (H, W, 3), Intrinsics, CameraPose, timestamp, None); WorldObject seen_by
    empty sets; tuples where the dataclass has tuples."""
    raise NotImplementedError("T5-01")
