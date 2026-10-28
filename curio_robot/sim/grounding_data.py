"""Sim training and eval data for the grounder: rendered human-view frames
with marks, a request, and the exact answer letter. Task T2-07.

    uv run python -m curio_robot.sim.grounding_data out/grounding --n 300 --seed 0"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from curio_robot.grounding.marks import place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, build_request
from curio_robot.perspective import pick
from curio_robot.sim import scenarios
from curio_robot.sim.world import SimWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, ObjectMap, WorldObject

KINDS = ("relation", "deictic", "color")
HABITS = (0.15, 0.85)  # the p_own_frame shown in relation prompts
MAX_ATTEMPTS = 1000  # guard against a kind that never yields an example


@dataclass(frozen=True, slots=True)
class GroundingExample:
    id: str
    kind: str
    text: str
    image: np.ndarray  # the marked, cropped image (render_marks output)
    request: GroundingRequest  # built with build_request on that image
    answer: str  # the correct letter
    target_id: str  # the object the person meant


def _build_map(world: SimWorld) -> ObjectMap:
    """Build an ObjectMap from the current world state."""
    frame = world.render_camera(HUMAN_CAM)
    robot_frame = world.render_camera(ROBOT_CAM)

    counts: dict[str, int] = {}
    objects: list[WorldObject] = []
    for color, pos in world.block_positions():
        counts[color] = counts.get(color, 0) + 1
        objects.append(WorldObject(f"{color}_{counts[color]}", color, pos, set()))

    return ObjectMap(objects, robot_frame.pose, frame.pose)


def make_example(world, rng, kind, example_id) -> GroundingExample | None:
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")

    if kind == "relation":
        scenarios.perspective(world, rng)
        p = rng.choice(HABITS)
        frame_is_speaker = rng.random() < p

        side = rng.choice(["left", "right"])
        use_marker = rng.random() < 0.5

        # The pair shares one colour; the first block is one of the pair.
        color = world.block_positions()[0][0]
        if use_marker:
            marker = "my" if frame_is_speaker else "your"
            text = f"Hand me the {color} one on {marker} {side}"
        else:
            text = f"Hand me the {color} one on the {side}"

        object_map = _build_map(world)
        pair = object_map.objects[:2]

        if frame_is_speaker:
            viewer_pose = object_map.human_pose
        else:
            viewer_pose = object_map.robot_pose

        target = pick(pair, side, viewer_pose)

        frame = world.render_camera(HUMAN_CAM)
        marks = place_marks(object_map, frame)

        target_mark = next((m for m in marks if m.object_id == target.object_id), None)
        if target_mark is None:
            return None

        image, _ = render_marks(frame, marks)

        request = build_request(text, object_map, marks, image, p_own_frame=p)

        answer = next(l for l, oid in request.letters.items() if oid == target.object_id)

        return GroundingExample(example_id, kind, text, image, request, answer, target.object_id)

    elif kind == "deictic":
        scene = scenarios.cluttered(world, rng)

        blocks = world.block_positions()
        if not blocks:
            return None
        idx = rng.integers(len(blocks))
        _, target_pos = blocks[idx]

        # Re-aim the person at the target so it sits dead centre in their view.
        world.set_human(scene.human_eye, target_pos)

        text = "Hand me that one"

        object_map = _build_map(world)
        target_obj = object_map.objects[idx]

        frame = world.render_camera(HUMAN_CAM)
        marks = place_marks(object_map, frame)

        target_mark = next((m for m in marks if m.object_id == target_obj.object_id), None)
        if target_mark is None:
            return None

        image, _ = render_marks(frame, marks)

        request = build_request(text, object_map, marks, image, p_own_frame=None)

        answer = next(l for l, oid in request.letters.items() if oid == target_obj.object_id)

        return GroundingExample(example_id, kind, text, image, request, answer, target_obj.object_id)

    elif kind == "color":
        scenarios.cluttered(world, rng)

        blocks = world.block_positions()
        color_counts: dict[str, int] = {}
        for color, _ in blocks:
            color_counts[color] = color_counts.get(color, 0) + 1

        unique_colors = [c for c, count in color_counts.items() if count == 1]
        if not unique_colors:
            return None

        target_color = rng.choice(unique_colors)

        target_idx = None
        for i, (color, _) in enumerate(blocks):
            if color == target_color:
                target_idx = i
                break

        if target_idx is None:
            return None

        text = f"Hand me the {target_color} one"

        object_map = _build_map(world)
        target_obj = object_map.objects[target_idx]

        frame = world.render_camera(HUMAN_CAM)
        marks = place_marks(object_map, frame)

        target_mark = next((m for m in marks if m.object_id == target_obj.object_id), None)
        if target_mark is None:
            return None

        image, _ = render_marks(frame, marks)

        request = build_request(text, object_map, marks, image, p_own_frame=None)

        answer = next(l for l, oid in request.letters.items() if oid == target_obj.object_id)

        return GroundingExample(example_id, kind, text, image, request, answer, target_obj.object_id)

    return None


def write_dataset(out_dir: Path, n: int, seed: int = 0, kinds: tuple[str, ...] = KINDS) -> Path:
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    world = SimWorld(320, 240)
    rng = np.random.default_rng(seed)

    examples: list[GroundingExample] = []

    for i in range(n):
        kind = kinds[i % len(kinds)]
        example_id = f"{seed}-{i:05d}"

        for _ in range(MAX_ATTEMPTS):
            ex = make_example(world, rng, kind, example_id)
            if ex is not None:
                examples.append(ex)
                break
        else:
            raise RuntimeError(f"could not make {kind} example")

        Image.fromarray(ex.image).save(images_dir / f"{example_id}.png")

    data_path = out_dir / "data.jsonl"
    with data_path.open("w") as f:
        for ex in examples:
            # Deep copy so the stored request keeps its data URL; the JSONL gets the path.
            messages = copy.deepcopy(ex.request.messages)
            messages[0]["content"][0]["image_url"]["url"] = f"images/{ex.id}.png"
            messages.append({"role": "assistant", "content": ex.answer})

            row = {
                "id": ex.id,
                "kind": ex.kind,
                "text": ex.text,
                "answer": ex.answer,
                "target_id": ex.target_id,
                "letters": ex.request.letters,
                "prompt_version": ex.request.prompt_version,
                "image": f"images/{ex.id}.png",
                "messages": messages,
            }
            f.write(json.dumps(row) + "\n")

    return data_path


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounding dataset")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--n", type=int, default=300, help="Number of examples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    data_path = write_dataset(Path(args.out_dir), args.n, args.seed)
    print(f"wrote {args.n} examples to {data_path}")


if __name__ == "__main__":
    main()
