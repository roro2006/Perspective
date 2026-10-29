"""Three-pane frames for the teaser video, slides and booth loop:
robot view | your view | twin, with the request and the outcome as captions."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curio_robot.geometry import project
from curio_robot.perception import SegmentationDetector
from curio_robot.pipeline import step
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.world import SimWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM

BG = (18, 20, 24)
FG = (240, 240, 236)
OK = (90, 200, 120)
BAD = (235, 90, 80)
CAM_COLOR = {ROBOT_CAM: (255, 170, 60), HUMAN_CAM: (80, 200, 235)}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ring(draw: ImageDraw.ImageDraw, uv, r: float, color, width: int = 3) -> None:
    u, v = uv
    draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=width)


def render_shot(kind: str, seed: int, curio_on: bool, pane=(640, 480)) -> Image.Image:
    rng = np.random.default_rng(seed)
    with SimWorld(*pane) as world:
        scenario = SCENARIOS[kind](world, rng)
        frames = world.capture()
        result = step(
            frames, scenario.request, SegmentationDetector(world.block_colors), use_human_view=curio_on
        )
        target = result.target
        correct = target.position is not None and (
            np.linalg.norm(target.position[:2] - scenario.expected_xy) < 0.02
        )
        if target.position is not None:
            world.move_gripper_to(target.position)
        twin = world.render_camera("overview")

        panes = []
        for cam in (ROBOT_CAM, HUMAN_CAM):
            img = Image.fromarray(frames[cam].rgb)
            d = ImageDraw.Draw(img)
            used = cam == ROBOT_CAM or curio_on
            for o in result.object_map.objects:
                if cam in o.seen_by:
                    uv = project(o.position, frames[cam].intrinsics, frames[cam].pose)
                    if uv:
                        _ring(d, uv, 16, CAM_COLOR[cam])
            if not used:
                img = Image.blend(img, Image.new("RGB", img.size, BG), 0.65)
            panes.append(img)

        img = Image.fromarray(twin.rgb)
        d = ImageDraw.Draw(img)
        goal = project(np.array([*scenario.expected_xy, 0.0125]), twin.intrinsics, twin.pose)
        if goal:
            _ring(d, goal, 20, OK, 2)
        if target.position is not None:
            uv = project(target.position, twin.intrinsics, twin.pose)
            if uv:
                _ring(d, uv, 14, OK if correct else BAD, 4)
        panes.append(img)

        w, h = pane
        header, footer = 70, 64
        canvas = Image.new("RGB", (3 * w, h + header + footer), BG)
        for i, p in enumerate(panes):
            canvas.paste(p, (i * w, header))
        d = ImageDraw.Draw(canvas)
        big, small = _font(30), _font(20)
        mode = "CURIO ON: robot + your view" if curio_on else "CURIO OFF: robot's view only"
        d.text((20, 18), f'"{scenario.request}"', font=big, fill=FG)
        d.text((3 * w - 20, 24), mode, font=small, fill=OK if curio_on else BAD, anchor="ra")
        for i, label in enumerate(("ROBOT VIEW", "YOUR VIEW", "TWIN")):
            d.text((i * w + 14, header + 10), label, font=small, fill=FG)
        verdict = ("Correct" if correct else "Wrong") if target.object_id else "Not found"
        d.text(
            (20, h + header + 18),
            f"{verdict}: {target.reason}",
            font=big,
            fill=OK if correct else BAD,
        )
        return canvas
