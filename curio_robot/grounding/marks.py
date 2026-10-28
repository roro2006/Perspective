"""Set-of-marks: every candidate object gets a capital letter drawn on the
human-view frame, and the VLM answers with one letter. Task T2-03."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from curio_robot.geometry import project
from curio_robot.types import CameraFrame, ObjectMap

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CROP_MARGIN_PX = 40
MAX_SIDE_PX = 448


@dataclass(frozen=True, slots=True)
class Mark:
    letter: str
    object_id: str
    pixel: tuple[float, float]  # (u, v) in the ORIGINAL frame


@dataclass(frozen=True, slots=True)
class Crop:
    """How the rendered image maps back to the original frame:
    original = (x0, y0) + rendered / scale."""

    x0: int
    y0: int
    scale: float


def place_marks(object_map: ObjectMap, frame: CameraFrame) -> list[Mark]:
    """One Mark per object whose position projects inside the frame
    (geometry.project with frame.intrinsics and frame.pose; drop None and
    pixels outside [0, width) x [0, height)). Objects sorted by object_id,
    letters assigned A, B, C... in that order, at most 26 (extra objects are
    dropped)."""
    width = frame.intrinsics.width
    height = frame.intrinsics.height

    marks: list[Mark] = []
    for obj in sorted(object_map.objects, key=lambda o: o.object_id):
        pixel = project(obj.position, frame.intrinsics, frame.pose)
        if pixel is None:
            continue
        u, v = pixel
        if not (0 <= u < width and 0 <= v < height):
            continue
        marks.append(Mark(LETTERS[len(marks)], obj.object_id, (u, v)))
        if len(marks) == len(LETTERS):
            break

    return marks


def render_marks(frame: CameraFrame, marks: list[Mark]) -> tuple[np.ndarray, Crop]:
    """Draw each letter at its pixel (a filled white disc of radius 10 with a
    black outline, the letter in black, centred), then crop and downscale.

    Crop box: x0 = max(0, floor(min_u - CROP_MARGIN_PX)),
    x1 = min(width, ceil(max_u + CROP_MARGIN_PX)), same for y with v and
    height. No marks -> the whole frame (0, 0, width, height).
    Downscale (never upscale): scale = min(1, MAX_SIDE_PX / max(w, h)) with
    w = x1 - x0, h = y1 - y0; new size (max(1, round(w * scale)),
    max(1, round(h * scale))), PIL bilinear.
    Returns (uint8 HxWx3 image, Crop(x0, y0, scale)). Doesn't modify frame.rgb."""
    width = frame.intrinsics.width
    height = frame.intrinsics.height

    img = Image.fromarray(frame.rgb.copy())
    draw = ImageDraw.Draw(img)
    for mark in marks:
        u, v = mark.pixel
        draw.ellipse((u - 10, v - 10, u + 10, v + 10), fill="white", outline="black")
        draw.text((u, v), mark.letter, fill="black", anchor="mm")

    if not marks:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        us = [m.pixel[0] for m in marks]
        vs = [m.pixel[1] for m in marks]
        x0 = max(0, math.floor(min(us) - CROP_MARGIN_PX))
        x1 = min(width, math.ceil(max(us) + CROP_MARGIN_PX))
        y0 = max(0, math.floor(min(vs) - CROP_MARGIN_PX))
        y1 = min(height, math.ceil(max(vs) + CROP_MARGIN_PX))

    w = x1 - x0
    h = y1 - y0
    scale = min(1.0, MAX_SIDE_PX / max(w, h))
    cropped = img.crop((x0, y0, x1, y1))
    if scale < 1.0:
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        cropped = cropped.resize(new_size, Image.BILINEAR)

    return np.asarray(cropped, dtype=np.uint8), Crop(x0, y0, float(scale))
