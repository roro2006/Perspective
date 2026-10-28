"""Set-of-marks: every candidate object gets a capital letter drawn on the
human-view frame, and the VLM answers with one letter. Task T2-03."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    raise NotImplementedError("T2-03")


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
    raise NotImplementedError("T2-03")
