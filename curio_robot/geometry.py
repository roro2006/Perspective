from __future__ import annotations

import numpy as np

from curio_robot.types import CameraPose, Intrinsics

# Detections are lifted onto a plane at roughly the height of a block's centre.
OBJECT_PLANE_Z = 0.0125


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> Intrinsics:
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return Intrinsics(fx=f, fy=f, cx=width / 2, cy=height / 2, width=width, height=height)


def pixel_ray(pixel: tuple[float, float], k: Intrinsics, pose: CameraPose) -> np.ndarray:
    """World-space unit direction of the ray through an image pixel (v points down)."""
    u, v = pixel
    d_cam = np.array([(u - k.cx) / k.fx, -(v - k.cy) / k.fy, -1.0])
    d = pose.rotation @ d_cam
    return d / np.linalg.norm(d)


def lift_to_plane(
    pixel: tuple[float, float], k: Intrinsics, pose: CameraPose, z: float = OBJECT_PLANE_Z
) -> np.ndarray | None:
    """Intersect the pixel's ray with the horizontal plane at height z."""
    d = pixel_ray(pixel, k, pose)
    if abs(d[2]) < 1e-6:
        return None
    s = (z - pose.position[2]) / d[2]
    if s <= 0:
        return None
    return pose.position + s * d


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose columns are the camera axes (x right, y up, z backward)."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.column_stack([right, true_up, -forward])


def project(point: np.ndarray, k: Intrinsics, pose: CameraPose) -> tuple[float, float] | None:
    """World point -> pixel (u, v); None if behind the camera."""
    d = pose.rotation.T @ (np.asarray(point, float) - pose.position)
    if d[2] >= 0:
        return None
    return (k.cx + k.fx * d[0] / -d[2], k.cy - k.fy * d[1] / -d[2])
