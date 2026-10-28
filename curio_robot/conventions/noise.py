"""How likely head-pose calibration error swapped the left/right order of two
objects. Corrections on such pairs are discounted (docs §2). Task T1-03."""

from __future__ import annotations

import numpy as np

from curio_robot.types import CameraPose

DEFAULT_POS_SIGMA = 0.01  # metres, matches the A/B's realistic calibration error
DEFAULT_ROT_SIGMA_DEG = 2.0


def flip_probability(
    a: np.ndarray,
    b: np.ndarray,
    viewer: CameraPose,
    *,
    pos_sigma: float = DEFAULT_POS_SIGMA,
    rot_sigma_deg: float = DEFAULT_ROT_SIGMA_DEG,
    samples: int = 400,
    seed: int = 0,
) -> float:
    """Fraction of noisy viewer poses under which the sign of
    lateral(a) - lateral(b) differs from the noise-free sign.

    lateral(p) = dot(p - viewer.position, viewer.right), as in perspective.py.
    Noisy poses: geometry.perturb_pose(viewer, rng, pos_sigma, rot_sigma_deg)
    with rng = np.random.default_rng(seed), `samples` draws. Deterministic for
    a given seed. Return 0.5 if the noise-free difference is exactly 0.
    Must run in under 20 ms for samples=400 (vectorize or keep it simple)."""
    raise NotImplementedError("T1-03")
