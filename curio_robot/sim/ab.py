"""The A/B experiment behind the pitch's results slide: the same rendered
scene, resolved with Curio on (robot + human view) and off (robot view only)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from curio_robot.geometry import perturb_pose
from curio_robot.perception import SegmentationDetector
from curio_robot.pipeline import step
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.world import SimWorld
from curio_robot.types import HUMAN_CAM

SUCCESS_RADIUS = 0.02  # metres from the correct block: within SO-101 grasp tolerance


def run(
    trials: int, seed: int = 0, pos_noise: float = 0.0, rot_noise_deg: float = 0.0
) -> dict:
    rng = np.random.default_rng(seed)
    with SimWorld(320, 240) as world:
        results: dict[str, dict[str, list[bool]]] = {
            kind: {"on": [], "off": []} for kind in SCENARIOS
        }
        for kind, make in SCENARIOS.items():
            for _ in range(trials):
                world.home()
                scenario = make(world, rng)
                frames = world.capture()
                if pos_noise or rot_noise_deg:
                    frames[HUMAN_CAM] = dataclasses.replace(
                        frames[HUMAN_CAM],
                        pose=perturb_pose(frames[HUMAN_CAM].pose, rng, pos_noise, rot_noise_deg),
                    )
                detector = SegmentationDetector(world.block_colors)
                for mode, use_human in (("on", True), ("off", False)):
                    target = step(frames, scenario.request, detector, use_human_view=use_human).target
                    ok = target.position is not None and (
                        np.linalg.norm(target.position[:2] - scenario.expected_xy) < SUCCESS_RADIUS
                    )
                    results[kind][mode].append(bool(ok))
        summary = {
            kind: {mode: round(100 * float(np.mean(v)), 1) for mode, v in modes.items()}
            for kind, modes in results.items()
        }
        return {
            "trials_per_scenario": trials,
            "seed": seed,
            "human_pose_noise": {"position_m": pos_noise, "rotation_deg": rot_noise_deg},
            "success_radius_m": SUCCESS_RADIUS,
            "success_rate_pct": summary,
        }


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
