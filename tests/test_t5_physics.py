"""Physics world and scripted pick (task T5-04)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from curio_robot.collab.tasks import fetch_scene
from curio_robot.sim.expert import pick
from curio_robot.sim.physics import TIMESTEP, PhysWorld
from curio_robot.sim.world import BLOCK_HALF, HOME_QPOS, BlockSpec
from curio_robot.types import HUMAN_CAM, ROBOT_CAM

N_SCENES = 100
GATE = 0.90


@pytest.fixture(scope="module")
def world():
    with PhysWorld(64, 48) as w:
        yield w


def test_blocks_settle_on_the_table(world):
    scene = fetch_scene(np.random.default_rng(0))
    world.reset(scene)
    world.step(HOME_QPOS, int(0.5 / TIMESTEP))
    for i, block in enumerate(scene.blocks):
        p = world.block_position(i)
        assert abs(p[2] - BLOCK_HALF) < 0.002
        assert np.linalg.norm(p[:2] - block.xy) < 0.002
        assert not world.lifted(i)


def test_a_commanded_joint_reaches_its_target(world):
    world.reset(fetch_scene(np.random.default_rng(1)))
    ctrl = HOME_QPOS.copy()
    ctrl[0] = 1.0  # shoulder_pan, from 1.5
    ctrl[3] = 0.6  # wrist_flex, from 1.0
    world.step(ctrl, int(1.0 / TIMESTEP))
    q = world.arm_qpos()
    assert abs(q[0] - 1.0) < 0.02
    assert abs(q[3] - 0.6) < 0.02


def test_cameras_render(world):
    world.reset(fetch_scene(np.random.default_rng(2)))
    for name in (ROBOT_CAM, HUMAN_CAM):
        frame = world.render_camera(name, segmentation=True)
        assert frame.rgb.shape == (48, 64, 3)
        assert frame.segmentation is not None


def test_expert_picks_in_most_scenes(world):
    """One target per scene, cycling through all five block indices (pairs
    and the single), so each index is the target in 20 of the 100 scenes."""
    wins = 0
    for seed in range(N_SCENES):
        scene = fetch_scene(np.random.default_rng(seed))
        i = seed % len(scene.blocks)
        world.reset(scene)
        result = pick(world, i)
        wins += result.success
        if result.success:
            assert result.ctrl_log.shape == (result.steps, 6)
            assert world.lifted(i)
    assert wins / N_SCENES >= GATE, f"{wins}/{N_SCENES}"


def test_out_of_reach_block_is_not_picked(world):
    scene = fetch_scene(np.random.default_rng(3))
    far = dataclasses.replace(scene, blocks=(BlockSpec("red", (0.6, 0.0)), *scene.blocks[1:]))
    world.reset(far)
    result = pick(world, 0)
    assert not result.success
    assert result.steps == 0
    assert not world.lifted(0)
