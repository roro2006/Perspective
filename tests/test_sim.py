import numpy as np

from curio_robot.sim import ab
from curio_robot.sim.world import SimWorld


def test_arm_reaches_a_workspace_point():
    world = SimWorld(64, 48)
    assert world.move_gripper_to(np.array([0.25, 0.05, 0.0125]))


def test_ab_smoke():
    result = ab.run(trials=3, seed=1)
    rates = result["success_rate_pct"]
    assert set(rates) == {"perspective", "occlusion", "cluttered"}
    assert rates["occlusion"]["on"] >= rates["occlusion"]["off"]
