import numpy as np

from curio_robot.sim import ab
from curio_robot.sim.world import SimWorld


def test_arm_reaches_a_workspace_point():
    with SimWorld(64, 48) as world:
        assert world.move_gripper_to(np.array([0.25, 0.05, 0.0125]))


def test_ab_smoke():
    result = ab.run(trials=3, seed=1)
    rates = result["success_rate_pct"]
    assert set(rates) == {"perspective", "occlusion", "cluttered"}
    assert rates["occlusion"]["on"] >= rates["occlusion"]["off"]


def test_live_runtime_picks_the_speakers_left_before_the_vlm_answers():
    from curio_robot.sim.live import run_live

    r = run_live("perspective", seed=3, vlm_latency=0.3, quiet=True)
    assert r["success"], r
    w = r["waterfall_ms"]
    assert w["motion_start"] < 0 < w["vlm_reply"] < w["arrived"]
