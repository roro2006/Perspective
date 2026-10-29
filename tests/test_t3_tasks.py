"""T3-04: the multi-step fetch scene and its ground-truth map."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from curio_robot.collab.tasks import BLOCK_Z, PAIR_HALF, apply_scene, fetch_scene, true_map
from curio_robot.sim.scenarios import COLORS, MIN_GAP, WORKSPACE_X, WORKSPACE_Y
from curio_robot.sim.world import ROBOT_CAM_POS, SimWorld


def xy(block) -> np.ndarray:
    return np.asarray(block.xy, float)


@pytest.mark.parametrize("seed", range(20))
def test_default_scene_shape(seed):
    s = fetch_scene(np.random.default_rng(seed))
    assert s.kind == "fetch" and len(s.blocks) == 5
    colors = [b.color for b in s.blocks]
    assert colors[0] == colors[1] and colors[2] == colors[3] and colors[0] != colors[2]
    assert colors[4] not in colors[:4] and set(colors) <= set(COLORS)
    assert sorted(Counter(colors).values()) == [1, 2, 2]
    for b in s.blocks:
        assert WORKSPACE_X[0] <= b.xy[0] <= WORKSPACE_X[1]
        assert WORKSPACE_Y[0] <= b.xy[1] <= WORKSPACE_Y[1]
    for i in range(5):
        for j in range(i + 1, 5):
            assert np.linalg.norm(xy(s.blocks[i]) - xy(s.blocks[j])) > MIN_GAP
    for a, b in ((0, 1), (2, 3)):
        gap = np.linalg.norm(xy(s.blocks[a]) - xy(s.blocks[b]))
        assert 2 * PAIR_HALF[0] - 1e-9 <= gap <= 2 * PAIR_HALF[1] + 1e-9
    assert s.occluder is None and s.occluder_yaw == 0.0
    assert np.shape(s.human_eye) == (3,) and np.shape(s.human_look) == (3,)


def test_reproducible():
    a, b = fetch_scene(np.random.default_rng(3)), fetch_scene(np.random.default_rng(3))
    assert [(x.color, tuple(x.xy)) for x in a.blocks] == [(x.color, tuple(x.xy)) for x in b.blocks]
    assert np.allclose(a.human_eye, b.human_eye)


@pytest.mark.parametrize("seed", range(10))
def test_occluder_sits_between_the_single_and_the_robot(seed):
    s = fetch_scene(np.random.default_rng(seed), occlude=True)
    single, occ = xy(s.blocks[4]), np.asarray(s.occluder, float)
    assert np.linalg.norm(occ - single) == pytest.approx(0.045)
    cam = ROBOT_CAM_POS[:2]
    assert np.linalg.norm(occ - cam) < np.linalg.norm(single - cam)
    d = occ - single
    assert s.occluder_yaw == pytest.approx(float(np.arctan2(d[1], d[0])))


def test_bad_arguments():
    with pytest.raises(ValueError):
        fetch_scene(np.random.default_rng(0), n_pairs=3, n_single=2)
    with pytest.raises(ValueError):
        fetch_scene(np.random.default_rng(0), n_single=0, occlude=True)


def test_other_sizes():
    s = fetch_scene(np.random.default_rng(1), n_pairs=1, n_single=2)
    assert sorted(Counter(b.color for b in s.blocks).values()) == [1, 1, 2]


def test_true_map():
    s = fetch_scene(np.random.default_rng(2))
    m = true_map(s)
    c0, c1, c4 = s.blocks[0].color, s.blocks[2].color, s.blocks[4].color
    assert [o.object_id for o in m.objects] == [
        f"{c0}_1", f"{c0}_2", f"{c1}_1", f"{c1}_2", f"{c4}_1"]
    for o, b in zip(m.objects, s.blocks):
        assert o.color == b.color and np.allclose(o.position, [*b.xy, BLOCK_Z])
        assert o.seen_by == set()
    assert np.allclose(m.human_pose.position, s.human_eye)
    assert np.allclose(m.robot_pose.position, ROBOT_CAM_POS)
    toward = s.human_look - s.human_eye
    assert np.dot(m.human_pose.forward, toward / np.linalg.norm(toward)) == pytest.approx(1.0)


def test_apply_scene():
    s = fetch_scene(np.random.default_rng(4), occlude=True)
    world = SimWorld(64, 48)
    apply_scene(world, s)
    placed = world.block_positions()
    assert sorted(c for c, _ in placed) == sorted(b.color for b in s.blocks)
    for b in s.blocks:
        assert any(c == b.color and np.allclose(p[:2], b.xy, atol=1e-6) for c, p in placed)
