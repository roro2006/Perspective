"""Recorded pick demonstrations (task T5-05): a small recording round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot.collab.tasks import TaskScene, fetch_scene
from curio_robot.sim import expert
from curio_robot.sim.demos import (
    FORMAT,
    FPS,
    MARKER_COLOUR,
    MARKER_MODES,
    MAX_FRAMES,
    WRIST,
    draw_marker,
    load_episode,
    load_info,
    marker_uv,
    record,
    render_wrist,
    words_for,
)
from curio_robot.sim.physics import ARM_JOINTS, PhysWorld
from curio_robot.sim.world import BlockSpec

W, H = 160, 120
N = 3
FRAME_KEYS = {"episode_index", "frame_index", "timestamp", "state", "action", "target_pixel",
              "words", "images"}
EPISODE_KEYS = {"episode_index", "seed", "target_index", "target_id", "words", "frame",
                "target_pixel", "success", "reason", "length"}


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    out = tmp_path_factory.mktemp("demos")
    summary = record(N, seed=0, out_dir=out, width=W, height=H)
    return out, summary


def test_summary_and_info(recording):
    out, summary = recording
    assert summary["episodes"] == N
    assert summary["attempts"] == summary["successes"] + summary["failed"]
    info = load_info(out)
    assert info["format"] == FORMAT
    assert info["fps"] == FPS
    assert info["image_size"] == {"width": W, "height": H}
    assert info["state_names"] == list(ARM_JOINTS) == info["action_names"]
    assert info["total_episodes"] == N
    assert info["total_frames"] == summary["frames"]


def test_every_frame_has_every_field(recording):
    out, _ = recording
    total = 0
    for i in range(N):
        ep = load_episode(out, i, images=True)
        meta, frames = ep["episode"], ep["frames"]
        assert set(meta) == EPISODE_KEYS
        assert meta["episode_index"] == i
        assert meta["success"] is True  # failed picks are skipped by default
        assert meta["length"] == len(frames) > 20
        assert meta["words"].startswith("Hand me the ")
        total += len(frames)
        for k, frame in enumerate(frames):
            assert FRAME_KEYS <= set(frame)
            assert frame["frame_index"] == k
            assert frame["timestamp"] == pytest.approx(k / FPS)
            assert len(frame["state"]) == 6 and len(frame["action"]) == 6
            assert frame["words"] == meta["words"]
            uv = frame["target_pixel"]
            assert uv is None or (len(uv) == 2 and 0 <= uv[0] < W and 0 <= uv[1] < H)
            for cam in ("robot", "human", WRIST):
                assert (out / frame["images"][cam]).exists()
                assert frame[f"{cam}_rgb"].shape == (H, W, 3)
        assert frames[0]["target_pixel"] is not None  # the target starts in view
        # The arm moves: the state at the end differs from the start.
        assert np.abs(np.subtract(frames[-1]["state"], frames[0]["state"])).max() > 0.1
    assert total == load_info(out)["total_frames"]


def test_refuses_to_overwrite(recording):
    out, _ = recording
    with pytest.raises(FileExistsError):
        record(1, seed=0, out_dir=out, width=W, height=H)


def test_words_name_the_side_in_the_persons_frame():
    # Person at x=0.7 facing -x: their left is -y.
    scene = TaskScene(
        "fetch",
        (BlockSpec("red", (0.2, 0.08)), BlockSpec("red", (0.2, -0.08)),
         BlockSpec("blue", (0.15, 0.0))),
        np.array([0.7, 0.0, 0.35]), np.array([0.2, 0.0, 0.0]), None, 0.0,
    )
    assert words_for(scene, 0) == ("Hand me the red one on my right", "speaker")
    assert words_for(scene, 1) == ("Hand me the red one on my left", "speaker")
    assert words_for(scene, 2) == ("Hand me the blue one", None)


def test_draw_marker_rings_the_target_and_keeps_the_input():
    rgb = np.zeros((64, 64, 3), np.uint8)
    out = draw_marker(rgb, (32.0, 32.0), radius=10, width=2)
    assert not rgb.any()  # the input is not touched
    assert tuple(out[32, 32 + 10]) == MARKER_COLOUR  # on the ring (pixel centre 42.5 ~ 32 + 10)
    assert not out[32, 32].any()  # the centre stays clear, so the block shows through
    assert not out[0, 0].any()
    ring = (out == MARKER_COLOUR).all(axis=-1)
    assert 40 < ring.sum() < 200  # a thin ring, not a disc


def test_draw_marker_handles_missing_and_edge_targets():
    rgb = np.full((32, 48, 3), 7, np.uint8)
    assert np.array_equal(draw_marker(rgb, None), rgb)
    edge = draw_marker(rgb, (0.0, 0.0), radius=6, width=2)  # partly off the image
    assert edge.shape == rgb.shape and (edge == MARKER_COLOUR).all(axis=-1).any()


def test_marker_uv_modes():
    episode = {"target_pixel": [10.0, 20.0]}
    frame = {"target_pixel": [12.0, 5.0]}
    assert marker_uv(episode, frame) == [10.0, 20.0]
    assert marker_uv(episode, frame, "track") == [12.0, 5.0]
    assert set(MARKER_MODES) == {"start", "track"}
    with pytest.raises(ValueError):
        marker_uv(episode, frame, "none")


def test_varied_words_keep_the_target_and_vary_the_wording():
    import numpy as np

    from curio_robot.collab.tasks import fetch_scene
    from curio_robot.sim.demos import words_varied

    texts = set()
    for seed in range(40):
        scene = fetch_scene(np.random.default_rng(seed))
        for i in range(len(scene.blocks)):
            text, frame = words_varied(scene, i, np.random.default_rng([seed, i]))
            assert scene.blocks[i].color in text
            assert (frame is None) == (i == 4)  # block 4 is the unique colour
            texts.add(text.split(" the ")[0] + ("|your" if " your " in text else ""))
    assert {"Hand me", "Hand me|your"} <= texts and len(texts) >= 4


def test_info_lists_the_wrist_camera(recording):
    out, _ = recording
    assert load_info(out)["cameras"] == ["robot", "human", WRIST]


def test_without_wrist_the_folder_has_two_cameras(tmp_path):
    record(1, seed=0, out_dir=tmp_path, width=W, height=H, wrist=False)
    assert load_info(tmp_path)["cameras"] == ["robot", "human"]
    frame = load_episode(tmp_path, 0, images=True)["frames"][0]
    assert set(frame["images"]) == {"robot", "human"} and f"{WRIST}_rgb" not in frame


def test_wrist_view_sees_the_jaw_and_keeps_the_near_plane():
    from curio_robot.collab.tasks import fetch_scene
    from curio_robot.sim.physics import PhysWorld

    with PhysWorld(W, H) as world:
        world.reset(fetch_scene(np.random.default_rng(0)))
        znear = world.model.vis.map.znear
        robot_before = world.render_camera("robot").rgb
        rgb = render_wrist(world)
        assert world.model.vis.map.znear == znear  # restored for the other cameras
        assert np.array_equal(world.render_camera("robot").rgb, robot_before)
    assert rgb.shape == (H, W, 3)
    # The yellow fixed jaw fills part of the bottom of the view (red and green high, blue low).
    bottom = rgb[H // 2:].astype(int)
    jaw = (bottom[..., 0] > 100) & (bottom[..., 1] > 80) & (bottom[..., 2] < 60)
    assert jaw.mean() > 0.02


def test_recovery_demos_mark_their_off_target_frames(tmp_path):
    summary = record(2, seed=3000, out_dir=tmp_path, width=W, height=H, recovery=1.0, wrist=False)
    assert summary["episodes"] == 2
    assert load_info(tmp_path)["recovery"] == 1.0
    for i in range(2):
        ep = load_episode(tmp_path, i)
        skip = ep["episode"]["skipped_frames"]
        assert ep["episode"]["recovery"]
        if skip:
            flags = [f["train"] for f in ep["frames"]]
            assert flags == [False] * skip + [True] * (len(flags) - skip)
        else:  # a plan the expert can't shift runs as a plain pick
            assert all("train" not in f for f in ep["frames"])


def test_noisy_pick_labels_the_clean_command_and_closes_only_straddling():
    rng = np.random.default_rng(3000)
    scene = fetch_scene(rng)
    i = int(rng.integers(0, len(scene.blocks)))
    world = PhysWorld(W, H)
    log = []
    try:
        world.reset(scene)

        def on_tick(w, executed, label):
            log.append((executed, label, expert.straddles(w, i)))

        result = expert.noisy_pick(world, i, np.random.default_rng(1), 0.01, on_tick)
    finally:
        world.close()
    assert result.success
    assert any(np.abs(ex - lab).max() > 1e-3 for ex, lab, _ in log)  # noise was executed
    first_close = next(k for k, (_, lab, _) in enumerate(log) if lab[5] < expert.JAW_OPEN)
    assert log[first_close - 1][2]  # the jaws straddled the block on the tick before
    assert all(np.allclose(ex, lab) for ex, lab, _ in log[first_close:])


def test_noisy_demos_stay_within_the_horizon(tmp_path):
    summary = record(2, seed=5000, out_dir=tmp_path, width=W, height=H, noise=0.012, wrist=False)
    assert summary["episodes"] == 2
    assert load_info(tmp_path)["noise"] == 0.012
    for i in range(2):
        ep = load_episode(tmp_path, i)
        assert len(ep["frames"]) <= MAX_FRAMES
        assert {"noise_m", "retries", "settle_s"} <= ep["episode"].keys()
