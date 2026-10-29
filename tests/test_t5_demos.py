"""Recorded pick demonstrations (task T5-05): a small recording round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot.collab.tasks import TaskScene
from curio_robot.sim.demos import FORMAT, FPS, load_episode, load_info, record, words_for
from curio_robot.sim.physics import ARM_JOINTS
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
            for cam in ("robot", "human"):
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
