"""T5-28: pi0 data relabelling and the gate harness. CPU only: a fake policy and a stub
grounder stand in for pi0 and the served VLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pi0_data as D
import pi0_eval as E

from curio_robot.collab.tasks import fetch_scene, true_map
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.layer.robot_words import check
from curio_robot.sim import demos
from curio_robot.sim.physics import PhysWorld

SIZE = 32


class Stub:
    """A grounder that always answers with the given object id."""

    def __init__(self) -> None:
        self.answer: str | None = None

    def score(self, request) -> GroundingAnswer:
        return GroundingAnswer({self.answer: 1.0} if self.answer else {}, None, 1.0, "stub")


class Fake:
    """A policy that holds the arm still and records every task text it is given."""

    def __init__(self) -> None:
        self.cameras = ["robot", "wrist"]
        self.tasks: list[str] = []
        self.seeds: list[int] = []
        self.seen: set[str] = set()

    def reset(self, seed: int) -> None:
        self.seeds.append(seed)

    def act(self, images, state, task):
        self.tasks.append(task)
        self.seen |= set(images)
        assert all(D.ring_pixels(im) == 0 for im in images.values())
        return np.asarray(state, float)


@pytest.fixture(scope="module")
def world():
    w = PhysWorld(E.IMAGE_SIZE, E.IMAGE_SIZE)
    yield w
    w.close()


@pytest.fixture(scope="module")
def trials(world):
    kept, dropped = E.make_gate_trials(6, E.GATE_SEED, world)
    return kept, dropped


# --- data ----------------------------------------------------------------------------------


def _meta(seed: int, index: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    scene = fetch_scene(rng)
    i = int(rng.integers(0, len(scene.blocks)))
    words, frame = demos.words_for(scene, i)
    return {"episode_index": index, "seed": seed, "target_index": i,
            "target_id": true_map(scene).objects[i].object_id, "words": words, "frame": frame}


def test_relabel_gives_robot_words_that_pick_out_the_true_target():
    for seed in range(7000, 7030):
        meta = _meta(seed)
        row = D.relabel_episode(meta)
        om = D.episode_map(meta)
        assert row["robot_words"].startswith("pick up the ")
        assert "my " not in row["robot_words"] and " me" not in row["robot_words"]
        assert check(row["robot_words"], om) == meta["target_id"]
        assert row["unambiguous"] and row["person_words"] == meta["words"]


def test_relabel_rejects_a_target_not_in_the_rebuilt_scene():
    meta = {**_meta(7000), "target_id": "purple_9"}
    with pytest.raises(ValueError):
        D.relabel_episode(meta)


def _fake_demos(root: Path, n_frames: int = 3) -> Path:
    """A two-episode demos folder with clean frames (what demos.record stores)."""
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    metas = [_meta(7000, 0), _meta(7001, 1)]
    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    info = {"total_episodes": 2, "image_size": {"width": SIZE, "height": SIZE},
            "cameras": ["robot", "human", "wrist"]}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    for e in range(2):
        (root / "images" / f"episode_{e:06d}").mkdir(parents=True)
        with (root / "data" / f"episode_{e:06d}.jsonl").open("w") as f:
            for k in range(n_frames):
                paths = {}
                for c, shade in (("robot", 60), ("human", 120), ("wrist", 180)):
                    rel = f"images/episode_{e:06d}/{c}_{k:04d}.png"
                    Image.fromarray(np.full((SIZE, SIZE, 3), shade + k, np.uint8)).save(root / rel)
                    paths[c] = rel
                f.write(json.dumps({"state": [k] * 6, "action": [k + 0.5] * 6,
                                    "images": paths}) + "\n")
    return root


def test_frames_carry_robot_words_robot_cameras_and_a_padded_chunk(tmp_path):
    root = _fake_demos(tmp_path / "demos")
    rows = D.relabel(root)
    tasks = {r["episode_index"]: r["robot_words"] for r in rows}
    ds = D.Pi0Frames(root, [0, 1], chunk=4, tasks=tasks)
    assert len(ds) == 6
    s = ds[2]  # episode 0, last frame
    assert s["task"] == tasks[0] and s["task"] != rows[0]["person_words"]
    assert set(s) == {D.STATE, D.ACTION, "action_is_pad", "task", *D.CAMERA_KEYS.values()}
    assert not any("camera2" in k or "human" in k for k in s)
    assert s[D.ACTION].shape == (4, 6) and s["action_is_pad"].tolist() == [False, True, True, True]
    assert np.allclose(s[D.ACTION][:, 0], 2.5)
    img = s[D.CAMERA_KEYS["robot"]]
    assert img.shape == (3, SIZE, SIZE) and img.dtype == np.float32
    assert np.isclose(img.max(), 62 / 255)  # robot shade, not the person's; no ring drawn
    assert ds.stats()[D.STATE]["mean"].shape == (6,)
    with pytest.raises(KeyError):
        D.Pi0Frames(root, [0, 1], chunk=4, tasks={0: "x"})


def test_ring_pixels_sees_the_smolvla_ring_and_nothing_else():
    clean = np.full((64, 64, 3), 100, np.uint8)
    assert D.ring_pixels(clean) == 0
    assert D.ring_pixels(demos.draw_marker(clean, (32.0, 32.0))) > 50


def test_split_holds_out_the_last_tenth():
    train, held = D.split(800)
    assert train[-1] == 719 and held == list(range(720, 800))


# --- eval harness --------------------------------------------------------------------------


def test_gate_trials_use_words_whose_robot_reading_picks_the_partner(trials):
    from curio_robot import perspective

    kept, dropped = trials
    assert [gt.seed for gt in kept] + dropped == list(range(51000, 51006))
    assert any(gt.rule == "distance" for gt in kept)  # seed 51004: the pair is along the view
    for gt in kept:
        om = true_map(gt.trial.scene)
        pair = [om.objects[gt.trial.target], om.objects[gt.trial.partner]]
        rel = {"left": "left", "right": "right", "to me": "nearest", "from me": "farthest"}
        reading = next(v for k, v in rel.items() if gt.words.endswith(k))
        assert perspective.pick(pair, reading, om.robot_pose) is pair[1]
        assert gt.words.startswith(f"Hand me the {pair[0].color} one")


def test_conditions_give_raw_words_curio_words_and_oracle_words(world, trials):
    kept, _ = trials
    gt = kept[0]
    om, _, _ = E.scene_view(world, gt.trial)
    target, partner = (om.objects[i].object_id for i in (gt.trial.target, gt.trial.partner))
    g = Stub()
    g.answer = target
    right = E.goal_row(world, gt, grounder=g)
    g.answer = partner
    wrong = E.goal_row(world, gt, grounder=g)
    g.answer = None
    none = E.goal_row(world, gt, grounder=g)

    assert right["curio_words"] == right["oracle_words"] == check_words(right, om, target)
    assert wrong["curio_words"] != wrong["oracle_words"]
    assert check(wrong["curio_words"], om) == partner
    assert none["goal_id"] is None and none["error"]
    assert E.inputs("pi0", "raw", right) == (gt.words, None)
    assert E.inputs("pi0", "curio", wrong) == (wrong["curio_words"], None)
    assert E.inputs("pi0", "oracle", wrong) == (right["oracle_words"], None)
    assert E.inputs("pi0", "curio", none) == (None, None)
    assert E.inputs("smolvla", "raw", right) == (gt.words, None)
    assert E.inputs("smolvla", "oracle", right) == (gt.words, right["oracle_ring"])
    assert right["curio_ring"] == pytest.approx(right["oracle_ring"], abs=0.02)
    assert E.inputs("smolvla", "curio", none) == (None, None)


def check_words(row, om, target):
    assert check(row["oracle_words"], om) == target
    return row["oracle_words"]


def test_rollout_feeds_the_task_and_scores_a_still_arm_as_nothing(world, trials):
    kept, _ = trials
    gt = kept[1]
    fake = Fake()
    g = {"person_words": gt.words, "curio_words": "pick up the x block", "oracle_words": "o",
         "goal_id": "x", "curio_ring": None, "oracle_ring": None}
    row = E.run_condition(world, fake, "pi0", "curio", gt, g)
    assert row["outcome"] == "nothing" and row["frames"] == round(E.HORIZON_S * demos.FPS)
    assert set(fake.tasks) == {"pick up the x block"} and fake.seeds == [gt.seed]
    assert fake.seen == {"robot", "wrist"}
    skipped = E.run_condition(world, fake, "pi0", "curio", gt, {**g, "curio_words": None})
    assert skipped["outcome"] == "no_goal" and fake.seeds == [gt.seed]


def _rows(policy, rates: dict[str, int], n: int = 100) -> list[dict]:
    return [{"policy": policy, "condition": c, "seed": s,
             "outcome": "target" if s < k else "partner"}
            for c, k in rates.items() for s in range(n)]


def test_gate_lines_and_verdicts():
    seeds = list(range(100))
    ok = E.gate(_rows("pi0", {"raw": 20, "curio": 60, "oracle": 65}), seeds)
    assert ok["verdict"] == "PASS" and [x["pass"] for x in ok["lines"]] == [True] * 3
    assert ok["pi0"]["B"]["rate"] == 0.6 and ok["lines"][0]["value"] == 0.4
    lo, hi = ok["lines"][0]["ci95"]
    assert lo <= 0.4 <= hi and hi - lo < 0.3
    assert "smolvla_run5" not in ok

    small_gain = E.gate(_rows("pi0", {"raw": 40, "curio": 60, "oracle": 65}), seeds)
    assert small_gain["verdict"] == "FAIL (line 1)"
    costly = E.gate(_rows("pi0", {"raw": 10, "curio": 50, "oracle": 65}), seeds)
    assert [x["pass"] for x in costly["lines"]] == [True, False, True]
    weak = E.gate(_rows("pi0", {"raw": 0, "curio": 28, "oracle": 29}), seeds)
    assert weak["verdict"].startswith("FAIL (line 3")
    edge = E.gate(_rows("pi0", {"raw": 0, "curio": 30, "oracle": 30}), seeds)
    assert edge["verdict"] == "PASS"  # >= at the boundaries

    both = E.gate(_rows("pi0", {"raw": 20, "curio": 60, "oracle": 65})
                  + _rows("smolvla", {"raw": 5, "curio": 50, "oracle": 55}), seeds)
    assert both["smolvla_run5"]["B"]["rate"] == 0.5
    with pytest.raises(ValueError):
        E.gate(_rows("pi0", {"raw": 1, "curio": 1}), seeds)


def test_report_markdown_has_every_line_and_policy():
    seeds = list(range(100))
    res = E.gate(_rows("pi0", {"raw": 20, "curio": 60, "oracle": 65})
                 + _rows("smolvla", {"raw": 5, "curio": 50, "oracle": 55}), seeds)
    res.update(seeds=list(range(51000, 51100)), rules={"side": 84, "distance": 16},
               adapter="r16-v4", checkpoints={"pi0": "ckpt"},
               grounding={"n": 100, "correct": 93, "errors": 0, "resolve_ms_p50": 120.0})
    md = E.markdown(res)
    for text in ("PASS", "| 1 |", "| 2 |", "| 3 |", "SmolVLA run 5", "0.600", "0.500",
                 "93/100"):
        assert text in md
