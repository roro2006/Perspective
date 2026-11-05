"""The run-5 demos relabelled for pi0: robot-frame words, robot cameras, no ring. Task T5-28.

pi0 only ever learns what the Curio layer will hand it: the robot-words instruction for the
episode's true target (curio_robot.layer.robot_words.instruction, from the scene rebuilt from
the episode seed), the robot camera and the wrist camera. It never sees the person's camera,
the person's own words or the magenta target ring.

The ring question: the demos store clean frames. smolvla_dataset draws the ring onto camera1
when it loads a sample (demos.draw_marker); it is not in the PNGs (checked: no ring-coloured
pixel in 612 sampled images of out/demos800b, 108 once draw_marker adds it), so nothing has to
be regenerated. ring_pixels() is the check, and the CLI repeats it.

Camera names follow pi0_base's own inputs: the robot view is base_0_rgb and the wrist view
left_wrist_0_rgb; right_wrist_0_rgb is left out (pi0 masks a missing camera anyway).

Runs in either env (no torch here):

    uv run python scripts/pi0_data.py --demos C:/Users/Rohan/dev/Perspective/out/demos800b \\
        --out C:/Users/Rohan/dev/Perspective/out/t5_28/data

writes relabel.jsonl (one line per episode: person words, robot words, relation, whether
check() reads the words back to the target) and summary.json. A LeRobotDataset v3 on disk
needs lerobot[dataset] (datasets, pyarrow), which the lerobot venv lacks and this card may not
install, so, as smolvla_dataset does, Pi0Frames reads the demos folder directly and yields
what LeRobotDataset would with pi0's delta timestamps: one state, the images in [0, 1] CHW, an
action chunk padded with the last action plus action_is_pad, and the task text."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a plain script from the lerobot venv
    sys.path.insert(0, str(REPO))

from curio_robot.collab.tasks import fetch_scene, true_map
from curio_robot.layer.robot_words import check, instruction, relation
from curio_robot.sim import demos

EVAL_FRACTION = 0.1  # as smolvla_dataset: the last 10% of episodes are held out
STATE = "observation.state"
ACTION = "action"
CAMERA_KEYS = {  # demos camera -> pi0_base input name; the person's camera is not an input
    "robot": "observation.images.base_0_rgb",
    demos.WRIST: "observation.images.left_wrist_0_rgb",
}
RING_TOLERANCE = 30  # a ring pixel is within this of (255, 0, 255) on every channel


def split(n_episodes: int, eval_fraction: float = EVAL_FRACTION) -> tuple[list[int], list[int]]:
    """(train, eval) episode indices: the last ceil(n * fraction) are held out."""
    n_eval = math.ceil(n_episodes * eval_fraction)
    return list(range(n_episodes - n_eval)), list(range(n_episodes - n_eval, n_episodes))


def episode_map(meta: dict):
    """The episode's ObjectMap, rebuilt from its seed as demos.record built the scene."""
    return true_map(fetch_scene(np.random.default_rng(meta["seed"])))


def relabel_episode(meta: dict) -> dict:
    """One relabel line: the robot-words task for the episode's true target."""
    om = episode_map(meta)
    target = meta["target_id"]
    if not any(o.object_id == target for o in om.objects):
        raise ValueError(f"episode {meta['episode_index']}: {target} not in its scene")
    words = instruction(target, om)
    phrase = relation(target, om)
    return {
        "episode_index": meta["episode_index"],
        "seed": meta["seed"],
        "target_id": target,
        "person_words": meta["words"],
        "robot_words": words,
        "relation": "none" if phrase is None else (phrase or "colour"),
        "unambiguous": check(words, om) == target,
    }


def relabel(demos_dir: Path) -> list[dict]:
    with (Path(demos_dir) / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        metas = [json.loads(line) for line in f]
    return [relabel_episode(m) for m in sorted(metas, key=lambda m: m["episode_index"])]


def summarize(rows: list[dict], lengths: list[int], train_flags: list[int]) -> dict:
    train, held_out = split(len(rows))
    return {
        "episodes": len(rows),
        "frames": int(sum(lengths)),
        "train_frames": int(sum(train_flags[i] for i in train)),
        "eval_frames": int(sum(train_flags[i] for i in held_out)),
        "train_episodes": [train[0], train[-1]] if train else [],
        "eval_episodes": [held_out[0], held_out[-1]] if held_out else [],
        "relations": dict(Counter(r["relation"] for r in rows)),
        "unambiguous": sum(r["unambiguous"] for r in rows),
        "distinct_tasks": len({r["robot_words"] for r in rows}),
        "cameras": CAMERA_KEYS,
    }


def ring_pixels(rgb: np.ndarray) -> int:
    """How many pixels have the target ring's colour (demos.MARKER_COLOUR)."""
    diff = np.abs(np.asarray(rgb, int) - np.asarray(demos.MARKER_COLOUR, int))
    return int((diff <= RING_TOLERANCE).all(axis=-1).sum())


def _chw(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(rgb, np.float32).transpose(2, 0, 1)) / 255.0


class Pi0Frames:
    """Every trainable frame of the chosen episodes as a pi0 training sample (numpy arrays;
    torch's default collate turns them into tensors). Joint data is loaded up front; PNGs are
    read per sample, so DataLoader workers share the decoding."""

    def __init__(self, demos_dir: Path, episodes: list[int], chunk: int,
                 tasks: dict[int, str], cameras: list[str] | None = None):
        self.demos_dir = Path(demos_dir)
        self.chunk = chunk
        self.cameras = cameras or list(CAMERA_KEYS)
        unknown = [c for c in self.cameras if c not in CAMERA_KEYS]
        if unknown:
            raise ValueError(f"not pi0 inputs: {unknown}")
        self.episodes = [demos.load_episode(self.demos_dir, i) for i in episodes]
        missing = [i for i in episodes if i not in tasks]
        if missing:
            raise KeyError(f"no robot words for episodes {missing[:5]}")
        self.tasks = [tasks[i] for i in episodes]
        # Recovery demos (T5-15) mark their off-target opening frames "train": false.
        self.index = [(e, f) for e, ep in enumerate(self.episodes)
                      for f, fr in enumerate(ep["frames"]) if fr.get("train", True)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int) -> dict:
        e, f = self.index[k]
        frames = self.episodes[e]["frames"]
        steps = np.arange(f, f + self.chunk)
        actions = [frames[min(s, len(frames) - 1)]["action"] for s in steps]
        sample = {
            STATE: np.asarray(frames[f]["state"], np.float32),
            ACTION: np.asarray(actions, np.float32),
            "action_is_pad": steps >= len(frames),
            "task": self.tasks[e],
        }
        for cam in self.cameras:
            rgb = np.asarray(Image.open(self.demos_dir / frames[f]["images"][cam]).convert("RGB"))
            sample[CAMERA_KEYS[cam]] = _chw(rgb)
        return sample

    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Mean/std/min/max of state and action over these episodes (for the normalizer)."""
        out = {}
        for key, field in ((STATE, "state"), (ACTION, "action")):
            x = np.asarray([fr[field] for ep in self.episodes for fr in ep["frames"]], np.float32)
            out[key] = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}
        return out


def loader_options(batch_size: int, workers: int, shuffle: bool) -> dict:
    """torch DataLoader keyword arguments for pi0 batches (the generator is the caller's).

    pin_memory stays off (T5-30). With pinned batches, lerobot's device step copies them to the
    RX 7900 XT with non_blocking=True, and inside T5-28's training loop (GPU still busy with the
    previous step) the action chunk the normalizer saw was often not the batch's: in an exact
    rerun, 5 of 8 checked steps scored a different loss than the same batch, noise and time
    re-read after a synchronise, and the action differed by up to 9 normalized units. pi0
    learned those corrupted targets (training loss 0.001, while the saved weights score about 1
    on clean batches and miss the demos by 0.47 rad open loop). Unpinned: 0 of 60 steps."""
    return {"batch_size": batch_size, "shuffle": shuffle, "num_workers": workers,
            "persistent_workers": workers > 0, "pin_memory": False, "drop_last": shuffle}


def load_tasks(relabel_path: Path) -> dict[int, str]:
    """episode index -> robot words, from relabel.jsonl."""
    with Path(relabel_path).open(encoding="utf-8") as f:
        return {r["episode_index"]: r["robot_words"] for r in map(json.loads, f)}


def check_ring(demos_dir: Path, every: int = 20, frame_every: int = 10) -> dict:
    """Ring-coloured pixels in a sample of the stored images (every camera)."""
    info = demos.load_info(demos_dir)
    images = ring = 0
    for i in range(0, info["total_episodes"], every):
        for fr in demos.load_episode(demos_dir, i)["frames"][::frame_every]:
            for rel in fr["images"].values():
                ring += ring_pixels(np.asarray(Image.open(Path(demos_dir) / rel).convert("RGB")))
                images += 1
    return {"images_checked": images, "ring_pixels": ring}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    t0 = time.perf_counter()
    rows = relabel(args.demos)
    lengths, flags = [], []
    for r in rows:
        frames = demos.load_episode(args.demos, r["episode_index"])["frames"]
        lengths.append(len(frames))
        flags.append(sum(fr.get("train", True) for fr in frames))
    summary = summarize(rows, lengths, flags)
    summary["ring_check"] = check_ring(args.demos)
    if summary["ring_check"]["ring_pixels"]:
        raise SystemExit(f"ring pixels in the stored frames: {summary['ring_check']}")
    summary["demos"] = str(args.demos)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "relabel.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
