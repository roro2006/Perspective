"""Curio pick demonstrations as SmolVLA training samples. Task T5-07.

Runs in the lerobot venv, not the project env:

    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe scripts/smolvla_dataset.py \\
        --demos out/demos300 --root out/lerobot/curio_pick [--marker start] [--overwrite]

SmolVLA's base checkpoint was trained on observation.images.camera1..3, observation.state (6)
and action (6), so the samples use those names: camera1 is the robot view with the target ring
drawn on it (demos.draw_marker, at conversion time), camera2 the person's view, camera3 absent
(the policy drops a missing camera). The task text is the spoken words. The last EVAL_FRACTION
of episodes are held out for evaluation.

Two ways to the same samples:
- `convert` writes a local LeRobotDataset v3 (images as images, no video, no hub push). It
  needs lerobot[dataset] (datasets, pyarrow, pandas, av), which the lerobot venv lacks as of
  T5-07; installing them was not approved, so this path is written against lerobot 0.6.1's
  API but not yet run.
- `CurioPickFrames` reads the demos folder directly and yields what LeRobotDataset would with
  SmolVLA's delta_timestamps: one state, two images in [0, 1] CHW, a 50-step action chunk
  padded with the last action plus action_is_pad, and the task. scripts/smolvla_train.py
  trains on it."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run as a plain script from the lerobot venv
    sys.path.insert(0, str(REPO))

from curio_robot.sim import demos

REPO_ID = "curio/curio_pick"  # local only: never pushed
EVAL_FRACTION = 0.1
ROBOT_TYPE = "so101"
STATE = "observation.state"
ACTION = "action"
CAMERA_KEYS = {"robot": "observation.images.camera1", "human": "observation.images.camera2"}


def split(n_episodes: int, eval_fraction: float = EVAL_FRACTION) -> tuple[list[int], list[int]]:
    """(train, eval) episode indices: the last ceil(n * fraction) are held out."""
    n_eval = math.ceil(n_episodes * eval_fraction)
    return list(range(n_episodes - n_eval)), list(range(n_episodes - n_eval, n_episodes))


def features(info: dict) -> dict:
    """LeRobot feature spec for a demos folder's info.json."""
    h, w = info["image_size"]["height"], info["image_size"]["width"]
    image = {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {
        STATE: {"dtype": "float32", "shape": (6,), "names": info["state_names"]},
        ACTION: {"dtype": "float32", "shape": (6,), "names": info["action_names"]},
        CAMERA_KEYS["robot"]: dict(image),
        CAMERA_KEYS["human"]: dict(image),
    }


class CurioPickFrames(torch.utils.data.Dataset):
    """Every frame of the chosen episodes as a LeRobot-style training sample.

    Joint data is loaded up front (small); PNGs are read per sample, so DataLoader workers
    share the decoding."""

    def __init__(self, demos_dir: Path, episodes: list[int], chunk: int, marker: str = "start"):
        self.demos_dir = Path(demos_dir)
        self.chunk = chunk
        self.marker = marker
        self.episodes = [demos.load_episode(self.demos_dir, i) for i in episodes]
        self.index = [(e, f) for e, ep in enumerate(self.episodes) for f in range(len(ep["frames"]))]

    def __len__(self) -> int:
        return len(self.index)

    def _image(self, rel: str) -> np.ndarray:
        return np.asarray(Image.open(self.demos_dir / rel).convert("RGB"))

    def __getitem__(self, k: int) -> dict:
        e, f = self.index[k]
        ep = self.episodes[e]
        frames = ep["frames"]
        frame = frames[f]
        steps = np.arange(f, f + self.chunk)
        actions = np.asarray([frames[min(s, len(frames) - 1)]["action"] for s in steps], np.float32)
        robot = demos.draw_marker(self._image(frame["images"]["robot"]),
                                  demos.marker_uv(ep["episode"], frame, self.marker))
        human = self._image(frame["images"]["human"])
        return {
            STATE: torch.tensor(frame["state"], dtype=torch.float32),
            ACTION: torch.from_numpy(actions),
            "action_is_pad": torch.from_numpy(steps >= len(frames)),
            CAMERA_KEYS["robot"]: _chw(robot),
            CAMERA_KEYS["human"]: _chw(human),
            "task": ep["episode"]["words"],
        }

    def stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Mean/std/min/max of state and action over these episodes (for the normalizer)."""
        out = {}
        for key, field in ((STATE, "state"), (ACTION, "action")):
            x = np.asarray([fr[field] for ep in self.episodes for fr in ep["frames"]], np.float32)
            out[key] = {
                "mean": torch.from_numpy(x.mean(0)),
                "std": torch.from_numpy(x.std(0)),
                "min": torch.from_numpy(x.min(0)),
                "max": torch.from_numpy(x.max(0)),
            }
        return out


def _chw(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0


def convert(demos_dir: Path, root: Path, marker: str, overwrite: bool) -> dict:
    """Write a LeRobotDataset under root and a side file <root>.curio.json (split, marker,
    words) beside it; return a summary. Needs lerobot[dataset]."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = demos.load_info(demos_dir)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    dataset = LeRobotDataset.create(
        REPO_ID, fps=info["fps"], features=features(info), root=root, robot_type=ROBOT_TYPE,
        use_videos=False, image_writer_threads=4,
    )
    words: list[str] = []
    try:
        for i in range(info["total_episodes"]):
            ep = demos.load_episode(demos_dir, i, images=True)
            meta = ep["episode"]
            for frame in ep["frames"]:
                uv = demos.marker_uv(meta, frame, marker)
                dataset.add_frame({
                    STATE: np.asarray(frame["state"], np.float32),
                    ACTION: np.asarray(frame["action"], np.float32),
                    CAMERA_KEYS["robot"]: demos.draw_marker(frame["robot_rgb"], uv),
                    CAMERA_KEYS["human"]: frame["human_rgb"],
                    "task": meta["words"],
                })
            dataset.save_episode()
            words.append(meta["words"])
    finally:
        dataset.finalize()

    train, held_out = split(len(words))
    side = {
        "demos": str(demos_dir),
        "repo_id": REPO_ID,
        "marker": marker_settings(marker),
        "train_episodes": train,
        "eval_episodes": held_out,
        "words": words,
    }
    side_path = root.parent / f"{root.name}.curio.json"
    side_path.write_text(json.dumps(side, indent=1), encoding="utf-8")
    size_mb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
    return {"episodes": len(words), "frames": dataset.meta.total_frames,
            "train": len(train), "eval": len(held_out), "size_mb": round(size_mb, 1)}


def marker_settings(mode: str) -> dict:
    return {"mode": mode, "radius": demos.MARKER_RADIUS, "width": demos.MARKER_WIDTH,
            "colour": list(demos.MARKER_COLOUR)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--marker", choices=demos.MARKER_MODES, default="start")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    summary = convert(args.demos, args.root, args.marker, args.overwrite)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
