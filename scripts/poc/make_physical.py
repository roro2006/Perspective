"""T5-20: re-render both videos with physically simulated robots.

The live scenes reuse the grounder answers the gate logged (out/physical/grounding.json): the
grounder is not asked again. Each scene's robot is rolled out once from its nominal start under
physics, its beat times are moved onto when it really arrived, and the recorded states are
played back through the T5-18 / T5-19 overlays. Scripted reel scenes keep their scripted
targets; reach, memory and pointing walk there with the G1 / Go2. Encoded with ffmpeg (x264).

    uv run python scripts/poc/make_physical.py                  # both videos
    uv run python scripts/poc/make_physical.py --stills         # PNG stills only
    uv run python scripts/poc/make_physical.py --which reel --only visibility,reach
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import compose as C
import make_reel as MR
import make_video as MV
import numpy as np
import reel_kit as R
from make_reel import outcome
from make_video import FPS, LABELS, Sim
from phys_eval import OUT, PhysSim, gate_scenes
from phys_reel import PhysMemory, PhysPointing, PhysReach, reel_live
from phys_run import rollout
from PIL import Image
from reel_scenes_a import Gaze
from reel_scenes_b import Habits, MovingFrame, Placement
from scene_base import make_request

SMALL = (1280, 720)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


class Encoder:
    """Raw RGB frames into ffmpeg (libx264, yuv420p), resized when the size differs."""

    def __init__(self, path: Path, size=(C.W, C.H), crf: int = 18) -> None:
        self.size = size
        self.proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags",
             "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0]:
            frame = np.asarray(Image.fromarray(frame).resize(self.size, Image.LANCZOS))
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit("ffmpeg failed")


class PlaybackSim(PhysSim):
    """PhysSim with the T5-18 Sim's render/project, so its overlay code runs unchanged."""

    render = Sim.render
    project = Sim.project


def load_answers() -> tuple[dict, dict]:
    """(answers, info per scene) from the gate's grounding log."""
    g = json.loads((OUT / "grounding.json").read_text())
    answers, info = {}, {}
    for q in g["queries"]:
        answers.setdefault(q["scene"], {})[q["query"]] = q["choice"]
        info.setdefault(q["scene"], {})[q["query"]] = {
            "letter": q["letter"], "p": q["p"], "latency_ms": q["latency_ms"],
            "error": q["error"]}
    return answers, info


def prepare(scene, answers: dict) -> PlaybackSim:
    """Roll the robot out from its nominal start, retime the beats, open a playback sim."""
    traj = rollout(scene, answers, seed=None)
    scene.retime(traj)
    print(f"  {scene.name}: {scene.duration:.1f} s, legs {traj.legs}, errors "
          f"{ {k: round(v, 3) for k, v in traj.errors.items()} }, fell {traj.fell}", flush=True)
    sim = PlaybackSim(scene)
    sim.answers = answers
    return sim


def views_for(sim: PlaybackSim) -> dict:
    """The marked head-camera views the inset shows (re-rendered; no grounder call)."""
    out = {}
    for q in sim.scene.queries:
        req, _, view = make_request(sim.scene, q, sim.small, sim.model, sim.data)
        out[q.key] = (view, req.letters)
    return out


# ---- the PoC video ---------------------------------------------------------------------


def poc(stills: bool, only: list[str], answers_all: dict, info_all: dict) -> None:
    MV.OUT = OUT  # overlay/explainer stills land in out/physical
    full = [] if stills else [Encoder(OUT / "curio_poc_physical.mp4"),
                              Encoder(OUT / "curio_poc_physical_720p.mp4", SMALL, 23)]
    MV.explainer(full, stills)
    tiles = []
    for scene in [s for s in gate_scenes()[:6] if not only or s.name in only]:
        print(scene.name, flush=True)
        answers, info = answers_all.get(scene.name, {}), info_all.get(scene.name, {})
        sim = prepare(scene, answers)
        images = views_for(sim)
        clip = None if stills else Encoder(OUT / f"scene_{scene.name}.mp4")
        last = MV.render_scene(sim, answers, info, images, full + ([clip] if clip else []),
                               stills)
        if clip:
            clip.close()
        geo = [q for q in scene.queries if q.frame == "speaker"]
        good = all(answers.get(q.key) == q.correct for q in geo)
        outcome_text = "picked the one they meant" if good else "grounder picked wrong"
        tiles.append((last, LABELS[scene.name], good, outcome_text))
        sim.close()
    if stills:
        Image.fromarray(C.end_grid(tiles, 3.0)).save(OUT / "still_grid.png")
        return
    for i in range(6 * FPS):
        frame = C.end_grid(tiles, i / FPS)
        for s in full:
            s.write(frame)
    for s in full:
        s.close()


# ---- the reel ------------------------------------------------------------------------------


def reel_scenes() -> list:
    vis, yourframe = reel_live()
    return [vis, PhysReach(), Gaze(), PhysPointing(), PhysMemory(), Placement(), yourframe,
            MovingFrame(), Habits()]


def reel(stills: bool, only: list[str], answers_all: dict) -> None:
    MR.OUT = OUT
    (OUT / "stills").mkdir(parents=True, exist_ok=True)
    full = [] if stills else [Encoder(OUT / "curio_reel_physical.mp4"),
                              Encoder(OUT / "curio_reel_physical_720p.mp4", SMALL, 23)]
    for i in range(round(MR.COLD * FPS)):
        if stills and i not in (40, 130):
            continue
        frame = R.cold_open(i / FPS)
        for s in full:
            s.write(frame)
    tiles = []
    for scene in [s for s in reel_scenes() if not only or s.name in only]:
        print(scene.name, flush=True)
        if hasattr(scene, "kind"):
            answers = answers_all.get(scene.name, {})
            sim = prepare(scene, answers)
            views = {k: v[0] for k, v in views_for(sim).items()}
        else:  # scripted scene, unchanged from T5-19
            sim = Sim(scene)
            sim.answers, views = {}, {}
        clip = None if stills else Encoder(OUT / f"reel_{scene.name}.mp4")
        last = MR.render_scene(sim, views, full + ([clip] if clip else []), stills)
        if clip:
            clip.close()
        good, text = outcome(scene, sim.answers)
        tiles.append((last, scene.word, good, text))
        sim.close()
    for i in range(round(MR.PAYOFF * FPS)):
        if stills and i != 240:
            continue
        frame = R.payoff(tiles, i / FPS)
        if stills:
            Image.fromarray(frame).save(OUT / "stills" / "payoff.png")
        for s in full:
            s.write(frame)
    for s in full:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["both", "poc", "reel"])
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--fake", action="store_true",
                    help="preview with each query's correct answer (no grounding log needed)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    only = [s for s in args.only.split(",") if s]
    if args.fake:
        answers = {s.name: {q.key: q.correct for q in s.queries} for s in gate_scenes()}
        info = {s.name: {q.key: {"letter": "A", "p": 1.0, "latency_ms": 0.0, "error": None}
                         for q in s.queries} for s in gate_scenes()}
    else:
        answers, info = load_answers()
    t0 = time.perf_counter()
    if args.which in ("both", "poc"):
        poc(args.stills, only, answers, info)
    if args.which in ("both", "reel"):
        reel(args.stills, only, answers)
    print(f"done in {time.perf_counter() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
