"""T5-19: render the versatility reel, "Whatever you mean" (~75 s, 1920x1080, 30 fps).

A 5 s cold open, nine scenes and a 10 s payoff grid. The visibility and your-frame scenes ask
the served grounder (Qwen3-VL-4B Q8_0 + the r16 v2 adapter on llama-server :8081) once each,
in the speaker's frame (the geometry condition) and in the robot's own frame, and the robot
goes to whatever it picked; every query and answer goes to out/poc_reel/grounding.json. The
other scenes are scripted: the robot goes to the right target by construction and nothing
is asked. Reuses the T5-18 kit (Sim, make_request, compose, encode_cv2) without changing it.

    uv run python scripts/poc/make_reel.py                      # everything (grounder on)
    uv run python scripts/poc/make_reel.py --stills --no-ground # preview PNGs, no grounder
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import compose as C
import mujoco
import numpy as np
import reel_kit as R
from kit import cam_pose
from make_video import CV2_PY, ENCODER, FPS, SERVER, URL, Sim
from PIL import Image, ImageDraw
from reel_scenes_a import scenes as scenes_a
from reel_scenes_b import scenes as scenes_b
from scene_base import make_request

from curio_robot.grounding.client import VllmGrounder

OUT = Path("C:/Users/Rohan/dev/Perspective/out/poc_reel")
COLD, PAYOFF = 5.0, 10.0
SMALL_W, SMALL_H = 1280, 720
GAZE = (210, 235, 255)


class Encoder:
    def __init__(self, path: Path, w: int = C.W, h: int = C.H) -> None:
        self.size = (w, h)
        self.proc = subprocess.Popen([CV2_PY, ENCODER, str(path), str(w), str(h), str(FPS)],
                                     stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0]:
            frame = np.asarray(Image.fromarray(frame).resize(self.size, Image.LANCZOS))
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit("encoder failed")


def all_scenes() -> list[R.ReelScene]:
    return scenes_a() + scenes_b()


# ---- grounding -----------------------------------------------------------------------------


def ground(sim: Sim, grounder: VllmGrounder | None, log: list) -> tuple[dict, dict]:
    """Ask the live grounder each of the scene's queries once; (answers, marked views)."""
    scene = sim.scene
    answers, views = {}, {}
    if grounder is None:
        return answers, views
    for q in scene.queries:
        req, img, view = make_request(scene, q, sim.small, sim.model, sim.data)
        ans = grounder.score(req)
        choice = max(ans.probs, key=ans.probs.get) if ans.probs else None
        answers[q.key] = choice
        views[q.key] = view
        name = f"grounder_{scene.name}_{q.key}.png"
        Image.fromarray(img).save(OUT / name)
        p = ans.probs.get(choice, 0.0) if choice else 0.0
        log.append({"scene": scene.name, "query": q.key, "frame": q.frame, "text": q.text,
                    "prompt": req.messages[0]["content"][1]["text"], "letters": req.letters,
                    "emitted_letter": ans.letter, "probs": ans.probs, "choice": choice,
                    "correct": q.correct, "ok": choice == q.correct, "p": round(p, 4),
                    "latency_ms": round(ans.latency_ms, 1), "source": ans.source,
                    "error": ans.error, "image": name})
        print(f"  {scene.name}/{q.key}: {choice} (correct {q.correct}) p={p:.2f} "
              f"{ans.latency_ms:.0f} ms {ans.error or ''}", flush=True)
    return answers, views


# ---- frames --------------------------------------------------------------------------------


def head_view(sim: Sim) -> np.ndarray:
    sim.small.update_scene(sim.data, camera=sim.scene.inset_cam())
    if sim.scene.fog:
        sim.small.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
    return sim.small.render().copy()


def flashback(sim: Sim, t: float, length: float) -> np.ndarray:
    """The memory flashback: the head camera full screen, warm-tinted, vignetted."""
    sim.big.update_scene(sim.data, camera=sim.scene.inset_cam())
    rgb = sim.big.render().astype(float) / 255.0
    grey = rgb @ np.array([0.3, 0.59, 0.11])
    grey = np.clip(grey * 1.6, 0, 1)
    tint = np.stack([grey * 1.0 + 0.08, grey * 0.88 + 0.04, grey * 0.68], -1)
    yy, xx = np.mgrid[0:C.H, 0:C.W]
    vig = 1 - 0.55 * (((xx - C.W / 2) / C.W) ** 2 + ((yy - C.H / 2) / C.H) ** 2) * 2.2
    out = np.clip(tint * vig[..., None], 0, 1)
    a = max(0.0, min(1.0, t / 0.2, (length - t) / 0.15))
    return (out * 255 * a).astype(np.uint8)


def proj(sim: Sim, p):
    if p is None:
        return None
    px = sim.project(p)
    if px is None:
        return None
    return float(px[0]), float(px[1])


def draw_beats(sim: Sim, d: ImageDraw.ImageDraw, t: float, beats) -> float:
    """Draw every live beat; returns how dark the 'dip' beat wants the frame (0-1)."""
    scene = sim.scene
    dark = 0.0
    for b in beats:
        end = scene.duration + 1 if b.t1 >= scene.duration - 1e-6 else b.t1
        a = C.alpha_of(t, b.t0, end, 0.2)
        if a <= 0:
            continue
        if b.kind == "say":
            C.subtitle(d, b.text, a)
        elif b.kind in ("x", "ok"):
            px = proj(sim, b.pos)
            if px is not None:
                x, y = float(np.clip(px[0], 80, C.W - 80)), float(np.clip(px[1], 160, C.H - 250))
                px = (C.W - 500.0 if x > C.W - 480 and y < 420 else x, y)
                r = 30 + 18 * min(1.0, (t - b.t0) / 0.25)
                C.check_mark(d, px, int(r), b.kind == "ok", a)
        elif b.kind in ("ray", "badray"):
            p0, p1 = proj(sim, b.pos), proj(sim, b.pos2)
            if p0 and p1:
                R.ray(d, p0, p1, C.GREEN if b.kind == "ray" else C.RED, 7, a)
        elif b.kind == "gaze":
            pose = cam_pose(sim.model, sim.data, scene.inset_cam())
            p0, p1 = proj(sim, pose.position), proj(sim, scene.gaze_hit(pose))
            if p0 and p1:
                R.ray(d, p0, p1, GAZE, 5, 0.6 * a)
        elif b.kind == "pulse":
            px = proj(sim, b.pos)
            if px is not None:
                R.pulse(d, px, t - b.t0, b.text, a)
        elif b.kind == "trail":
            for k, p in enumerate(scene.trail_points(t)):
                px = proj(sim, p)
                if px is not None:
                    r = 9 - 0.6 * k
                    d.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r),
                              fill=(80, 200, 220, int(200 * a * (1 - k / 11))))
        elif b.kind == "dip":
            mid, half = (b.t0 + b.t1) / 2, (b.t1 - b.t0) / 2
            dark = max(dark, 1 - abs(t - mid) / half)
    return dark


def frame_at(sim: Sim, t: float, beats, views: dict) -> tuple[np.ndarray, np.ndarray]:
    """(composited frame, raw cinematic render) at time t."""
    scene = sim.scene
    rgb = sim.render(t, sim.answers)
    flash = next((b for b in beats if b.kind == "flash" and b.t0 <= t < b.t1), None)
    if flash is not None:
        return flashback(sim, t - flash.t0, flash.t1 - flash.t0), rgb
    base = Image.fromarray(rgb).convert("RGBA")
    if "geo" in views:
        view, label = views["geo"], "Person's head camera"
    else:
        view, label = head_view(sim), "Person's head camera"
    C.inset(base, view, label, C.alpha_of(t, 0.15, scene.duration + 1, 0.3))
    lay = Image.new("RGBA", (C.W, C.H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    C.caption(d, scene.caption, C.alpha_of(t, 0.05, scene.duration + 1, 0.4))
    dark = draw_beats(sim, d, t, beats)
    base.alpha_composite(lay)
    out = np.asarray(base.convert("RGB"))
    if dark > 0:
        out = (out * (1 - dark)).astype(np.uint8)
    return out, rgb


def render_scene(sim: Sim, views, sinks, stills: bool) -> np.ndarray:
    scene = sim.scene
    beats = scene.beats(sim.answers)
    n = round(scene.duration * FPS)
    picks = {round(f * (n - 1)) for f in (0.05, 0.3, 0.45, 0.65, 0.85, 1.0)}
    last = None
    for i in range(n):
        if stills and i not in picks:
            continue
        frame, last = frame_at(sim, i / FPS, beats, views)
        if i in picks:
            Image.fromarray(frame).save(OUT / "stills" / f"{scene.name}_{i:03d}.png")
        for s in sinks:
            s.write(frame)
    return last


def outcome(scene: R.ReelScene, answers: dict) -> tuple[bool, str]:
    """Scripted scenes are right by construction. A live scene is right when the speaker-frame
    pick is the target; if it isn't, the tile says what it really did."""
    q = next((q for q in scene.queries if q.key == "geo"), None)
    if q is None:
        return True, ""
    a = answers.get(q.key)
    if a == q.correct:
        return True, ""
    return False, "no answer" if a is None else f"picked the other {scene.cand(a).label}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="PNG stills only, no video")
    ap.add_argument("--no-ground", action="store_true", help="preview: don't call the grounder")
    ap.add_argument("--only", default="", help="comma-separated scene names")
    args = ap.parse_args()
    (OUT / "stills").mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    grounder = None if args.no_ground else VllmGrounder(URL, "qwen-vl", name="r16-v2",
                                                        timeout_s=60)
    scenes = [s for s in all_scenes() if not args.only or s.name in args.only.split(",")]
    full = [] if args.stills else [Encoder(OUT / "curio_reel.mp4"),
                                   Encoder(OUT / "curio_reel_720p.mp4", SMALL_W, SMALL_H)]
    for i in range(round(COLD * FPS)):
        if args.stills and i not in (40, 100, 130):
            continue
        frame = R.cold_open(i / FPS)
        if args.stills:
            Image.fromarray(frame).save(OUT / "stills" / f"cold_{i:03d}.png")
        for s in full:
            s.write(frame)

    if grounder is not None:
        warm = next((s for s in scenes if s.queries), None)
        if warm is not None:
            sim = Sim(warm)
            for _ in range(2):  # warm-up, discarded (as bench_grounding does)
                grounder.score(make_request(warm, warm.queries[0], sim.small, sim.model,
                                            sim.data)[0])
            sim.close()

    log, tiles = [], []
    for scene in scenes:
        print(scene.name, flush=True)
        sim = Sim(scene)
        sim.answers, views = ground(sim, grounder, log)
        clip = None if args.stills else Encoder(OUT / f"scene_{scene.name}.mp4")
        last = render_scene(sim, views, full + ([clip] if clip else []), args.stills)
        if clip:
            clip.close()
        good, text = outcome(scene, sim.answers)
        tiles.append((last, scene.word, good, text))
        sim.close()

    for i in range(round(PAYOFF * FPS)):
        if args.stills and i != 240:
            continue
        frame = R.payoff(tiles, i / FPS)
        if args.stills or i == 240:
            Image.fromarray(frame).save(OUT / "stills" / "payoff.png")
        for s in full:
            s.write(frame)
    for s in full:
        s.close()
    if grounder is not None:
        (OUT / "grounding.json").write_text(json.dumps(
            {"server": SERVER, "url": URL, "model": "qwen-vl (Qwen3-VL-4B-Instruct Q8_0)",
             "adapter": "curio-ground-r16-v2-f16.gguf",
             "condition": "geometry (speaker head frame); 'naive' = robot's own frame",
             "live_scenes": sorted({e["scene"] for e in log}),
             "scripted_scenes": [s.name for s in scenes if not s.queries],
             "queries": log, "runtime_s": round(time.perf_counter() - t_start, 1)}, indent=2))
    print(f"done in {time.perf_counter() - t_start:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
