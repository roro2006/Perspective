"""T5-18: render the simulation-only proof-of-concept video.

1. For each scene, pose the person, render their head camera, build the object map in their
   head frame (the geometry condition) and ask the served grounder (Qwen3-VL-4B Q8_0 + the
   r16 v2 adapter on llama-server :8081). The warehouse also asks in the robot's own frame.
   Every query and answer goes to out/poc/grounding.json.
2. Script the robot to whatever the grounder chose (min-jerk) and render the cinematic camera
   with overlays; wrong picks are shown as they happened.
3. Encode each scene clip and the full cut (cv2 mp4v, 1920x1080, 30 fps).

    uv run python scripts/poc/make_video.py                 # everything
    uv run python scripts/poc/make_video.py --stills        # a few PNG frames per scene only
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
from kit import cam_pose, set_cine
from PIL import Image, ImageDraw
from scene_base import GROUND_H, GROUND_W, Scene, make_request
from scenes_b import FaceToFace, all_scenes

from curio_robot.geometry import intrinsics_from_fovy, project
from curio_robot.grounding.client import VllmGrounder

OUT = Path("C:/Users/Rohan/dev/Perspective/out/poc")
CV2_PY = "C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe"
ENCODER = str(Path(__file__).with_name("encode_cv2.py"))
URL = "http://127.0.0.1:8081/v1"
FPS = 30
SERVER = ("llama-server (Vulkan) -m Qwen3VL-4B-Instruct-Q8_0.gguf --mmproj "
          "mmproj-Qwen3VL-4B-Instruct-F16.gguf --lora curio-ground-r16-v2-f16.gguf -c 8192 "
          "--jinja --parallel 1 --alias qwen-vl --port 8081")
LABELS = {"warehouse": "Warehouse AMR", "dog": "Rescue robot dog", "drone": "Delivery drone",
          "humanoid": "Home humanoid", "rov": "Underwater ROV", "space": "Space free-flyer"}


class Encoder:
    def __init__(self, path: Path) -> None:
        self.proc = subprocess.Popen([CV2_PY, ENCODER, str(path), str(C.W), str(C.H), str(FPS)],
                                     stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit("encoder failed")


class Sim:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_string(scene.xml())
        self.data = mujoco.MjData(self.model)
        scene.setup(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.big = mujoco.Renderer(self.model, C.H, C.W)
        self.small = mujoco.Renderer(self.model, GROUND_H, GROUND_W)

    def render(self, t: float, answers: dict) -> np.ndarray:
        self.scene.animate(self.model, self.data, t, answers)
        eye, target = self.scene.camera(t)
        set_cine(self.model, self.data, eye, target)
        mujoco.mj_forward(self.model, self.data)
        self.big.update_scene(self.data, camera="cine")
        if self.scene.fog:
            self.big.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
        return self.big.render().copy()

    def project(self, point) -> tuple[float, float] | None:
        fovy = float(self.model.cam_fovy[self.model.camera("cine").id])
        k = intrinsics_from_fovy(fovy, C.W, C.H)
        return project(np.asarray(point, float), k, cam_pose(self.model, self.data, "cine"))

    def close(self) -> None:
        self.big.close()
        self.small.close()


# ---- grounding -----------------------------------------------------------------------------


def ground(sim: Sim, grounder: VllmGrounder, log: list) -> tuple[dict, dict, dict]:
    scene = sim.scene
    answers, info, images = {}, {}, {}
    for q in scene.queries:
        req, img, view = make_request(scene, q, sim.small, sim.model, sim.data)
        ans = grounder.score(req)
        choice = max(ans.probs, key=ans.probs.get) if ans.probs else None
        answers[q.key] = choice
        letter = next((k for k, v in req.letters.items() if v == choice), None)
        info[q.key] = {"letter": letter, "p": ans.probs.get(choice, 0.0) if choice else 0.0,
                       "latency_ms": ans.latency_ms, "error": ans.error}
        images[q.key] = (view, req.letters)
        Image.fromarray(img).save(OUT / f"grounder_{scene.name}_{q.key}.png")
        log.append({"scene": scene.name, "query": q.key, "frame": q.frame, "text": q.text,
                    "prompt": req.messages[0]["content"][1]["text"], "letters": req.letters,
                    "emitted_letter": ans.letter, "probs": ans.probs, "choice": choice,
                    "correct": q.correct, "ok": choice == q.correct,
                    "latency_ms": round(ans.latency_ms, 1), "source": ans.source,
                    "error": ans.error, "image": f"grounder_{scene.name}_{q.key}.png"})
        print(f"  {scene.name}/{q.key}: {choice} (correct {q.correct}) "
              f"p={info[q.key]['p']:.2f} {ans.latency_ms:.0f} ms {ans.error or ''}", flush=True)
    return answers, info, images


# ---- frames --------------------------------------------------------------------------------


def overlay(sim: Sim, rgb: np.ndarray, t: float, events, images, answers) -> Image.Image:
    scene = sim.scene
    base = Image.fromarray(rgb).convert("RGBA")
    lay = Image.new("RGBA", (C.W, C.H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    C.caption(d, scene.caption, C.alpha_of(t, 0.05, scene.duration + 1, 0.4))
    chips = [e for e in events if e.kind == "chip" and e.t0 <= t <= e.t1]
    active = chips[-1].oid if chips else None
    shown = active or next((q.key for q in scene.queries if q.frame == "speaker"), None)
    if shown in images:
        img, letters = images[shown]
        a = C.alpha_of(t, 0.2, scene.duration + 1, 0.3)
        frame = next(q.frame for q in scene.queries if q.key == shown)
        label = ("Person's head camera (grounder input)" if frame == "speaker"
                 else "Robot's own camera (robot-only reading)")
        base.alpha_composite(lay)
        C.inset(base, img, label, a)
        lay = Image.new("RGBA", (C.W, C.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for letter, oid in sorted(letters.items()):
            dz = 0.12 if scene.name == "space" else 0.35
            px = sim.project(np.asarray(scene.cand(oid).pos) + [0, 0, dz])
            if px is not None and 0 < px[0] < C.W and 0 < px[1] < C.H:
                C.letter_disc(d, px, letter, a * 0.95)
    for i, e in enumerate(chips):
        C.chip(d, e.text, e.good, C.alpha_of(t, e.t0, e.t1, 0.2), 410 + 58 * i)
    for e in events:
        a = C.alpha_of(t, e.t0, e.t1, 0.2)
        if a <= 0:
            continue
        if e.kind == "say":
            C.subtitle(d, e.text, a, scene.small)
        elif e.kind == "caption2":
            C.big_center(d, e.text, a, (255, 225, 120))
        elif e.kind == "verdict" and e.oid:
            px = sim.project(np.asarray(scene.cand(e.oid).pos) + [0, 0, 0.5])
            if px is not None:
                px = (float(np.clip(px[0], 80, C.W - 80)), float(np.clip(px[1], 200, C.H - 260)))
                r = 30 + 18 * min(1.0, (t - e.t0) / 0.25)
                C.check_mark(d, px, int(r), e.good, a)
    base.alpha_composite(lay)
    return base


def render_scene(sim: Sim, answers, info, images, sinks, stills: bool) -> np.ndarray:
    scene = sim.scene
    events = scene.events(answers, info)
    n = round(scene.duration * FPS)
    last = None
    picks = {0, n // 3, 2 * n // 3, n - 1}
    for i in range(n):
        if stills and i not in picks:
            continue
        t = i / FPS
        rgb = sim.render(t, answers)
        frame = np.asarray(overlay(sim, rgb, t, events, images, answers).convert("RGB"))
        if stills:
            Image.fromarray(frame).save(OUT / f"still_{scene.name}_{i:03d}.png")
        for s in sinks:
            s.write(frame)
        last = rgb
    return last


def explainer(sinks, stills: bool) -> None:
    for i in range(4 * FPS):
        if stills and i not in (30, 100):
            continue
        frame = C.title_card([("On my left.", 0.0, 2.0), ("Whose left?", 2.0, 4.0)], i / FPS)
        for s in sinks:
            s.write(frame)
    sim = Sim(FaceToFace())
    for i in range(4 * FPS):
        if stills and i not in (20, 90):
            continue
        t = i / FPS
        rgb = sim.render(t, {})
        base = Image.fromarray(rgb).convert("RGBA")
        lay = Image.new("RGBA", (C.W, C.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        a = C.alpha_of(t, 0.3, 9, 0.4)
        for x, z, col, text in ((0.75, 1.1, (52, 152, 219), "your left"),
                                (-0.28, 1.45, (243, 156, 18), "its right")):
            p0, p1 = sim.project((x, 0.25, z)), sim.project((x, -0.55, z))
            if p0 and p1:
                C.arrow(d, p0, p1, col, 12, a)
                C.shadow_text(d, (p0[0], p0[1] - 60), text, C.font(44, True), col, a, "mm")
        C.shadow_text(d, (C.W / 2, 90), "your left = its right", C.font(64, True), a=a,
                      anchor="mm")
        f = C.font(50, True)
        txt = "Every robot that takes directions gets this backwards."
        a2 = C.alpha_of(t, 1.0, 9, 0.4)
        tw = d.textlength(txt, font=f)
        C.pill(d, (C.W / 2 - tw / 2 - 30, C.H - 178, C.W / 2 + tw / 2 + 30, C.H - 98), a2)
        d.text((C.W / 2, C.H - 138), txt, font=f, fill=(255, 255, 255, int(255 * a2)),
               anchor="mm")
        base.alpha_composite(lay)
        frame = np.asarray(base.convert("RGB"))
        if stills:
            Image.fromarray(frame).save(OUT / f"still_table_{i:03d}.png")
        for s in sinks:
            s.write(frame)
    sim.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true", help="PNG stills only, no video")
    ap.add_argument("--only", default="", help="comma-separated scene names")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    grounder = VllmGrounder(URL, "qwen-vl", name="r16-v2", timeout_s=60)
    scenes = [s for s in all_scenes() if not args.only or s.name in args.only.split(",")]

    full = None if args.stills else Encoder(OUT / "curio_poc.mp4")
    sinks = [full] if full else []
    explainer(sinks, args.stills)

    log, tiles = [], []
    warm = Sim(scenes[0])
    for _ in range(2):  # warm-up, discarded (as bench_grounding does)
        grounder.score(make_request(warm.scene, warm.scene.queries[0], warm.small, warm.model,
                                    warm.data)[0])
    warm.close()
    for scene in scenes:
        print(scene.name, flush=True)
        sim = Sim(scene)
        answers, info, images = ground(sim, grounder, log)
        clip = None if args.stills else Encoder(OUT / f"scene_{scene.name}.mp4")
        last = render_scene(sim, answers, info, images, sinks + ([clip] if clip else []),
                            args.stills)
        if clip:
            clip.close()
        geo = [q for q in scene.queries if q.frame == "speaker"]
        good = all(answers.get(q.key) == q.correct for q in geo)
        outcome = "picked the one they meant" if good else "grounder picked wrong"
        if any(answers.get(q.key) is None for q in geo):
            outcome = "grounder gave no answer"
        tiles.append((last, LABELS[scene.name], good, outcome))
        sim.close()

    if not args.stills:
        for i in range(6 * FPS):
            full.write(C.end_grid(tiles, i / FPS))
        full.close()
    else:
        Image.fromarray(C.end_grid(tiles, 3.0)).save(OUT / "still_grid.png")
    (OUT / "grounding.json").write_text(json.dumps(
        {"server": SERVER, "url": URL, "model": "qwen-vl (Qwen3-VL-4B-Instruct Q8_0)",
         "adapter": "curio-ground-r16-v2-f16.gguf", "condition": "geometry (speaker head frame)",
         "queries": log, "runtime_s": round(time.perf_counter() - t_start, 1)}, indent=2))
    print(f"done in {time.perf_counter() - t_start:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
