"""T5-34 gate: the live dog demo, run once on the demo laptop.

    uv run python scripts/dog_demo_gate.py            # writes out/t5_34/gate.json and gate.md

Same code the viewer runs: DogWorld (live stepping, kinematic people), decide (resolver on the
floor map's rooms) and DogDemo (tick + draw with insets) through a headless stand-in for
MuJoCo's passive viewer. Lines Q, D, E and L are the card's, unchanged."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curio_robot.demo import dog

OUT = ROOT / "out" / "t5_34"
SEEDS = range(54000, 54010)
GOAL_TOL = 0.35  # T5-20's
CAP_S = 40.0  # T5-20's cap per leg
REQUESTS = {side: f"Survivor's in the room on my {side}." for side in ("left", "right")}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def walk(w: dog.DogWorld, room: str) -> dict:
    goal = w.go(room)
    while w.walking and w.sim_t < CAP_S:
        w.step(0.1)
    err = float(np.linalg.norm(w.dog_xy() - goal))
    return {"room": room, "err_m": round(err, 3), "t_s": round(w.sim_t, 2), "fell": w.fell,
            "timed_out": not w.dog.arrived and not w.fell, "min_z": round(w.min_z, 3)}


def line_q(w: dog.DogWorld) -> dict:
    runs = []
    for room in w.rooms:
        for seed in SEEDS:
            w.home(seed)
            r = walk(w, room) | {"seed": seed}
            r["ok"] = r["err_m"] <= GOAL_TOL and not r["fell"] and not r["timed_out"]
            runs.append(r)
    k, falls = sum(r["ok"] for r in runs), sum(r["fell"] for r in runs)
    return {"need": ">= 19 of 20 within 0.35 m, 0 falls", "ok": k, "n": len(runs),
            "falls": falls, "ci95": wilson(k, len(runs)), "pass": k >= 19 and falls == 0,
            "runs": runs}


def cases():
    for turned in (False, True):
        for side in ("left", "right"):
            yield turned, side


def set_facing(w: dog.DogWorld, turned: bool) -> None:
    w.facing = np.pi if turned else 0.0
    w.home()


def line_d(w: dog.DogWorld) -> dict:
    rows = []
    for turned, side in cases():
        set_facing(w, turned)
        for curio in (True, False):
            d = dog.decide(w, REQUESTS[side], curio)
            rows.append({"turned": turned, "words": REQUESTS[side], "curio": curio,
                         "target": d.target.object_id, "reason": d.target.reason,
                         "truth": d.oracle_id, "verdict": d.verdict})
    on = [r for r in rows if r["curio"]]
    off = [r for r in rows if not r["curio"]]
    k_on = sum(r["verdict"] == "correct" for r in on)
    return {"need": "Curio on correct 4 of 4 (off reported as it comes)", "on_correct": k_on,
            "off_correct": sum(r["verdict"] == "correct" for r in off), "n": len(on),
            "pass": k_on == len(on), "rows": rows}


def line_e(w: dog.DogWorld) -> dict:
    runs = []
    for turned, side in cases():
        for seed in list(SEEDS)[:3]:
            w.facing = np.pi if turned else 0.0
            w.home(seed)
            d = dog.decide(w, REQUESTS[side], True)
            r = walk(w, d.target.object_id)
            ok = not r["fell"] and dog.arrived_at(w, d.oracle_id)
            runs.append(r | {"turned": turned, "side": side, "seed": seed,
                             "truth": d.oracle_id, "ok": ok})
    k, falls = sum(r["ok"] for r in runs), sum(r["fell"] for r in runs)
    return {"need": ">= 11 of 12 in the true room (0.5 m, nearer than the other), 0 falls",
            "ok": k, "n": len(runs), "falls": falls, "ci95": wilson(k, len(runs)),
            "pass": k >= 11 and falls == 0, "runs": runs}


class HeadlessViewer:
    """What DogDemo touches of mujoco.viewer's handle: lock, user_scn, cam, viewport, and the
    image and text overlays (kept, not drawn)."""

    def __init__(self) -> None:
        self.user_scn = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), 100)
        self.cam = mujoco.MjvCamera()
        self.viewport = mujoco.MjrRect(0, 0, 1600, 900)
        self.images: list = []
        self.texts: list = []

    def lock(self):
        return contextlib.nullcontext()

    def set_images(self, imgs) -> None:
        self.images = imgs

    def set_texts(self, texts) -> None:
        self.texts = texts


def demo_walk(w: dog.DogWorld, paced: bool, hz: float = 30.0) -> dict:
    """One request through DogDemo's own tick and draw at the viewer's loop rate."""
    v = HeadlessViewer()
    demo = dog.DogDemo(v, w)
    demo.paced = paced
    demo.load()
    period = 1.0 / hz
    t0 = time.perf_counter()
    demo.command("")  # the scene's own request
    frames, last = 0, t0
    while w.walking and time.perf_counter() - t0 < 120:
        now = time.perf_counter()
        demo.tick(now - last)
        last = now
        demo.draw(now)
        frames += 1
        if paced:
            time.sleep(max(0.0, period - (time.perf_counter() - now)))
    wall = time.perf_counter() - t0
    return {"paced": paced, "sim_s": round(w.sim_t, 2), "wall_s": round(wall, 2),
            "rate": round(w.sim_t / wall, 3), "frames": frames, "fell": w.fell,
            "insets": len(v.images)}


def line_l(w: dog.DogWorld) -> dict:
    set_facing(w, False)
    cap = [demo_walk(w, paced=False) for _ in range(3)]
    paced = demo_walk(w, paced=True)
    med = float(np.median([r["rate"] for r in cap]))
    return {"need": "simulated / wall >= 1.0, median of 3 runs, insets on",
            "measured_as": "the loop's capacity (pacing off): the shown demo is paced to the "
                           "wall clock, which caps its own rate at 1.0 by design",
            "median_rate": round(med, 3), "pass": med >= 1.0, "capacity_runs": cap,
            "paced_run": paced}


def main() -> None:
    if not dog.have_assets():
        sys.exit("Go2 assets missing (see third_party/README.md)")
    OUT.mkdir(parents=True, exist_ok=True)
    w = dog.DogWorld()
    t0 = time.perf_counter()
    gate = {"card": "T5-34", "Q": line_q(w)}
    print("Q", gate["Q"]["ok"], "/", gate["Q"]["n"], "falls", gate["Q"]["falls"], flush=True)
    if gate["Q"]["pass"]:
        gate["D"] = line_d(w)
        print("D on", gate["D"]["on_correct"], "off", gate["D"]["off_correct"], flush=True)
        gate["E"] = line_e(w)
        print("E", gate["E"]["ok"], "/", gate["E"]["n"], "falls", gate["E"]["falls"], flush=True)
        gate["L"] = line_l(w)
        print("L median", gate["L"]["median_rate"], flush=True)
    else:
        gate["note"] = "Q failed: the dog demo is unqualified; D, E and L not run"
    lines = [k for k in ("Q", "D", "E", "L") if k in gate]
    gate["pass"] = len(lines) == 4 and all(gate[k]["pass"] for k in lines)
    gate["wall_s"] = round(time.perf_counter() - t0, 1)
    w.close()
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    md = ["# T5-34 gate (run once)", "", f"Overall: **{'PASS' if gate['pass'] else 'FAIL'}**",
          ""]
    for k in lines:
        g = gate[k]
        res = {"Q": lambda g: f"{g['ok']}/{g['n']} {g['ci95']}, falls {g['falls']}",
               "D": lambda g: f"on {g['on_correct']}/{g['n']}, off {g['off_correct']}/{g['n']}",
               "E": lambda g: f"{g['ok']}/{g['n']} {g['ci95']}, falls {g['falls']}",
               "L": lambda g: (f"median {g['median_rate']}x (capacity); paced run "
                               f"{g['paced_run']['rate']}x")}[k](g)
        md.append(f"- **{k}** {'PASS' if g['pass'] else 'FAIL'}: {res}  (need {g['need']})")
    (OUT / "gate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
