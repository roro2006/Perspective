"""T5-20: evaluation. `walkers` runs the standalone checks (each legged walker to 10 random
targets on flat ground); `gate` runs the card's gate: the live grounder is asked once per
scene, then 10 seeded physical runs reuse that answer. Writes out/physical/walkers.json,
gate.json and grounding.json (the answers the videos reuse).

    uv run python scripts/poc/phys_eval.py walkers
    uv run python scripts/poc/phys_eval.py gate          # needs llama-server on :8081
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import compose as C
import mujoco
import numpy as np
from make_video import SERVER, URL
from phys_core import attach
from phys_g1 import FALL_Z as G1_FALL
from phys_g1 import G1Walker, g1_spec
from phys_go2 import FALL_Z as GO2_FALL
from phys_go2 import Go2Walker, go2_spec
from phys_run import CAP, rollout
from PIL import Image
from scene_base import GROUND_H, GROUND_W, make_request

OUT = Path("C:/Users/Rohan/dev/Perspective/out/physical")
FLOOR = """<mujoco><option timestep="0.002"/><worldbody><light pos="0 0 3" dir="0 0 -1"/>
<geom name="floor" type="plane" size="30 30 0.1"/></worldbody></mujoco>"""
RUNS = 10
GOAL_TOL, REACH_TOL, NEED = 0.35, 0.10, 9


class PhysSim:
    """make_video.Sim for a physical scene: the model comes from the scene's MjSpec."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self.model = scene.spec().compile()
        self.data = mujoco.MjData(self.model)
        scene.setup(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.big = mujoco.Renderer(self.model, C.H, C.W)
        self.small = mujoco.Renderer(self.model, GROUND_H, GROUND_W)

    def close(self) -> None:
        self.big.close()
        self.small.close()


def gate_scenes() -> list:
    from phys_reel import reel_live
    from phys_scenes import poc_scenes
    return poc_scenes() + reel_live()


# ---- standalone walker checks ------------------------------------------------------------


def walker_model(kind: str):
    parent = mujoco.MjSpec.from_string(FLOOR)
    if kind == "g1":
        attach(parent, g1_spec(), "pelvis", "g1_")
    else:
        attach(parent, go2_spec(), "base", "go2_")
    m = parent.compile()
    return m, mujoco.MjData(m)


def _step(m, d, w) -> None:
    w.control()
    mujoco.mj_step(m, d)
    w.after_step()


def walkers(n: int = 10) -> dict:
    """Each walker to n random targets 1-4 m away in a random direction, one after another."""
    out = {}
    for kind, cls, fall in (("g1", G1Walker, G1_FALL), ("go2", Go2Walker, GO2_FALL)):
        m, d = walker_model(kind)
        w = cls(m, d)
        w.place(0, 0, 0)
        mujoco.mj_forward(m, d)
        rng = np.random.default_rng(0)
        rows, minz, t = [], 9.0, 0.0
        for k in range(n):
            x, y, _ = w.pose()
            a, r = rng.uniform(-np.pi, np.pi), rng.uniform(1.0, 4.0)
            goal = np.array([x + r * np.cos(a), y + r * np.sin(a)])
            heading = float(rng.uniform(-np.pi, np.pi)) if kind == "g1" else None
            w.go([goal], heading)
            t0 = t
            while not w.arrived and t - t0 < CAP:
                _step(m, d, w)
                t += m.opt.timestep
                minz = min(minz, w.height())
            for _ in range(250):  # settle 0.5 s on the spot before measuring
                _step(m, d, w)
                t += m.opt.timestep
                minz = min(minz, w.height())
            x, y, _ = w.pose()
            err = float(np.hypot(x - goal[0], y - goal[1]))
            rows.append({"target": [round(float(v), 3) for v in goal], "heading": heading,
                         "time_s": round(t - t0, 2), "err_m": round(err, 3),
                         "ok": err <= GOAL_TOL and t - t0 < CAP})
            print(f"{kind} {k}: dist {r:.2f} err {err:.3f} m in {t - t0:.1f} s", flush=True)
        out[kind] = {"runs": rows, "min_base_z": round(minz, 3), "fell": minz < fall,
                     "reached": sum(r["ok"] for r in rows)}
        print(kind, out[kind]["reached"], "of", n, "min z", round(minz, 3), flush=True)
    return out


# ---- the gate ------------------------------------------------------------------------------


def ground(sim: PhysSim, grounder, log: list, out: Path) -> dict:
    """Ask each of the scene's queries once (as make_video.ground does) and log it."""
    scene, answers = sim.scene, {}
    for q in scene.queries:
        req, img, _ = make_request(scene, q, sim.small, sim.model, sim.data)
        ans = grounder.score(req)
        choice = max(ans.probs, key=ans.probs.get) if ans.probs else None
        answers[q.key] = choice
        letter = next((k for k, v in req.letters.items() if v == choice), None)
        name = f"grounder_{scene.name}_{q.key}.png"
        Image.fromarray(img).save(out / name)
        p = ans.probs.get(choice, 0.0) if choice else 0.0
        log.append({"scene": scene.name, "query": q.key, "frame": q.frame, "text": q.text,
                    "prompt": req.messages[0]["content"][1]["text"], "letters": req.letters,
                    "emitted_letter": ans.letter, "letter": letter, "probs": ans.probs,
                    "choice": choice, "correct": q.correct, "ok": choice == q.correct,
                    "p": round(p, 4), "latency_ms": round(ans.latency_ms, 1),
                    "source": ans.source, "error": ans.error, "image": name})
        print(f"  {scene.name}/{q.key}: {choice} (correct {q.correct}) p={p:.2f} "
              f"{ans.latency_ms:.0f} ms {ans.error or ''}", flush=True)
    return answers


def judge(scene, answers: dict, traj) -> dict:
    """The card's gate for one run: every scored leg ends within its tolerance, no leg ran
    out of time, no fall."""
    errs = {k: round(float(v), 4) for k, v in traj.errors.items()}
    ok_legs = all(v <= (REACH_TOL if k == "reach" else GOAL_TOL) for k, v in errs.items())
    expected = {lg.key for lg in scene.legs(answers) if lg.goal is not None}
    ok = ok_legs and expected <= set(errs) and not traj.timed_out and not traj.fell
    return {"ok": bool(ok), "errors": errs, "timed_out": traj.timed_out, "fell": traj.fell,
            "min_base_z": round(traj.min_z, 3) if traj.min_z < 9 else None,
            "end_s": traj.end}


def run_gate(out: Path, only: list[str]) -> dict:
    from curio_robot.grounding.client import VllmGrounder
    grounder = VllmGrounder(URL, "qwen-vl", name="r16-v2", timeout_s=60)
    scenes = [s for s in gate_scenes() if not only or s.name in only]
    warm = PhysSim(scenes[0])
    for _ in range(2):  # warm-up, discarded (as bench_grounding does)
        grounder.score(make_request(warm.scene, warm.scene.queries[0], warm.small, warm.model,
                                    warm.data)[0])
    warm.close()
    log, table = [], {}
    for scene in scenes:
        print(scene.name, flush=True)
        sim = PhysSim(scene)
        answers = ground(sim, grounder, log, out)
        sim.close()
        runs = []
        for seed in range(RUNS):
            r = judge(scene, answers, rollout(scene, answers, seed=seed, record=False))
            runs.append({"seed": seed, **r})
            print(f"    seed {seed}: {'ok' if r['ok'] else 'FAIL'} {r['errors']} "
                  f"fell={r['fell']} timeout={r['timed_out']}", flush=True)
        n_ok = sum(r["ok"] for r in runs)
        falls = sum(r["fell"] for r in runs)
        table[scene.name] = {"robot": scene.kind, "answers": answers, "success": n_ok,
                             "falls": falls, "pass": n_ok >= NEED and falls == 0, "runs": runs}
        print(f"  {scene.name}: {n_ok}/{RUNS}, falls {falls}", flush=True)
    (out / "grounding.json").write_text(json.dumps(
        {"server": SERVER, "url": URL, "model": "qwen-vl (Qwen3-VL-4B-Instruct Q8_0)",
         "adapter": "curio-ground-r16-v2-f16.gguf",
         "condition": "geometry (speaker head frame); 'naive' = robot's own frame",
         "note": "asked once per scene; the answers drive all 10 gate runs and the videos",
         "answers": {k: v["answers"] for k, v in table.items()}, "queries": log}, indent=2))
    return {"gate": {"runs": RUNS, "goal_tol_m": GOAL_TOL, "reach_tol_m": REACH_TOL,
                     "need": NEED}, "scenes": table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["walkers", "gate"])
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.what == "walkers":
        res = walkers()
        (OUT / "walkers.json").write_text(json.dumps(res, indent=2))
    else:
        res = run_gate(OUT, [s for s in args.only.split(",") if s])
        res["runtime_s"] = round(time.perf_counter() - t0, 1)
        (OUT / "gate.json").write_text(json.dumps(res, indent=2))
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
