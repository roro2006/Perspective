"""T5-32: the Go2 drop-in gate. Curio's goal drives kouylty's pretrained navigation policy
(scripts/poc/phys_go2_nav.py) instead of T5-20's path follower.

    uv run python scripts/go2_nav_eval.py port    # P1 and P2 only (CPU)
    uv run python scripts/go2_nav_eval.py gate    # P1, P2, then the Curio lines if both pass

The Curio lines need the shared grounder on :8081 (queried without the GPU lock; this script
never starts or stops it). Writes out/t5_32/port.json, gate.json and gate.md.

Choices fixed before any gate run (also in tasks/notes/T5-32.md):
- P1: flat ground, 10 runs of 20 s; each run's (vx, vy, wz) is drawn uniformly from the
  walking policy's training range [-1, 1]^3 and redrawn every 10 s (Isaac Lab's resampling
  time); a fall is base z < 0.18 m (T5-20's).
- P2: flat ground, 20 runs; start yaw uniform, goal 1-4 m away in a uniform direction, goal
  heading = bearing from the start (simple heading). Reached = within 0.3 m of the goal at
  the end of the 20 s run.
- Curio: the T5-20 Go2 live scene (rescue dog, the only Go2 scene with grounder queries),
  10 runs per condition with T5-20's start jitter, seeds 54000-54009, grounded per run from
  the jittered start. The goal (from the grounder's pick or, for O, the truth) is handed to
  the scene's own leg code, as in T5-27's pose-adapter check: its map route through the door
  of the chosen room, whose last point is the goal. The navigation policy gets those points
  one at a time; nothing else steers it. A run ends when the dog has stopped (mean speed
  < 0.1 m/s over 1 s) or after 40 s (T5-20's cap). Success: stopped, no fall, within 0.5 m
  of the true room's goal point and closer to it than to the other room's.
- PF: T5-20's path follower (phys_go2.Go2Walker) on condition B's goals, same seeds and the
  same success rule, for comparison; T5-20's own gate figure is quoted too."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "poc"))

import mujoco
from phys_core import attach
from phys_go2 import FALL_Z, Go2Walker, go2_spec
from phys_go2_nav import Go2NavWalker
from phys_run import Run, rollout
from phys_scenes import PhysDog

OUT = Path("C:/Users/Rohan/dev/Perspective/out/t5_32")
T520_GATE = Path("C:/Users/Rohan/dev/Perspective/out/physical/gate.json")
URL = "http://127.0.0.1:8081/v1"
FLOOR = """<mujoco><option timestep="0.002"/><worldbody><light pos="0 0 3" dir="0 0 -1"/>
<geom name="floor" type="plane" size="40 40 0.1"/></worldbody></mujoco>"""
P1_RUNS, P1_SECONDS, P1_RESAMPLE_S, P1_SEED = 10, 20.0, 10.0, 32100
P2_RUNS, P2_SECONDS, P2_TOL, P2_NEED, P2_SEED = 20, 20.0, 0.3, 16, 32200
CMD_RANGE = 1.0
SEEDS = tuple(range(54000, 54010))
SUCCESS_TOL = 0.5
TRUE, DECOY = "room_n", "room_s"
GATE = {"b_minus_a": 0.50, "b_over_o": 0.85, "falls": 0}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, mid - half), 3), round(min(1.0, mid + half), 3))


def rate(hits: list[bool]) -> dict:
    k, n = int(sum(hits)), len(hits)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else 0.0, "ci95": list(wilson(k, n))}


# ---- P1 and P2 -----------------------------------------------------------------------------


def flat_walker():
    parent = mujoco.MjSpec.from_string(FLOOR)
    attach(parent, go2_spec(), "base", "go2_")
    m = parent.compile()
    d = mujoco.MjData(m)
    return m, d, Go2NavWalker(m, d)


def _step(m, d, w) -> None:
    w.control()
    mujoco.mj_step(m, d)
    w.after_step()


def p1() -> dict:
    rows = []
    for k in range(P1_RUNS):
        rng = np.random.default_rng(P1_SEED + k)
        m, d, w = flat_walker()
        w.place(0, 0, 0)
        mujoco.mj_forward(m, d)
        cmds, minz = [], 9.0
        steps, every = round(P1_SECONDS / m.opt.timestep), round(P1_RESAMPLE_S / m.opt.timestep)
        for i in range(steps):
            if i % every == 0:
                w.cmd = rng.uniform(-CMD_RANGE, CMD_RANGE, 3).astype(np.float32)
                cmds.append([round(float(c), 3) for c in w.cmd])
            _step(m, d, w)
            minz = min(minz, w.height())
        rows.append({"seed": P1_SEED + k, "cmds": cmds, "min_base_z": round(minz, 3),
                     "fell": minz < FALL_Z})
        print(f"P1 {k}: cmds {cmds} min z {minz:.3f}", flush=True)
    falls = sum(r["fell"] for r in rows)
    return {"runs": rows, "falls": falls, "pass": falls == 0}


def p2() -> dict:
    rows = []
    for k in range(P2_RUNS):
        rng = np.random.default_rng(P2_SEED + k)
        m, d, w = flat_walker()
        yaw0 = float(rng.uniform(-np.pi, np.pi))
        w.place(0, 0, yaw0)
        mujoco.mj_forward(m, d)
        a, r = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(1.0, 4.0))
        goal = (r * np.cos(a), r * np.sin(a))
        w.go([goal])
        minz, best = 9.0, 9.0
        for _ in range(round(P2_SECONDS / m.opt.timestep)):
            _step(m, d, w)
            minz = min(minz, w.height())
            x, y, _ = w.pose()
            best = min(best, float(np.hypot(x - goal[0], y - goal[1])))
        x, y, _ = w.pose()
        err = float(np.hypot(x - goal[0], y - goal[1]))
        fell = minz < FALL_Z
        rows.append({"seed": P2_SEED + k, "start_yaw": round(yaw0, 3), "dist": round(r, 3),
                     "goal": [round(float(v), 3) for v in goal], "err_m": round(err, 3),
                     "min_err_m": round(best, 3), "min_base_z": round(minz, 3), "fell": fell,
                     "ok": err <= P2_TOL and not fell})
        print(f"P2 {k}: dist {r:.2f} err {err:.2f} (min {best:.2f}) min z {minz:.3f}",
              flush=True)
    n_ok = sum(r["ok"] for r in rows)
    falls = sum(r["fell"] for r in rows)
    return {"runs": rows, "reached": rate([r["ok"] for r in rows]), "falls": falls,
            "pass": n_ok >= P2_NEED and falls == 0}


# ---- the Curio lines -----------------------------------------------------------------------


class NavDog(PhysDog):
    """The T5-20 rescue-dog scene with the navigation policy (or, nav=False, T5-20's path
    follower) and the final pose kept for scoring."""

    def __init__(self, nav: bool = True) -> None:
        super().__init__()
        self.nav = nav
        self.end_xy: tuple[float, float] | None = None

    def make_robot(self, model, data):
        return Go2NavWalker(model, data) if self.nav else Go2Walker(model, data)

    def leg_error(self, run, leg) -> float:
        x, y, _ = run.robot.pose()
        self.end_xy = (x, y)
        return super().leg_error(run, leg)


def goal_point(scene, oid: str) -> tuple[float, float]:
    return scene.legs({"geo": oid})[0].goal


def handoff(scene, target_id: str, world_xyz) -> object:
    """The scene with the chosen target placed where the pose adapter puts it (world frame),
    as T5-27's same_legs check does; its legs() then give the route and the goal."""
    moved = copy.copy(scene)
    moved.candidates = [replace(c, pos=tuple(float(v) for v in world_xyz))
                        if c.oid == target_id else c for c in scene.candidates]
    return moved


def score_run(scene, traj, truth: dict) -> dict:
    x, y = scene.end_xy if scene.end_xy is not None else (np.nan, np.nan)
    d_true = float(np.hypot(x - truth[TRUE][0], y - truth[TRUE][1]))
    d_decoy = float(np.hypot(x - truth[DECOY][0], y - truth[DECOY][1]))
    stopped = scene.end_xy is not None and not traj.timed_out
    ok = stopped and not traj.fell and d_true <= SUCCESS_TOL and d_true < d_decoy
    return {"end_xy": [round(float(x), 3), round(float(y), 3)], "d_true": round(d_true, 3),
            "d_decoy": round(d_decoy, 3), "stopped": bool(stopped),
            "timed_out": bool(traj.timed_out), "fell": bool(traj.fell),
            "min_base_z": round(traj.min_z, 3), "end_s": traj.end, "ok": bool(ok)}


def ground_run(scene, seed: int, grounder) -> dict:
    """Both readings from this seed's jittered start: A (T5-20 naive, robot frame) and B (the
    layer's resolve on the head camera, then the pose adapter)."""
    from scene_base import GROUND_H, GROUND_W, camera_frame, make_request, object_map

    from curio_robot.layer import adapters, resolve
    from curio_robot.layer.goal import base_from_camera
    from curio_robot.types import ROBOT_CAM

    run = Run(scene, {}, seed)
    m, d = run.model, run.data
    renderer = mujoco.Renderer(m, GROUND_H, GROUND_W)
    try:
        q = {qq.key: qq for qq in scene.queries}
        req = make_request(scene, q["naive"], renderer, m, d)[0]
        ans = grounder.score(req)
        naive = max(ans.probs, key=ans.probs.get) if ans.probs else None
        head = camera_frame(renderer, m, d, scene.person + "head_cam", "human", scene.fog)
        robot = camera_frame(renderer, m, d, scene.robot_cam, "robot", scene.fog)
        om = object_map(scene, q["geo"].ids, scene.head_pose(m, d), robot.pose)
        goal = resolve(q["geo"].text, head, {ROBOT_CAM: robot}, om, None, grounder=grounder)
        world = adapters.pose(goal, base_from_camera(om.robot_pose))
    finally:
        renderer.close()
    return {"A": {"target": naive, "p": round(float(ans.probs.get(naive, 0.0)), 4)
                  if naive else None, "xyz": list(scene.cand(naive).pos) if naive else None},
            "B": {"target": goal.target_id, "p": round(goal.confidence, 4),
                  "xyz": [round(float(v), 4) for v in world],
                  "xyz_robot": [round(float(v), 4) for v in goal.target_xyz_robot],
                  "adapter": goal.adapter}}


def curio(grounder) -> dict:
    base_scene = NavDog()
    truth = {oid: goal_point(base_scene, oid) for oid in (TRUE, DECOY)}
    rows: dict[str, list] = {"A": [], "B": [], "O": [], "PF": []}
    for seed in SEEDS:
        g = ground_run(NavDog(), seed, grounder)
        picks = {"A": g["A"], "B": g["B"],
                 "O": {"target": TRUE, "xyz": list(base_scene.cand(TRUE).pos)}}
        picks["PF"] = picks["B"]
        for cond in ("A", "B", "O", "PF"):
            pick = picks[cond]
            scene = NavDog(nav=cond != "PF")
            if pick["target"] is None:
                rows[cond].append({"seed": seed, "target": None, "ok": False, "fell": False})
                continue
            moved = handoff(scene, pick["target"], pick["xyz"])
            leg = moved.legs({"geo": pick["target"]})[0]
            traj = rollout(moved, {"geo": pick["target"]}, seed=seed, record=False)
            r = score_run(moved, traj, truth)
            rows[cond].append({"seed": seed, "target": pick["target"], "p": pick.get("p"),
                               "goal": [round(float(v), 3) for v in leg.goal], **r})
            print(f"{cond} seed {seed}: {pick['target']} -> end {r['end_xy']} d_true "
                  f"{r['d_true']} ok={r['ok']} fell={r['fell']} t={r['end_s']}", flush=True)
        rows.setdefault("grounding", []).append({"seed": seed, **g})
    res = {c: rate([r["ok"] for r in rows[c]]) for c in ("A", "B", "O", "PF")}
    falls = {c: int(sum(r["fell"] for r in rows[c])) for c in ("A", "B", "O", "PF")}
    a, b, o = res["A"]["rate"], res["B"]["rate"], res["O"]["rate"]
    curio_falls = falls["A"] + falls["B"] + falls["O"]
    lines = {"1_b_ge_a_plus_0.50": b >= a + GATE["b_minus_a"],
             "2_b_ge_0.85_o": b >= GATE["b_over_o"] * o,
             "3_zero_falls": curio_falls == 0}
    return {"truth_goal_points": truth, "rates": res, "falls": falls, "lines": lines,
            "pass": all(lines.values()), "runs": rows}


def t520_reference() -> dict:
    try:
        dog = json.loads(T520_GATE.read_text(encoding="utf-8"))["scenes"]["dog"]
    except (OSError, KeyError, ValueError) as e:
        return {"error": str(e)}
    return {"source": str(T520_GATE), "success": dog["success"], "runs": len(dog["runs"]),
            "falls": dog["falls"], "rule": "within 0.35 m of the leg goal, seeds 0-9",
            "rate": rate([r["ok"] for r in dog["runs"]])}


# ---- report --------------------------------------------------------------------------------


def _r(d: dict) -> str:
    return f"{d['k']}/{d['n']} = {d['rate']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"


def markdown(res: dict) -> str:
    port = res["port"]
    out = ["# T5-32 gate: Go2 navigation-policy drop-in", "",
           f"Verdict: **{'PASS' if res['pass'] else 'FAIL'}**"
           + ("" if port["pass"] else " (on the port: P1/P2 failed, no Curio lines run)"), "",
           "## Preconditions", "",
           (f"- P1 walking: {P1_RUNS} x {P1_SECONDS:.0f} s, falls {port['p1']['falls']} -> "
            f"**{'PASS' if port['p1']['pass'] else 'FAIL'}**"),
           (f"- P2 navigation: reached {_r(port['p2']['reached'])} (need {P2_NEED}/{P2_RUNS}), "
            f"falls {port['p2']['falls']} -> **{'PASS' if port['p2']['pass'] else 'FAIL'}**"),
           ""]
    if "curio" in res:
        c = res["curio"]
        out += ["## Curio lines (rescue dog, seeds 54000-54009)", "",
                "| condition | success (Wilson 95%) | falls |", "|---|---|---|"]
        names = {"A": "A raw (robot-frame reading)", "B": "B Curio (layer pose adapter, v4)",
                 "O": "O oracle", "PF": "T5-20 path follower on B's goals"}
        for k in ("A", "B", "O", "PF"):
            out.append(f"| {names[k]} | {_r(c['rates'][k])} | {c['falls'][k]} |")
        out += ["", *[f"- line {k}: **{'PASS' if v else 'FAIL'}**" for k, v in c["lines"].items()]]
    ref = res.get("t520_reference", {})
    if "rate" in ref:
        out += ["", (f"T5-20's own gate on the dog scene (path follower, {ref['rule']}): "
                     f"{_r(ref['rate'])}, falls {ref['falls']}.")]
    out += ["", f"Runtime {res.get('runtime_s', 0):.0f} s."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["port", "gate"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    port = {"p1": p1(), "p2": p2()}
    port["pass"] = port["p1"]["pass"] and port["p2"]["pass"]
    (OUT / "port.json").write_text(json.dumps(port, indent=2))
    if args.what == "port":
        print(f"P1 {port['p1']['pass']} P2 {port['p2']['pass']}")
        return
    res: dict = {"port": port, "t520_reference": t520_reference()}
    if port["pass"]:
        from curio_robot.grounding.client import VllmGrounder
        from curio_robot.layer.resolve import production_adapter
        grounder = VllmGrounder(URL, "qwen-vl", name=production_adapter(), timeout_s=60)
        res["curio"] = curio(grounder)
        res["pass"] = res["curio"]["pass"]
    else:
        res["pass"] = False
    res["runtime_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "gate.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / "gate.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))


if __name__ == "__main__":
    main()
