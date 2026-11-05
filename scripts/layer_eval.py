"""T5-27: the gate for Curio as a plug-in layer (tasks/T5-27-curio-layer.md).

    uv run python scripts/layer_eval.py cpu      # lines 1 and 2: equivalence, words checker
    uv run python scripts/layer_eval.py live     # lines 3 and 4 on the shared :8081 server
    uv run python scripts/layer_eval.py report   # gate.json and gate.md from the saved rows

`cpu` needs no grounder: a replay grounder answers with fixed probabilities (T5-22's cached
bench5c scores, T5-20's logged scene answers), the same answer to the layer and to the
existing code, and the check is that the layer asks the identical request and returns the
same target, ring image and pose goal. `live` uses the production adapter already served on
:8081 (never starts or stops a server, never takes the GPU lock). The reader for line 3 is
bench.conditions "robot-only" given every mapped object the robot image shows (the layer's
map knows where a hidden block is) and no habit line; the same reader with the human's words
is the reported comparison, and the bench's visible-only robot-only layout is reported too."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from curio_robot.bench import items as bench_items
from curio_robot.bench.conditions import build
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.layer import adapters, resolve
from curio_robot.layer import robot_words as rw
from curio_robot.layer.goal import base_from_camera
from curio_robot.layer.resolve import (
    GroundingUnavailable,
    geometry_request,
    production_adapter,
)
from curio_robot.sim.demos import draw_marker
from curio_robot.types import ROBOT_CAM, ObjectMap, WorldObject

OUT = Path("C:/Users/Rohan/dev/Perspective/out")
BENCH = OUT / "bench5c"
T522_SCORES = OUT / "t5_22" / "scores.json"
T520_GROUNDING = OUT / "physical" / "grounding.json"
DEST = OUT / "t5_27"
URL = "http://127.0.0.1:8081/v1"
SMOLVLA_TRIALS = 50
POSE_TOL_M = 1e-9  # the base-frame round trip is exact to float rounding, nothing more
GATE = {"words_unambiguous": 0.99, "reader": 0.95, "p95_ms": 250.0}


class Replay:
    """A grounder that answers every request with the same object-id probabilities."""

    def __init__(self) -> None:
        self.probs: dict[str, float] = {}
        self.requests: list = []

    def score(self, request) -> GroundingAnswer:
        self.requests.append(request)
        return GroundingAnswer(dict(self.probs), None, 0.0, "replay")


def argmax(probs: dict[str, float]) -> str | None:
    return max(probs, key=probs.get) if probs else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def ref_pixel(xyz, frame) -> tuple[float, float] | None:
    """demos.target_pixel for a known position (bench items have no physics world)."""
    from curio_robot.geometry import project

    uv = project(xyz, frame.intrinsics, frame.pose)
    if uv is None or not (0 <= uv[0] < frame.intrinsics.width
                          and 0 <= uv[1] < frame.intrinsics.height):
        return None
    return (round(float(uv[0]), 2), round(float(uv[1]), 2))


# ---- line 1: equivalence (CPU) -------------------------------------------------------------


def equiv_bench(items) -> list[dict]:
    scores = json.loads(T522_SCORES.read_text(encoding="utf-8"))["items"]
    g, rows = Replay(), []
    for item in items:
        g.probs = scores[item.id]["probs"]
        goal = resolve(item.text, item.human, {ROBOT_CAM: item.robot}, item.object_map, None,
                       grounder=g, p_own_frame=item.p_own_frame)
        layer_req = g.requests[-1]
        existing_req = build("geometry", item)
        existing_pick = argmax(g.score(existing_req).probs)  # score.vlm_predict's rule
        target_obj = next(o for o in item.object_map.objects if o.object_id == goal.target_id)
        ref_ring = draw_marker(item.robot.rgb, ref_pixel(target_obj.position, item.robot))
        world_xyz = adapters.pose(goal, base_from_camera(item.object_map.robot_pose))
        rows.append({
            "id": item.id, "kind": item.kind,
            "request": layer_req.messages == existing_req.messages,
            "target": goal.target_id == existing_pick,
            "ring": bool(np.array_equal(adapters.ring(goal, item.robot), ref_ring)),
            "pose": bool(np.allclose(world_xyz, target_obj.position, rtol=0, atol=POSE_TOL_M)),
            "ask": goal.ask})
    return rows


def equiv_smolvla(n: int = SMOLVLA_TRIALS) -> list[dict]:
    """smolvla_eval's own scenes and marker path against the ring adapter."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from smolvla_eval import EVAL_SEED, make_trials

    from curio_robot.collab.tasks import true_map
    from curio_robot.sim import demos
    from curio_robot.sim.physics import PhysWorld
    from curio_robot.types import HUMAN_CAM

    rows, g = [], Replay()
    world = PhysWorld()
    try:
        for trial in make_trials(n, EVAL_SEED, world):
            world.reset(trial.scene)  # as smolvla_eval.rollout does
            robot, human = world.render_camera(ROBOT_CAM), world.render_camera(HUMAN_CAM)
            existing = demos.draw_marker(robot.rgb, demos.target_pixel(world, trial.target,
                                                                       robot))
            objs = true_map(trial.scene).objects
            om = ObjectMap([WorldObject(o.object_id, o.color, world.block_position(i))
                            for i, o in enumerate(objs)], robot.pose, human.pose)
            g.probs = {objs[trial.target].object_id: 1.0}
            goal = resolve(trial.words, human, {ROBOT_CAM: robot}, om, None, grounder=g)
            rows.append({"seed": trial.seed, "target": goal.target_id == objs[trial.target]
                         .object_id, "ring": bool(np.array_equal(adapters.ring(goal, robot),
                                                                 existing))})
    finally:
        world.close()
    return rows


def equiv_scenes() -> list[dict]:
    """Every T5-20 live scene and query: the request, the target, the ring and the goals
    the scene's own leg code computes from the layer's pose output."""
    sys.path.insert(0, str(ROOT / "scripts" / "poc"))
    import mujoco
    from phys_eval import gate_scenes
    from scene_base import GROUND_H, GROUND_W, camera_frame, make_request, object_map

    logged = json.loads(T520_GROUNDING.read_text(encoding="utf-8"))
    probs_of = {(q["scene"], q["query"]): q["probs"] for q in logged["queries"]}
    rows, g = [], Replay()
    for scene in gate_scenes():
        model = scene.spec().compile()
        data = mujoco.MjData(model)
        scene.setup(model, data)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, GROUND_H, GROUND_W)
        goals, layer_answers = {}, {}
        try:
            for q in scene.queries:
                g.probs = probs_of[(scene.name, q.key)]
                existing_req = make_request(scene, q, renderer, model, data)[0]
                existing_pick = argmax(g.score(existing_req).probs)
                head = camera_frame(renderer, model, data, scene.person + "head_cam", "human",
                                    scene.fog) if q.frame == "speaker" else None
                robot = camera_frame(renderer, model, data, scene.robot_cam, "robot", scene.fog)
                om = object_map(scene, q.ids, scene.head_pose(model, data), robot.pose)
                goal = resolve(q.text, head, {ROBOT_CAM: robot}, om, None, grounder=g)
                req_same = g.requests[-1].messages == existing_req.messages
                cand = np.asarray(scene.cand(goal.target_id).pos, float)
                ring_same = np.array_equal(adapters.ring(goal, robot),
                                           draw_marker(robot.rgb, ref_pixel(cand, robot)))
                goals[q.key] = goal
                layer_answers[q.key] = goal.target_id
                rows.append({"scene": scene.name, "query": q.key, "frame": goal.frame,
                             "depth_prompt": q.depth, "request": req_same,
                             "target": goal.target_id == existing_pick,
                             "ring": bool(ring_same)})
        finally:
            renderer.close()
        legs_ok = same_legs(scene, logged["answers"][scene.name], goals, om.robot_pose)
        for r in rows:
            if r["scene"] == scene.name:
                r["pose"] = legs_ok
    return rows


def same_legs(scene, t520_answers: dict, goals: dict, robot_pose) -> bool:
    """The scene's legs from T5-20's answers vs from the layer: the layer's targets placed
    where the pose adapter says they are (world frame), then the scene's own leg code."""
    base = base_from_camera(robot_pose)
    moved = copy.copy(scene)
    where = {gl.target_id: tuple(float(v) for v in adapters.pose(gl, base))
             for gl in goals.values()}
    moved.candidates = [replace(c, pos=where.get(c.oid, c.pos)) for c in scene.candidates]
    old = scene.legs(t520_answers)
    new = moved.legs({k: gl.target_id for k, gl in goals.items()})
    if [lg.key for lg in old] != [lg.key for lg in new]:
        return False
    for a, b in zip(old, new):
        if (a.goal is None) != (b.goal is None) or a.heading != b.heading:
            return False
        if a.goal is not None and not np.allclose(a.goal, b.goal, rtol=0, atol=POSE_TOL_M):
            return False
        if isinstance(a.route, list) and not np.allclose(
                np.asarray(a.route, float), np.asarray(b.route, float), rtol=0, atol=POSE_TOL_M):
            return False
    return True


# ---- line 2: words checker (CPU) -----------------------------------------------------------


def words_rows(items) -> list[dict]:
    rows = []
    for item in items:
        words = rw.instruction(item.target_id, item.object_map)
        rows.append({"id": item.id, "kind": item.kind, "text": item.text, "words": words,
                     "hit": rw.check(words, item.object_map) == item.target_id})
    return rows


# ---- lines 3 and 4: live -------------------------------------------------------------------


def reader_request(item, text: str, visible_only: bool = False):
    """bench.conditions robot-only with `text` and no habit line; every mapped object the
    robot image shows gets a letter unless visible_only (the bench's own layout)."""
    shown = item.robot_visible if visible_only else tuple(o.object_id
                                                          for o in item.object_map.objects)
    return build("robot-only", replace(item, text=text, p_own_frame=None, robot_visible=shown))


def live(items, url: str) -> dict:
    from curio_robot.grounding.client import VllmGrounder

    grounder = VllmGrounder(url, "qwen-vl", name=production_adapter(), timeout_s=60)
    for _ in range(2):  # warm-up, discarded (as bench_grounding does)
        grounder.score(geometry_request(items[0].text, items[0].object_map, items[0].human))
    rows, t0 = [], time.perf_counter()
    for i, item in enumerate(items):
        ts = time.perf_counter()
        error = None
        try:
            goal = resolve(item.text, item.human, {ROBOT_CAM: item.robot}, item.object_map,
                           None, grounder=grounder, p_own_frame=item.p_own_frame)
        except GroundingUnavailable as e:
            goal, error = None, str(e)
        ms = (time.perf_counter() - ts) * 1000
        words = rw.instruction(item.target_id, item.object_map)
        picks = {}
        for name, req in (("robot_words", reader_request(item, words)),
                          ("human_words", reader_request(item, item.text)),
                          ("robot_words_visible_only", reader_request(item, words, True))):
            ans = grounder.score(req)
            picks[name] = {"pick": argmax(ans.probs), "error": ans.error}
        rows.append({
            "id": item.id, "kind": item.kind, "target": item.target_id, "text": item.text,
            "words": words, "resolve_ms": round(ms, 1), "error": error,
            "goal_target": goal.target_id if goal else None,
            "confidence": goal.confidence if goal else None,
            "ask": goal.ask if goal else None,
            **{f"{k}_pick": v["pick"] for k, v in picks.items()},
            **{f"{k}_error": v["error"] for k, v in picks.items() if v["error"]}})
        if i % 50 == 0:
            print(f"{i}/{len(items)} {ms:.0f} ms {rows[-1]}", flush=True)
    return {"url": url, "adapter": production_adapter(),
            "runtime_s": round(time.perf_counter() - t0, 1), "rows": rows}


# ---- report --------------------------------------------------------------------------------


def rate(hits: list[bool]) -> dict:
    k, n = sum(hits), len(hits)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci95": [round(lo, 4), round(hi, 4)]}


def by_kind(rows: list[dict], hit) -> dict:
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    return {k: rate([hit(r) for r in rows if r["kind"] == k]) for k in kinds}


def report() -> dict:
    cpu = json.loads((DEST / "cpu.json").read_text(encoding="utf-8"))
    lv = json.loads((DEST / "live.json").read_text(encoding="utf-8"))
    bench, scenes, smol = cpu["bench"], cpu["scenes"], cpu["smolvla"]
    checks = ("target", "ring", "pose")
    eq_bench = rate([all(r[c] for c in checks) for r in bench])
    names = list(dict.fromkeys(r["scene"] for r in scenes))
    eq_scene = rate([all(all(r[c] for c in checks) for r in scenes if r["scene"] == s)
                     for s in names])
    eq_smol = rate([r["target"] and r["ring"] for r in smol])
    line1 = eq_bench["k"] == eq_bench["n"] and eq_scene["k"] == eq_scene["n"] \
        and eq_smol["k"] == eq_smol["n"]
    words = rate([r["hit"] for r in cpu["words"]])
    rows = lv["rows"]
    reader = rate([r["robot_words_pick"] == r["target"] for r in rows])
    human = rate([r["human_words_pick"] == r["target"] for r in rows])
    visible = rate([r["robot_words_visible_only_pick"] == r["target"] for r in rows])
    geo = rate([r["goal_target"] == r["target"] for r in rows])
    e2e = rate([r["goal_target"] == r["target"] and r["robot_words_pick"] == r["target"]
                for r in rows])
    ms = np.array([r["resolve_ms"] for r in rows])
    p95 = float(np.percentile(ms, 95))
    under = rate([m <= GATE["p95_ms"] for m in ms])
    gate = {
        "1_equivalence": {"pass": line1, "bench5c_items": eq_bench, "t5_20_scenes": eq_scene,
                          "t5_20_queries": rate([all(r[c] for c in checks) for r in scenes]),
                          "smolvla_eval_trials": eq_smol,
                          "requests_identical": {
                              "bench5c": rate([r["request"] for r in bench]),
                              "t5_20_queries": rate([r["request"] for r in scenes]),
                              "t5_20_depth_prompt_queries": sum(r["depth_prompt"]
                                                                for r in scenes)}},
        "2_words_unambiguous": {"pass": words["rate"] >= GATE["words_unambiguous"], **words},
        "3_reader_robot_words": {"pass": reader["rate"] >= GATE["reader"], **reader,
                                 "by_kind": by_kind(rows, lambda r: r["robot_words_pick"]
                                                    == r["target"])},
        "4_latency": {"pass": p95 <= GATE["p95_ms"], "p95_ms": round(p95, 1),
                      "p50_ms": round(float(np.percentile(ms, 50)), 1),
                      "max_ms": round(float(ms.max()), 1), "share_le_250ms": under},
    }
    reported = {
        "reader_human_words": {**human, "by_kind": by_kind(
            rows, lambda r: r["human_words_pick"] == r["target"])},
        "reader_robot_words_visible_marks_only": {**visible, "by_kind": by_kind(
            rows, lambda r: r["robot_words_visible_only_pick"] == r["target"])},
        "resolve_geometry_accuracy": geo,
        "end_to_end_resolve_then_words_reader": e2e,
        "ask_rate": rate([r["ask"] is not None for r in rows]),
        "relations": relation_counts(cpu["words"]),
    }
    result = {"card": "T5-27", "adapter": lv["adapter"], "url": lv["url"],
              "live_runtime_s": lv["runtime_s"], "gate": gate, "reported": reported,
              "pass": all(v["pass"] for v in gate.values())}
    (DEST / "gate.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (DEST / "gate.md").write_text(markdown(result), encoding="utf-8")
    return result


def relation_counts(rows: list[dict]) -> dict[str, int]:
    def phrase(words: str) -> str:
        return next((p for p in rw.RELATIONS if words.endswith(" " + p)), "(unique colour)")
    names = [phrase(r["words"]) for r in rows]
    return {p: names.count(p) for p in dict.fromkeys(names)}


def _r(d: dict) -> str:
    return f"{d['rate']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] ({d['k']}/{d['n']})"


def markdown(res: dict) -> str:
    g, rep = res["gate"], res["reported"]
    one, lat = g["1_equivalence"], g["4_latency"]
    pf = {True: "PASS", False: "FAIL"}
    lines = [
        f"# T5-27 gate: {'PASS' if res['pass'] else 'FAIL'}",
        "",
        f"Grounder {res['adapter']} on {res['url']} (shared server). Wilson 95% CIs.",
        "",
        (f"1. {pf[one['pass']]} equivalence (CPU, replayed grounder answers): bench5c items "
        f"{_r(one['bench5c_items'])}; T5-20 live scenes {_r(one['t5_20_scenes'])}, queries "
        f"{_r(one['t5_20_queries'])}; smolvla_eval trials {_r(one['smolvla_eval_trials'])}. "
        f"Identical requests: bench5c {_r(one['requests_identical']['bench5c'])}, T5-20 "
        f"{_r(one['requests_identical']['t5_20_queries'])} "
        f"({one['requests_identical']['t5_20_depth_prompt_queries']} T5-20 queries use the "
        "warehouse's scene-only depth prompt)."),
        (f"2. {pf[g['2_words_unambiguous']['pass']]} words unambiguous (rule checker): "
        f"{_r(g['2_words_unambiguous'])} (bar 0.99)"),
        (f"3. {pf[g['3_reader_robot_words']['pass']]} v4 reader, robot view, robot words: "
        f"{_r(g['3_reader_robot_words'])} (bar 0.95)"),
        (f"4. {pf[lat['pass']]} resolve latency p95 {lat['p95_ms']} ms (bar 250; p50 "
        f"{lat['p50_ms']}, max {lat['max_ms']}); share <= 250 ms {_r(lat['share_le_250ms'])}"),
        "",
        "## Reported, not gated",
        f"- same reader, human's original words: {_r(rep['reader_human_words'])}",
        (f"- robot words, bench robot-only layout (visible marks only): "
        f"{_r(rep['reader_robot_words_visible_marks_only'])}"),
        (f"- resolve (geometry) accuracy: {_r(rep['resolve_geometry_accuracy'])}; end to end "
        f"(resolve, then words to the reader): {_r(rep['end_to_end_resolve_then_words_reader'])}"),
        f"- ask rate (tau 0.95): {_r(rep['ask_rate'])}",
        f"- relations used: {rep['relations']}",
        "",
        "## Reader by kind (robot words | human words | visible-only)",
    ]
    hw, vo = rep["reader_human_words"]["by_kind"], rep["reader_robot_words_visible_marks_only"][
        "by_kind"]
    for k, v in g["3_reader_robot_words"]["by_kind"].items():
        lines.append(f"- {k}: {v['rate']:.2f} | {hw[k]['rate']:.2f} | {vo[k]['rate']:.2f} "
                     f"(n {v['n']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["cpu", "live", "report"])
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.what == "report":
        res = report()
        print((DEST / "gate.md").read_text(encoding="utf-8"))
        print("pass" if res["pass"] else "fail")
        return
    items = bench_items.load(BENCH)
    if args.what == "cpu":
        out = {"bench": equiv_bench(items), "words": words_rows(items),
               "scenes": equiv_scenes(), "smolvla": equiv_smolvla()}
        (DEST / "cpu.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        for k in ("bench", "scenes", "smolvla"):
            bad = [r for r in out[k] if not all(r.get(c, True) for c in
                                                ("target", "ring", "pose"))]
            print(k, len(out[k]), "mismatches", len(bad), bad[:3])
        print("words", sum(r["hit"] for r in out["words"]), "/", len(out["words"]))
    else:
        (DEST / "live.json").write_text(json.dumps(live(items, args.url), indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
