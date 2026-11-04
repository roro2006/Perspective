"""Promotion gate for a flywheel-retrained grounder (the GATE in tasks/T5-24). Task T5-24.

    uv run python scripts/flywheel_gate.py make-test --out out/t5_24/fb_test
    uv run python scripts/flywheel_gate.py score --name r16-v2 \\
        --lora C:/Users/Rohan/dev/llama/models/lora/curio-ground-r16-v2-f16.gguf \\
        --json out/t5_24/scores_v2.json
    uv run python scripts/flywheel_gate.py score --name r16-v3 \\
        --lora out/t5_24/curio-ground-r16-v3-f16.gguf --json out/t5_24/scores_v3.json
    uv run python scripts/flywheel_gate.py decide --v2 out/t5_24/scores_v2.json \\
        --v3 out/t5_24/scores_v3.json --candidate out/t5_24/curio-ground-r16-v3-f16.gguf \\
        --out out/t5_24/gate [--register]

`score` serves one adapter on a fresh llama-server under the GPU lock and scores (a)
bench5c in the geometry condition, the target-2 deictic held-out rows, (b) the front/behind
held-out set and (c) the T5-18 warehouse "behind me" query. `decide` applies the gate and,
only on promote with --register, copies the GGUF into llama/models/lora and makes it the
registry's production adapter."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("C:/Users/Rohan/dev/Perspective/out")
BENCH = OUT / "bench5c"
TARGET2 = OUT / "grounding" / "data.jsonl"
TARGET2_EVAL_IDS = OUT / "lora" / "qwen3vl4b-r16" / "eval_ids.json"
FB_TEST = OUT / "t5_24" / "fb_test"
REGISTRY = ROOT / "curio_robot" / "flywheel" / "registry.json"


def argmax(probs: dict[str, float]) -> str | None:
    return max(probs, key=probs.get) if probs else None


def score_bench(grounder) -> list[dict]:
    from curio_robot.bench import items as bench_items
    from curio_robot.bench.conditions import build

    rows = []
    for item in bench_items.load(BENCH):
        ans = grounder.score(build("geometry", item))
        pick = argmax(ans.probs)
        rows.append({"id": item.id, "kind": item.kind, "target": item.target_id, "pick": pick,
                     "hit": pick == item.target_id, "error": ans.error})
    return rows


def score_deictic(grounder) -> list[dict]:
    from curio_robot.sim.ground_eval import load_requests

    ids = json.loads(TARGET2_EVAL_IDS.read_text(encoding="utf-8"))
    rows = []
    for example, request in load_requests(TARGET2, ids):
        if example.kind != "deictic":
            continue
        target = request.letters[example.answer]
        ans = grounder.score(request)
        pick = argmax(ans.probs)
        rows.append({"id": example.id, "kind": "deictic", "target": target, "pick": pick,
                     "hit": pick == target, "error": ans.error})
    return rows


def score_front_behind(grounder) -> list[dict]:
    from curio_robot.flywheel.scenes import load_items, request_for

    rows = []
    for item in load_items(FB_TEST):
        ans = grounder.score(request_for(item))
        pick = argmax(ans.probs)
        rows.append({"id": item.id, "kind": item.kind, "pool": item.pool, "frame": item.frame,
                     "relation": item.relation, "text": item.text, "target": item.target_id,
                     "pick": pick, "p_target": ans.probs.get(item.target_id, 0.0),
                     "hit": pick == item.target_id, "error": ans.error})
    return rows


def warehouse_requests():
    """The T5-18 warehouse queries, rebuilt the way make_video.py builds them."""
    sys.path.insert(0, str(ROOT / "scripts" / "poc"))
    import mujoco
    from scene_base import GROUND_H, GROUND_W, make_request
    from scenes_a import Warehouse

    scene = Warehouse()
    model = mujoco.MjModel.from_xml_string(scene.xml())
    data = mujoco.MjData(model)
    scene.setup(model, data)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, GROUND_H, GROUND_W)
    try:
        return {q.key: (q, make_request(scene, q, renderer, model, data)[0])
                for q in scene.queries if q.depth}
    finally:
        renderer.close()


def score_warehouse(grounder) -> dict:
    out = {}
    for key, (q, req) in warehouse_requests().items():
        ans = grounder.score(req)
        out[key] = {"text": q.text, "frame": q.frame, "correct": q.correct,
                    "pick": argmax(ans.probs), "probs": ans.probs,
                    "prompt": req.messages[0]["content"][1]["text"], "error": ans.error}
    return out


def score_all(url: str, name: str) -> dict:
    from curio_robot.grounding.client import VllmGrounder

    grounder = VllmGrounder(url, "qwen-vl", name=name, timeout_s=60)
    warm = warehouse_requests()["geo"][1]
    for _ in range(2):
        grounder.score(warm)
    result = {"adapter": name, "bench5c": score_bench(grounder),
              "deictic": score_deictic(grounder), "front-behind": score_front_behind(grounder),
              "warehouse": score_warehouse(grounder)}
    errors = sum(r["error"] is not None for k in ("bench5c", "deictic", "front-behind")
                 for r in result[k])
    result["errors"] = errors
    return result


def cmd_make_test(args) -> None:
    from curio_robot.flywheel.scenes import FBWorld, heldout_set, save_items

    if (args.out / "items.jsonl").exists():
        raise SystemExit(f"error: {args.out} exists; the test set is generated once")
    with FBWorld() as world:
        items = heldout_set(world, args.n)
    print(f"{len(items)} items -> {save_items(items, args.out)}")


def cmd_score(args) -> None:
    from curio_robot.flywheel import serve

    if args.url:
        result = score_all(args.url, args.name)
    else:
        log = args.json.with_suffix(".server.log")
        with serve.gpu_lock(f"scoring {args.name}"), serve.llama_server(
                args.lora, args.port, log) as url:
            result = score_all(url, args.name)
    result["lora"] = str(args.lora)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    fb = result["front-behind"]
    print(f"{args.name}: bench5c {sum(r['hit'] for r in result['bench5c'])}/"
          f"{len(result['bench5c'])}, front-behind {sum(r['hit'] for r in fb)}/{len(fb)}, "
          f"warehouse geo -> {result['warehouse']['geo']['pick']}, errors {result['errors']}")


def subsets(result: dict) -> list[tuple[str, list[dict]]]:
    fb = result["front-behind"]
    return [(f"front-behind {k}={v}", [r for r in fb if r[k] == v])
            for k, vals in (("pool", ("train", "heldout")), ("frame", ("speaker", "robot")),
                            ("relation", ("behind", "front"))) for v in vals]


def cmd_decide(args) -> None:
    from curio_robot.flywheel.gate import compare, decide, format_table

    v2 = json.loads(args.v2.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3.read_text(encoding="utf-8"))
    verdict = decide(v2, v3)
    extra = [compare(name, rows, [r for r in v3["front-behind"]
                                  if r["id"] in {x["id"] for x in rows}])
             for name, rows in subsets(v2)]
    wh = {k: {"v2": v2["warehouse"][k], "v3": v3["warehouse"][k]} for k in v2["warehouse"]}
    lines = [(f"# T5-24 gate: {'PROMOTE' if verdict['promote'] else 'REJECT'} "
             f"{v3['adapter']} over {v2['adapter']}"), "", *[f"- {r}" for r in verdict["reasons"]],
             "", format_table(verdict["rows"]), "", "Front/behind breakdown (not gated):", "",
             format_table(extra), "", "Warehouse (reported, not gated):", ""]
    for key, both in wh.items():
        for name in ("v2", "v3"):
            w = both[name]
            p = w["probs"].get(w["correct"], 0.0)
            lines.append(f"- {key} ({w['frame']} frame) {name}: picked {w['pick']}, "
                         f"P(correct {w['correct']}) = {p:.2f}")
    report = "\n".join(lines) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate.md").write_text(report, encoding="utf-8")
    (args.out / "gate.json").write_text(json.dumps(
        {**verdict, "breakdown": extra, "warehouse": wh}, indent=1), encoding="utf-8")
    print(report)
    if verdict["promote"] and args.register:
        register(args.candidate, v3["adapter"])
    elif args.register:
        print("not registered: the gate rejected the candidate")


def register(candidate: Path, name: str) -> None:
    from curio_robot.flywheel import serve

    dest = serve.LORA_DIR / candidate.name
    if dest.exists():
        raise SystemExit(f"error: {dest} exists; never overwrite a served adapter")
    shutil.copyfile(candidate, dest)
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    reg["adapters"].append({"name": name, "gguf": candidate.name, "card": "T5-24",
                            "note": "flywheel: v2 data + front/behind corrections; gate PROMOTE"})
    reg["production"] = name
    REGISTRY.write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8")
    print(f"registered {name}: {dest}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="flywheel promotion gate (T5-24)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-test", help="generate the front/behind held-out set once")
    m.add_argument("--out", type=Path, default=FB_TEST)
    m.add_argument("--n", type=int, default=200)
    s = sub.add_parser("score", help="score one adapter on every gate set")
    s.add_argument("--name", required=True)
    s.add_argument("--lora", type=Path, required=True)
    s.add_argument("--json", type=Path, required=True)
    s.add_argument("--url", default=None, help="use this server instead of starting one")
    s.add_argument("--port", type=int, default=8091)
    d = sub.add_parser("decide", help="apply the gate to two score files")
    d.add_argument("--v2", type=Path, required=True)
    d.add_argument("--v3", type=Path, required=True)
    d.add_argument("--candidate", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--register", action="store_true")
    args = p.parse_args(argv)
    {"make-test": cmd_make_test, "score": cmd_score, "decide": cmd_decide}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
