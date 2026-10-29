"""curio-sim: render demo shots and run the A/B experiment.

    uv run curio-sim shot perspective --seed 3 --out out/shots
    uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2
    uv run curio-sim live perspective --seed 3 --vlm-latency 0.6
    uv run curio-sim live perspective --grounder-url http://127.0.0.1:8081/v1
    uv run curio-sim learn --user all --users 50 --noise --baseline
    uv run curio-sim learn --live --user robot-frame --requests 6
    uv run curio-sim collab --tasks 5 --conditions view-off,view-on --user mixed --export out/dataset
    uv run curio-sim critic phase1 --out out/critic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from curio_robot.critic import cli as critic_cli
from curio_robot.grounding.client import DEFAULT_URL, VllmGrounder
from curio_robot.sim import ab
from curio_robot.sim.ground_eval import DEFAULT_MAX_P95_MS, run_ground_eval
from curio_robot.sim.learn import learning_curve, live_learning
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.shots import render_shot
from curio_robot.sim.users import POPULATION

LIVE_SEED = 1  # a stage-friendly default: seed 0's robot-frame person says "mine" once by chance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="curio-sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("shot", help="render Curio off/on frames for one scenario")
    s.add_argument("scenario", choices=sorted(SCENARIOS))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("out/shots"))
    s.add_argument("--size", default="640x480", help="pane size, e.g. 640x480")

    a = sub.add_parser("ab", help="success rate with vs. without the human view")
    a.add_argument("--trials", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--pose-noise", type=float, default=0.0, help="head-cam position sigma, m")
    a.add_argument("--rot-noise", type=float, default=0.0, help="head-cam rotation sigma, deg")
    a.add_argument("--out", type=Path, default=Path("out/ab.json"))

    v = sub.add_parser("live", help="the live runtime on a scenario, with a latency waterfall")
    v.add_argument("scenario", choices=sorted(SCENARIOS))
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--runs", type=int, default=1, help="seeds seed..seed+runs-1")
    v.add_argument("--vlm-latency", type=float, default=0.6, help="mock VLM delay, s")
    v.add_argument("--llm-url", help="OpenAI-compatible endpoint instead of the mock")
    v.add_argument("--grounder-url", help="target 2: grounder endpoint (llama-server / vLLM)")
    v.add_argument("--off", action="store_true", help="Curio off: robot camera only")

    lr = sub.add_parser("learn", help="learning curves: simulated users, error rate per request")
    lr.add_argument("--user", choices=["own-frame", "robot-frame", "mixed", "all"], default="all")
    lr.add_argument("--requests", type=int, default=15)
    lr.add_argument("--users", type=int, default=50)
    lr.add_argument("--seed", type=int, default=None, help=f"default 0 ({LIVE_SEED} with --live)")
    lr.add_argument("--noise", action="store_true")
    lr.add_argument("--baseline", action="store_true")
    lr.add_argument("--out", type=Path, default=Path("out/learn.json"))
    lr.add_argument("--live", action="store_true",
                    help="one person through the full runtime, with a transcript")

    g = sub.add_parser("ground-eval", help="score a grounder on a dataset and gate it")
    g.add_argument("--data", type=Path, required=True, help="data.jsonl from grounding_data")
    g.add_argument("--url", default=DEFAULT_URL, help="vLLM base URL, OpenAI-compatible")
    g.add_argument("--ids", type=Path, help="JSON file holding a list of ids to evaluate")
    g.add_argument("--current", type=Path, help="previous result JSON to gate against")
    g.add_argument("--max-p95-ms", type=float, default=DEFAULT_MAX_P95_MS,
                   help="latency budget, ms")
    g.add_argument("--out", type=Path, default=Path("out/eval/ground_eval.json"))

    c = sub.add_parser("collab", help="run collaboration tasks per condition, print the scoreboard")
    c.add_argument("--user", choices=["own-frame", "robot-frame", "mixed"], default="mixed")
    c.add_argument("--tasks", type=int, default=5, help="seeds seed..seed+tasks-1 per condition")
    c.add_argument("--goals", type=int, default=4)
    c.add_argument("--conditions", default="view-off,view-on", help="comma-separated")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--deictic", type=float, default=0.0, help="p_deictic")
    c.add_argument("--occlude", action="store_true")
    c.add_argument("--grounder-url", default=None)
    c.add_argument("--out", type=Path, default=Path("out/collab"))
    c.add_argument("--export", type=Path, default=None, help="write the dataset here")

    critic_cli.add_parser(sub)

    args = parser.parse_args(argv)
    if args.cmd == "shot":
        pane = tuple(int(v) for v in args.size.split("x"))
        args.out.mkdir(parents=True, exist_ok=True)
        for on in (False, True):
            path = args.out / f"{args.scenario}_seed{args.seed}_{'on' if on else 'off'}.png"
            render_shot(args.scenario, args.seed, on, pane).save(path)
            print(path)
    elif args.cmd == "live":
        _live(args)
    elif args.cmd == "learn":
        _learn(args)
    elif args.cmd == "ground-eval":
        _ground_eval(args)
    elif args.cmd == "collab":
        _collab(args)
    elif args.cmd == "critic":
        critic_cli.run(args)
    else:
        result = ab.run(args.trials, args.seed, args.pose_noise, args.rot_noise)
        ab.save(result, args.out)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


def _live(args) -> None:
    from curio_robot.sim.live import run_live
    from curio_robot.sim.world import SimWorld

    world = SimWorld(320, 240)
    results = []
    for seed in range(args.seed, args.seed + args.runs):
        r = run_live(
            args.scenario, seed, vlm_latency=args.vlm_latency, llm_url=args.llm_url,
            use_human_view=not args.off, quiet=args.runs > 1, world=world,
            grounder_url=args.grounder_url,
        )
        results.append(r)
        if args.runs == 1:
            print(json.dumps(r, indent=2))
    if args.runs > 1:
        def p50(stage):
            v = [r["waterfall_ms"][stage] for r in results if stage in r["waterfall_ms"]]
            return round(float(np.median(v)), 1) if v else None

        print(json.dumps({
            "scenario": args.scenario,
            "runs": args.runs,
            "curio": "off" if args.off else "on",
            "success_pct": round(100 * float(np.mean([r["success"] for r in results])), 1),
            "p50_ms_from_speech_end": {
                s: p50(s) for s in (
                    "motion_start", "final_transcript", "vlm_reply", "ground_sent",
                    "ground_reply", "arrived",
                )
            },
        }, indent=2))


def _learn(args) -> None:
    if args.live:
        _learn_live(args)
        return
    seed = 0 if args.seed is None else args.seed
    pos_noise = 0.01 if args.noise else 0.0
    rot_noise_deg = 2.0 if args.noise else 0.0

    if args.user == "all":
        users = list(POPULATION)
    else:
        users = [u for u in POPULATION if u.name == args.user]

    results = []
    for user in users:
        res = learning_curve(
            user,
            requests=args.requests,
            users=args.users,
            seed=seed,
            pos_noise=pos_noise,
            rot_noise_deg=rot_noise_deg,
            learn=True,
        )
        results.append(res)
        if args.baseline:
            res_base = learning_curve(
                user,
                requests=args.requests,
                users=args.users,
                seed=seed,
                pos_noise=pos_noise,
                rot_noise_deg=rot_noise_deg,
                learn=False,
            )
            results.append(res_base)

    print("user, mode, err@1, err@3, err@5, err@last, ask@1, ask@3, ask@5, ask@last, p_own")
    for r in results:
        print(_format_learn_row(r, args.requests))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


def _format_learn_row(r: dict, requests: int) -> str:
    err = r["error_rate"]
    ask = r["ask_rate"]
    indices = [0, 2, 4, requests - 1]
    indices = [i for i in indices if i < requests]

    err_strs = [f"{err[i] * 100:.1f}%" for i in indices]
    ask_strs = [f"{ask[i] * 100:.1f}%" for i in indices]

    mode = "learn" if r["learn"] else "baseline"
    p_own = f"{r['final_p_own_frame']:.2f}"

    return f"{r['user']}, {mode}, {', '.join(err_strs)}, {', '.join(ask_strs)}, {p_own}"


def _learn_live(args) -> None:
    """One simulated person per user type, live; --users, --noise, --baseline don't apply."""
    users = list(POPULATION) if args.user == "all" else [u for u in POPULATION if u.name == args.user]
    seed = LIVE_SEED if args.seed is None else args.seed
    for user in users:
        print(f"== {user.name} (live, {args.requests} requests, seed {seed})")
        r = live_learning(user, requests=args.requests, seed=seed)
        print(f"outcomes: {', '.join(r['outcomes'])}")
        print(f"p_own_frame: {' -> '.join(f'{p:.2f}' for p in r['p_own_frame'])}")


def _ground_eval(args) -> None:
    grounder = VllmGrounder(args.url)
    ids = json.loads(args.ids.read_text()) if args.ids else None
    current = json.loads(args.current.read_text()) if args.current else None
    result, verdict = run_ground_eval(
        args.data, grounder, ids=ids, current=current, max_p95_ms=args.max_p95_ms
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"n {result['n']}  accuracy {result['accuracy']:.3f}")
    for kind in sorted(result["by_kind"]):
        stats = result["by_kind"][kind]
        print(f"  {kind}: {stats['accuracy']:.3f} (n {stats['n']})")
    print(f"p50 {result['p50_ms']:.0f} ms  p95 {result['p95_ms']:.0f} ms")
    if verdict is not None:
        ok, message = verdict
        print(f"gate {'PASS' if ok else 'FAIL'}: {message}")
        if not ok:
            sys.exit(1)


def _collab(args) -> None:
    from curio_robot.collab.export import export_episodes
    from curio_robot.collab.metrics import summarize
    from curio_robot.collab.report import format_table
    from curio_robot.collab.run import run_task
    from curio_robot.collab.session import save_sessions
    from curio_robot.sim.world import SimWorld

    user = next(u for u in POPULATION if u.name == args.user)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    world = SimWorld(320, 240)

    sessions = []
    for condition in conditions:
        for seed in range(args.seed, args.seed + args.tasks):
            session = run_task(
                user, condition, seed,
                n_goals=args.goals,
                p_deictic=args.deictic,
                occlude=args.occlude,
                world=world,
                grounder_url=args.grounder_url,
                quiet=True,
            )
            sessions.append(session)
            print(f"{condition} seed {seed}: {len(session.done)}/{len(session.goals)} goals, "
                  f"{len(session.steps)} turns")

    args.out.mkdir(parents=True, exist_ok=True)
    save_sessions(sessions, args.out / "sessions.jsonl")

    summary = summarize(sessions)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(format_table(summary))

    if args.export:
        counts = export_episodes(sessions, args.export, overwrite=True)
        print(f"exported {args.export}: {counts}")
