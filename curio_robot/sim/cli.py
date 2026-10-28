"""curio-sim: render demo shots and run the A/B experiment.

    uv run curio-sim shot perspective --seed 3 --out out/shots
    uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2
    uv run curio-sim live perspective --seed 3 --vlm-latency 0.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from curio_robot.sim import ab
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.shots import render_shot


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
    v.add_argument("--off", action="store_true", help="Curio off: robot camera only")

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
                s: p50(s) for s in ("motion_start", "final_transcript", "vlm_reply", "arrived")
            },
        }, indent=2))
