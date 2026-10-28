"""curio-sim: render demo shots and run the A/B experiment.

    uv run curio-sim shot perspective --seed 3 --out out/shots
    uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    args = parser.parse_args(argv)
    if args.cmd == "shot":
        pane = tuple(int(v) for v in args.size.split("x"))
        args.out.mkdir(parents=True, exist_ok=True)
        for on in (False, True):
            path = args.out / f"{args.scenario}_seed{args.seed}_{'on' if on else 'off'}.png"
            render_shot(args.scenario, args.seed, on, pane).save(path)
            print(path)
    else:
        result = ab.run(args.trials, args.seed, args.pose_noise, args.rot_noise)
        ab.save(result, args.out)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
