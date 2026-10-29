"""curio-sim bench: generate the perspective benchmark and score grounders on
it, per request kind, against the no-VLM rule. Task T5-03."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curio_robot.bench import items as bench_items
from curio_robot.bench.conditions import CONDITIONS
from curio_robot.bench.score import rule_predict, score, vlm_predict


def add_parser(sub: argparse._SubParsersAction) -> None:
    b = sub.add_parser("bench", help="perspective benchmark: generate items, score grounders")
    bs = b.add_subparsers(dest="bench_cmd", required=True)

    g = bs.add_parser("gen", help="render and save benchmark items")
    g.add_argument("--n-per-kind", type=int, default=50)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", type=Path, default=Path("out/bench5"))

    s = bs.add_parser("score", help="score the rule and, with --url, a grounder per condition")
    s.add_argument("--data", type=Path, default=Path("out/bench5"))
    s.add_argument("--url", default=None, help="OpenAI-compatible grounder endpoint")
    s.add_argument("--conditions", default=",".join(CONDITIONS), help="comma-separated")
    s.add_argument("--json", type=Path, default=None)


def run(args: argparse.Namespace) -> None:
    if args.bench_cmd == "gen":
        from curio_robot.sim.world import SimWorld

        items = bench_items.generate(SimWorld(320, 240), args.n_per_kind, args.seed)
        path = bench_items.save(items, args.out)
        print(f"{len(items)} items -> {path}")
        return

    items = bench_items.load(args.data)
    results = {"rule": score(items, rule_predict)}
    if args.url:
        from curio_robot.grounding.client import VllmGrounder

        grounder = VllmGrounder(args.url)
        for condition in (c for c in args.conditions.split(",") if c):
            results[condition] = score(items, vlm_predict(grounder, condition))
    print(format_table(results))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


def format_table(results: dict[str, dict]) -> str:
    """Markdown: one row per kind plus overall, one column per scorer."""
    names = list(results)
    kinds = list(next(iter(results.values()))["by_kind"])
    lines = ["| kind | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for kind in kinds:
        n = results[names[0]]["by_kind"][kind]["n"]
        accs = " | ".join(f"{results[c]['by_kind'][kind]['acc']:.2f}" for c in names)
        lines.append(f"| {kind} | {n} | {accs} |")
    overall = " | ".join(f"{results[c]['acc']:.2f}" for c in names)
    lines.append(f"| overall | {results[names[0]]['n']} | {overall} |")
    return "\n".join(lines)
