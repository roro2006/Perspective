"""T5-23 gate: teach by doing through the head camera, 50 episodes per family.

    uv run python scripts/ego_eval.py                  # all 200, 12 worker processes
    uv run python scripts/ego_eval.py --n 3 --workers 4

Each episode records one demonstration, extracts the plan from the head camera, and runs it on
a new layout with Curio (person's frame) and with the baseline (robot's frame). Writes
results.jsonl, summary.json and summary.md to --out."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from curio_robot.sim.ego_exec import episode
from curio_robot.sim.ego_tasks import FAMILIES, make_episode
from curio_robot.sim.ego_world import EgoWorld

OUT = Path("C:/Users/Rohan/dev/Perspective/out/t5_23")
GATE = {"exact": 0.90, "success": 0.80, "c_gap": 0.40}
_world: EgoWorld | None = None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _init() -> None:
    global _world
    _world = EgoWorld()


def _job(args: tuple[str, int]) -> dict:
    family, seed = args
    t = time.time()
    assert _world is not None
    out = episode(_world, make_episode(family, seed))
    out["wall_s"] = round(time.time() - t, 1)
    return out


def _rate(rows: list[dict], key) -> dict:
    k = sum(1 for r in rows if key(r))
    lo, hi = wilson(k, len(rows))
    return {"k": k, "n": len(rows), "rate": k / len(rows) if rows else 0.0,
            "ci": [round(lo, 3), round(hi, 3)]}


def summarise(rows: list[dict]) -> dict:
    table = {}
    for fam in (*FAMILIES, "all"):
        sel = [r for r in rows if fam == "all" or r["family"] == fam]
        if not sel:
            continue
        table[fam] = {
            "exact": _rate(sel, lambda r: r["exact"]),
            "curio": _rate(sel, lambda r: r["curio"]["success"]),
            "baseline": _rate(sel, lambda r: r["baseline"]["success"]),
            "curio_fail": dict(Counter(r["curio"]["reason"].split(":")[-1].strip()
                                       for r in sel if not r["curio"]["success"])),
        }
    allr = table["all"]
    c = table.get("c_beside")
    gate = {
        "exact": allr["exact"]["rate"] >= GATE["exact"],
        "success": allr["curio"]["rate"] >= GATE["success"],
        "c_gap": c is not None and c["baseline"]["rate"] <= c["curio"]["rate"] - GATE["c_gap"],
    }
    return {"table": table, "gate": gate, "pass": all(gate.values())}


def markdown(s: dict) -> str:
    def cell(x: dict) -> str:
        return f"{x['k']}/{x['n']} = {x['rate']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"

    lines = ["| family | plan exact match | Curio success | baseline success | Curio failures |",
             "|---|---|---|---|---|"]
    for fam, row in s["table"].items():
        fails = ", ".join(f"{k} {v}" for k, v in row["curio_fail"].items()) or "-"
        lines.append(f"| {fam} | {cell(row['exact'])} | {cell(row['curio'])} | "
                     f"{cell(row['baseline'])} | {fails} |")
    g = s["gate"]
    verdict = {k: "PASS" if v else "FAIL" for k, v in g.items()}
    lines += ["", (f"Gate: exact >= 0.90 {verdict['exact']}; success >= 0.80 "
                   f"{verdict['success']}; (c) baseline <= Curio - 0.40 {verdict['c_gap']}. "
                   f"Overall: {'PASS' if s['pass'] else 'FAIL'}.")]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="episodes per family")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (dev runs: 1000)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(f, s) for f in FAMILIES for s in range(args.seed0, args.seed0 + args.n)]
    rows = []
    t = time.time()
    with Pool(args.workers, initializer=_init) as pool, \
            open(args.out / "results.jsonl", "w") as fh:
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{len(rows)}/{len(jobs)}] {r['family']} {r['seed']} exact={r['exact']} "
                  f"curio={r['curio']['success']} base={r['baseline']['success']} "
                  f"{r['curio']['reason']}", flush=True)
    rows.sort(key=lambda r: (r["family"], r["seed"]))
    s = summarise(rows)
    s["wall_minutes"] = round((time.time() - t) / 60, 1)
    (args.out / "summary.json").write_text(json.dumps(s, indent=2))
    (args.out / "summary.md").write_text(markdown(s) + "\n")
    print(markdown(s))


if __name__ == "__main__":
    main()
