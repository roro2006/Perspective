"""Score belief-aware grounding against the current-state-only pipeline on false-belief
scenes and true-belief controls, and check the card's gate. Task T5-21.

    uv run python scripts/belief_eval.py --url http://127.0.0.1:8081/v1 --out out/t5_21

Needs llama-server with the r16 v2 adapter (docs/bringup-7900xt.md). Items are generated
from belief_bench.SEEDS; every grounder call is logged in <out>/gate.json."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from curio_robot import belief_bench as bb
from curio_robot.grounding.client import VllmGrounder
from curio_robot.sim.world import SimWorld

GATE_BELIEF = 0.85
GATE_GAP = 0.30
GATE_CONTROL_SLACK = 0.03
GATE_FLAG = 0.95


def evaluate(items: list[bb.BeliefItem], grounder: bb.Grounder) -> list[dict]:
    rows = []
    for item in items:
        systems = {
            "control": ("current", "belief", "reference", "belief-raw"),
            "false-belief": ("current", "belief", "belief-raw"),
        }.get(item.kind, ("current", "belief"))
        row = {
            "id": item.id, "kind": item.kind, "text": item.text, "frame": item.frame,
            "target": item.target_id, "moved": item.moved_id, "mover": item.mover,
            "away": item.away, "target_moved": item.target_moved,
        }
        for system in systems:
            res = bb.run(item, grounder, system)
            row[system] = {
                "chosen": res.object_id, "ok": res.object_id == item.target_id,
                "used_belief": res.used_belief, "note": res.note, "error": res.error,
                "p": round(res.probs.get(res.object_id, 0.0), 4) if res.object_id else 0.0,
                "goal": None if res.goal is None else [round(float(x), 4) for x in res.goal],
            }
        row["flag_ok"] = (row["belief"]["note"] is not None) == item.target_moved
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    def ok(kind: str, system: str, **where) -> list[bool]:
        return [
            r[system]["ok"] for r in rows
            if r["kind"] == kind and all(r[k] == v for k, v in where.items())
        ]

    fb_b, fb_c = ok("false-belief", "belief"), ok("false-belief", "current")
    ct_b, ct_c, ct_r = ok("control", "belief"), ok("control", "current"), ok("control", "reference")
    flags = [r["flag_ok"] for r in rows if r["kind"] in ("false-belief", "control")]
    gap = bb.paired_diff_ci(fb_b, fb_c)
    r_name = "current" if sum(ct_c) >= sum(ct_r) else "reference"
    r_hits = ct_c if r_name == "current" else ct_r
    ctrl_diff = bb.paired_diff_ci(ct_b, r_hits)
    s = {
        "false_belief": {"belief": bb.rate(fb_b), "current": bb.rate(fb_c)},
        "fb_gap": {"diff": gap[0], "ci95": [gap[1], gap[2]]},
        "control": {"belief": bb.rate(ct_b), "current": bb.rate(ct_c), "reference": bb.rate(ct_r)},
        "control_vs_R": {"R": r_name, "diff": ctrl_diff[0], "ci95": [ctrl_diff[1], ctrl_diff[2]]},
        "flag": bb.rate(flags),
        "belief_raw_ablation": {"false_belief": bb.rate(ok("false-belief", "belief-raw")),
                                "control": bb.rate(ok("control", "belief-raw"))},
        "present": {"belief": bb.rate(ok("present", "belief")),
                    "current": bb.rate(ok("present", "current"))},
        "breakdown_false_belief": {},
    }
    for key, values in (("target_moved", (True, False)), ("away", ("look-away", "leave")),
                        ("mover", ("robot", "person")), ("frame", ("speaker", "robot"))):
        for v in values:
            s["breakdown_false_belief"][f"{key}={v}"] = {
                "belief": bb.rate(ok("false-belief", "belief", **{key: v})),
                "current": bb.rate(ok("false-belief", "current", **{key: v})),
            }
    b_acc, c_acc = s["false_belief"]["belief"]["acc"], s["false_belief"]["current"]["acc"]
    r_acc = max(s["control"]["current"]["acc"], s["control"]["reference"]["acc"])
    s["gate"] = {
        "1_belief_fb>=0.85": b_acc >= GATE_BELIEF,
        "2_current_fb<=belief-0.30": c_acc <= b_acc - GATE_GAP,
        "3_control_belief>=R-0.03": s["control"]["belief"]["acc"] >= r_acc - GATE_CONTROL_SLACK,
        "4_flag>=0.95": s["flag"]["acc"] >= GATE_FLAG,
    }
    s["gate"]["pass"] = all(s["gate"].values())
    return s


def table(s: dict) -> str:
    def cell(r: dict) -> str:
        return f"{r['acc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ({r['k']}/{r['n']})"

    lines = [
        "| set | belief-aware | current-state only | v2 marked ref |",
        "|---|---|---|---|",
        (f"| false-belief | {cell(s['false_belief']['belief'])} | "
        f"{cell(s['false_belief']['current'])} | |"),
        (f"| control | {cell(s['control']['belief'])} | {cell(s['control']['current'])} | "
        f"{cell(s['control']['reference'])} |"),
        (f"| present (after move) | {cell(s['present']['belief'])} | "
        f"{cell(s['present']['current'])} | |"),
        "",
        (f"FB gap belief - current: {s['fb_gap']['diff']:.3f} "
        f"[{s['fb_gap']['ci95'][0]:.3f}, {s['fb_gap']['ci95'][1]:.3f}]"),
        (f"Control belief - R ({s['control_vs_R']['R']}): {s['control_vs_R']['diff']:.3f} "
        f"[{s['control_vs_R']['ci95'][0]:.3f}, {s['control_vs_R']['ci95'][1]:.3f}]"),
        f"Note flag: {cell(s['flag'])}",
        (f"Ablation, belief-aware with the words as said: false-belief "
        f"{cell(s['belief_raw_ablation']['false_belief'])}, control "
        f"{cell(s['belief_raw_ablation']['control'])}"),
        "",
    ]
    lines += [f"{k}: {'PASS' if v else 'FAIL'}" for k, v in s["gate"].items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--out", type=Path, default=Path("out/t5_21"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    with SimWorld(320, 240) as world:
        items = bb.generate(world)
    print(f"{len(items)} items in {time.perf_counter() - t0:.0f} s")
    grounder = VllmGrounder(args.url, "qwen-vl", timeout_s=args.timeout)
    rows = evaluate(items, grounder)
    summary = summarise(rows)
    print(table(summary))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "seeds": bb.SEEDS, "counts": bb.COUNTS},
                   indent=2),
        encoding="utf-8",
    )
    (args.out / "gate.md").write_text(table(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
