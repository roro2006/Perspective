"""T5-35 gate: the dog demo's seeded building layouts, run once on the demo laptop.

    uv run python scripts/dog_scenes_gate.py          # writes out/t5_35/gate.json and gate.md

Same code the viewer runs (DogWorld, decide, the layout's routes). Lines Q, D, E and the
showcase rule are the card's, unchanged."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curio_robot.demo import dog
from curio_robot.demo.dog_scenes import NEW_KINDS

OUT = ROOT / "out" / "t5_35"
GOAL_TOL = 0.35  # T5-20's
CAP_S = 40.0
Q_SEEDS, D_SEEDS = range(10), range(50)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def walk(w: dog.DogWorld, room: str) -> dict:
    w.home()
    goal = w.go(room)
    while w.walking and w.sim_t < CAP_S:
        w.step(0.1)
    err = float(np.linalg.norm(w.dog_xy() - goal))
    return {"room": room, "err_m": round(err, 3), "t_s": round(w.sim_t, 2), "fell": w.fell,
            "timed_out": not w.dog.arrived and not w.fell}


def line_q(kind: str) -> dict:
    runs = []
    for seed in Q_SEEDS:
        w = dog.DogWorld(kind, seed)
        for room in w.rooms:
            r = walk(w, room) | {"seed": seed}
            r["ok"] = r["err_m"] <= GOAL_TOL and not r["fell"] and not r["timed_out"]
            runs.append(r)
        w.close()
    k, n, falls = sum(r["ok"] for r in runs), len(runs), sum(r["fell"] for r in runs)
    return {"need": ">= 95% within 0.35 m, 0 falls", "ok": k, "n": n, "falls": falls,
            "ci95": wilson(k, n), "pass": k >= 0.95 * n and falls == 0, "runs": runs}


def line_d(kind: str) -> dict:
    rows = []
    for seed in D_SEEDS:
        w = dog.DogWorld(kind, seed)
        on, off = dog.decide(w, w.request, True), dog.decide(w, w.request, False)
        rows.append({"seed": seed, "words": w.request, "truth": on.oracle_id,
                     "on": on.target.object_id, "on_verdict": on.verdict,
                     "off": off.target.object_id, "off_verdict": off.verdict})
        w.close()
    k_on = sum(r["on_verdict"] == "correct" for r in rows)
    k_off = sum(r["off_verdict"] == "correct" for r in rows)
    return {"need": "Curio on correct >= 49 of 50 (off reported)", "on_correct": k_on,
            "off_correct": k_off, "n": len(rows), "on_ci95": wilson(k_on, len(rows)),
            "off_ci95": wilson(k_off, len(rows)), "pass": k_on >= 49, "rows": rows}


def end_to_end(kind: str, seed: int) -> dict:
    w = dog.DogWorld(kind, seed)
    d = dog.decide(w, w.request, True)
    r = walk(w, d.target.object_id) if d.target.object_id else {"fell": False}
    ok = bool(d.oracle_id) and not r["fell"] and dog.arrived_at(w, d.oracle_id)
    w.close()
    return r | {"seed": seed, "truth": d.oracle_id, "ok": ok}


def line_e(kind: str) -> dict:
    runs = [end_to_end(kind, s) for s in Q_SEEDS]
    k, falls = sum(r["ok"] for r in runs), sum(r["fell"] for r in runs)
    return {"need": ">= 9 of 10 in the true room (0.5 m, nearest goal), 0 falls", "ok": k,
            "n": len(runs), "falls": falls, "ci95": wilson(k, len(runs)),
            "pass": k >= 9 and falls == 0, "runs": runs}


def showcase(kind: str, d: dict, e: dict) -> list[int]:
    """The card's rule: the two lowest seeds where off is wrong and on is right, and the walk
    succeeded (E's run for seeds 0-9, one confirming walk above)."""
    e_ok = {r["seed"]: r["ok"] for r in e["runs"]}
    picks = []
    for r in d["rows"]:
        if r["off_verdict"] != "wrong" or r["on_verdict"] != "correct":
            continue
        s = r["seed"]
        if s in e_ok and not e_ok[s]:
            continue
        if s not in e_ok and not end_to_end(kind, s)["ok"]:
            continue
        picks.append(s)
        if len(picks) == 2:
            break
    return picks


def main() -> None:
    if not dog.have_assets():
        sys.exit("Go2 assets missing (see third_party/README.md)")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    gate: dict = {"card": "T5-35", "kinds": {}}
    for kind in NEW_KINDS:
        g = {"Q": line_q(kind)}
        print(kind, "Q", g["Q"]["ok"], "/", g["Q"]["n"], "falls", g["Q"]["falls"], flush=True)
        g["D"] = line_d(kind)
        print(kind, "D on", g["D"]["on_correct"], "off", g["D"]["off_correct"], flush=True)
        g["E"] = line_e(kind)
        print(kind, "E", g["E"]["ok"], "/", g["E"]["n"], flush=True)
        g["pass"] = all(g[k]["pass"] for k in ("Q", "D", "E"))
        g["offered"] = g["Q"]["pass"]
        g["showcase"] = showcase(kind, g["D"], g["E"]) if g["pass"] else []
        print(kind, "showcase", g["showcase"], flush=True)
        gate["kinds"][kind] = g
    gate["pass"] = all(g["pass"] for g in gate["kinds"].values())
    gate["wall_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    md = ["# T5-35 gate (run once)", "", f"Overall: **{'PASS' if gate['pass'] else 'FAIL'}**",
          "", "| kind | Q walks | D Curio on | D Curio off | E end to end | showcase |",
          "|---|---|---|---|---|---|"]
    for kind, g in gate["kinds"].items():
        q, d, e = g["Q"], g["D"], g["E"]
        md.append(f"| {kind} | {q['ok']}/{q['n']} {q['ci95']}, falls {q['falls']} "
                  f"({'PASS' if q['pass'] else 'FAIL'}) | {d['on_correct']}/{d['n']} "
                  f"({'PASS' if d['pass'] else 'FAIL'}) | {d['off_correct']}/{d['n']} | "
                  f"{e['ok']}/{e['n']} {e['ci95']}, falls {e['falls']} "
                  f"({'PASS' if e['pass'] else 'FAIL'}) | {g['showcase']} |")
    (OUT / "gate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
