"""T5-36 gate: survivor variety in the dog demo's buildings, run once on the demo laptop.

    uv run python scripts/dog_survivors_gate.py       # writes out/t5_36/gate.json and gate.md

Lines U, V, Q and E are the card's, unchanged. U compares against T5-35's recorded rows."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curio_robot.demo import dog, dog_scenes
from curio_robot.demo.dog_scenes import NEW_KINDS, POSES

OUT = ROOT / "out" / "t5_36"
T535 = ROOT / "out" / "t5_35" / "gate.json"
GOAL_TOL, CAP_S = 0.35, 40.0


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(max(0.0, c - h)), 3), round(float(min(1.0, c + h)), 3)]


def walk(w: dog.DogWorld, room: str) -> dict:
    goal = w.go(room)
    while w.walking and w.sim_t < CAP_S:
        w.step(0.1)
    return {"room": room, "err_m": round(float(np.linalg.norm(w.dog_xy() - goal)), 3),
            "fell": w.fell, "timed_out": not w.dog.arrived and not w.fell,
            "contacts": w.survivor_contacts}


def line_u(kind: str, recorded: list[dict]) -> dict:
    same = 0
    for row in recorded:
        w = dog.DogWorld(kind, row["seed"])
        on, off = dog.decide(w, w.request, True), dog.decide(w, w.request, False)
        same += (w.request == row["words"] and on.oracle_id == row["truth"]
                 and on.target.object_id == row["on"] and off.target.object_id == row["off"])
        w.close()
    return {"need": "50 of 50 equal T5-35", "same": same, "n": len(recorded),
            "pass": same == 50 and len(recorded) == 50}


def line_v(kind: str) -> dict:
    counts, outfits, poses = {}, set(), set()
    for seed in range(50):
        lay = dog_scenes.make(kind, seed)
        n = sum(sv.meant for sv in lay.survivors)
        counts[n] = counts.get(n, 0) + 1
        outfits |= {sv.outfit.name for sv in lay.survivors}
        poses |= {sv.pose for sv in lay.survivors}
    ok = set(counts) >= {1, 2, 3} and len(outfits) >= 5 and poses == set(POSES)
    return {"need": "counts 1, 2, 3; >= 5 outfits; all 4 poses", "counts": counts,
            "outfits": sorted(outfits), "poses": sorted(poses), "pass": ok}


def line_q(kind: str) -> dict:
    runs = []
    for seed in range(10):
        w = dog.DogWorld(kind, seed)
        for room in w.rooms:
            w.home()
            r = walk(w, room) | {"seed": seed}
            r["ok"] = r["err_m"] <= GOAL_TOL and not r["fell"] and not r["timed_out"]
            runs.append(r)
        w.close()
    k, n = sum(r["ok"] for r in runs), len(runs)
    falls, contacts = sum(r["fell"] for r in runs), sum(r["contacts"] for r in runs)
    return {"need": ">= 95% within 0.35 m, 0 falls, 0 survivor contacts", "ok": k, "n": n,
            "falls": falls, "contacts": contacts, "ci95": wilson(k, n),
            "pass": k >= 0.95 * n and falls == 0 and contacts == 0, "runs": runs}


def line_e(kind: str) -> dict:
    runs = []
    for seed in range(10):
        for turned in (False, True):
            w = dog.DogWorld(kind, seed)
            if turned:
                w.turn_firefighter()
            d = dog.decide(w, w.request, True)
            r = walk(w, d.target.object_id)
            here = w.survivors_in(d.oracle_id)
            ok = (not r["fell"] and dog.arrived_at(w, d.oracle_id) and 1 <= here <= 3
                  and r["contacts"] == 0)
            runs.append(r | {"seed": seed, "turned": turned, "truth": d.oracle_id,
                             "survivors_there": here, "ok": ok})
            w.close()
    k = sum(r["ok"] for r in runs)
    falls, contacts = sum(r["fell"] for r in runs), sum(r["contacts"] for r in runs)
    return {"need": ">= 19 of 20 in the meant room with 1-3 survivors, 0 falls, 0 contacts",
            "ok": k, "n": len(runs), "falls": falls, "contacts": contacts,
            "ci95": wilson(k, len(runs)),
            "pass": k >= 19 and falls == 0 and contacts == 0, "runs": runs}


def main() -> None:
    if not dog.have_assets():
        sys.exit("Go2 assets missing (see third_party/README.md)")
    recorded = json.loads(T535.read_text())["kinds"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    gate: dict = {"card": "T5-36", "kinds": {}}
    for kind in NEW_KINDS:
        g = {"U": line_u(kind, recorded[kind]["D"]["rows"]), "V": line_v(kind),
             "Q": line_q(kind), "E": line_e(kind)}
        g["pass"] = all(g[k]["pass"] for k in "UVQE")
        print(kind, {k: g[k]["pass"] for k in "UVQE"}, flush=True)
        gate["kinds"][kind] = g
    gate["pass"] = all(g["pass"] for g in gate["kinds"].values())
    gate["wall_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    md = ["# T5-36 gate (run once)", "", f"Overall: **{'PASS' if gate['pass'] else 'FAIL'}**",
          "", "| kind | U unchanged | V variety | Q walks | E end to end |", "|---|---|---|---|---|"]
    for kind, g in gate["kinds"].items():
        u, v, q, e = g["U"], g["V"], g["Q"], g["E"]
        md.append(
            f"| {kind} | {u['same']}/{u['n']} | counts {dict(sorted(v['counts'].items()))}, "
            f"{len(v['outfits'])} outfits, {len(v['poses'])} poses "
            f"({'PASS' if v['pass'] else 'FAIL'}) | {q['ok']}/{q['n']}, falls {q['falls']}, "
            f"contacts {q['contacts']} ({'PASS' if q['pass'] else 'FAIL'}) | "
            f"{e['ok']}/{e['n']}, falls {e['falls']}, contacts {e['contacts']} "
            f"({'PASS' if e['pass'] else 'FAIL'}) |")
    (OUT / "gate.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
