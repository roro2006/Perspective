"""T5-22: does asking back pay? Score bench5c once with the served grounder (geometry
condition, r16 v2 adapter), cache the letter probabilities, then replay the ask-back policy
(curio_robot.clarify) against simulated answerers on that cache: tune tau_act on the dev half,
report the test half against the gate in tasks/T5-22-ask-back.md.

    uv run python scripts/ask_back_eval.py score --url http://127.0.0.1:8081/v1   # GPU, once
    uv run python scripts/ask_back_eval.py report                                # CPU only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curio_robot import clarify, perspective
from curio_robot.bench.items import BLOCK_HALF, BOX_HALF, BOX_OFFSET, BenchItem
from curio_robot.sim.world import ROBOT_CAM_POS

OUT = Path("C:/Users/Rohan/dev/Perspective/out/t5_22")
DATA = Path("C:/Users/Rohan/dev/Perspective/out/bench5c")
TAUS = (*(round(0.5 + 0.05 * i, 2) for i in range(10)), 0.97, 0.98, 0.99)
MAX_DEV_ASK_RATE = 0.25
MAX_DEV_MEAN_Q = 0.33
NOISE_SEEDS = range(20)
NOISE = 0.1
BOOT = 2000
GATE = {"acc": 0.97, "ask_rate": 0.30, "mean_q": 0.4}
HIDDEN_KINDS = ("hidden", "hidden-pair")
BOX_NEAR = 0.06  # m: the box sits BOX_OFFSET (0.045) beyond the block it hides


# ---- scoring (GPU, once) -------------------------------------------------------------------


def score_items(items: list[BenchItem], url: str) -> dict:
    from curio_robot.bench.conditions import build
    from curio_robot.grounding.client import VllmGrounder

    grounder = VllmGrounder(url, "qwen-vl", name="r16-v2", timeout_s=60)
    for _ in range(2):  # warm-up, discarded (as bench_grounding does)
        grounder.score(build("geometry", items[0]))
    out, t0 = {}, time.perf_counter()
    for i, item in enumerate(items):
        ans = grounder.score(build("geometry", item))
        out[item.id] = {"probs": ans.probs, "letter": ans.letter,
                        "latency_ms": round(ans.latency_ms, 1), "error": ans.error}
        if i % 50 == 0:
            print(f"{i}/{len(items)} {item.kind} {ans.probs}", flush=True)
    return {"condition": "geometry", "adapter": "curio-ground-r16-v2-f16.gguf", "url": url,
            "runtime_s": round(time.perf_counter() - t0, 1), "items": out}


# ---- candidates in the speaker's frame -----------------------------------------------------


def box_centre(item: BenchItem, object_id: str) -> np.ndarray:
    """Where the box stands if it hides `object_id` (as fetch_scene / _box_in_front place it)."""
    obj = next(o for o in item.object_map.objects if o.object_id == object_id)
    xy = obj.position[:2]
    toward = ROBOT_CAM_POS[:2] - xy
    return xy + BOX_OFFSET * toward / np.linalg.norm(toward)


def box_of(item: BenchItem, object_id: str) -> clarify.Landmark:
    c = box_centre(item, object_id)
    return clarify.Landmark("box", (float(c[0]), float(c[1])), BOX_HALF[1], BOX_NEAR)


def _in_shadow(xy: np.ndarray, box: np.ndarray) -> bool:
    """Is a block at xy behind the box as the robot camera sees it (top view)?"""
    cam = ROBOT_CAM_POS[:2]
    ray = xy - cam
    length = float(np.linalg.norm(ray))
    u = ray / length
    along = float((box - cam) @ u)
    return along < length and float(np.linalg.norm(box - cam - along * u)) < BOX_HALF[1] + BLOCK_HALF


def boxed(item: BenchItem) -> list[str]:
    """The block the box stands in front of. Items don't store the box, so each block the robot
    can't see is tried as the boxed one and scored by how many other blocks' visibility it
    fails to explain; fewest wins, ties to the one nearest the robot camera. Uses the robot's
    view only, never the target (it matches the generator's box on 83 of 100 hidden items)."""
    objs = item.object_map.objects
    hidden = {o.object_id for o in objs if o.object_id not in item.robot_visible}
    best: tuple[tuple[int, float], str] | None = None
    for h in (o for o in objs if o.object_id in hidden):
        box = box_centre(item, h.object_id)
        bad = sum(_in_shadow(o.position[:2], box) != (o.object_id in hidden)
                  for o in objs if o is not h)
        key = (bad, float(np.linalg.norm(h.position[:2] - ROBOT_CAM_POS[:2])))
        if best is None or key < best[0]:
            best = (key, h.object_id)
    return [] if best is None else [best[1]]


def candidates_for(item: BenchItem) -> list[clarify.Candidate]:
    m = item.object_map
    speaker = m.human_pose if m.human_pose is not None else m.robot_pose
    boxes = [] if item.kind not in HIDDEN_KINDS else [box_of(item, hid) for hid in boxed(item)]
    out = []
    for o in m.objects:
        rels = tuple(dict.fromkeys(
            r for b in boxes if (r := clarify.landmark_relation(o.position, speaker.position, b))))
        out.append(clarify.Candidate(o.object_id, o.color, perspective.lateral(o, speaker),
                                     perspective.horizontal_distance(o, speaker), rels))
    return out


# ---- replay --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    item_id: str
    kind: str
    base_ok: bool
    ok: bool
    n_q: int
    asked: tuple[tuple[str, bool], ...]
    p_max: float


def split(items: list[BenchItem]) -> tuple[list[BenchItem], list[BenchItem]]:
    """Within each kind, file order: even positions dev, odd test."""
    seen: dict[str, int] = {}
    dev, test = [], []
    for item in items:
        i = seen.get(item.kind, 0)
        seen[item.kind] = i + 1
        (dev if i % 2 == 0 else test).append(item)
    return dev, test


def replay(items: list[BenchItem], scores: dict, tau: float, noise_seed: int | None = None
           ) -> list[Run]:
    runs = []
    for idx, item in enumerate(items):
        probs = scores[item.id]["probs"]
        if noise_seed is None:
            answer = clarify.truthful(item.target_id)
        else:
            rng = np.random.default_rng([noise_seed, idx])
            answer = clarify.noisy(item.target_id, rng, NOISE)
        d = clarify.decide(candidates_for(item), probs, answer, tau)
        base = clarify.argmax(probs) if probs else None
        runs.append(Run(item.id, item.kind, base == item.target_id, d.choice == item.target_id,
                        len(d.asked), tuple((q.text, a) for q, a in d.asked),
                        max(d.prior.values()) if d.prior else 0.0))
    return runs


def summary(runs: list[Run]) -> dict:
    n = len(runs)
    return {"n": n, "base_acc": sum(r.base_ok for r in runs) / n,
            "acc": sum(r.ok for r in runs) / n,
            "ask_rate": sum(r.n_q > 0 for r in runs) / n,
            "mean_q": sum(r.n_q for r in runs) / n}


def tune(dev: list[BenchItem], scores: dict) -> tuple[float, list[dict]]:
    """The gate's rule: best dev accuracy under the ask budget; ties -> smallest tau."""
    table = [{"tau": t, **summary(replay(dev, scores, t))} for t in TAUS]
    ok = [r for r in table
          if r["ask_rate"] <= MAX_DEV_ASK_RATE and r["mean_q"] <= MAX_DEV_MEAN_Q]
    pool = ok or table
    best = max(r["acc"] for r in pool)
    return min(r["tau"] for r in pool if r["acc"] == best), table


# ---- statistics ----------------------------------------------------------------------------


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def boot_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def by_kind(runs: list[Run]) -> dict[str, dict]:
    kinds = list(dict.fromkeys(r.kind for r in runs))
    return {k: summary([r for r in runs if r.kind == k]) for k in kinds}


def evaluate(items: list[BenchItem], scores: dict) -> dict:
    dev, test = split(items)
    tau, dev_table = tune(dev, scores)
    truth = replay(test, scores, tau)
    s = summary(truth)
    n = s["n"]
    noisy_runs = [replay(test, scores, tau, seed) for seed in NOISE_SEEDS]
    noisy_ok = np.array([[r.ok for r in runs] for runs in noisy_runs], float)  # seeds x items
    noisy_acc = float(noisy_ok.mean())
    base = np.array([r.base_ok for r in truth], float)
    diff = noisy_ok.mean(axis=0) - base  # per item, averaged over seeds
    q = np.array([r.n_q for r in truth], float)
    result = {
        "tau_act": tau, "k": clarify.MAX_QUESTIONS, "eps": clarify.EPS,
        "dev_table": dev_table,
        "test": {
            **s,
            "base_acc_ci": wilson(base.sum(), n),
            "acc_ci": wilson(sum(r.ok for r in truth), n),
            "ask_rate_ci": wilson(sum(r.n_q > 0 for r in truth), n),
            "mean_q_ci": boot_ci(q),
            "noisy_acc": noisy_acc,
            "noisy_acc_seed0": float(noisy_ok[0].mean()),
            "noisy_acc_ci": wilson(noisy_acc * n, n),
            "noisy_minus_base": float(diff.mean()),
            "noisy_minus_base_ci": boot_ci(diff),
            "noisy_ask_rate": float(np.mean([summary(r)["ask_rate"] for r in noisy_runs])),
            "noisy_mean_q": float(np.mean([summary(r)["mean_q"] for r in noisy_runs])),
        },
        "by_kind": by_kind(truth),
        "noisy_by_kind": {k: float(np.mean([by_kind(r)[k]["acc"] for r in noisy_runs]))
                          for k in by_kind(truth)},
        "dev": summary(replay(dev, scores, tau)),
        "asked": [{"id": r.item_id, "kind": r.kind, "p_max": round(r.p_max, 3),
                   "base_ok": r.base_ok, "ok": r.ok, "questions": r.asked}
                  for r in truth if r.n_q],
        "missed": [{"id": r.item_id, "kind": r.kind, "p_max": round(r.p_max, 3),
                    "questions": r.asked} for r in truth if not r.ok],
    }
    t = result["test"]
    result["gate"] = {
        "acc>=0.97": t["acc"] >= GATE["acc"],
        "ask_rate<=0.30": t["ask_rate"] <= GATE["ask_rate"],
        "mean_q<=0.4": t["mean_q"] <= GATE["mean_q"],
        "noisy>=baseline": noisy_acc >= t["base_acc"],
    }
    result["gate"]["pass"] = all(result["gate"].values())
    return result


def report(r: dict) -> str:
    t = r["test"]

    def ci(key):
        lo, hi = t[key + "_ci"]
        return f"[{lo:.3f}, {hi:.3f}]"

    g = r["gate"]
    rows = [  # (label, key, format, gate text, gate key)
        ("no-ask baseline acc", "base_acc", ".3f", "", None),
        ("ask-back acc (truthful)", "acc", ".3f", ">= 0.97", "acc>=0.97"),
        ("asked on", "ask_rate", ".3f", "<= 0.30", "ask_rate<=0.30"),
        ("mean questions", "mean_q", ".3f", "<= 0.4", "mean_q<=0.4"),
        ("ask-back acc (noisy, 20 seeds)", "noisy_acc", ".3f", ">= baseline", "noisy>=baseline"),
        ("noisy - baseline", "noisy_minus_base", "+.3f", "", None),
    ]
    lines = [f"tau_act = {r['tau_act']} (tuned on dev), k = {r['k']}, eps = {r['eps']}", "",
             f"| test (n={t['n']}) | value | 95% CI | gate | pass |", "|---|---|---|---|---|"]
    for label, key, fmt, gate, gkey in rows:
        ok = "" if gkey is None else str(g[gkey])
        lines.append(f"| {label} | {t[key]:{fmt}} | {ci(key)} | {gate} | {ok} |")
    lines += ["", "| kind | n | baseline | ask-back | noisy | asked |", "|---|---|---|---|---|---|"]
    for k, s in r["by_kind"].items():
        lines.append(f"| {k} | {s['n']} | {s['base_acc']:.2f} | {s['acc']:.2f} | "
                     f"{r['noisy_by_kind'][k]:.2f} | {s['ask_rate']:.2f} |")
    lines.append(f"\nGATE: {'PASS' if g['pass'] else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    from curio_robot.bench import items as bench_items

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--url", default="http://127.0.0.1:8081/v1")
    e = sub.add_parser("report")
    for p in (s, e):
        p.add_argument("--data", type=Path, default=DATA)
        p.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = bench_items.load(args.data)
    if args.cmd == "score":
        path = args.out / "scores.json"
        if path.exists():
            raise SystemExit(f"{path} exists: every item is scored once")
        path.write_text(json.dumps(score_items(items, args.url), indent=1), encoding="utf-8")
        print(f"-> {path}")
        return
    scores = json.loads((args.out / "scores.json").read_text(encoding="utf-8"))["items"]
    result = evaluate(items, scores)
    (args.out / "report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    text = report(result)
    (args.out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
