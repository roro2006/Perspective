"""Latency and accuracy of a served grounder on freshly rendered sim examples.

    uv run python scripts/bench_grounding.py --url http://127.0.0.1:8081/v1 --n 50

Renders examples with grounding_data.make_example (kinds in turn), scores each
with VllmGrounder and prints p50/p95 latency and accuracy, overall and per
kind. "emitted" is the letter the model wrote; "argmax" is the top letter by
probability, which is what fusion uses. Task T2-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from curio_robot.grounding.client import VllmGrounder
from curio_robot.sim.grounding_data import KINDS, MAX_ATTEMPTS, make_example
from curio_robot.sim.world import SimWorld


def _example(world: SimWorld, rng: np.random.Generator, kind: str, example_id: str):
    for _ in range(MAX_ATTEMPTS):
        ex = make_example(world, rng, kind, example_id)
        if ex is not None:
            return ex
    raise RuntimeError(f"no {kind} example after {MAX_ATTEMPTS} attempts")


def bench(grounder: VllmGrounder, n: int, seed: int, warmup: int = 2) -> dict:
    """Score n examples; the first `warmup` calls are discarded (model load, caches)."""
    rng = np.random.default_rng(seed)
    world = SimWorld(320, 240)
    for i in range(warmup):
        grounder.score(_example(world, rng, KINDS[i % len(KINDS)], f"warm-{i}").request)
    rows = []
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        ex = _example(world, rng, kind, f"{seed}-{i:05d}")
        ans = grounder.score(ex.request)
        argmax = max(ans.probs, key=ans.probs.get) if ans.probs else None
        rows.append({
            "id": ex.id, "kind": kind, "answer": ex.answer, "target": ex.target_id,
            "emitted": ans.letter, "argmax_ok": argmax == ex.target_id,
            "emitted_ok": ans.letter == ex.answer, "latency_ms": ans.latency_ms,
            "source": ans.source, "error": ans.error,
        })
    return {"rows": rows, "summary": _summary(rows)}


def _summary(rows: list[dict]) -> dict:
    def stats(rs: list[dict]) -> dict:
        lat = [r["latency_ms"] for r in rs]
        return {
            "n": len(rs),
            "emitted_acc": float(np.mean([r["emitted_ok"] for r in rs])),
            "argmax_acc": float(np.mean([r["argmax_ok"] for r in rs])),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "errors": sum(r["source"] == "unavailable" for r in rs),
        }

    out = {"all": stats(rows)}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            out[kind] = stats(rs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="qwen-vl")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=10.0, help="per request, s")
    parser.add_argument("--choice-param", default="guided_choice")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    grounder = VllmGrounder(
        args.url, args.model, timeout_s=args.timeout, choice_param=args.choice_param
    )
    result = bench(grounder, args.n, args.seed)
    print(f"{'kind':<9} {'n':>3} {'emitted':>8} {'argmax':>7} {'p50 ms':>7} {'p95 ms':>7} {'err':>4}")
    for kind, s in result["summary"].items():
        print(f"{kind:<9} {s['n']:>3} {s['emitted_acc']:>8.2f} {s['argmax_acc']:>7.2f} "
              f"{s['p50_ms']:>7.0f} {s['p95_ms']:>7.0f} {s['errors']:>4}")
    first_error = next((r["error"] for r in result["rows"] if r["error"]), None)
    if first_error:
        print("first error:", first_error)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
