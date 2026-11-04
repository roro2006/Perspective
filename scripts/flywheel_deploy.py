"""Simulated deployment of the production grounder on front/behind requests. Task T5-24.

    uv run python scripts/flywheel_deploy.py --out out/t5_24/deploy [--corrections 450]

Serves the registry's production adapter on a fresh llama-server (under the GPU lock), or
uses --url. Episodes take seeds from scenes.DEPLOY_SEEDS in order until --corrections
corrections are logged or --max-episodes run. Only corrections and a CONFIRM_RATE sample of
correct answers reach the store; meta.json counts everything."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from curio_robot.flywheel import serve
from curio_robot.flywheel.deploy import CONFIRM_RATE, run_episode
from curio_robot.flywheel.log import EventStore
from curio_robot.flywheel.scenes import DEPLOY_SEEDS, FBWorld, make_item, request_for
from curio_robot.grounding.client import VllmGrounder

REGISTRY = Path(__file__).resolve().parent.parent / "curio_robot" / "flywheel" / "registry.json"


def production() -> tuple[str, Path]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = next(a for a in reg["adapters"] if a["name"] == reg["production"])
    return entry["name"], serve.LORA_DIR / entry["gguf"]


def deploy(url: str, name: str, store: EventStore, args) -> dict:
    grounder = VllmGrounder(url, "qwen-vl", name=name, timeout_s=60)
    rng = np.random.default_rng(args.log_seed)
    counts: Counter = Counter()
    by_cell: Counter = Counter()
    t0 = time.perf_counter()
    with FBWorld() as world:
        warm = make_item(world, DEPLOY_SEEDS[-1], "warm")  # last seed: never reached below
        for _ in range(2):
            grounder.score(request_for(warm))
        n = min(args.max_episodes, len(DEPLOY_SEEDS) - 1)
        for episode, seed in enumerate(DEPLOY_SEEDS[:n]):
            item = make_item(world, seed, f"dep-{seed}")
            out = run_episode(item, episode, grounder.score, name, rng, store, args.confirm_rate)
            counts["episodes"] += 1
            counts["correct"] += out.correct
            counts["no_answer"] += out.pick is None
            by_cell[f"{item.frame}/{item.relation}/{'ok' if out.correct else 'wrong'}"] += 1
            if out.event is not None:
                counts[out.event.outcome] += 1
                counts[f"resolved_by_{out.event.resolved_by}"] += 1
            if episode % 100 == 0:
                print(f"{episode}: {dict(counts)}", flush=True)
            if counts["corrected"] >= args.corrections:
                break
    counts_d = dict(counts)
    return {"grounder": name, "seeds": [DEPLOY_SEEDS[0], DEPLOY_SEEDS[0] + counts["episodes"] - 1],
            "confirm_rate": args.confirm_rate, "accuracy": counts["correct"] / counts["episodes"],
            "counts": counts_d, "by_cell": dict(sorted(by_cell.items())),
            "minutes": (time.perf_counter() - t0) / 60}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="simulated deployment with corrections (T5-24)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corrections", type=int, default=450)
    p.add_argument("--max-episodes", type=int, default=3000)
    p.add_argument("--confirm-rate", type=float, default=CONFIRM_RATE)
    p.add_argument("--log-seed", type=int, default=24, help="rng for sampling and corrections")
    p.add_argument("--url", default=None, help="use this server instead of starting one")
    p.add_argument("--port", type=int, default=8091)
    args = p.parse_args(argv)

    store = EventStore(args.out)
    if store.path.exists():
        raise SystemExit(f"error: {store.path} exists; deployment logs are append-once")
    args.out.mkdir(parents=True, exist_ok=True)
    name, lora = production()
    if args.url:
        meta = deploy(args.url, name, store, args)
    else:
        with serve.gpu_lock("deployment"), serve.llama_server(
                lora, args.port, args.out / "server.log") as url:
            meta = deploy(url, name, store, args)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
