"""Retraining data and the retrain itself for the flywheel. Tasks T5-24 and T5-26.

    uv run python scripts/bench_lora_data.py out/t5_24/v2data --seeds 11-19 \\
        --deictic-from out/grounding/data.jsonl                 # v2's data, rebuilt
    uv run python scripts/flywheel_data.py build --v2data out/t5_24/v2data \\
        --events out/t5_24/deploy --out out/t5_24/v3data
    uv run python scripts/flywheel_data.py train --data out/t5_24/v3data/data.jsonl \\
        --out out/t5_24/lora-v3 --gguf out/t5_24/curio-ground-r16-v3-f16.gguf
    uv run python scripts/flywheel_data.py build-v4 --v2data out/t5_26/v2data \\
        --events out/t5_24/deploy --fresh out/t5_26/deictic31 --out out/t5_26/v4data

`build` checks the rebuilt v2 data against the v2 adapter's data_sha256 (so v3 = exactly
v2's data + the flywheel rows), and refuses any event from the held-out seed range or with
held-out phrasing. `build-v4` (T5-26) adds deictic replay to the same rows: the target-2
deictic rows that are neither held out nor already in v2's data, then fresh grounding_data
deictic rows (seed 31) until deictic is at least a quarter of all rows. It refuses the
deictic test seed (41) and both front/behind test ranges. `train` runs train_lora.py with
its defaults (the v2 recipe) in the ROCm venv under the GPU lock, offline, then converts the
adapter to GGUF. It never writes into llama/models/lora: only the gate's promotion does."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from curio_robot.bench.items import HELDOUT_KINDS
from curio_robot.flywheel import serve
from curio_robot.flywheel.gate import DEICTIC_TEST_SEED, DEICTIC_TRAIN_SEED, FB2_SEEDS
from curio_robot.flywheel.log import EventStore, to_rows
from curio_robot.flywheel.scenes import DEPLOY_SEEDS, HELDOUT_SEEDS, heldout_phrasing
from curio_robot.sim.grounding_data import load_dataset, write_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = Path("C:/Users/Rohan/dev/Perspective/out")
V2_CONFIG = OUT / "lora" / "qwen3vl4b-r16-v2" / "train_config.json"
TARGET2 = OUT / "grounding" / "data.jsonl"
TARGET2_EVAL_IDS = OUT / "lora" / "qwen3vl4b-r16" / "eval_ids.json"
FB_TEST_SEEDS = (HELDOUT_SEEDS, FB2_SEEDS)  # T5-24's and T5-26's front/behind tests
MIN_DEICTIC_SHARE = 0.25
ROCM_PY = Path("C:/Users/Rohan/dev/rocm-train/.venv/Scripts/python.exe")
BASE_MODEL = Path("C:/Users/Rohan/dev/rocm-train/models/Qwen3-VL-4B-Instruct")
CONVERTER = serve.LLAMA / "src" / "convert_lora_to_gguf.py"
PROTECTED = ("curio-ground-r16-f16.gguf", "curio-ground-r16-v2-f16.gguf")


def check_events(events) -> None:
    """ValueError if any event could leak the held-out test into training."""
    for e in events:
        if any(e.seed in r for r in FB_TEST_SEEDS) or e.seed not in DEPLOY_SEEDS:
            raise ValueError(f"{e.event_id}: seed {e.seed} is not a deployment seed")
        if heldout_phrasing(e.utterance):
            raise ValueError(f"{e.event_id}: held-out phrasing {e.utterance!r}")
        if e.kind in HELDOUT_KINDS:
            raise ValueError(f"{e.event_id}: kind {e.kind} is held out")


def build_data(v2data: Path, events_dir: Path, out: Path, v2_sha: str | None) -> dict:
    v2_path = v2data / "data.jsonl"
    if v2_sha is not None:
        from_disk = _sha256(v2_path)
        if from_disk != v2_sha:
            raise SystemExit(f"error: rebuilt v2 data sha {from_disk} != v2's {v2_sha}")
    v2_rows = load_dataset(v2_path)
    events = EventStore(events_dir).load()
    check_events(events)
    fw_rows = to_rows(events)
    ids = [r["id"] for r in v2_rows + fw_rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("error: duplicate row ids")
    (out / "images").mkdir(parents=True, exist_ok=True)
    for src, rows in ((v2data, v2_rows), (events_dir, fw_rows)):
        for r in rows:
            shutil.copyfile(src / r["image"], out / r["image"])
    rows = v2_rows + fw_rows
    (out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                    encoding="utf-8")
    meta = {"v2_rows": len(v2_rows), "v2_sha256": v2_sha, "flywheel_rows": len(fw_rows),
            "flywheel_outcomes": dict(Counter(e.outcome for e in events)),
            "flywheel_frames": dict(Counter(e.meta.get("frame") for e in events)),
            "n": len(rows), "kinds": dict(Counter(r["kind"] for r in rows))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _train_lora():
    """scripts/train_lora.py as a module (its top level needs no torch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_lora", ROOT / "scripts" / "train_lora.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_lora", mod)  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def fresh_needed(n_rows: int, n_deictic: int, share: float = MIN_DEICTIC_SHARE) -> int:
    """The fewest deictic rows to add so deictic is >= share of all rows."""
    return max(0, math.ceil((share * n_rows - n_deictic) / (1 - share) - 1e-9))


def check_deictic_seed(seed: int) -> None:
    """ValueError for a seed that belongs to a test set (T5-26 seed ranges)."""
    if seed == DEICTIC_TEST_SEED or any(seed in r for r in FB_TEST_SEEDS):
        raise ValueError(f"seed {seed} is reserved for a test set")


def fresh_deictic(out: Path, n: int, seed: int = DEICTIC_TRAIN_SEED) -> list[dict]:
    """n deictic rows from grounding_data at this seed, written once to out and reused
    (generation is deterministic, so a rerun gives the same rows)."""
    check_deictic_seed(seed)
    if n == 0:
        return []
    path = out / "data.jsonl"
    if not path.is_file():
        write_dataset(out, n, seed=seed, kinds=("deictic",))
    rows = load_dataset(path)
    want = [f"{seed}-{i:05d}" for i in range(n)]
    if [r["id"] for r in rows] != want:
        raise SystemExit(f"error: {path} does not hold {n} rows of seed {seed}; remove it")
    return rows


def replay_rows(target2: Path, held_out: set[str], have: set[str]) -> list[dict]:
    """Target-2 deictic rows that are neither held out nor already in the data."""
    return [r for r in load_dataset(target2)
            if r["kind"] == "deictic" and r["id"] not in held_out and r["id"] not in have]


def build_v4(v2data: Path, events_dir: Path, fresh_dir: Path, out: Path, v2_sha: str | None,
             target2: Path = TARGET2, eval_ids: Path = TARGET2_EVAL_IDS,
             seed: int = DEICTIC_TRAIN_SEED, share: float = MIN_DEICTIC_SHARE) -> dict:
    """v2's data + the flywheel rows + deictic replay (target-2, then fresh rows at seed)
    until deictic is >= share of all rows. Writes out/data.jsonl, images and meta.json."""
    check_deictic_seed(seed)
    v2_path = v2data / "data.jsonl"
    if v2_sha is not None and _sha256(v2_path) != v2_sha:
        raise SystemExit(f"error: rebuilt v2 data sha {_sha256(v2_path)} != v2's {v2_sha}")
    v2_rows = load_dataset(v2_path)
    events = EventStore(events_dir).load()
    check_events(events)
    fw_rows = to_rows(events)
    held_out = set(json.loads(eval_ids.read_text(encoding="utf-8")))
    base = v2_rows + fw_rows
    replay = replay_rows(target2, held_out, {r["id"] for r in base})
    so_far = base + replay
    n_fresh = fresh_needed(len(so_far), sum(r["kind"] == "deictic" for r in so_far), share)
    fresh = fresh_deictic(fresh_dir, n_fresh, seed)
    groups = ((v2data, v2_rows), (events_dir, fw_rows), (target2.parent, replay),
              (fresh_dir, fresh))
    rows = [r for _, g in groups for r in g]
    check_v4_rows(rows, held_out, {r["id"] for r in v2_rows}, share)
    (out / "images").mkdir(parents=True, exist_ok=True)
    for src, group in groups:
        for r in group:
            shutil.copyfile(src / r["image"], out / r["image"])
    (out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                    encoding="utf-8")
    n_deictic = sum(r["kind"] == "deictic" for r in rows)
    meta = {"v2_rows": len(v2_rows), "v2_sha256": v2_sha, "flywheel_rows": len(fw_rows),
            "replay_rows": len(replay), "fresh_rows": len(fresh), "fresh_seed": seed,
            "n": len(rows), "deictic": n_deictic, "deictic_share": n_deictic / len(rows),
            "kinds": dict(Counter(r["kind"] for r in rows))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def check_v4_rows(rows: list[dict], held_out: set[str], v2_ids: set[str], share: float) -> None:
    """SystemExit on duplicate ids, a target-2 held-out id that could be trained on, a row
    of the deictic test seed, or a deictic share below share. v2's data carries some
    held-out target-2 ids (as it did for v2); they pass only because train_lora's by-id
    split keeps each of them out of training."""
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("error: duplicate row ids")
    is_eval_id = _train_lora().is_eval_id
    leaked = sorted(i for i in set(ids) & held_out if i not in v2_ids or not is_eval_id(i))
    if leaked:
        raise SystemExit(f"error: target-2 held-out ids in the data: {leaked[:5]}")
    test = [i for i in ids if i.startswith(f"{DEICTIC_TEST_SEED}-")]
    if test:
        raise SystemExit(f"error: rows of the deictic test seed: {test[:5]}")
    got = sum(r["kind"] == "deictic" for r in rows) / len(rows)
    if got < share:
        raise SystemExit(f"error: deictic share {got:.3f} < {share}")


def train(data: Path, out: Path, gguf: Path) -> dict:
    """train_lora.py defaults, then convert. Offline: nothing may be fetched."""
    if gguf.name in PROTECTED or gguf.parent.resolve() == serve.LORA_DIR.resolve():
        raise SystemExit(f"error: refusing to write {gguf}; promotion copies it there")
    env = {**os.environ, "MIOPEN_FIND_MODE": "FAST", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with serve.gpu_lock("training"), (out / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(ROCM_PY), str(ROOT / "scripts" / "train_lora.py"), "--data", str(data),
                        "--model", str(BASE_MODEL), "--out", str(out)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    minutes = (time.perf_counter() - t0) / 60
    subprocess.run([str(ROCM_PY), str(CONVERTER), str(out), "--base", str(BASE_MODEL),
                    "--outtype", "f16", "--outfile", str(gguf)],
                   env=env, cwd=CONVERTER.parent, check=True)
    return {"train_minutes": minutes, "gguf": str(gguf)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="flywheel retraining data and run (T5-24, T5-26)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, text in (("build", "v2's data plus the flywheel rows"),
                       ("build-v4", "v2's data, the flywheel rows and deictic replay")):
        b = sub.add_parser(name, help=text)
        b.add_argument("--v2data", type=Path, required=True)
        b.add_argument("--events", type=Path, required=True)
        b.add_argument("--out", type=Path, required=True)
        b.add_argument("--no-sha-check", action="store_true")
        if name == "build-v4":
            b.add_argument("--fresh", type=Path, required=True,
                           help="where the fresh deictic rows are written (once)")
            b.add_argument("--seed", type=int, default=DEICTIC_TRAIN_SEED)
    t = sub.add_parser("train", help="train_lora.py defaults, then GGUF")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--gguf", type=Path, required=True)
    t.add_argument("--owner", default=serve.OWNER, help="card id stamped into the GPU lock")
    args = p.parse_args(argv)
    if args.cmd.startswith("build"):
        sha = None if args.no_sha_check else json.loads(
            V2_CONFIG.read_text(encoding="utf-8"))["data_sha256"]
        if args.cmd == "build":
            meta = build_data(args.v2data, args.events, args.out, sha)
        else:
            meta = build_v4(args.v2data, args.events, args.fresh, args.out, sha, seed=args.seed)
        print(json.dumps(meta, indent=1))
    else:
        serve.OWNER = args.owner
        print(json.dumps(train(args.data, args.out, args.gguf), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
