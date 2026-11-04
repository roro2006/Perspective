"""Retraining data and the retrain itself for the flywheel. Task T5-24.

    uv run python scripts/bench_lora_data.py out/t5_24/v2data --seeds 11-19 \\
        --deictic-from out/grounding/data.jsonl                 # v2's data, rebuilt
    uv run python scripts/flywheel_data.py build --v2data out/t5_24/v2data \\
        --events out/t5_24/deploy --out out/t5_24/v3data
    uv run python scripts/flywheel_data.py train --data out/t5_24/v3data/data.jsonl \\
        --out out/t5_24/lora-v3 --gguf out/t5_24/curio-ground-r16-v3-f16.gguf

`build` checks the rebuilt v2 data against the v2 adapter's data_sha256 (so v3 = exactly
v2's data + the flywheel rows), and refuses any event from the held-out seed range or with
held-out phrasing. `train` runs train_lora.py with its defaults (the v2 recipe) in the ROCm
venv under the GPU lock, offline, then converts the adapter to GGUF. It never writes into
llama/models/lora: only the gate's promotion does."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from curio_robot.bench.items import HELDOUT_KINDS
from curio_robot.flywheel import serve
from curio_robot.flywheel.log import EventStore, to_rows
from curio_robot.flywheel.scenes import DEPLOY_SEEDS, HELDOUT_SEEDS, heldout_phrasing
from curio_robot.sim.grounding_data import load_dataset

ROOT = Path(__file__).resolve().parent.parent
V2_CONFIG = Path("C:/Users/Rohan/dev/Perspective/out/lora/qwen3vl4b-r16-v2/train_config.json")
ROCM_PY = Path("C:/Users/Rohan/dev/rocm-train/.venv/Scripts/python.exe")
BASE_MODEL = Path("C:/Users/Rohan/dev/rocm-train/models/Qwen3-VL-4B-Instruct")
CONVERTER = serve.LLAMA / "src" / "convert_lora_to_gguf.py"
PROTECTED = ("curio-ground-r16-f16.gguf", "curio-ground-r16-v2-f16.gguf")


def check_events(events) -> None:
    """ValueError if any event could leak the held-out test into training."""
    for e in events:
        if e.seed in HELDOUT_SEEDS or e.seed not in DEPLOY_SEEDS:
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
    p = argparse.ArgumentParser(description="flywheel retraining data and run (T5-24)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="v2's data plus the flywheel rows")
    b.add_argument("--v2data", type=Path, required=True)
    b.add_argument("--events", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--no-sha-check", action="store_true")
    t = sub.add_parser("train", help="train_lora.py defaults, then GGUF")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--gguf", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "build":
        sha = None if args.no_sha_check else json.loads(
            V2_CONFIG.read_text(encoding="utf-8"))["data_sha256"]
        print(json.dumps(build_data(args.v2data, args.events, args.out, sha), indent=1))
    else:
        print(json.dumps(train(args.data, args.out, args.gguf), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
