"""Training data for the phase-2 grounding LoRA, made from benchmark items. Task T5-06.

    uv run python scripts/bench_lora_data.py out/grounding_t5 --seeds 11-19 \\
        [--deictic-from out/grounding/data.jsonl]

The target-2 adapter was trained on single-view examples (relation, deictic,
colour) and is weak on the benchmark's free wording. Here every example is a
benchmark item (bench.items.make_item) rendered as the geometry condition, which
is exactly the live pipeline request, so the adapter trains on what it serves.
Rows use the grounding_data JSONL format, so train_lora.py reads them unchanged.

Seeds: the benchmark test set is seed 0 (out/bench5b) and the end-to-end trials
are seeds 70000+; training seeds must avoid both (default 11-19). Each seed
draws kind-major from one rng, like bench.items.generate, but with per-kind
counts (MIX) weighted towards free wording and hidden pairs. With
--deictic-from, deictic rows ("hand me that one") from the target-2 dataset are
copied in, so the new adapter keeps the gaze skill the benchmark does not test.
train_lora.py holds out 10% of rows by id hash as the validation slice."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from curio_robot.bench.conditions import build
from curio_robot.bench.items import BenchItem, make_item
from curio_robot.sim.grounding_data import load_dataset
from curio_robot.sim.world import SimWorld

# Items per kind per seed. Free wording and hidden pairs are the weak kinds; the
# others stay in at lower weight so the adapter does not forget them.
MIX: dict[str, int] = {
    "free": 112,
    "hidden-pair": 90,
    "marked": 56,
    "habit": 56,
    "color": 34,
    "hidden": 34,
}
FORBIDDEN_SEEDS = frozenset({0})  # the benchmark test set (out/bench5b)
E2E_SEED = 70000  # scripts/e2e_eval.py trials start here
DATA_URL_PREFIX = "data:image/png;base64,"


def parse_seeds(spec: str) -> list[int]:
    """'11-19' or '11,12,15' -> seeds; refuses the benchmark and e2e seeds."""
    seeds: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    bad = [s for s in seeds if s in FORBIDDEN_SEEDS or s >= E2E_SEED]
    if bad:
        raise ValueError(f"seeds {bad} overlap the benchmark (0) or the e2e trials (70000+)")
    return seeds


def item_row(item: BenchItem, row_id: str, images: Path) -> dict | None:
    """The geometry-condition request for one item as a grounding_data row, its
    PNG written to images/<row_id>.png; None if the target has no mark."""
    request = build("geometry", item)
    answer = next((k for k, v in request.letters.items() if v == item.target_id), None)
    if answer is None:
        return None
    messages = json.loads(json.dumps(request.messages))  # deep copy
    image_part = messages[0]["content"][0]
    url = image_part["image_url"]["url"]
    if not url.startswith(DATA_URL_PREFIX):
        raise ValueError(f"{item.id}: image is not a PNG data URL")
    rel = f"images/{row_id}.png"
    (images.parent / rel).write_bytes(base64.b64decode(url[len(DATA_URL_PREFIX):]))
    image_part["image_url"]["url"] = rel
    messages.append({"role": "assistant", "content": answer})
    return {
        "id": row_id,
        "kind": item.kind,
        "text": item.text,
        "answer": answer,
        "target_id": item.target_id,
        "letters": request.letters,
        "prompt_version": request.prompt_version,
        "image": rel,
        "messages": messages,
    }


def bench_rows(seeds: list[int], mix: dict[str, int], images: Path) -> tuple[list[dict], int]:
    """(rows, skipped) for every seed, kind-major per seed from one rng."""
    rows: list[dict] = []
    skipped = 0
    with SimWorld(320, 240) as world:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for kind, n in mix.items():
                for i in range(n):
                    item = make_item(world, rng, kind, f"{kind}-{i:04d}")
                    row = item_row(item, f"s{seed}-{kind}-{i:04d}", images)
                    if row is None:
                        skipped += 1
                    else:
                        rows.append(row)
    return rows, skipped


def deictic_rows(source: Path, n: int, images: Path) -> list[dict]:
    """The first n deictic rows of a grounding_data set, images copied over. Ids
    are kept, so train_lora's by-id split holds out the same rows as before."""
    picked = [r for r in load_dataset(source) if r["kind"] == "deictic"][:n]
    for row in picked:
        shutil.copyfile(source.parent / row["image"], images.parent / row["image"])
    return picked


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="benchmark-style grounding LoRA data (T5-06)")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--seeds", default="11-19", help="e.g. 11-19 or 11,12; never 0 or 70000+")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every MIX count")
    p.add_argument("--deictic-from", type=Path, default=None, help="a grounding_data data.jsonl")
    p.add_argument("--n-deictic", type=int, default=600)
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    mix = {k: max(1, round(v * args.scale)) for k, v in MIX.items()}
    images = args.out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows, skipped = bench_rows(seeds, mix, images)
    if args.deictic_from is not None:
        rows += deictic_rows(args.deictic_from, args.n_deictic, images)
    path = args.out_dir / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {"seeds": seeds, "mix_per_seed": mix, "skipped_no_mark": skipped,
            "deictic_from": str(args.deictic_from), "n": len(rows),
            "kinds": dict(Counter(r["kind"] for r in rows))}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
