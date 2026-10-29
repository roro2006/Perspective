"""End-to-end target-5 experiment: a grounder places the ring, SmolVLA picks. Task T5-09.

A person asks for a block: "the red one on my left" (kind my), "... on your left" (your),
"... on the left" with a learned habit (habit), or a unique colour (color). Four setups decide
where the ring goes on the robot image, then the fine-tuned SmolVLA picks in closed loop in
PhysWorld:

- words:  no ring, the policy gets only the words;
- base:   the base Qwen3-VL grounder picks the object (geometry condition, the live Curio
          request), ring on it;
- lora:   the same with the target-2 LoRA served;
- oracle: ring on the true target.

Two stages, because the grounder server and the policy do not share the 7900 XT comfortably:

    # project env, one llama-server running (docs/target5-perspective-policy.md, T5-09)
    uv run python scripts/e2e_eval.py ground --name base --url http://127.0.0.1:8081/v1 \\
        --n 200 --out out/e2e/run1
    # lerobot env, server stopped
    <lerobot venv>/python scripts/e2e_eval.py rollout --out out/e2e/run1 \\
        --checkpoint out/smolvla/run1/checkpoints/002500/pretrained_model --device cuda

`ground` writes trials.jsonl (the same for every grounder: scene seed, kind, words, target),
choices_oracle.jsonl, choices_words.jsonl and choices_<name>.jsonl. `--url fake` picks the
first letter (no server) for tests and CPU smoke runs. `rollout` reuses smolvla_eval's
rollout, ring and policy loading; it appends to rollouts.jsonl (a rerun skips finished
rollouts) and writes summary.json: per setup, grounding accuracy and pick outcome
(target / partner / other / nothing), overall and per kind.

Scenes are fetch_scene without the box from seeds 70000+ (demos 1000-1300, smolvla_eval
50000+). Pair kinds keep only scenes where the person's and the robot's side disagree (as the
benchmark does), so a frame-blind grounder is wrong half the time; scenes whose target or
partner would fall outside the policy's robot image are skipped. The grounder sees the
benchmark's 320x240 renders; the ring is drawn at the chosen block's start pixel in the
policy's own image size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
for _p in (REPO, SCRIPTS):  # run as a plain script from either venv
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import smolvla_eval as sve  # needs the paths above; no torch at import

from curio_robot import perspective
from curio_robot.bench.conditions import build
from curio_robot.bench.items import (
    FREE_LEADS,
    FREE_NOUNS,
    FREE_TEMPLATES,
    HABITS,
    MIN_DISTANCE_GAP,
    MIN_VISIBLE_PX,
    PAIR_BLOCKS,
    SINGLE_INDEX,
    BenchItem,
)
from curio_robot.collab.tasks import TaskScene, fetch_scene, true_map
from curio_robot.geometry import intrinsics_from_fovy, project
from curio_robot.grounding.client import GroundingAnswer, VllmGrounder
from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.sim.physics import PhysWorld
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, CameraFrame, Intrinsics, ObjectMap

E2E_SEED = 70000  # demos 1000-1300, smolvla_eval 50000+
KINDS = ("my", "your", "habit", "color", "free")
BENCH_KIND = {"my": "marked", "your": "marked", "habit": "habit", "color": "color",
              "free": "free"}
SETUPS = ("words", "base", "lora", "oracle")  # table order; base/lora come from `ground`
FIXED_SETUPS = ("words", "oracle")  # written by every `ground` run
GROUND_SIZE = (320, 240)  # the benchmark's render size (bench gen: SimWorld(320, 240))
POLICY_SIZE = "256x256"  # demos300 and the fine-tuned checkpoint
MAX_SEED_TRIES = 50  # seeds tried per trial before giving up
OUTCOMES = ("target", "partner", "other", "nothing")
TRIALS = "trials.jsonl"
ROLLOUTS = "rollouts.jsonl"


# ---- grounders ---------------------------------------------------------------------------


class FirstLetterGrounder:
    """Always the first letter: deterministic and server-free, for tests and smoke runs."""

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        letter = min(request.letters)
        return GroundingAnswer({request.letters[letter]: 1.0}, letter, 0.0, "fake")


def make_grounder(url: str, model: str, name: str, timeout: float):
    if url == "fake":
        return FirstLetterGrounder()
    return VllmGrounder(url, model, name=name, timeout_s=timeout)


# ---- trials ------------------------------------------------------------------------------


def _pixel(point: np.ndarray, k: Intrinsics, frame: CameraFrame) -> list[float] | None:
    uv = project(point, k, frame.pose)
    if uv is None or not (0 <= uv[0] < k.width and 0 <= uv[1] < k.height):
        return None
    return [round(float(uv[0]), 2), round(float(uv[1]), 2)]


def _words(kind: str, rng: np.random.Generator, omap: ObjectMap, target: int):
    """(text, frame, p_own_frame, target), or None when the frame would not matter.
    Draws from rng (habit and free only) after the scene and the target index, like bench
    items. "free" uses the benchmark's free-wording templates; for nearest/farthest the
    target is whichever block of the pair the relation picks, so it may change."""
    obj = omap.objects[target]
    if kind == "color":
        return f"Hand me the {obj.color} one", None, None, target
    if kind == "free":
        return _free_words(rng, omap, obj)
    p = HABITS[int(rng.integers(0, 2))] if kind == "habit" else None
    if kind == "habit":
        frame = "speaker" if rng.random() < p else "robot"
    else:
        frame = "speaker" if kind == "my" else "robot"
    cands = omap.by_color(obj.color)
    viewer = omap.human_pose if frame == "speaker" else omap.robot_pose
    side = "left" if perspective.pick(cands, "left", viewer) is obj else "right"
    by_human = perspective.pick(cands, side, omap.human_pose)
    if by_human is perspective.pick(cands, side, omap.robot_pose):
        return None
    whose = {"my": "my ", "your": "your ", "habit": "the "}[kind]
    return f"Hand me the {obj.color} one on {whose}{side}", frame, p, target


def _free_words(rng: np.random.Generator, omap: ObjectMap, obj):
    """Free wording as bench.items draws it: template, lead, noun, then the exact answer."""
    relation, frame, template = FREE_TEMPLATES[int(rng.integers(0, len(FREE_TEMPLATES)))]
    lead = FREE_LEADS[int(rng.integers(0, len(FREE_LEADS)))]
    noun = FREE_NOUNS[int(rng.integers(0, len(FREE_NOUNS)))]
    cands = omap.by_color(obj.color)
    viewer = omap.human_pose if frame == "speaker" else omap.robot_pose
    other = omap.robot_pose if frame == "speaker" else omap.human_pose
    side = ""
    if relation == "side":
        side = "left" if perspective.pick(cands, "left", viewer) is obj else "right"
        if perspective.pick(cands, side, omap.human_pose) is perspective.pick(
            cands, side, omap.robot_pose
        ):
            return None
        chosen = obj
    else:
        chosen = perspective.pick(cands, relation, viewer)
        a, b = (perspective.horizontal_distance(o, viewer) for o in cands)
        if abs(a - b) < MIN_DISTANCE_GAP or perspective.pick(cands, relation, other) is chosen:
            return None
    text = template.format(lead=lead, color=obj.color, noun=noun, side=side)
    return text, frame, None, omap.objects.index(chosen)


def make_trial(world: PhysWorld, seed: int, kind: str, trial_id: str,
               policy_k: Intrinsics) -> tuple[dict, BenchItem] | None:
    """The trial record and the benchmark item for one scene seed; None if the seed is unusable
    (pair not ambiguous, or target or partner outside the policy's robot image)."""
    rng = np.random.default_rng(seed)
    scene = fetch_scene(rng)
    target = SINGLE_INDEX if kind == "color" else int(rng.integers(0, PAIR_BLOCKS))
    world.reset(scene)
    human = world.render_camera(HUMAN_CAM)
    robot = world.render_camera(ROBOT_CAM, segmentation=True)
    omap = ObjectMap(true_map(scene).objects, robot.pose, human.pose)
    words = _words(kind, rng, omap, target)
    if words is None:
        return None
    text, frame, p, target = words
    colour = omap.objects[target].color
    partner = next((i for i, o in enumerate(omap.objects) if o.color == colour and i != target),
                   None)
    centres = [world.block_position(i) for i in range(len(scene.blocks))]
    policy_px = [_pixel(c, policy_k, robot) for c in centres]
    if any(policy_px[i] is None for i in (target, partner) if i is not None):
        return None
    visible = tuple(o.object_id for i, o in enumerate(omap.objects)
                    if int((robot.segmentation == world.block_ids()[i]).sum()) >= MIN_VISIBLE_PX)
    record = {
        "id": trial_id, "seed": seed, "kind": kind, "words": text, "frame": frame,
        "p_own_frame": p, "target": target, "partner": partner,
        "object_ids": [o.object_id for o in omap.objects],
        "pixels_ground": [_pixel(c, robot.intrinsics, robot) for c in centres],
        "pixels_policy": policy_px,
    }
    item = BenchItem(trial_id, BENCH_KIND[kind], text, frame, p, omap.objects[target].object_id,
                     omap, visible, human, _no_seg(robot))
    return record, item


def _no_seg(frame: CameraFrame) -> CameraFrame:
    return CameraFrame(frame.camera, frame.rgb, frame.intrinsics, frame.pose, frame.timestamp,
                       None)


def iter_trials(world: PhysWorld, n: int, seed: int, kinds: tuple[str, ...], policy_k: Intrinsics):
    """n trials, kinds in turn; each takes the next seed that gives a usable scene."""
    s = seed
    for k in range(n):
        kind = kinds[k % len(kinds)]
        for _ in range(MAX_SEED_TRIES):
            made = make_trial(world, s, kind, f"e2e-{k:04d}", policy_k)
            s += 1
            if made is not None:
                yield made
                break
        else:
            raise RuntimeError(f"no usable {kind} scene in {MAX_SEED_TRIES} seeds before {s}")


def policy_intrinsics(world: PhysWorld, size: str) -> Intrinsics:
    w, h = (int(x) for x in size.lower().split("x"))
    return intrinsics_from_fovy(float(world.model.cam_fovy[world.model.camera(ROBOT_CAM).id]),
                                w, h)


# ---- stage 1: ground ---------------------------------------------------------------------


def _choice(record: dict, answer: GroundingAnswer | None, fixed: int | None) -> dict:
    if answer is not None:
        oid = max(answer.probs, key=answer.probs.get) if answer.probs else None
        fixed = None if oid is None else record["object_ids"].index(oid)
    out = {"id": record["id"], "choice": fixed,
           "object_id": None if fixed is None else record["object_ids"][fixed],
           "pixel_ground": None if fixed is None else record["pixels_ground"][fixed],
           "correct": None if answer is None and fixed is None else fixed == record["target"]}
    if answer is not None:
        out.update(probs={k: round(v, 4) for k, v in answer.probs.items()},
                   letter=answer.letter, latency_ms=round(answer.latency_ms, 1),
                   error=answer.error)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ground(out: Path, name: str, grounder, n: int, seed: int = E2E_SEED,
           kinds: tuple[str, ...] = KINDS, policy_size: str = POLICY_SIZE) -> dict:
    """Build the trials, ask the grounder for each, write the choice files; return the
    grounding accuracy per kind. trials.jsonl must match an existing one (same --n, --seed)."""
    if name in FIXED_SETUPS:
        raise ValueError(f"grounder name {name!r} is reserved")
    out.mkdir(parents=True, exist_ok=True)
    records, choices = [], []
    with PhysWorld(*GROUND_SIZE) as world:
        k = policy_intrinsics(world, policy_size)
        for record, item in iter_trials(world, n, seed, kinds, k):
            records.append(record)
            choices.append(_choice(record, grounder.score(build("geometry", item)), None))
    errors = sum(c.get("error") is not None for c in choices)
    if records and errors == len(records):  # server down: write nothing a rollout could read
        raise SystemExit(f"every grounder request failed, e.g. {choices[0]['error']}")
    if errors:
        print(f"warning: {errors} grounder requests failed (scored as wrong)", file=sys.stderr)
    trials = out / TRIALS
    if trials.exists() and read_jsonl(trials) != records:
        raise SystemExit(f"{trials} holds other trials; use the same --n/--seed or a new --out")
    _write_jsonl(trials, records)
    _write_jsonl(out / "choices_oracle.jsonl", [_choice(r, None, r["target"]) for r in records])
    _write_jsonl(out / "choices_words.jsonl", [_choice(r, None, None) for r in records])
    _write_jsonl(out / f"choices_{name}.jsonl", choices)
    return {"name": name, "n": len(records), "errors": errors,
            **_acc(records, choices)}


def _acc(records: list[dict], choices: list[dict]) -> dict:
    hits: dict[str, list[bool]] = {}
    for r, c in zip(records, choices, strict=True):
        hits.setdefault(r["kind"], []).append(bool(c["correct"]))
    every = [h for v in hits.values() for h in v]
    return {"acc": _rate(every), "by_kind": {k: _rate(v) for k, v in hits.items()}}


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


# ---- stage 2: rollout --------------------------------------------------------------------


def load_choices(out: Path, setups: list[str] | None) -> dict[str, dict[str, dict]]:
    """setup -> trial id -> choice row, for the setups given or every choices_*.jsonl."""
    if setups is None:
        found = {p.stem.removeprefix("choices_") for p in out.glob("choices_*.jsonl")}
        setups = [s for s in SETUPS if s in found] + sorted(found - set(SETUPS))
    return {s: {c["id"]: c for c in read_jsonl(out / f"choices_{s}.jsonl")} for s in setups}


def scene_for(record: dict) -> TaskScene:
    """The trial's scene, rebuilt from its seed (fetch_scene draws first)."""
    scene = fetch_scene(np.random.default_rng(record["seed"]))
    ids = [o.object_id for o in true_map(scene).objects]
    if ids != record["object_ids"]:
        raise RuntimeError(f"{record['id']}: seed {record['seed']} gives another scene")
    return scene


def outcome(picked: int | None, record: dict) -> str:
    if picked is None:
        return "nothing"
    if picked == record["target"]:
        return "target"
    return "partner" if picked == record["partner"] else "other"


def rollout_one(world: PhysWorld, model, record: dict, setup: str, choice: dict,
                horizon: float, ctrl: str, frames_dir: Path | None) -> dict:
    """One closed-loop pick with the ring on the chosen block (none for words or no answer).
    smolvla_eval puts the ring on Trial.target under "marker", so the chosen block goes
    there; the outcome is then scored against the true target here."""
    chosen = None if setup == "words" else choice["choice"]
    trial = sve.Trial(record["seed"], scene_for(record),
                      record["target"] if chosen is None else chosen,
                      -1 if record["partner"] is None else record["partner"], record["words"])
    r = sve.rollout(world, model, trial, "words" if chosen is None else "marker", horizon, ctrl,
                    frames_dir)
    return {
        "id": record["id"], "setup": setup, "seed": record["seed"], "kind": record["kind"],
        "words": record["words"], "target": record["target"], "partner": record["partner"],
        "choice": chosen, "grounded": None if setup == "words" else chosen == record["target"],
        "ring": r["ring"], "picked_block": r["picked_block"],
        "outcome": outcome(r["picked_block"], record), "t_lift": r["t_lift"],
        "ring_block_max_lift_m": r["target_max_lift_m"], "frames": r["frames"],
        "queries": r["queries"], "query_s": r["query_s"], "rollout_s": r["rollout_s"],
    }


def _rates(rows: list[dict]) -> dict:
    grounded = [r["grounded"] for r in rows if r["grounded"] is not None]
    return {"n": len(rows), "grounding_acc": _rate(grounded),
            **{o: _rate([r["outcome"] == o for r in rows]) for o in OUTCOMES}}


def summarize(rows: list[dict], setups: list[str]) -> dict:
    """Per setup: n, grounding accuracy (None for words), outcome rates; and per kind."""
    out = {}
    for s in setups:
        mine = [r for r in rows if r["setup"] == s]
        kinds = list(dict.fromkeys(r["kind"] for r in mine))
        out[s] = {**_rates(mine),
                  "by_kind": {k: _rates([r for r in mine if r["kind"] == k]) for k in kinds}}
    return out


def format_table(summary: dict) -> str:
    """Markdown: one row per setup (and per setup x kind)."""
    cols = ("n", "grounding_acc", *OUTCOMES)
    lines = ["| setup | kind | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]

    def cell(v) -> str:
        return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

    for s, d in summary.items():
        for kind, row in [("all", d), *d["by_kind"].items()]:
            lines.append(f"| {s} | {kind} | " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_rollouts(args: argparse.Namespace) -> dict:
    out: Path = args.out
    records = read_jsonl(out / TRIALS)[: args.n]
    choices = load_choices(out, args.setups)
    path = out / ROLLOUTS
    rows = read_jsonl(path) if path.exists() else []
    done = {(r["setup"], r["id"]) for r in rows}
    policy, pre, post, (h, w) = sve.load_policy(args.checkpoint, args.device, args.n_action_steps)
    with PhysWorld(w, h) as world, path.open("a", encoding="utf-8") as log:
        for setup in choices:
            frames = out / "frames" / setup
            frames.mkdir(parents=True, exist_ok=True)
            for k, record in enumerate(records):
                if (setup, record["id"]) in done:
                    continue
                row = rollout_one(world, (policy, pre, post), record, setup,
                                  choices[setup][record["id"]], args.horizon, args.ctrl,
                                  frames if k == 0 else None)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    ids = {r["id"] for r in records}
    rows = [r for r in rows if r["id"] in ids and r["setup"] in choices]
    report = {"checkpoint": str(args.checkpoint), "device": args.device, "n": len(records),
              "horizon_s": args.horizon, "ctrl": args.ctrl,
              "n_action_steps": policy.config.n_action_steps,
              "summary": summarize(rows, list(choices))}
    (out / "summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# ---- CLI ---------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("ground", help="trials and one grounder's choices (project env)")
    g.add_argument("--name", required=True, help="setup name, e.g. base or lora")
    g.add_argument("--url", required=True, help="OpenAI-compatible endpoint, or 'fake'")
    g.add_argument("--model", default="qwen-vl")
    g.add_argument("--timeout", type=float, default=10.0, help="per request, s")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=E2E_SEED)
    g.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    g.add_argument("--policy-size", default=POLICY_SIZE, help="WxH of the policy's images")
    g.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("rollout", help="SmolVLA picks for every setup (lerobot env)")
    r.add_argument("--out", type=Path, required=True, help="the ground stage's --out")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--device", default="cuda")
    r.add_argument("--n", type=int, default=None, help="first n trials (default: all)")
    r.add_argument("--setups", nargs="+", default=None, help="default: every choices file")
    r.add_argument("--horizon", type=float, default=sve.HORIZON_S, help="seconds per rollout")
    r.add_argument("--ctrl", choices=("ramp", "hold"), default="ramp")
    r.add_argument("--n-action-steps", type=int, default=None)
    args = parser.parse_args()
    if args.cmd == "ground":
        grounder = make_grounder(args.url, args.model, args.name, args.timeout)
        print(json.dumps(ground(args.out, args.name, grounder, args.n, args.seed,
                                tuple(args.kinds), args.policy_size), indent=1))
        return
    sve._hide_gpus_for_cpu(args.device)  # before torch is imported
    report = run_rollouts(args)
    print(format_table(report["summary"]))


if __name__ == "__main__":
    main()
