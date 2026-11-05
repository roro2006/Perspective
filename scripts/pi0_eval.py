"""T5-28: the gate for Curio driving pi0 (tasks/T5-28-pi0-plugin.md).

Runs in the lerobot venv, offline, in three steps (rows are appended to out/t5_28/*.jsonl
and skipped when already there, so an interrupted step continues where it stopped):

    python scripts/pi0_eval.py goals        # resolve() on every trial, live :8081 grounder
    python scripts/pi0_eval.py rollouts --policy pi0 --checkpoint .../pretrained_model
    python scripts/pi0_eval.py rollouts --policy smolvla --checkpoint .../pretrained_model
    python scripts/pi0_eval.py report       # gate.json and gate.md

T5-30 reran the gate unchanged on fresh seeds: --seed 53000 --out .../out/t5_30 --card T5-30.

Trials: smolvla_eval.make_trials(100, 51000) at 256 x 256 (seeds 51000-51099; the target is
one block of a same-colour pair). The person's words must be in a frame where the robot's own
reading picks the partner. smolvla_eval's "on my left/right" (demos.words_for) is that on 84
of the 100 scenes; on the other 16 the pair lies along the person's line of sight, so both
viewers' "left" is the same block, and the person says "Hand me the <colour> one closest to
me" / "farthest from me" instead (bench FREE_TEMPLATES wording, >= MIN_DISTANCE_GAP apart),
which the robot's own reading (nearest the robot) turns into the partner. A scene where
neither rule works would be dropped and reported (none in 51000-51099). Rule fixed before
any training or evaluation.

Conditions (pi0's task text; it never sees a ring or the person's camera):
- raw (A): the person's words as said;
- curio (B): layer.resolve on the head camera (production adapter, r16-v4 on :8081, the
  geometry request), then robot_words(goal) from the same object map;
- oracle (O): robot-words instruction for the true target.
SmolVLA run 5, for context, on the same trials with the person's words as its task text (as
it was trained): raw = no ring, curio = the ring adapter's pixel for B's goal, oracle = the
ring on the true target (smolvla_eval "marker"). If the grounder gives no answer, B has no
goal and the trial counts as not "target" for both policies.

Rollouts follow smolvla_eval: 10 Hz observations, each action ramped over 5 control ticks,
lerobot's action queue (pi0: 10 of a 30-action chunk per call), 6 s horizon, the same
flow-matching seed (the trial seed) in every condition. "target" = the target block stays
lifted >= 3 cm for 0.5 s. Gate lines (point estimates): B >= A + 0.25; B >= 0.85 x O;
O >= 0.30. 95% Wilson CIs for each rate, paired bootstrap 95% CIs for B - A and B - 0.85 O."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from smolvla_eval import HORIZON_S, Trial, _Lifts, _outcome, make_trials

from curio_robot import perspective
from curio_robot.bench.items import MIN_DISTANCE_GAP
from curio_robot.collab.tasks import true_map
from curio_robot.flywheel.gate import paired_bootstrap, wilson
from curio_robot.layer.adapters import ring_pixel
from curio_robot.layer.resolve import GroundingUnavailable, production_adapter
from curio_robot.layer.resolve import resolve as layer_resolve
from curio_robot.layer.robot_words import instruction
from curio_robot.layer.robot_words import robot_words as words_adapter
from curio_robot.sim import demos
from curio_robot.sim.expert import SUBSTEPS
from curio_robot.sim.physics import PhysWorld
from curio_robot.sim.world import BLOCK_HALF
from curio_robot.types import HUMAN_CAM, ROBOT_CAM, ObjectMap, WorldObject

OUT = Path("C:/Users/Rohan/dev/Perspective/out/t5_28")
GATE_SEED = 51000
GATE_N = 100
IMAGE_SIZE = 256  # the demos' (and both policies') image size; make_trials depends on it
CONDITIONS = ("raw", "curio", "oracle")
LABELS = {"raw": "A", "curio": "B", "oracle": "O"}
GAIN_MIN = 0.25  # line 1
SHARE_MIN = 0.85  # line 2
ORACLE_MIN = 0.30  # line 3
EPS = 1e-9
POLICIES = ("pi0", "smolvla")


@dataclass(frozen=True, slots=True)
class GateTrial:
    trial: Trial  # smolvla_eval's trial (its words field is smolvla_eval's own wording)
    words: str  # what the person says here
    rule: str  # "side" (words_for) or "distance"

    @property
    def seed(self) -> int:
        return self.trial.seed


def person_words(trial: Trial) -> tuple[str, str] | None:
    """(words, rule) in a frame where the robot's own reading picks the partner, or None."""
    om = true_map(trial.scene)
    target, partner = om.objects[trial.target], om.objects[trial.partner]
    pair = [target, partner]
    words, _ = demos.words_for(trial.scene, trial.target)
    side = "left" if words.endswith("left") else "right"
    if perspective.pick(pair, side, om.robot_pose).object_id == partner.object_id:
        return words, "side"
    d_t, d_p = (perspective.horizontal_distance(o, om.human_pose) for o in pair)
    if abs(d_t - d_p) < MIN_DISTANCE_GAP:
        return None
    relation, phrase = ("nearest", "closest to me") if d_t < d_p else ("farthest",
                                                                        "farthest from me")
    if perspective.pick(pair, relation, om.robot_pose).object_id != partner.object_id:
        return None
    return f"Hand me the {target.color} one {phrase}", "distance"


def make_gate_trials(n: int, seed: int, world: PhysWorld) -> tuple[list[GateTrial], list[int]]:
    """(kept trials, dropped seeds) from smolvla_eval's n trials starting at seed."""
    kept, dropped = [], []
    for trial in make_trials(n, seed, world):
        said = person_words(trial)
        if said is None:
            dropped.append(trial.seed)
        else:
            kept.append(GateTrial(trial, *said))
    return kept, dropped


def scene_view(world: PhysWorld, trial: Trial):
    """(object map at the settled block positions, robot frame, head frame) after a reset,
    built as layer_eval builds it for smolvla_eval's scenes."""
    world.reset(trial.scene)
    robot, human = world.render_camera(ROBOT_CAM), world.render_camera(HUMAN_CAM)
    objs = true_map(trial.scene).objects
    om = ObjectMap([WorldObject(o.object_id, o.color, world.block_position(i))
                    for i, o in enumerate(objs)], robot.pose, human.pose)
    return om, robot, human


def goal_row(world: PhysWorld, gt: GateTrial, grounder=None) -> dict:
    """Every condition's input for one trial: texts for pi0, ring pixels for SmolVLA."""
    om, robot, human = scene_view(world, gt.trial)
    target_id = om.objects[gt.trial.target].object_id
    row = {
        "seed": gt.seed, "rule": gt.rule, "person_words": gt.words,
        "target_id": target_id, "partner_id": om.objects[gt.trial.partner].object_id,
        "oracle_words": instruction(target_id, om),
        "oracle_ring": demos.target_pixel(world, gt.trial.target, robot),
    }
    t0 = time.perf_counter()
    try:
        goal = layer_resolve(gt.words, human, {ROBOT_CAM: robot}, om, None, grounder=grounder)
    except GroundingUnavailable as exc:
        row.update(goal_id=None, confidence=None, curio_words=None, curio_ring=None,
                   adapter=None, error=str(exc))
    else:
        ring = ring_pixel(goal, robot)
        row.update(goal_id=goal.target_id, confidence=round(goal.confidence, 4),
                   curio_words=words_adapter(goal, om),
                   curio_ring=None if ring is None else list(ring), adapter=goal.adapter,
                   error=None)
    row["resolve_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return row


def inputs(policy: str, condition: str, g: dict) -> tuple[str | None, list | None]:
    """(task text, ring pixel) for this policy and condition; text None = no goal (B only)."""
    if policy == "pi0":
        text = {"raw": g["person_words"], "curio": g["curio_words"],
                "oracle": g["oracle_words"]}[condition]
        return text, None
    ring = {"raw": None, "curio": g["curio_ring"], "oracle": g["oracle_ring"]}[condition]
    if condition == "curio" and g["goal_id"] is None:
        return None, None
    return g["person_words"], ring


class Policy(Protocol):
    cameras: list[str]  # demos cameras it takes: "robot", "human", "wrist"

    def reset(self, seed: int) -> None: ...

    def act(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """The next joint target (6,) for this observation."""
        ...


def observe(world: PhysWorld, cameras: list[str], ring) -> dict[str, np.ndarray]:
    images = {}
    if "robot" in cameras:
        images["robot"] = demos.draw_marker(world.render_camera(ROBOT_CAM).rgb, ring)
    if "human" in cameras:
        images["human"] = world.render_camera(HUMAN_CAM).rgb
    if demos.WRIST in cameras:
        images[demos.WRIST] = demos.render_wrist(world)
    return images


def rollout(world: PhysWorld, policy: Policy, gt: GateTrial, task: str, ring,
            horizon: float = HORIZON_S) -> dict:
    """One closed-loop episode as smolvla_eval.rollout runs it (ramped ctrl)."""
    trial = gt.trial
    world.reset(trial.scene)
    policy.reset(trial.seed)
    lifts, t0, start = _Lifts(len(trial.scene.blocks)), world.time, time.perf_counter()
    prev = world.ctrl()
    max_target_z = BLOCK_HALF
    f = 0
    for f in range(round(horizon * demos.FPS)):
        action = np.asarray(policy.act(observe(world, policy.cameras, ring), world.arm_qpos(),
                                       task), float)
        for k in range(1, demos.TICKS_PER_FRAME + 1):
            world.step(prev + (action - prev) * (k / demos.TICKS_PER_FRAME), SUBSTEPS)
            lifts.update(world, world.time - t0)
            max_target_z = max(max_target_z, float(world.block_position(trial.target)[2]))
        prev = action
        if lifts.picked is not None:
            break
    return {
        "outcome": _outcome(lifts.picked, trial), "picked_block": lifts.picked,
        "t_lift": None if lifts.t_lift is None else round(lifts.t_lift, 3),
        "target_max_lift_m": round(max_target_z - BLOCK_HALF, 4), "frames": f + 1,
        "rollout_s": round(time.perf_counter() - start, 2),
    }


def run_condition(world: PhysWorld, policy: Policy, name: str, condition: str, gt: GateTrial,
                  g: dict) -> dict:
    task, ring = inputs(name, condition, g)
    row = {"policy": name, "condition": condition, "seed": gt.seed, "task": task,
           "ring": ring}
    if task is None:
        return {**row, "outcome": "no_goal"}
    return {**row, **rollout(world, policy, gt, task, ring)}


# --- the gate, as pure functions over rows -------------------------------------------------


def hits(rows: list[dict], policy: str, condition: str, seeds: list[int]) -> list[bool]:
    by_seed = {r["seed"]: r for r in rows if r["policy"] == policy
               and r["condition"] == condition}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise ValueError(f"{policy}/{condition}: no rollout for seeds {missing[:5]}")
    return [by_seed[s]["outcome"] == "target" for s in seeds]


def rate(h: list[bool]) -> dict:
    k, n = sum(h), len(h)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else 0.0,
            "ci95": [round(lo, 3), round(hi, 3)]}


def outcome_counts(rows: list[dict], policy: str, condition: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if r["policy"] == policy and r["condition"] == condition:
            out[r["outcome"]] = out.get(r["outcome"], 0) + 1
    return out


def gate(rows: list[dict], seeds: list[int]) -> dict:
    """The three lines on pi0's rows, plus SmolVLA's rates when its rows are there."""
    h = {c: hits(rows, "pi0", c, seeds) for c in CONDITIONS}
    r = {c: rate(h[c]) for c in CONDITIONS}
    a, b, o = (np.mean(h[c]) if seeds else 0.0 for c in CONDITIONS)
    gain_ci = paired_bootstrap(h["raw"], h["curio"])
    share_ci = paired_bootstrap([SHARE_MIN * x for x in h["oracle"]], h["curio"])
    lines = [
        {"line": 1, "rule": "B >= A + 0.25", "value": round(float(b - a), 3),
         "ci95": [round(x, 3) for x in gain_ci], "pass": bool(b + EPS >= a + GAIN_MIN)},
        {"line": 2, "rule": "B >= 0.85 x O", "value": round(float(b - SHARE_MIN * o), 3),
         "ci95": [round(x, 3) for x in share_ci], "pass": bool(b + EPS >= SHARE_MIN * o)},
        {"line": 3, "rule": "O >= 0.30", "value": round(float(o), 3),
         "ci95": r["oracle"]["ci95"], "pass": bool(o + EPS >= ORACLE_MIN)},
    ]
    if all(x["pass"] for x in lines):
        verdict = "PASS"
    elif not lines[2]["pass"]:
        verdict = "FAIL (line 3: pi0's competence, not Curio's)"
    else:
        verdict = "FAIL (line " + ", ".join(str(x["line"]) for x in lines if not x["pass"]) + ")"
    out = {"n": len(seeds), "pi0": {LABELS[c]: {**r[c], "outcomes": outcome_counts(
        rows, "pi0", c)} for c in CONDITIONS}, "lines": lines, "verdict": verdict}
    if any(x["policy"] == "smolvla" for x in rows):
        out["smolvla_run5"] = {LABELS[c]: {**rate(hits(rows, "smolvla", c, seeds)),
                                           "outcomes": outcome_counts(rows, "smolvla", c)}
                               for c in CONDITIONS}
    return out


def markdown(res: dict) -> str:
    lines = [f"# {res.get('card', 'T5-28')} gate: Curio drives pi0 -- **{res['verdict']}**", "",
             (f"{res['n']} scenes, seeds {res['seeds'][0]}-{res['seeds'][-1]} "
              f"({res['rules']['side']} with 'on my left/right', {res['rules']['distance']} "
              f"with 'closest to me / farthest from me'); grounder {res['adapter']}; "
              f"pi0 checkpoint {res['checkpoints'].get('pi0')}."), "",
             "| condition | pi0 target rate [95% CI] | SmolVLA run 5 [95% CI] |", "|---|---|---|"]
    smol = res.get("smolvla_run5", {})
    for c in CONDITIONS:
        p = res["pi0"][LABELS[c]]
        s = smol.get(LABELS[c])
        s_txt = f"{s['rate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]" if s else "not run"
        lines.append(f"| {LABELS[c]} {c} | {p['rate']:.3f} [{p['ci95'][0]:.3f}, "
                     f"{p['ci95'][1]:.3f}] | {s_txt} |")
    lines += ["", "| line | rule | value | paired 95% CI | result |", "|---|---|---|---|---|"]
    for x in res["lines"]:
        lines.append(f"| {x['line']} | {x['rule']} | {x['value']:+.3f} | [{x['ci95'][0]:+.3f}, "
                     f"{x['ci95'][1]:+.3f}] | {'PASS' if x['pass'] else 'FAIL'} |")
    lines += ["", "Line 1 and 2 values are B - A and B - 0.85 O; line 3 is O itself (Wilson CI).",
              "", "Outcomes (target / partner / other / nothing / no_goal):", ""]
    for name, block in (("pi0", res["pi0"]), ("SmolVLA run 5", smol)):
        for label, v in block.items():
            lines.append(f"- {name} {label}: {v['outcomes']}")
    g = res["grounding"]
    lines += ["", (f"Curio goals: {g['correct']}/{g['n']} on the true target, {g['errors']} "
                   f"without an answer; resolve p50 {g['resolve_ms_p50']} ms."), ""]
    return "\n".join(lines)


# --- policies (torch only from here on) ----------------------------------------------------


class Pi0Runner:
    """A fine-tuned pi0 checkpoint behind the Policy interface (bf16 weights, no autocast)."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from pi0_data import CAMERA_KEYS, STATE
        from pi0_train import assert_loaded

        self.torch, self.keys, self.state_key = torch, CAMERA_KEYS, STATE
        self.policy = PI0Policy.from_pretrained(str(checkpoint))
        assert_loaded(self.policy, str(checkpoint))  # its load swallows errors
        self.policy.eval()
        self.cameras = [c for c, k in CAMERA_KEYS.items()
                        if k in self.policy.config.input_features]
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})

    def reset(self, seed: int) -> None:
        self.policy.reset()
        self.torch.manual_seed(seed)

    def act(self, images, state, task) -> np.ndarray:
        torch = self.torch
        batch = {self.state_key: torch.tensor(state, dtype=torch.float32)[None], "task": [task],
                 **{self.keys[c]: _chw(torch, images[c])[None] for c in self.cameras}}
        with torch.inference_mode():
            return self.post(self.policy.select_action(self.pre(batch))).squeeze(0).numpy()


class SmolvlaRunner:
    """A SmolVLA checkpoint (smolvla_eval.load_policy) behind the Policy interface."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import smolvla_eval
        import torch

        self.torch, self.keys, self.state_key = torch, smolvla_eval.CAMERA_KEYS, smolvla_eval.STATE
        self.policy, self.pre, self.post, _ = smolvla_eval.load_policy(checkpoint, device, None)
        self.cameras = smolvla_eval.policy_cameras(self.policy)

    reset = Pi0Runner.reset
    act = Pi0Runner.act


def _chw(torch, rgb: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0


# --- CLI -----------------------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def cmd_goals(args, world: PhysWorld, trials: list[GateTrial]) -> None:
    path = args.out / "goals.jsonl"
    done = {r["seed"] for r in _read(path)}
    for gt in trials:
        if gt.seed not in done:
            _append(path, goal_row(world, gt))


def cmd_rollouts(args, world: PhysWorld, trials: list[GateTrial]) -> None:
    goals = {r["seed"]: r for r in _read(args.out / "goals.jsonl")}
    missing = [gt.seed for gt in trials if gt.seed not in goals]
    if missing:
        raise SystemExit(f"run `goals` first: no goal for seeds {missing[:5]}")
    path = args.out / "rollouts.jsonl"
    done = {(r["policy"], r["condition"], r["seed"]) for r in _read(path)}
    runner = (Pi0Runner if args.policy == "pi0" else SmolvlaRunner)(args.checkpoint, args.device)
    _append(args.out / "checkpoints.jsonl", {"policy": args.policy,
                                             "checkpoint": str(args.checkpoint)})
    for condition in CONDITIONS:
        for gt in trials:
            if (args.policy, condition, gt.seed) not in done:
                _append(path, run_condition(world, runner, args.policy, condition, gt,
                                            goals[gt.seed]))


def cmd_report(args, trials: list[GateTrial], dropped: list[int]) -> dict:
    seeds = [gt.seed for gt in trials]
    goals = [g for g in _read(args.out / "goals.jsonl") if g["seed"] in set(seeds)]
    res = gate(_read(args.out / "rollouts.jsonl"), seeds)
    ckpts = {r["policy"]: r["checkpoint"] for r in _read(args.out / "checkpoints.jsonl")}
    ms = sorted(g["resolve_ms"] for g in goals)
    res.update(
        card=args.card,
        seeds=seeds, dropped_seeds=dropped, adapter=production_adapter(), checkpoints=ckpts,
        rules={k: sum(gt.rule == k for gt in trials) for k in ("side", "distance")},
        grounding={"n": len(goals), "correct": sum(g["goal_id"] == g["target_id"] for g in goals),
                   "errors": sum(g["goal_id"] is None for g in goals),
                   "resolve_ms_p50": ms[len(ms) // 2] if ms else None},
    )
    (args.out / "gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (args.out / "gate.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("goals", "rollouts", "report"))
    parser.add_argument("--policy", choices=POLICIES, default="pi0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=GATE_N)
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    parser.add_argument("--card", default="T5-28", help="the card whose gate this is (report title)")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):  # see pi0_train.utf8_stdout
        stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.step == "rollouts" and args.checkpoint is None:
        parser.error("rollouts needs --checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    world = PhysWorld(IMAGE_SIZE, IMAGE_SIZE)
    try:
        trials, dropped = make_gate_trials(args.n, args.seed, world)
        if args.step == "goals":
            cmd_goals(args, world, trials)
        elif args.step == "rollouts":
            cmd_rollouts(args, world, trials)
        else:
            cmd_report(args, trials, dropped)
    finally:
        world.close()


if __name__ == "__main__":
    main()
