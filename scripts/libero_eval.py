"""T5-31: Curio drives LIBERO-Spatial policies trained by others (tasks/T5-31-libero-dropin.md).

Runs in C:/Users/Rohan/dev/libero-venv (scripts/libero_setup.md), offline, step by step. Rows
are appended to out/t5_31/*.jsonl and skipped when already there, so a stopped step continues:

    python scripts/libero_eval.py check                    # libero_words vs the BDDL files
    python scripts/libero_eval.py smoke --policy smolvla   # harness check on init states 5-9
    python scripts/libero_eval.py p --policy smolvla       # precondition P (lerobot-eval)
    python scripts/libero_eval.py goals                    # trials, head renders, resolve()
    python scripts/libero_eval.py rollouts --policy smolvla
    python scripts/libero_eval.py report                   # gate.json and gate.md

P is LeRobot's own `lerobot-eval` on libero_spatial (its default env config, batch size 1,
5 episodes per task, seed 1000), so its init states are 0-4 of every task. The only overrides
are SmolVLA's camera rename map (its cameras are camera1/camera2; LeRobot's LIBERO docs pass the
same --rename_map) and two this machine needs: SmolVLA's load_vlm_weights=false (the
checkpoint holds the backbone; true would fetch SmolVLM2's weights again) and pi0.5's
compile_model=false (torch.compile has no Triton backend on Windows ROCm). The Curio rollouts parse the same
arguments into the same EvalPipelineConfig and call lerobot_eval.rollout on the same envs and
processors; only the task text and the init state change.

Trials: init states from index 10 on, tasks 0-9 in turn at each index, kept when
libero_scenes.choose_request finds a seat and words for which the robot's own reading picks
the other bowl, until 100 are kept. The env is reset exactly as a rollout resets it (same
LiberoEnv, seed 1000 + init index), the person's head camera is rendered there, and
layer.resolve grounds the words on it against the live :8081 grounder (production adapter).
Conditions (the policy's task text):
- A raw: the person's words;
- B curio: libero_words(task, the bowl resolve chose); the person's words if resolve gave no
  answer or chose something that isn't a bowl. B acts on the top choice even when the Goal
  carries an ask-back question (nobody is there to answer); asks are counted;
- O oracle: the scene's own LIBERO-Spatial task string.
Success is LIBERO's own check_success for the scene's task (bowl 1 on the plate). The torch
seed is set to the trial seed before every rollout, so the conditions share their noise.

Gate (per policy that passes P): B >= A + 0.25; B >= 0.85 x O; grounding picks the true bowl
on >= 0.90 of the trials. Wilson 95% CIs for rates, paired bootstrap (10,000, seed 0) for
B - A and B - 0.85 O."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import libero_scenes as scenes

from curio_robot.flywheel.gate import paired_bootstrap, wilson
from curio_robot.layer import libero_words as lw

OUT = Path("C:/Users/Rohan/dev/Perspective/out/t5_31")
SUITE = "libero_spatial"
LIBERO_CONFIG = "C:/Users/Rohan/dev/libero-venv/libero_config"  # scripts/libero_setup.md
P_EPISODES = 5  # per task: init states 0-4
P_MIN = 0.60
CURIO_FIRST_INIT = 10
CURIO_LAST_INIT = 49  # LIBERO ships 50 init states per task
CURIO_N = 100
SMOKE_INITS = range(5, 10)  # used by neither P nor the Curio trials
BASE_SEED = 1000  # lerobot-eval's default seed; trial seed = BASE_SEED + init index
GAIN_MIN = 0.25  # line 1
SHARE_MIN = 0.85  # line 2
GROUND_MIN = 0.90  # line 3
EPS = 1e-9
N_EXAMPLES = 5
CONDITIONS = ("raw", "curio", "oracle")
LABELS = {"raw": "A", "curio": "B", "oracle": "O"}
CHECKPOINTS = {  # repo, revision (pinned at download, scripts/libero_setup.md)
    "smolvla": ("lerobot/smolvla_libero", "a24f1f380461b665158d52acd140152c79e8b36d"),
    "pi05": ("lerobot/pi05_libero_finetuned_v044", "7152f13e34c6c115d808aea0e5d08aca74eac14b"),
}
POLICY_OVERRIDES = {
    "smolvla": ["--policy.load_vlm_weights=false", "--rename_map=" + json.dumps(
        {"observation.images.image": "observation.images.camera1",
         "observation.images.image2": "observation.images.camera2"})],
    "pi05": ["--policy.compile_model=false"],
}


# --- pure parts (tested on CPU) --------------------------------------------------------------


def trial_row(task: int, init: int, positions: dict[str, np.ndarray]) -> dict | None:
    """A Curio trial for one scene and init state, or None when no seat and words work."""
    target, other = lw.TARGET, next(b for b in lw.BOWLS if b != lw.TARGET)
    req = scenes.choose_request(positions[target], positions[other], rotate=task + init)
    if req is None:
        return None
    return {"id": f"t{task}_i{init}", "task": task, "init": init, "seed": BASE_SEED + init,
            "seat": req.seat, "phrase": req.phrase, "words": req.words,
            "person_margin": req.person_margin, "robot_margin": req.robot_margin,
            "positions": {k: [round(float(x), 5) for x in v] for k, v in positions.items()}}


def select_trials(rows: list[tuple[int, int, dict | None]], n: int = CURIO_N
                  ) -> tuple[list[dict], list[str]]:
    """(kept trials, skipped ids) from (task, init, trial_row) in visiting order."""
    kept, skipped = [], []
    for task, init, row in rows:
        if len(kept) == n:
            break
        if row is None:
            skipped.append(f"t{task}_i{init}")
        else:
            kept.append(row)
    return kept, skipped


def visit_order(first: int = CURIO_FIRST_INIT, last: int = CURIO_LAST_INIT):
    """(task, init) pairs: every task at init `first`, then every task at first + 1, ..."""
    for init in range(first, last + 1):
        for task in range(len(lw.TASKS)):
            yield task, init


def curio_bowl(goal_id: str | None, back: dict[str, str]) -> str | None:
    """The LIBERO bowl the Goal points at, or None (no goal, or not a bowl)."""
    if goal_id is None:
        return None
    name = back.get(goal_id)
    return name if name in lw.BOWLS else None


def task_text(condition: str, trial: dict, goal: dict) -> str:
    if condition == "raw":
        return trial["words"]
    if condition == "oracle":
        return lw.TASKS[trial["task"]]
    if condition == "curio":
        bowl = goal.get("bowl")
        return trial["words"] if bowl is None else lw.libero_words(trial["task"], bowl)
    raise ValueError(f"unknown condition {condition!r}")


Runner = Callable[[int, int, int, str], dict]  # (task, init, seed, text) -> {"success", ...}


def run_trials(policy: str, trials: list[dict], goals: dict[str, dict], runner: Runner,
               done: set[tuple[str, str, str]], emit: Callable[[dict], None]) -> None:
    """Every condition on every trial, conditions outermost; rows already done are skipped."""
    for condition in CONDITIONS:
        for t in trials:
            if (policy, condition, t["id"]) in done:
                continue
            text = task_text(condition, t, goals[t["id"]])
            t0 = time.time()
            out = runner(t["task"], t["init"], t["seed"], text)
            emit({"policy": policy, "condition": condition, "id": t["id"], "task": t["task"],
                  "init": t["init"], "text": text, "success": bool(out["success"]),
                  "steps": out.get("steps"), "s": round(time.time() - t0, 1),
                  **({"positions_ok": out["positions_ok"]} if "positions_ok" in out else {})})


def rate(h: list[bool]) -> dict:
    k, n = int(sum(h)), len(h)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else 0.0,
            "ci95": [round(lo, 3), round(hi, 3)]}


def hits(rows: list[dict], policy: str, condition: str, ids: list[str]) -> list[bool]:
    by_id = {r["id"]: r for r in rows if r["policy"] == policy and r["condition"] == condition}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{policy}/{condition}: no rollout for {missing[:5]}")
    return [bool(by_id[i]["success"]) for i in ids]


def grounding(trials: list[dict], goals: dict[str, dict]) -> dict:
    correct = [goals[t["id"]].get("bowl") == lw.TARGET for t in trials]
    return {**rate(correct), "errors": sum(goals[t["id"]].get("goal_id") is None for t in trials),
            "not_a_bowl": sum(goals[t["id"]].get("goal_id") is not None
                              and goals[t["id"]].get("bowl") is None for t in trials),
            "asks": sum(goals[t["id"]].get("ask") is not None for t in trials)}


def policy_gate(rows: list[dict], ids: list[str], ground: dict) -> dict:
    """Lines 1-3 for one policy's rows (line 3 is the policy-independent grounding rate)."""
    h = {c: hits(rows, rows[0]["policy"] if rows else "", c, ids) for c in CONDITIONS}
    r = {c: rate(h[c]) for c in CONDITIONS}
    a, b, o = (float(np.mean(h[c])) if ids else 0.0 for c in CONDITIONS)
    gain_ci = paired_bootstrap(h["raw"], h["curio"])
    share_ci = paired_bootstrap([SHARE_MIN * x for x in h["oracle"]], h["curio"])
    lines = [
        {"line": 1, "rule": "B >= A + 0.25", "value": round(b - a, 3),
         "ci95": [round(x, 3) for x in gain_ci], "pass": bool(b + EPS >= a + GAIN_MIN)},
        {"line": 2, "rule": "B >= 0.85 x O", "value": round(b - SHARE_MIN * o, 3),
         "ci95": [round(x, 3) for x in share_ci], "pass": bool(b + EPS >= SHARE_MIN * o)},
        {"line": 3, "rule": "grounding >= 0.90", "value": ground["rate"],
         "ci95": ground["ci95"], "pass": bool(ground["rate"] + EPS >= GROUND_MIN)},
    ]
    failed = [x["line"] for x in lines if not x["pass"]]
    verdict = "PASS" if not failed else "FAIL (line " + ", ".join(map(str, failed)) + ")"
    return {"rates": {LABELS[c]: r[c] for c in CONDITIONS}, "lines": lines, "verdict": verdict}


def gate(p: dict[str, dict], rows: list[dict], trials: list[dict], goals: dict[str, dict]
         ) -> dict:
    """p: policy -> P summary ({"k", "n", ...}); rows: every Curio rollout row."""
    ids = [t["id"] for t in trials]
    ground = grounding(trials, goals)
    out: dict = {"n": len(ids), "grounding": ground, "policies": {}}
    for policy, ps in p.items():
        entry = {"P": {**rate([True] * ps["k"] + [False] * (ps["n"] - ps["k"])),
                       "pass": bool(ps["n"] and ps["k"] / ps["n"] + EPS >= P_MIN)}}
        mine = [r for r in rows if r["policy"] == policy]
        if not entry["P"]["pass"]:
            entry["verdict"] = "NOT RUN (P failed)"
            entry["diagnosis"] = ps.get("diagnosis", "")
        elif not mine:
            entry["verdict"] = "NOT RUN (no rollouts yet)"
        else:
            entry.update(policy_gate(mine, ids, ground))
        out["policies"][policy] = entry
    return out


def markdown(res: dict) -> str:
    g = res["grounding"]
    lines = ["# T5-31 gate: Curio drives LIBERO-Spatial policies trained by others", "",
             (f"{res['n']} trials (LIBERO-Spatial tasks 0-9, init states from "
              f"{CURIO_FIRST_INIT}; {len(res.get('skipped', []))} skipped, no seat worked); "
              f"grounder {res.get('adapter')}. Seats: {res.get('seats')}. Words: "
              f"{res.get('phrases')}."), "",
             (f"Grounding (line 3, same for both policies): {g['k']}/{g['n']} = {g['rate']:.3f} "
              f"[{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]; {g['errors']} without an answer, "
              f"{g['not_a_bowl']} not a bowl, {g['asks']} with an ask-back question."), ""]
    for policy, e in res["policies"].items():
        pr = e["P"]
        lines += [f"## {policy} -- **{e['verdict']}**", "",
                  (f"P (unmodified LIBERO-Spatial, init states 0-4, lerobot-eval): "
                   f"{pr['k']}/{pr['n']} = {pr['rate']:.3f} [{pr['ci95'][0]:.3f}, "
                   f"{pr['ci95'][1]:.3f}], {'pass' if pr['pass'] else 'FAIL'} (>= {P_MIN})."), ""]
        if "rates" not in e:
            if e.get("diagnosis"):
                lines += [f"Diagnosis: {e['diagnosis']}", ""]
            continue
        lines += ["| condition | success [95% Wilson CI] |", "|---|---|"]
        for c in CONDITIONS:
            r = e["rates"][LABELS[c]]
            lines.append(f"| {LABELS[c]} {c} | {r['k']}/{r['n']} = {r['rate']:.3f} "
                         f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
        lines += ["", "| line | rule | value | 95% CI | result |", "|---|---|---|---|---|"]
        for x in e["lines"]:
            lines.append(f"| {x['line']} | {x['rule']} | {x['value']:+.3f} | "
                         f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] | "
                         f"{'PASS' if x['pass'] else 'FAIL'} |")
        lines += ["", ("Lines 1 and 2: B - A and B - 0.85 O, paired bootstrap CIs. Line 3: "
                       "the grounding rate, Wilson CI."), ""]
    if res.get("examples"):
        lines += ["Head-camera renders the grounder saw: " + ", ".join(res["examples"]), ""]
    return "\n".join(lines)


# --- LIBERO / LeRobot (libero venv only) -----------------------------------------------------


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def snapshot(policy: str) -> str:
    from huggingface_hub import snapshot_download

    repo, rev = CHECKPOINTS[policy]
    return snapshot_download(repo, revision=rev, local_files_only=True)


def eval_args(policy: str, out_dir: Path) -> list[str]:
    """lerobot-eval's arguments for P; the Curio rollouts parse the same ones."""
    return [f"--policy.path={snapshot(policy)}", "--policy.device=cuda", "--env.type=libero",
            f"--env.task={SUITE}", "--eval.batch_size=1", f"--eval.n_episodes={P_EPISODES}",
            f"--seed={BASE_SEED}", f"--output_dir={out_dir}", *POLICY_OVERRIDES[policy]]


def check_tables() -> None:
    """libero_words' tables equal LIBERO's own task strings and BDDL placements."""
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[SUITE]()
    for i, task in enumerate(suite.tasks):
        if task.language != lw.TASKS[i]:
            raise SystemExit(f"task {i}: LIBERO says {task.language!r}")
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        got = lw.placements_from_bddl(bddl.read_text(encoding="utf-8"))
        if got != lw.PLACEMENTS[i]:
            raise SystemExit(f"task {i}: BDDL places the bowls on {got}")
        if "(On akita_black_bowl_1 plate_1)" not in bddl.read_text(encoding="utf-8"):
            raise SystemExit(f"task {i}: the goal is not bowl 1 on the plate")
    print(f"libero_words matches LIBERO's {len(suite.tasks)} {SUITE} tasks")


def env_config():
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    return LiberoEnvConfig(task=SUITE)  # lerobot-eval's defaults, as P runs them


def make_envs(cfg_env):
    from lerobot.envs.factory import make_env

    return make_env(cfg_env, n_envs=1, use_async_envs=False)[SUITE]


def libero_env(vec):
    return vec.envs[0].unwrapped  # lerobot.envs.libero.LiberoEnv


def positions(le) -> dict[str, np.ndarray]:
    data = le._env.sim.data._data
    names = [*lw.BOWLS, *scenes.OTHERS]
    return {n: data.body(f"{n}_main").xpos.copy() for n in names}


def reset_to(vec, init: int, seed: int) -> None:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    libero_env(vec).init_state_id = init
    vec.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})


def cmd_goals(args) -> None:
    """Trials (if not saved yet), then resolve() on each, saving the first head renders."""
    from PIL import Image

    from curio_robot.grounding.marks import place_marks, render_marks
    from curio_robot.layer.resolve import GroundingUnavailable, production_adapter, resolve

    check_tables()
    trials_path, goals_path = args.out / "trials.json", args.out / "goals.jsonl"
    envs = make_envs(env_config())
    try:
        if not trials_path.exists():
            visited = []
            for task, init in visit_order():
                vec = envs[task]
                reset_to(vec, init, BASE_SEED + init)
                visited.append((task, init, trial_row(task, init, positions(libero_env(vec)))))
                if sum(r is not None for *_, r in visited) == CURIO_N:
                    break
            kept, skipped = select_trials(visited)
            trials_path.write_text(json.dumps({"trials": kept, "skipped": skipped}, indent=1),
                                   encoding="utf-8")
        saved = json.loads(trials_path.read_text(encoding="utf-8"))
        done = {r["id"] for r in _read(goals_path)}
        examples = args.out / "examples"
        examples.mkdir(exist_ok=True)
        for k, t in enumerate(saved["trials"]):
            if t["id"] in done:
                continue
            vec = envs[t["task"]]
            reset_to(vec, t["init"], t["seed"])
            le = libero_env(vec)
            pos = positions(le)
            same = all(np.allclose(pos[n], t["positions"][n], atol=1e-4) for n in t["positions"])
            sim = le._env.sim  # hard resets rebuild the model, so one renderer per trial
            cam = scenes.HeadCamera(sim.model._model)
            try:
                head = cam.render(sim.data._data, scenes.head_pose(t["seat"]))
            finally:
                cam.close()
            om, back = scenes.object_map(pos, head.pose)
            t0 = time.perf_counter()
            try:
                goal = resolve(t["words"], head, {}, om, None, base=scenes.robot_base())
                g = {"goal_id": goal.target_id, "confidence": round(goal.confidence, 4),
                     "alternatives": goal.alternatives[:3], "ask": goal.ask}
            except GroundingUnavailable as e:
                g = {"goal_id": None, "error": str(e)}
            ms = round((time.perf_counter() - t0) * 1000, 1)
            row = {"id": t["id"], **g, "bowl": curio_bowl(g["goal_id"], back),
                   "target_curio_id": next(c for c, n in back.items() if n == lw.TARGET),
                   "resolve_ms": ms, "positions_match": bool(same), "adapter":
                   production_adapter()}
            if k < N_EXAMPLES:
                marks = place_marks(om, head)
                image, _ = render_marks(head, marks)
                Image.fromarray(head.rgb).save(examples / f"{t['id']}_head.png")
                Image.fromarray(image).save(examples / f"{t['id']}_marked.png")
            _append(goals_path, row)
    finally:
        for vec in envs.values():
            vec.close()


class LerobotRunner:
    """One policy behind lerobot_eval.rollout, built from lerobot-eval's own config."""

    def __init__(self, policy: str, out_dir: Path) -> None:
        import torch
        from lerobot.configs import parser
        from lerobot.configs.eval import EvalPipelineConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        def _identity(cfg):
            return cfg

        # parser.wrap reads the config class from the annotation, which must be the class
        # itself (this module's annotations are strings).
        _identity.__annotations__ = {"cfg": EvalPipelineConfig}
        _parse = parser.wrap()(_identity)
        argv = sys.argv
        sys.argv = ["lerobot_eval", *eval_args(policy, out_dir)]
        try:
            cfg = _parse()
        finally:
            sys.argv = argv
        self.cfg = cfg
        torch.backends.cudnn.benchmark = True
        self.policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        self.policy.eval()
        overrides = {"device_processor": {"device": str(self.policy.config.device)},
                     "rename_observations_processor": {"rename_map": cfg.rename_map}}
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=overrides)
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg.env,
                                                                   policy_cfg=cfg.policy)
        self.envs = make_envs(cfg.env)
        self.expected: dict[tuple[int, int], dict] = {}

    def __call__(self, task: int, init: int, seed: int, text: str) -> dict:
        import torch
        from lerobot.scripts.lerobot_eval import rollout
        from lerobot.utils.random_utils import set_seed

        vec = self.envs[task]
        le = libero_env(vec)
        le.init_state_id = init
        le.task_description = text
        seen: dict = {}

        def first_frame(env) -> None:
            if not seen:
                seen.update(positions(libero_env(env)))

        set_seed(seed)
        with torch.no_grad():
            data = rollout(vec, self.policy, self.env_pre, self.env_post, self.pre, self.post,
                           seeds=[seed], render_callback=first_frame)
        out = {"success": bool(data["success"][0].any()),
               "steps": int(data["done"][0].float().argmax()) + 1 if data["done"][0].any()
               else int(data["done"].shape[1])}
        want = self.expected.get((task, init))
        if want is not None:
            out["positions_ok"] = all(np.allclose(seen[n], want[n], atol=1e-4) for n in want)
        return out

    def close(self) -> None:
        for vec in self.envs.values():
            vec.close()


def cmd_smoke(args) -> None:
    """One rollout per init state in SMOKE_INITS on task 0 with its own words (neither P's
    nor the Curio trials' init states): does the harness run, how fast, does it succeed."""
    runner = LerobotRunner(args.policy, args.out / f"smoke_{args.policy}")
    try:
        for init in SMOKE_INITS[: args.n]:
            t0 = time.time()
            out = runner(0, init, BASE_SEED + init, lw.TASKS[0])
            _append(args.out / "smoke.jsonl", {"policy": args.policy, "task": 0, "init": init,
                                               **out, "s": round(time.time() - t0, 1)})
    finally:
        runner.close()


def cmd_p(args) -> None:
    """Precondition P: lerobot-eval itself, unmodified LIBERO-Spatial."""
    out_dir = args.out / f"p_{args.policy}"
    if (out_dir / "eval_info.json").exists():
        raise SystemExit(f"P already ran for {args.policy}: {out_dir}")
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_eval", *eval_args(args.policy, out_dir)]
    print(" ".join(cmd), flush=True)
    with (args.out / f"p_{args.policy}.log").open("w", encoding="utf-8") as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"lerobot-eval exited {code}; see p_{args.policy}.log")
    summarize_p(args.out, args.policy)


def summarize_p(out: Path, policy: str) -> dict:
    info = json.loads((out / f"p_{policy}" / "eval_info.json").read_text(encoding="utf-8"))
    per_task = {}
    k = n = 0
    for t in info["per_task"]:
        succ = [bool(s) for s in t["metrics"]["successes"]]
        per_task[t["task_id"]] = sum(succ)
        k, n = k + sum(succ), n + len(succ)
    res = {"policy": policy, "k": k, "n": n, "rate": round(k / n, 3) if n else 0.0,
           "per_task_successes": per_task, "pass": bool(n and k / n + EPS >= P_MIN)}
    (out / f"p_{policy}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return res


def cmd_rollouts(args) -> None:
    p_path = args.out / f"p_{args.policy}.json"
    if not p_path.exists():
        raise SystemExit(f"run P first for {args.policy}")
    if not json.loads(p_path.read_text(encoding="utf-8"))["pass"]:
        raise SystemExit(f"{args.policy} failed P: its Curio lines are not run")
    saved = json.loads((args.out / "trials.json").read_text(encoding="utf-8"))
    goals = {r["id"]: r for r in _read(args.out / "goals.jsonl")}
    missing = [t["id"] for t in saved["trials"] if t["id"] not in goals]
    if missing:
        raise SystemExit(f"run `goals` first: no goal for {missing[:5]}")
    path = args.out / "rollouts.jsonl"
    done = {(r["policy"], r["condition"], r["id"]) for r in _read(path)}
    runner = LerobotRunner(args.policy, args.out / f"curio_{args.policy}")
    runner.expected = {(t["task"], t["init"]): t["positions"] for t in saved["trials"]}
    try:
        run_trials(args.policy, saved["trials"], goals, runner, done,
                   lambda row: _append(path, row))
    finally:
        runner.close()


def cmd_report(args) -> dict:
    from curio_robot.layer.resolve import production_adapter

    saved = json.loads((args.out / "trials.json").read_text(encoding="utf-8"))
    goals = {r["id"]: r for r in _read(args.out / "goals.jsonl")}
    p = {}
    for policy in CHECKPOINTS:
        path = args.out / f"p_{policy}.json"
        if path.exists():
            p[policy] = json.loads(path.read_text(encoding="utf-8"))
            diag = args.out / f"p_{policy}_diagnosis.md"
            if diag.exists():
                p[policy]["diagnosis"] = diag.read_text(encoding="utf-8").strip()
    res = gate(p, _read(args.out / "rollouts.jsonl"), saved["trials"], goals)
    trials = saved["trials"]
    res.update(
        card="T5-31", adapter=production_adapter(), skipped=saved["skipped"],
        trial_ids=[t["id"] for t in trials],
        seats={s: sum(t["seat"] == s for t in trials) for s in scenes.SEATS},
        phrases={ph: sum(t["phrase"] == ph for t in trials) for ph in scenes.PHRASES},
        positions_match_goals=sum(goals[t["id"]]["positions_match"] for t in trials),
        resolve_ms_p50=float(np.median([goals[t["id"]]["resolve_ms"] for t in trials])),
        examples=sorted(str(f) for f in (args.out / "examples").glob("*_marked.png")),
        checkpoints={k: f"{r}@{v[:7]}" for k, (r, v) in CHECKPOINTS.items()},
    )
    (args.out / "gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (args.out / "gate.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("check", "smoke", "p", "goals", "rollouts", "report"))
    parser.add_argument("--policy", choices=tuple(CHECKPOINTS), default="smolvla")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n", type=int, default=1, help="smoke: rollouts (at most 5)")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    for key, value in (("HF_HUB_OFFLINE", "1"), ("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"),
                       ("HF_HUB_DISABLE_PROGRESS_BARS", "1"), ("MIOPEN_FIND_MODE", "FAST"),
                       ("TOKENIZERS_PARALLELISM", "false"), ("LIBERO_CONFIG_PATH", LIBERO_CONFIG)):
        os.environ.setdefault(key, value)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "check":
        check_tables()
    elif args.step == "smoke":
        cmd_smoke(args)
    elif args.step == "p":
        cmd_p(args)
    elif args.step == "goals":
        cmd_goals(args)
    elif args.step == "rollouts":
        cmd_rollouts(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
