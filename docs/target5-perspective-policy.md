# Target 5: perspective grounding that a robot policy can act on

The claim: a robot policy follows "the red one on my left" only when a
perspective-aware grounding layer tells it which block that is. Target 5 builds
the two halves and the benchmark that joins them. Cards: `tasks/T5-xx`.

## 1. The benchmark (phase 1)
Simulated scenes with the person's view and the robot's view rendered together,
exact answers, and six request kinds (`curio_robot/bench/items.py`):

| kind | scene | words | what it tests |
|---|---|---|---|
| color | cluttered | "Hand me the blue one" (unique colour) | baseline grounding |
| marked | same-colour pair | "... on my left" / "... on your left" | the frame named in the words |
| habit | same-colour pair | "... on the left", with the person's habit | the frame from what we learned |
| hidden | box hides the target (a unique colour) from the robot | "Hand me the blue one" | objects only the person sees |
| hidden-pair | box hides one block of a pair from the robot; its partner stays visible | "... on my left" / "... on your left" | hidden objects where the frame still matters |
| free | same-colour pair | templated free wording (below) | wording a keyword rule cannot parse |

"hidden-pair" and "free" were added by T5-10 at the end of the kind list, so
the first four kinds' items are byte-identical to the original benchmark for
the same seed (`out/bench5` = the first 200 items of `out/bench5b`).

**hidden-pair**: the box is placed as for "hidden" (0.045 m from the target
toward the robot camera), but in front of a random pair block; draws are
rejected when the box overlaps a block, the target still shows >= 20 robot
pixels, the partner does not, or both frames pick the same block. Words are
always marked ("my"/"your"), so robot-only sees only the wrong block and
the answer needs both the person view and the right frame.

**free**: `"{lead} the {colour} {noun} {relation}"`, lead from "Hand me",
"Pass me", "Could you get me", "Grab"; noun from "one", "block", "cube",
"thing"; 22 relation templates (`items.FREE_TEMPLATES`):
- side, speaker: "to my left", "on my left-hand side", "the left red one from
  where I sit", "on the left as I see it";
- side, robot: "to your left", "on your left-hand side", "the left red one
  from where you are", "on the left as you see it";
- distance (`perspective.pick` nearest/farthest from the speaker's or the
  robot's camera): "closest to me/you", "nearest me/you", "nearer to me/you",
  "at my/your end of the table", "farthest from me/you", "further away from
  me/you", "on the far side from me/you".

Distance items need the pair >= 0.02 m apart in distance; every free item is
rejected unless the two frames give different blocks. No gaze or "that one".
On seed 0 the no-VLM rule scores 0.30 on free (no answer on the 27 distance
items, 15 of 23 side items right: it reads " my "/" your " and otherwise
assumes the speaker).

Each item stores the target block, whose frame the words meant, the target's
table position and its pixel in the robot camera (what a policy is told).

Rule scores (no VLM) on `out/bench5b` (50 per kind, seed 0): color 1.00,
marked 1.00, habit 0.76, hidden 1.00, hidden-pair 1.00, free 0.30, overall
0.84. The rule reads the exact map (it knows hidden blocks' positions), so
on hidden-pair it is an upper bound, not a competitor to robot-only.

## 2. Three ways to ground (phase 2)
`curio_robot/bench/conditions.py` turns an item into a VLM request:
- **geometry** (Curio today): the person's view, marked, plus each block's
  side computed from the registered head camera ("speaker's left, robot's
  right"). The VLM only has to choose the frame.
- **paired**: both views, marked with the same letters, and no geometry hints.
  The VLM has to work out perspective itself, as general models are asked to.
- **robot-only**: the robot's view only, as a robot without Curio has.

`curio_robot/bench/score.py` scores any grounder per condition and kind, plus
a no-VLM rule that resolves the frame from the words and the habit. Phase 2
retrains the grounding LoRA on benchmark-style data and scores it the same way.

## 3. A policy that can grasp (phase 3)
The sim moves the arm by setting joint angles; it cannot grasp. Phase 3 adds a
physics world with the SO-101's position actuators and free blocks, a scripted
expert that picks a block at a given position, and recorded demonstrations
(both camera images, joint state, action, target pixel, words). Phase 4 (after
your approval of the lerobot install and SmolVLA download) fine-tunes SmolVLA
on those demonstrations, conditioned on the target point.

### Fine-tuning (T5-07)
- **Data:** `uv run curio-sim demos --n 300 --seed 1000 --size 256x256 --out
  out/demos300`: 300 successful picks (301 attempts), 42 frames each at 10 Hz,
  12,600 frames, 958 MB of PNGs, 3.5 min. Words: 245 "on my left/right"
  (person's frame), 55 colour-only. 256x256 matches the base checkpoint's
  camera shape. Episodes 0-269 train (11,340 frames), 270-299 held out.
- **Target marker:** a magenta ring (radius 9 px, 2 px wide; `demos.MARKER_*`)
  on camera1 at the target's *start* pixel for the whole episode, drawn at
  conversion time (`demos.draw_marker`, `marker_uv`), so recordings stay raw.
  "start" mirrors run time, where grounding points once before the pick;
  `--marker track` follows the block instead.
- **Features:** camera1 = robot view + ring, camera2 = person's view, state
  and action 6 joints, task = the words. The base's camera3 is dropped
  (`input_features` lists two cameras, `empty_cameras=0`).
- **Format:** `scripts/smolvla_dataset.py` can write a LeRobotDataset v3
  (images, no video), but lerobot[dataset] (datasets, pyarrow, pandas, av) is
  not installed in the lerobot venv and installing was not approved. Training
  therefore reads the demos folder directly (`CurioPickFrames`, the samples
  LeRobotDataset would give SmolVLA) with lerobot-train's loop written out in
  `scripts/smolvla_train.py` (policy preset AdamW 1e-4, cosine with warmup,
  grad clip 10, VLM frozen: 99.9 M trainable parameters).
- **Smoke (100 steps, batch 32):** 4.2 s/step, data loading ~0 (4 workers),
  8.7 GB peak allocated / 8.9 GB reserved of 20 GB; loss 1.34 -> 0.27,
  held-out loss 0.44 (step 50) -> 0.35 (step 100). A checkpoint is 1.5 GB
  (policy plus optimizer state).
- **Full run:** started 2026-10-29 16:22, 2,500 steps at batch 32 (about 7
  epochs), eval every 250 steps, checkpoint every 500; expected end about
  19:30. Log `out/smolvla/run1/train_log.jsonl` (stdout/stderr beside it),
  checkpoints `out/smolvla/run1/checkpoints/<step>/pretrained_model`.

      set HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 WANDB_MODE=disabled
      <lerobot venv>/python scripts/smolvla_train.py --demos out/demos300
          --out out/smolvla/run1 --steps 2500 --batch-size 32 --workers 4
          --log-freq 25 --eval-freq 250 --save-freq 500   # --resume to continue

### Evaluation (T5-08)
`scripts/smolvla_eval.py` (lerobot venv) rolls a checkpoint out in closed loop
in `PhysWorld`, on scenes it never trained on.
- **Scenes:** `fetch_scene(default_rng(s))` without the box for s = 50000,
  50001, ... (demos used 1000-1300); the target is a random block of a
  same-colour pair, so the words (`demos.words_for`) always say "on my
  left/right". A seed is skipped if either pair block projects outside the
  robot image. Every condition runs on the same scenes, targets and policy
  noise seed.
- **Conditions:** `marker` (ring on the target, words name it), `words` (no
  ring), `swapped` (ring on the pair partner, words name the target: does the
  policy follow the ring or the words?).
- **Timing as recorded:** observe at 10 Hz (camera1 = robot view + ring at the
  ring block's start pixel, camera2 = person's view, 6 joints), call
  `policy.select_action`, and let each action cover 0.1 s (5 ticks of 20 ms,
  ctrl ramped linearly from the previous action as the expert's ramp was;
  `--ctrl hold` holds it). A model call returns a chunk and `n_action_steps`
  of it run open loop, as in lerobot's eval (the checkpoint's 50 by default:
  one call plans most of a pick; `--n-action-steps 10` re-observes every 1 s).
  Horizon 6 s (demo episodes are 4.1 s).
- **What counts as a pick:** a block stays lifted >= 3 cm for 0.5 s
  (`expert.LIFT_HEIGHT`, `HOLD_TIME`), checked every 20 ms tick; the rollout
  stops there. Outcome `target`, `partner`, `other` or `nothing`, plus time to
  lift (start of the held lift) and the target's highest lift.
- **Outputs** in `--out`: `results.json` (settings, per-condition rates, mean
  time to lift, seconds per model call and per rollout, every rollout) and
  `frames/<condition>_<frame>.png` for the first scene of each condition
  (camera1 as the policy saw it | person view | overview, every 10th frame and
  the last).
- **CPU smoke** (100-step smoke checkpoint, 2 scenes, `marker`, 3 s horizon):
  41 s per model call on CPU (8 threads), 41 s per rollout (one call, 30
  frames), 17 s to load; both picked nothing, as expected at 100 steps. The
  process exits cleanly (no DataLoader). CPU is only for checking the loop.
- **Real run (GPU, after training finishes):** from the repo root,

      set HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
      <lerobot venv>/python scripts/smolvla_eval.py
          --checkpoint out/smolvla/run1/checkpoints/002500/pretrained_model
          --n 50 --device cuda --out out/smolvla/eval_run1

### End to end (T5-09)
`scripts/e2e_eval.py` is the headline experiment: a person asks for a block,
a setup decides where the ring goes, and the fine-tuned SmolVLA picks in
closed loop in `PhysWorld`.
- **Requests** (kinds in turn): `my` ("the red one on my left"), `your`
  ("... on your left"), `habit` ("... on the left", with the person's habit
  0.15 or 0.85 shown to the grounder as in the benchmark) and `color` (unique
  colour). Scenes `fetch_scene(default_rng(s))` without the box, s = 70000+.
  Pair kinds keep only scenes where the person's and the robot's side pick
  different blocks (as `bench/items.py` does); scenes whose target or partner
  falls outside the policy's 256x256 robot image are skipped.
- **Setups:** `words` (no ring, words only), `base` (the base Qwen3-VL
  grounder, geometry condition: `bench.conditions.build("geometry", item)` on
  a `BenchItem` built from the PhysWorld scene at the benchmark's 320x240,
  i.e. the live Curio request), `lora` (the same, target-2 LoRA served),
  `oracle` (ring on the true target). The ring sits at the chosen block's
  start pixel in the policy image; the policy also gets the words.
- **Two stages,** because the grounder server and the policy do not share
  the GPU comfortably. `ground` (project env) writes `trials.jsonl` (seed,
  kind, words, frame, habit, target, partner, every block's pixel at both
  sizes), `choices_oracle.jsonl`, `choices_words.jsonl` and
  `choices_<name>.jsonl` (chosen block, its pixel, letter probabilities,
  latency). A second `ground` run must produce the same trials (same `--n`,
  `--seed`) or it stops. `rollout` (lerobot venv) reuses `smolvla_eval`'s
  policy loading, ring and rollout, appends to `rollouts.jsonl` (a rerun
  skips finished rollouts), and writes `summary.json` and a table: per setup
  and per kind, grounding accuracy and the pick outcome rates (target,
  partner, other, nothing). `--url fake` picks the first letter, for tests.
- **CPU smoke** (fake grounder, 8 trials; rollout with the 100-step smoke
  checkpoint, 1 trial x words/base/oracle, 1 s horizon): ground 1.3 s;
  rollout 41-45 s per model call, all `nothing` as expected; rings land on
  the chosen block. A rerun skipped all three rollouts.
- **Real run** after training, from the repo root (PowerShell; `$d` as in
  `docs/bringup-7900xt.md`):

      # 1. base grounder
      & "$d\vulkan\llama-server.exe" -m "$d\models\Qwen3VL-4B-Instruct-Q8_0.gguf" `
        --mmproj "$d\models\mmproj-Qwen3VL-4B-Instruct-F16.gguf" -dev Vulkan0 -ngl 99 -fa on `
        -c 8192 --jinja --parallel 1 --alias qwen-vl --host 127.0.0.1 --port 8081
      uv run python scripts/e2e_eval.py ground --name base `
        --url http://127.0.0.1:8081/v1 --n 200 --out out/e2e/run1
      # 2. stop the server (Ctrl+C), restart it with the LoRA (a fresh server:
      #    the prompt cache must not carry over), ground again
      & "$d\vulkan\llama-server.exe" <same arguments> `
        --lora "$d\models\lora\curio-ground-r16-f16.gguf"
      uv run python scripts/e2e_eval.py ground --name lora `
        --url http://127.0.0.1:8081/v1 --n 200 --out out/e2e/run1
      # 3. stop the server, then roll out every setup on the GPU
      $env:HF_HUB_OFFLINE = "1"
      <lerobot venv>/python scripts/e2e_eval.py rollout --out out/e2e/run1 `
        --checkpoint out/smolvla/run1/checkpoints/002500/pretrained_model --device cuda

  200 trials x 4 setups = 800 rollouts. Add `--n-action-steps 10` (and a
  separate `--out` copy of the ground files) to compare re-observing every
  second, as for T5-08.

## 4. Gates (fixed before results)
- Phase 1: benchmark generation is deterministic per seed; every item's target
  is the block its words and frame pick.
- Phase 2: geometry beats paired and robot-only on marked, habit and hidden
  items; the retrained LoRA keeps colour accuracy >= 0.95 and reaches >= 0.85
  on marked items in the geometry condition.
- Phase 3: the scripted expert picks the target (lifted >= 3 cm, held 0.5 s)
  in >= 90% of 100 random scenes.

## 5. Waves

| Wave | Cards | Agent |
|---|---|---|
| 0 | T5-00 contracts, tests, board | Claude |
| 1 | T5-01 bench items · T5-04 physics world and expert | Qwen · Claude agent |
| 2 | T5-02 conditions (needs 01) · T5-05 demonstrations (needs 04) | Qwen · Claude agent |
| 3 | T5-03 scoring and `curio-sim bench` (needs 02) | any |
| 4 | T5-06 LoRA retrain and phase 2 results (needs 03; GPU, after the Qwen waves) | Claude |
