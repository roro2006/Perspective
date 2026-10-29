# Target 5: perspective grounding that a robot policy can act on

The claim: a robot policy follows "the red one on my left" only when a
perspective-aware grounding layer tells it which block that is. Target 5 builds
the two halves and the benchmark that joins them. Cards: `tasks/T5-xx`.

## 1. The benchmark (phase 1)
Simulated scenes with the person's view and the robot's view rendered together,
exact answers, and four request kinds (`curio_robot/bench/items.py`):

| kind | scene | words | what it tests |
|---|---|---|---|
| color | cluttered | "Hand me the blue one" (unique colour) | baseline grounding |
| marked | same-colour pair | "... on my left" / "... on your left" | the frame named in the words |
| habit | same-colour pair | "... on the left", with the person's habit | the frame from what we learned |
| hidden | box hides the target from the robot | colour, plus side if needed | objects only the person sees |

Each item stores the target block, whose frame the words meant, the target's
table position and its pixel in the robot camera (what a policy is told).

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
