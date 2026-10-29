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
