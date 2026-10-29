# Target 4: the reaction critic, phase 1 (simulation)

The brief ("Curio Reaction Critic") argues that untrained people react to a
robot's mistake before anyone takes over: they look at the object they wanted,
they start to say "no", their head moves. A learned critic that reads those
reactions gives a dense, early training signal for any robot policy. Phase 1
builds that critic in simulation and checks whether it is worth going further.

Work breakdown and status: [`tasks/`](../tasks/README.md) (cards T4-xx).

## 1. What phase 1 can and cannot show
Everything here runs against a **simulated reaction model** that we wrote. A
critic trained on it learns that model. So phase 1 shows:
- the plumbing works (episodes, features, training, evaluation, the gate);
- a learned critic beats the hand-written gaze rule from target 3 on the same
  signals, including on a reaction style it never saw in training;
- how much early warning is available *if* people react roughly like this.

It does not show that real people react like this. The reaction timings below
are assumptions, chosen to be plausible and widely spread; phase 3 measures
them on people.

## 2. An episode
One wrong-or-right reach, sampled without MuJoCo so thousands are cheap
(`critic/episodes.py`):
- The scene is a target-3 fetch scene (`collab/tasks.fetch_scene`): two
  same-coloured pairs and a single block, a seated person.
- The person asks for one block of a pair. The arm goes to it (correct) or to
  the other block of the pair (wrong: the frame mix-up target 1 is about).
- The arm moves in a straight line from `HOME` at `runtime.motion.SPEED`
  (0.25 m/s) to hover above its target, then holds. Time 0 is the reach start.
- The person's head and speech follow a **reaction style** (section 3).
- 3 s at 10 Hz: 31 frames.

## 3. Reaction styles (`critic/reactions.py`)
Four families. Each style draws its numbers uniformly from the family's ranges:

| Family | notice (s) | looks at the wanted block | look delay (s) | says "no" after noticing (s) | head noise (deg) | glances / s |
|---|---|---|---|---|---|---|
| attentive | 0.2-0.5 | 80-95% | 0.1-0.3 | 0.4-0.8 | 1-3 | 0.05-0.2 |
| slow | 0.5-1.0 | 60-90% | 0.2-0.5 | 0.8-1.5 | 2-4 | 0.1-0.3 |
| distracted | 0.3-0.8 | 40-70% | 0.1-0.4 | 0.5-1.2 | 3-6 | 0.4-1.0 |
| silent | as attentive | as attentive | as attentive | never speaks | as attentive | as attentive |

Before noticing, people watch the gripper, with random glances at any block.
After noticing a wrong reach they may look at the block they wanted, then say
"no". The head follows the eyes with a 0.15 s lag. Correct reaches get the
same watching and glancing, so glances are the false-alarm source.

## 4. Features and critic
`critic/features.py` turns an episode into a causal per-frame, per-block
feature array (F, N, D): head angle to the block, how long the head has dwelt
on it, whether it is the arm's target, gripper distance and approach, whether
it matches the requested colour, reach progress, "no" heard and since when,
head speed. Only what a running robot would have at that frame.

`critic/model.py` is a small numpy model, no new dependencies: one shared
hidden layer scores each block, a softmax over the blocks matching the request
gives `intended`, and `p_wrong = 1 - intended[arm_target]`. Trained with
cross-entropy on the true wanted block at every frame.

`critic/baseline.py` is the target-3 rule as a critic: a 0.4 s dwell within
10 deg on another matching block, or a heard "no", means wrong.

## 5. Evaluation and the gate (`critic/evaluate.py`)
Per episode, the alarm is the first frame with `p_wrong >= threshold`. Pick the
smallest threshold reaching 90% precision on the validation set, then on the
test set report:

| Metric | Definition | Gate |
|---|---|---|
| precision | wrong reaches flagged / all flagged | >= 0.90 |
| recall | wrong reaches flagged / wrong reaches | >= 0.70 |
| lead_s | median over flagged wrong reaches with speech of (when "no" starts - alarm) | >= 0.4 |
| target_acc | flagged wrong reaches where the top `intended` at the alarm is the wanted block | >= 0.85 |
| auroc | episode score = max `p_wrong`, wrong = positive | reported |

The brief's gate had no recall bar; 70% is added here because a very high
threshold can reach 90% precision while flagging almost nothing.

Split: train on seed 0 with the attentive, slow and distracted families,
choose the threshold on seed 1 (same families), test on seed 2 with all four
families. Silent is held out from training and reported separately: it is the
case where the critic has to work from gaze alone.

## 6. Waves

| Wave | Cards | Agent |
|---|---|---|
| 0 | T4-00 contracts, tests, board | Claude |
| 1 | T4-01 reactions · T4-03 features · T4-04 model · T4-05 baseline · T4-06 evaluate | Qwen |
| 2 | T4-02 episodes (needs 01) | Qwen |
| 3 | T4-07 `curio-sim critic` (needs 02-06) | any |
| 4 | T4-08 phase 1 run and write-up (needs 07) | Claude |

Running the critic inside the live runtime (in place of the gaze rule) is a
follow-up after the gate, not part of phase 1.
