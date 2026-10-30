# Simulation results (Phase 0)

Run 2026-10-28, `uv run curio-sim ab --trials 200` (seed 0). Each trial renders
one scene and resolves the request twice: **Curio on** (robot camera + human
camera) and **off** (robot camera only). A trial succeeds when the chosen point
is within 2 cm of the correct block, which is within the SO-101 gripper's
tolerance. Detection is ground truth from segmentation, so these numbers
isolate **viewpoint and perspective**, not detector quality.

| Scenario | What it is | Off | On (perfect head pose) | On (1 cm / 2° head-pose error) | On (2 cm / 4°) |
|---|---|---|---|---|---|
| Perspective | Two same-colour blocks, "the X one on my left/right" | 21.5% | 98.0% | 80.5% | 61.0% |
| Occlusion | Box placed between the robot cam and the target | 0.0% | 100.0% | 40.0% | 13.5% |
| **Cluttered** | Nothing staged: random box, 4–5 blocks, mixed requests | **50.5%** | **97.5%** | **85.5%** | **75.0%** |

## How to use these on stage
- **Headline:** the cluttered row, with realistic calibration error. *"In random
  cluttered scenes, success goes from 50% to 86% with the human's view."*
- The occlusion row's 0% is **by construction** (we put the box there). Say
  "when the target is hidden from the robot," never "robots fail 100% of the
  time."
- These are simulation numbers. Label them "simulation" on the slide and pair
  them with the real-camera Track 2 numbers once measured.

## What we learned: calibration is the product problem
Objects seen *only* by the human camera are located only as well as we know
where that camera is. At half a metre, a 2° head-pose error moves the estimate
by about 1.7 cm, which is most of the grasp tolerance. That's why occlusion
falls from 100% to 40% with modest pose noise.

The fixes, in order (these are also the engineering-depth story for judges):
1. **Close the loop with the wrist camera.** The human view gets the gripper
   *near* the object; the SO-101's wrist camera then sees it and refines the
   grasp. Next thing to build in the sim.
2. **Use the robot as the calibration target.** The arm is visible in the
   human's view, and its joint angles are known exactly, so solve the head-cam
   pose against it every frame, not just against the ArUco board.
3. **Fuse over time.** Average an object's position across frames as the head
   moves; noise drops roughly with the square root of the number of views.

Re-run: `uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2`

## Learning from corrections (target 1)

Simulated users only, not people. Run with `uv run curio-sim learn --user all --baseline` (clean) and `--noise` (1 cm / 2 deg head-pose error). Seed 0, 50 simulated users per type, 15 requests each. Request 1 comes before any correction, so learned and baseline match there by construction.

| Simulated-user type | Clean: error at request 1 | Clean: error at request 15 | Noisy: error at request 1 | Noisy: error at request 15 | Ask rate at request 15, clean / noisy (learned) |
|---|---|---|---|---|---|
| own-frame | 4.0% / 4.0% | 10.0% / 10.0% | 2.0% / 2.0% | 6.0% / 6.0% | 0.0% / 0.0% |
| robot-frame | 78.0% / 78.0% | 2.0% / 82.0% | 80.0% / 80.0% | 2.0% / 82.0% | 6.0% / 4.0% |
| mixed | 42.0% / 42.0% | 4.0% / 44.0% | 38.0% / 38.0% | 0.0% / 34.0% | 80.0% / 82.0% |

Each error cell is learned / baseline. The baseline never asks.

Re-run:

    uv run curio-sim learn --user all --baseline --out out/learn-clean.json
    uv run curio-sim learn --user all --baseline --noise --out out/learn-noise.json

Robot-frame simulated users go from 78% wrong at request 1 to 2% wrong at request 15, where the rule baseline stays 82% wrong. Own-frame simulated users match the baseline exactly, and mixed simulated users are mostly asked (80%) rather than guessed, which is intended.

- Own-frame simulated users say the robot's frame 5% of the time, and the robot has no per-utterance cue for that, so about 5% error remains for them; the baseline shows the same 6-10%.
- Whether real people split between frames this way is open: **[VERIFY: cite an HRI perspective-taking study before this goes on a slide.]**
- These are simulation numbers, not evidence about real people.

## Working together (target 3)

Simulated people only. Each task is 4 requests in one session through the full
live runtime (rendering, perception, partial transcripts, motion, voice), on a
scene with two same-coloured pairs and a single block. 10 tasks per cell
(seeds 100-109, the same scenes in every condition), means per task. The
person looks at each block as they ask for it and takes 0.9 s to say "no" to a
wrong pick, in every condition; the conditions differ only in what the robot
uses. Definitions: `docs/target3-collaboration.md`, section 3.

Run 2026-10-29:

    uv run curio-sim collab --user own-frame --tasks 10 --goals 4 --conditions view-off,view-on,view-on+gaze --seed 100 --out out/collab/own-frame --export out/collab/own-frame/dataset
    (same with --user robot-frame and --user mixed)

**Means "my left" (own-frame):**

| metric | view-off | view-on | view-on+gaze |
|---|---|---|---|
| success | 0.80 | 1.00 | 1.00 |
| turns per task | 7.0 | 4.5 | 4.3 |
| corrections per task | 2.5 | 0.2 | 0.0 |
| questions per task | 0.5 | 0.3 | 0.3 |
| task time, s | 29.2 | 19.7 | 19.2 |
| arm moving while they talk (concurrent_frac) | 0.11 | 0.24 | 0.25 |
| arm starts before they finish, s (functional delay) | -0.92 | -1.40 | -1.45 |

**Means "the robot's left" (robot-frame):**

| metric | view-off | view-on | view-on+gaze |
|---|---|---|---|
| success | 0.90 | 1.00 | 1.00 |
| turns per task | 4.4 | 6.8 | 5.9 |
| corrections per task | 0.4 | 1.0 | 0.1 |
| questions per task | 0.0 | 1.8 | 1.8 |
| task time, s | 21.7 | 23.2 | 21.3 |
| concurrent_frac | 0.21 | 0.06 | 0.12 |

**Either, 50/50 (mixed):**

| metric | view-off | view-on | view-on+gaze |
|---|---|---|---|
| success | 0.90 | 1.00 | 1.00 |
| turns per task | 5.2 | 6.0 | 5.4 |
| corrections per task | 1.1 | 0.8 | 0.2 |
| questions per task | 0.1 | 1.2 | 1.2 |
| task time, s | 23.8 | 22.2 | 20.8 |

Reply time (speech end to the robot starting to speak) was 0.19-0.20 s in
every cell. Full tables, sessions and the exported dataset (90 episodes, 495
steps): `out/collab/<user>/` (not in git).

What this shows:
- **The person's view lets every task finish.** Success is 1.00 with the view
  on for all three kinds of person, 0.8-0.9 without it (goals given up after two
  corrections).
- **For people who mean their own left, it removes most of the work:**
  corrections 2.5 to 0.2 per task, turns 7.0 to 4.5, task time 29 to 20 s, and
  the arm moves while they are still talking about twice as much.
- **Gaze removes most of the remaining spoken corrections:** 0.0, 0.1 and 0.2
  per task with gaze on, against 0.2, 1.0 and 0.8 without. Across the 30 gaze
  tasks it re-targeted 17 times (2 own-frame, 9 robot-frame, 6 mixed), and all
  17 ended on the block the person wanted. It only acts on another reading of
  the current request.
- **The cost: people who mean the robot's left get asked.** The robot-camera
  baseline already reads "left" from the robot's side, which is exactly what
  these people mean, so for them view-off is the cheapest on turns. With the
  view on, the belief starts at "their left", gets corrected, and then asks
  "your left or mine?" (1.8 questions per task) until it is sure.

Limits:
- Simulated people, not real ones; the pilot protocol
  (`docs/pilot-protocol.md`) is how to get real numbers.
- **Each task starts from a fresh profile**, so these numbers show learning
  within 4 requests only. Carrying the profile across tasks (as across days with
  one person) should cut the robot-frame questions, as target 1's curves
  suggest; that run hasn't been done.
- The sim gives the robot the exact head pose; head-tracking error (which hurt
  occlusion badly in Phase 0) isn't applied here, so gaze is at its best case.
- No "that one" requests and no grounder in this run (the GPU was serving the
  coding model). 10 tasks per cell, means only, no intervals.

## Reaction critic (target 4, phase 1)

Simulated reactions only, written by us (`docs/target4-critic.md`, section 3);
this tests the plumbing and the ceiling, not real people. Run 2026-10-29:

    uv run curio-sim critic phase1 --out out/critic

Train 4000 episodes (seed 0; attentive, slow, distracted), pick the threshold
for 90% precision on 1000 (seed 1), test on 2000 (seed 2; all four families,
silent held out). Half the reaches go to the wrong block of a same-coloured
pair. 12 s on the CPU.

| metric | learned critic | gaze rule (target 3) | gate |
|---|---|---|---|
| precision | 0.91 | 0.84 | >= 0.90 |
| recall | 0.96 | 0.97 | >= 0.70 |
| false alarms on right reaches | 0.09 | 0.18 | |
| lead_s (median, before "no" starts) | -0.14 | -0.21 | >= 0.4 |
| names the wanted block | 1.00 | 1.00 | >= 0.85 |
| AUROC | 0.98 | 0.90 | |

**Gate: fail, on lead time.** The critic is a better detector than the rule
(half the false alarms, AUROC 0.98 against 0.90, and it still finds 86% of
wrong reaches for silent people it never trained on), but at 90% precision it
mostly fires on the spoken "no" (the transcript arrives 0.2 s after speech
starts), not before it.

Lead time against precision on the test set:

| threshold | precision | recall | lead_s |
|---|---|---|---|
| 0.60 | 0.74 | 0.97 | 0.31 |
| 0.70 | 0.83 | 0.97 | 0.19 |
| 0.81 | 0.91 | 0.96 | -0.14 |
| 0.90 | 0.98 | 0.94 | -0.23 |

Why:
- **The two candidates are close together from the person.** Seen from the
  person, the two blocks of a pair are a median 8.6 deg apart (10th-90th
  percentile 6.1-11.7), inside one 10 deg gaze cone and the same scale as head
  noise (1-6 deg). A head turning to the wanted block looks a lot like a head
  following the gripper to the other one.
- **Eye-tracker accuracy alone does not fix it.** Rerunning with head noise
  cut to 1.0-1.5 deg (about Aria eye tracking) gives precision 0.92 and lead
  0.04 s: better than with head direction, still short of 0.4 s.
- **Our reaction model leaves little room.** People start looking at the wanted
  block 0.1-0.5 s after noticing and start saying "no" 0.4-1.5 s after; with a
  0.15 s head lag the gaze signal leads speech by roughly 0.2-0.5 s at best, so
  0.4 s at 90% precision was close to the ceiling we built in.

What would change it:
- Real reaction timing (phase 3). If people look earlier, relative to speaking,
  than we assumed, the lead grows; if not, the gate is unreachable for this task.
- Act at a lower precision with a cheap action: at 0.74 precision the critic
  leads by 0.31 s, enough to slow the arm and ask, not to stop it.
- Features finer than a 10 deg cone (angle to each block relative to the
  gripper direction), and scenes where candidates are further apart.

The critic as a **training signal** (phase 2) does not need a lead: it needs
the wanted block named correctly (1.00 here) and few false alarms, both of
which it meets in simulation.

### Finer gaze features and a slow-and-ask operating point (T4-09)

Six features that measure the head against the gripper instead of a 10 deg
cone per block (how far the head has moved from the gripper toward each block,
smoothed and as a rate; a 4 deg dwell; angle rates). Rerun, same split:

| metric | critic (stop, 90% precision) | critic (slow and ask, 75%) | gaze rule | gate |
|---|---|---|---|---|
| precision | 0.92 | 0.77 | 0.84 | >= 0.90 |
| recall | 0.97 | 0.97 | 0.97 | >= 0.70 |
| false alarms on right reaches | 0.08 | 0.28 | 0.18 | |
| lead_s | -0.05 | 0.35 | -0.21 | >= 0.4 |
| names the wanted block | 1.00 | 1.00 | 1.00 | >= 0.85 |

**Gate: still fails on lead time** (-0.14 s before the change, -0.05 s after).
The slow-and-ask point warns 0.35 s before the "no", but slows 28% of right
reaches too.

What limits it: false alarms come mostly from the distracted family (22% of
their right reaches at the stop threshold, against 4% for attentive): a glance
at the other block of the pair looks like a deliberate look until it has lasted
longer than a glance. A per-person threshold (scratch experiment, family used
as a stand-in for knowing the person) gets attentive and slow people to about
0.2 s of lead at 90% precision, but not the distracted ones, and 0.07 s
overall. Under our reaction model, 0.4 s at 90% precision does not look
reachable with any features of the head alone; what decides it is how real
people time looking against speaking (phase 3).

### Phase 2a: the critic as a training signal (T4-10)

Does a policy learn faster from the critic than from spoken corrections? The
policy learns, per person, whose left/right they mean; every feedback source
drives the same learner on the same simulated people, scenes and words (25 per
person type and reaction family, 15 requests each). The gate was committed
before the first run (8497bd2; design doc section 7). Run 2026-10-29, 13 s:

    uv run curio-sim critic phase2

| feedback | wrong reaches | wrong, last 5 requests | labels per reach | labels correct |
|---|---|---|---|---|
| none | 0.41 | 0.42 | 0.00 | - |
| spoken ("no" = other one, silence = yes) | 0.24 | 0.22 | 1.00 | 0.90 |
| critic | 0.20 | 0.17 | 0.90 | 0.99 |
| critic + spoken | 0.20 | 0.17 | 0.90 | 0.99 |
| oracle (true wanted block) | 0.19 | 0.16 | 1.00 | 1.00 |

Wrong reaches over the last 5 requests:

| feedback | own-frame | robot-frame | mixed | attentive | slow | distracted | silent |
|---|---|---|---|---|---|---|---|
| spoken | 0.03 | 0.23 | 0.41 | 0.17 | 0.16 | 0.16 | 0.40 |
| critic + spoken | 0.03 | 0.05 | 0.42 | 0.18 | 0.15 | 0.15 | 0.18 |
| oracle | 0.03 | 0.04 | 0.42 | 0.17 | 0.16 | 0.16 | 0.16 |

**Gate: pass, narrowly.** critic + spoken cuts wrong reaches over the last 5
requests by 25.4% against spoken alone (bar: 25%), no person type is worse,
and its labels are 99% correct. Repeated on seeds 4 and 5: 26.0% and 27.6%,
both pass.

What it shows, and what it does not:
- **The whole gain is people who do not speak up.** Spoken feedback reads
  silence as "yes", so for silent people a wrong reach teaches the wrong frame
  (0.40 wrong at the end, the same as no learning). The critic reads their
  reaction instead and gets them to 0.18, close to the oracle's 0.16. For
  people who do say "no", critic and speech are equal (0.144 against 0.149 on
  seed 4).
- **Its labels are cleaner:** 99% correct against 90% for speech-with-silence.
- **The size of the gain is set by our assumptions.** Silent people are a
  quarter of the simulated population by design, and the critic was trained on
  the same reaction model. How many real people stay quiet when a robot gets it
  wrong, and whether their head shows it, is what phase 3 has to measure.
- Mixed people stay at 0.42 under every source, including the oracle: a policy
  that picks one frame per person cannot serve someone who uses both 50/50.
  Target 1 handles them by asking, which this learner does not do.

## Perspective benchmark (target 5, phase 1 and first phase 2 scores)

200 simulated items, 50 per kind, seed 0 (`docs/target5-perspective-policy.md`).
Each grounder is Qwen3-VL-4B Q8_0 on llama.cpp; "LoRA" is the target-2 adapter
(trained on single-view examples with geometry hints). Run 2026-10-29:

    uv run curio-sim bench gen --n-per-kind 50 --seed 0 --out out/bench5
    uv run curio-sim bench score --data out/bench5 --url http://127.0.0.1:8081/v1

Accuracy (the block the person meant):

| kind | rule (no VLM) | base: geometry | base: paired | base: robot-only | LoRA: geometry | LoRA: paired | LoRA: robot-only |
|---|---|---|---|---|---|---|---|
| color | 1.00 | 1.00 | 1.00 | 0.86 | 1.00 | 1.00 | 0.86 |
| marked ("my/your left") | 1.00 | 0.52 | 0.52 | 0.38 | **0.94** | 0.62 | 0.38 |
| habit ("the left" + habit) | 0.76 | 0.38 | 0.38 | 0.44 | 0.70 | 0.54 | 0.50 |
| hidden from the robot | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| overall | 0.94 | 0.72 | 0.72 | 0.42 | 0.91 | 0.79 | 0.43 |

What it shows:
- **Without the person's view a robot cannot do this.** Robot-only is 0.38-0.50
  on left/right requests and 0.00 on blocks only the person can see.
- **The base VLM cannot resolve perspective, with or without hints.** "my/your
  left" is at chance (0.52) even when every block is labelled with its side in
  both frames; geometry and paired give identical answers, which points to a
  letter-position default rather than reasoning.
- **The target-2 LoRA uses the geometry.** 0.94 on marked requests in the
  geometry condition, against 0.62 when it must infer perspective from two
  images. So the value is in computing perspective from the registered head
  camera, then having a tuned model read it.
- **The rule beats every VLM on these kinds** (0.94 overall), because these
  requests have exact geometric answers. The VLM earns its place on what the
  rule cannot parse ("that one", free wording), which this benchmark does not
  yet include.

Phase 2 gate check (design doc section 4), on the existing LoRA:
- colour >= 0.95 (1.00) and marked >= 0.85 in geometry (0.94): **met**.
- geometry beats paired and robot-only on marked (0.94 > 0.62, 0.38) and habit
  (0.70 > 0.54, 0.50): met. On hidden it ties paired (1.00 = 1.00): **not
  met** as written. The flaw is in the benchmark: hidden targets are a unique
  colour, so any condition with the person view finds them. A fair hidden kind
  needs a same-colour pair with one hidden.

## SmolVLA, first closed-loop picks (target 5, T5-07 and T5-08)

SmolVLA (450 M) fine-tuned on 270 scripted pick demonstrations (target ringed
on the robot camera, person view as a second camera), 2,500 steps at batch 32
on the RX 7900 XT (3 h; held-out loss 0.29 -> 0.17, flat after step 1,250).
Evaluated closed loop in the physics sim on 50 unseen scenes, same targets in
every condition. A pick = a block lifted >= 3 cm and held 0.5 s.

| condition | picked target | picked pair partner | picked other | nothing |
|---|---|---|---|---|
| ring on target, words name it | 0.12 | 0.00 | 0.00 | 0.88 |
| words only, no ring | 0.02 | 0.02 | 0.10 | 0.86 |
| ring on partner, words name target | 0.00 | **0.14** | 0.00 | 0.86 |

Re-planning every 10 steps instead of 50: ring 0.10, words 0.00, swapped
partner 0.06. About 0.28 s per policy call on the GPU.

What it shows:
- **The ring steers the policy, the words do not.** With the ring on the
  partner it picks the partner, never the named target; with no ring its few
  picks are scattered across blocks. That is the behaviour the design needs:
  the grounding layer decides, the policy executes.
- **Grasping is the bottleneck.** Most rollouts end with nothing lifted: the
  arm reaches the ringed block and closes beside or above it (target lifted
  1 cm on average). A 2.5 cm block is about 8 px wide in a 256x256 robot
  camera, which is too coarse for the last centimetres from that view alone.

Likely fixes, in order: add the SO-101 wrist camera as the third view (the
base model has three camera slots, and LeRobot SO-101 setups use one for
exactly this); more and more varied demonstrations (out/demos300v is
recorded); a shorter action chunk to match the 42-frame episodes.

## End to end: grounding then SmolVLA (target 5, T5-09)

200 requests on unseen scenes (40 each: "my left", "your left", "the left" +
habit, colour only, free wording). Each setup puts the ring somewhere, then the
same SmolVLA checkpoint (run 1) picks in closed loop. Run 2026-10-29, 19:30-19:48.

| setup | ring on the right block | picked target | picked partner | nothing |
|---|---|---|---|---|
| words only (no ring) | - | 0.00 | 0.01 | 0.98 |
| base grounder | 0.67 | 0.09 | 0.07 | 0.85 |
| tuned grounder (LoRA) | 0.85 | 0.12 | 0.04 | 0.84 |
| oracle ring | 1.00 | 0.14 | 0.00 | 0.86 |

Ring accuracy by request: base my 0.78, your 0.40, habit 0.45, colour 1.00,
free 0.70; tuned 0.97, 1.00, 0.70, 1.00, 0.57.

What it shows:
- **Without grounding the policy does nothing useful** (0 target picks in 200).
- **Better grounding gives more correct picks and fewer wrong ones:** target
  picks 0.09 -> 0.12 -> 0.14 and partner picks 0.07 -> 0.04 -> 0.00 from base to
  tuned to oracle. The tuned grounder reaches 86% of the oracle's pick rate.
- **The ceiling is the grasp,** not the grounding: even the oracle ring lifts the
  target only 14% of the time (run 1 policy; see the previous section).
- Free wording is the tuned grounder's weak kind (0.57 here, 0.64 on bench5b):
  the LoRA never trained on it.

Grounder scores on the extended benchmark (out/bench5b, 300 items), geometry
condition: base 0.64 overall, tuned 0.87 (rule 0.84); hidden-pair tuned 0.92
vs 0.74 from both images vs 0.00 robot-only; free wording tuned 0.64 vs rule 0.30.

## Grounder retrained on free wording and hidden pairs (T5-06)

The target-2 LoRA never saw free wording, so a second adapter (r16 v2) was trained
from the base Qwen3-VL-4B on benchmark items rendered as the geometry request, the
same request the live pipeline sends (`scripts/bench_lora_data.py`). Data: 4,038
examples, 3,438 from benchmark seeds 11-19 (free 1,008, hidden-pair 810, marked
504, habit 504, colour 306, hidden 306) plus 600 deictic rows from the target-2 set
so "that one" stays in. None share an image with bench5b (seed 0) and none come
from the e2e seeds (70000+). 3,589 train, 449 held out by id hash. Same settings
as the first adapter (r=16 on q/k/v/o, lr 2e-4, 3 epochs, accumulation 16): 675
steps in 56 minutes on the RX 7900 XT. Loss went from 1.47 to 0.13 and held-out
accuracy from 0.80 to 0.87 to 0.89 over the three epochs. Served as
`curio-ground-r16-v2-f16.gguf` on a fresh llama-server. Run 2026-10-29.

The gate was set before training: in the geometry condition on out/bench5b, free
>= 0.80, no other kind more than 0.03 below the old adapter, and p50 latency
within +10%. Paired and robot-only are reported but not gated.

| kind | n | old: geometry | **new: geometry** | old: paired | new: paired | old: robot-only | new: robot-only |
|---|---|---|---|---|---|---|---|
| color | 50 | 1.00 | **1.00** | 1.00 | 1.00 | 0.86 | 0.86 |
| marked | 50 | 0.94 | **1.00** | 0.62 | 0.86 | 0.38 | 0.62 |
| habit | 50 | 0.70 | **0.72** | 0.54 | 0.50 | 0.50 | 0.38 |
| hidden | 50 | 1.00 | **1.00** | 1.00 | 1.00 | 0.00 | 0.00 |
| hidden-pair | 50 | 0.92 | **1.00** | 0.74 | 0.80 | 0.00 | 0.00 |
| free | 50 | 0.64 | **0.90** | 0.52 | 0.58 | 0.52 | 0.30 |
| overall | 300 | 0.87 | **0.94** | 0.74 | 0.79 | 0.38 | 0.36 |

(The rule without a VLM scores 0.84 overall, with free at 0.30.)

**Gate: PASS.** Free reached 0.90 against a bar of 0.80. No kind dropped: colour
and hidden stayed the same, and marked, habit and hidden-pair rose. Latency was
measured on the same 449 held-out requests, each adapter on its own fresh
server: p50 110 ms for both, p95 127 ms for both, so +0%. In the e2e grounding
run the p50 was 115 ms for both adapters.

One regression, not gated: **deictic ("hand me that one") got worse.** On the
302 target-2 held-out examples the served new adapter scores deictic 0.45
against 0.55 for the old one. Relation improved from 0.90 to 0.94, colour stayed
at 1.00, and the overall score went from 0.82 to 0.80. Deictic was a third of
the old training data and is 15% here.

End to end, using the same 200 trials and the same SmolVLA run-1 checkpoint as the
previous section:

| setup | ring on the right block | picked target | picked partner | nothing |
|---|---|---|---|---|
| words only (no ring) | - | 0.00 | 0.01 | 0.98 |
| base grounder | 0.67 | 0.09 | 0.07 | 0.85 |
| tuned grounder (LoRA) | 0.85 | 0.12 | 0.04 | 0.84 |
| **retrained grounder (LoRA v2)** | **0.95** | **0.14** | **0.01** | 0.85 |
| oracle ring | 1.00 | 0.14 | 0.00 | 0.86 |

Ring accuracy by request for v2: my 1.00, your 1.00, habit 0.78, colour 1.00,
free 0.95 (the old adapter scored 0.57 on free).

What it shows:
- **Free wording is fixed on the benchmark's templates.** The score went from
  0.64 to 0.90 on bench5b and from 0.57 to 0.95 end to end. The training data
  uses the same 22 templates as the benchmark, so this is generalization to new
  scenes and combinations, not to new phrasings. Wording no template covers is
  still untested.
- **With v2 the ring is as good as the oracle's for picking.** Target picks match
  the oracle (0.14 each) and partner picks fall to 0.01. What remains is the
  grasp, as in the previous section.
- Habit stays near its ceiling. Its answer follows a frame the simulated person
  draws at random, so the best possible is about 0.85.
- Paired and robot-only also moved (robot-only marked 0.38 -> 0.62, free 0.52
  -> 0.30). The adapter now leans more on words, but those conditions remain
  far below geometry.
- Deictic should be won back before v2 replaces the old adapter in the live
  pipeline (more deictic rows in the mix, then a fresh check).

## SmolVLA run 2, wrist camera (T5-12)

Run 1 followed the ring but rarely closed the grasp. Run 2 changes four things
at once: the SO-101 wrist camera as a third view (camera3, no ring), 400 new
demonstrations with varied wording ("my", "your", "the" side words and free
wording; out/demos400w, seeds 3000+, 360 train / 40 held out), a 30-step action
chunk re-planned every 10 steps (run 1: 50 and 50), and bf16 autocast. That made
training about 5x faster (0.86 s/step), so it ran 10,000 steps at batch 32 (21
epochs) in 2.5 h on the RX 7900 XT. Held-out loss went from 0.22 at step 500 to
0.078, with little change after step 7,500. Run 2026-10-29/30. Same T5-08
evaluation as run 1: 50 unseen scenes, default settings, final checkpoint.

| condition | run 1: target | run 1: partner | **run 2: target** | **run 2: partner** | run 2: other | run 2: nothing |
|---|---|---|---|---|---|---|
| ring on target, words name it | 0.12 | 0.00 | **0.36** | 0.00 | 0.00 | 0.64 |
| words only, no ring | 0.02 | 0.02 | 0.02 | 0.04 | 0.04 | 0.90 |
| ring on partner, words name target | 0.00 | 0.14 | 0.00 | **0.44** | 0.00 | 0.56 |

**Gate (fixed before training): FAIL**, on the first line:
- ring on target: target picks 0.36 against a bar of 0.40. Not met (run 1: 0.12).
- ring on partner: partner picks 0.44 > target picks 0.00. Met.
- words only: target picks 0.02, below 0.36. Met.

Not gated, run after the gated evaluation: the same checkpoint with 30 actions
per call (one open-loop chunk, `--n-action-steps 30`) scores 0.40 / 0.02 / 0.46
partner. The difference from 0.36 is two scenes out of 50 (standard error about
0.07), so the evidence does not favour either setting, and the gate is scored at
the default setting it named.

End to end, on the same 200 trials and grounder choices as T5-09 and T5-06,
with the run-2 policy (out/e2e/run2):

| setup | ring on the right block | run 1: target | **run 2: target** | run 2: partner | run 2: nothing |
|---|---|---|---|---|---|
| words only (no ring) | - | 0.00 | 0.01 | 0.01 | 0.96 |
| retrained grounder (LoRA v2) | 0.95 | 0.14 | **0.42** | 0.01 | 0.57 |
| oracle ring | 1.00 | 0.14 | **0.44** | 0.00 | 0.56 |

Target picks by request, LoRA v2 / oracle: my 0.38 / 0.38, your 0.45 / 0.45,
habit 0.17 / 0.28, colour 0.55 / 0.53, free 0.55 / 0.57.

What it shows:
- **The grasp is about three times as good, but it is still the bottleneck.**
  Target picks with the ring rose from 0.12 to 0.36 on the T5-08 scenes and from
  0.14 to 0.44 end to end (oracle). In most failures the arm reaches the ringed
  block and closes just short of it or beside it: in 19 of the 32 failed
  ring-on-target rollouts the target moved more than 5 mm but was not lifted
  (run 1: 8 of 44).
- **The ring still decides, not the words.** With the ring on the partner the
  policy picks the partner 44% of the time and never the named target. Without a
  ring it almost never picks anything (0.02), even though it was trained on the
  run-time wording.
- **Grounding is no longer what limits the pipeline.** The LoRA v2 ring reaches
  95% of the oracle's target picks (0.42 against 0.44). The only clear gap is
  habit (0.17 against 0.28), where the grounder's ring is right 78% of the time.
- Four things changed together (wrist view, more and more varied data, chunk
  length, 4x the training steps), so this does not measure the wrist camera on
  its own.

## Held-out phrasing (T5-13)

out/bench5c adds 50 "free-heldout" items whose templates, leads and nouns never
appear in grounder training. Geometry condition:

| kind | rule | old LoRA | LoRA v2 |
|---|---|---|---|
| free (training templates) | 0.30 | 0.64 | 0.90 |
| free-heldout (unseen phrasing) | 0.32 | 0.56 | **0.80** |
| overall (350) | 0.77 | 0.82 | 0.92 |

The v2 gain mostly carries over to wording it never saw (0.90 -> 0.80 on new
templates), so it learned the relations, not only the templates.
