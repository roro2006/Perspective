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
