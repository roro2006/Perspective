# Target 1: learning each person's conventions from corrections

The robot learns, live and per person, which frame of reference you mean when
you say "left" without saying whose. It gets better within a session because
of your corrections, and it can only do this because it has your head pose:
without the human view there is only one frame, so nothing to learn.

Work breakdown and status: [`tasks/`](../tasks/README.md).

## 1. Why this is a real problem

The rule resolver assumes "left" means the speaker's left unless they say
"your". People aren't consistent: many instructing a robot take the robot's
perspective, and it varies by person and setup. **[VERIFY: cite an HRI
perspective-taking study before this goes on a slide.]** Explicit "my left" /
"your left" is unambiguous; a bare "on the left" is the case we learn.

## 2. Pieces

```
 text + live map ──► readings ──► belief.posterior ──► decide ──► act / hedge / ask
                     (T1-01)       (T1-02)              (T1-04)
 outcome (accepted / "no, the other one") ──► learn_from_outcome ──► belief'
                                               (T1-02, weighted by T1-03)
 belief per person ◄──► ProfileStore (T1-05)
 every decision ──► DecisionRecorder (T1-07) ──► training data for target 2
 SimUser (T1-06) ──► learning-curve harness (T1-09)
 all of it wired into the runtime orchestrator (T1-08)
```

### Readings (`conventions/readings.py`)
A **reading** is one way to interpret the request: which object it points at
and from whose viewpoint. For "the red one on the left" with two reds and the
human view on, there are two readings: speaker frame → one red, robot frame →
the other. "my"/"to me" or "your"/"to you" makes the frame explicit and leaves
one reading. No relation word means one reading per candidate, frame `None`.

### Belief (`conventions/belief.py`)
One convention for now: P(this person means their own frame), a Beta(a, b)
starting at Beta(1.6, 0.4): prior mean 0.8 (most people mean their own left) but worth only two observations, so it moves fast.
`posterior(readings)` turns readings into P(object): explicit or frame-less
readings weigh 1, speaker-frame readings weigh p, robot-frame readings 1 − p,
summed per object and normalized. Room to add conventions later (e.g. "nearest
to whom") as more Beta fields; keep the dict serialization forward-compatible.

### Learning from outcomes
- **Corrected** ("no, the other one" → the other block): the frame whose
  reading pointed at the wrong object gets evidence against it, weight 1.
- **Accepted** (arrived, no complaint within the window): weak evidence for the
  frame that pointed at the chosen object, weight 0.3.
- **No information**: both frames point at the same object, the frame was
  explicit, or the request had no relation word.
- **Perception discount**: multiply the weight by `max(0, 1 − 2·p_flip)`, where
  `p_flip` (T1-03) is the probability that head-pose calibration noise swapped
  the left/right order of the two objects. A correction on a pair that's nearly
  in line with the speaker's gaze teaches almost nothing.

### Decide (`conventions/decide.py`)
From the posterior: `act` if the top object has p ≥ 0.7; `hedge` (move toward
the probability-weighted centre) if p ≥ 0.5; otherwise `ask` ("Your left or
mine?"). Thresholds are a parameter; tuning them is part of T1-09.

### Records (`recorder.py`)
Every decision is one JSON line: text, stage, readings, posterior, decision,
belief snapshot, versions (`VERSIONS` dict), and later its outcome. Versioning
is what lets a correction be attributed to the model that made the mistake,
and what makes the log replayable as target-2 training data.

## 3. How we'll know it works
`curio-sim learn` (T1-09): simulated users with a fixed frame preference say
marker-free requests; the robot decides, the user accepts or corrects, the
belief updates. Plot error rate against request number per user type
(own-frame users, robot-frame users, 50/50). Success: robot-frame users go
from mostly wrong to rarely wrong within ~3 requests, with "your left or mine?"
filling the gap and fading by ~10; own-frame users never get worse than the
rule baseline; the 50/50 user gets asked instead of guessed at. A reference
implementation already meets these (75% wrong at request 1, ~2% by request 4,
asks down to ~8% by request 10). About 20% of random scenes have both frames
agree, so even the baseline is right sometimes. This is a test of the loop,
not evidence about real people.

## 4. Out of scope for target 1
Image input, the VLM's own probabilities (target 2 feeds `posterior` later),
gaze, multi-person tables.
