# Pilot study: working with Curio, with real people

The sim scoreboard (docs/target3-collaboration.md) tests the loop against
simulated people. This protocol runs the same fetch task with real people,
so each sim number has a real counterpart. It is a **pilot**: it checks that
the task, the measures and the setup work, and gives effect-size estimates for
planning a proper study. It is not evidence that Curio helps people in
general.

Prerequisites: the SO-101 arm doing scripted top-down picks, the head camera
(phone on a head strap) calibrated against the ArUco board, whisper-server and
the grounder running, and `curio-sim collab`'s session recording working on
the real runtime.

## 1. Questions
1. With the person's view on, do people finish the fetch task with fewer turns,
   corrections and clarifying questions, and wait less, than with the robot's
   camera only?
2. Does gaze feedback (head direction) cut corrections further without wrong
   re-targets?
3. Does the robot get better across repeated tasks with the same person (the
   learned "left" and the profile carrying over)?
4. How do people rate workload and trust in each condition?

## 2. Design
- **Within-subjects**, three conditions: `view-off`, `view-on`,
  `view-on+gaze`. Each participant does all three.
- **Order counterbalanced** across participants (all 6 orders of the three
  conditions, cycled), so learning the task doesn't favour one condition.
- **Per condition**: 3 fetch tasks of 4 objects each, a fresh scene per task,
  same scene generator for every condition (different seeds per task, same
  seed list across conditions).
- The participant's profile is reset at the start of each condition, so each
  condition starts from the same belief; within a condition it carries over
  (question 3).
- Participants aren't told which condition is running; the operator switches
  it from the laptop. (They may notice the head camera matters; that's fine.)

## 3. Task script
The operator says: "The robot will hand you blocks. Ask for them one at a time
in whatever words feel natural. If it goes for the wrong one, tell it." A
card shows the four target blocks for the task in order (by colour and a
photo of the table from the participant's seat, with the targets circled), so
participants choose their own words: some will say "my left", some "the left",
some "that one".

Scene: two same-coloured pairs plus one single block, as in the sim; in one of
the three tasks per condition a box hides the single block from the robot's
camera.

## 4. Participants and consent
- **Who**: 6 to 12 adults, recruited from people around the project who
  volunteer; no one who worked on Curio's code. Record handedness and whether
  English is their first language (it affects "left"/"right" wording).
- **Consent first**, in writing, before anything is recorded. The form says:
  what is recorded (below), why, who can see it, how long it is kept, that
  they can stop at any time without giving a reason, and that they can ask for
  their data to be deleted afterwards.
- **What is recorded**: the session records (what was said as text, timings,
  which object the robot chose), questionnaire answers, and, only if they tick
  a separate box, the audio and the head-camera and robot-camera video.
- **Storage**: on the project machine only, under a pseudonym (`P01`, `P02`,
  ...). The pseudonym-to-name key is kept separately, on paper or in a
  separate encrypted file, and destroyed when the pilot is written up.
  Nothing is uploaded to cloud services or used to train models without a
  further, specific consent.
- **Deletion**: on request, delete every file under that pseudonym and note
  the deletion (date, pseudonym only).
- **Safety**: the arm runs at reduced speed, the workspace is marked on the
  table, the operator keeps a hand on the stop control, and participants keep
  their hands outside the marked area while the arm moves. Stop and log any
  contact.
- If the project is run through a university or company, check whether its
  ethics review is required before recruiting; this document is the input to
  that review, not a substitute for it.

## 5. Measures
Per task, from the session records (the same code as the sim scoreboard):
success, completion time, turns, corrections, questions, human idle time,
robot idle fraction, concurrency, functional delay and reply time.

Per condition, after its three tasks:
- **NASA-TLX** (raw, six subscales, 0-100), for workload.
- **Trust**: a short trust-in-automation scale (for example the 6-item short
  form of Jian et al.), 7-point.
- One open question: "What was annoying or helpful about the robot just now?"

At the end: which condition they preferred and why.

Operator log, per task: anything unusual (tracking lost, recalibration,
participant confused by the instructions, safety stop).

## 6. Analysis plan (fixed before running)
- Per participant, average each measure over the three tasks of a condition.
- Report each measure per condition as median and range across participants,
  and the paired difference `view-on` − `view-off` and `view-on+gaze` −
  `view-on` with a bootstrap 95% interval. With 6 to 12 people, report these as
  estimates for planning, not as significance tests.
- Learning (question 3): corrections and questions by task number (1, 2, 3)
  within each condition.
- Gaze safety: count gaze re-targets that went to the wrong object; any rate
  above a few percent is a finding to fix before a larger study.
- Read every open answer; group them by theme.
- Compare against the sim scoreboard for the same scene settings: where real
  and sim disagree, that's what to change in the simulated person.

## 7. Claims we will and won't make
- **Will**: "In a pilot with N people, median corrections per task went from
  A to B with the person's view on (paired difference C, 95% interval D to
  E)", with the sample size and the pilot label next to every number.
- **Will**: what went wrong and what we changed because of it.
- **Won't**: that Curio works for people in general, that it is safe for
  unsupervised use, or that it beats other systems. None of that follows from
  a pilot of this size.

## 8. Before the first participant
1. Run the whole protocol on two team members (they don't count as
   participants) and fix anything confusing.
2. Check the recording: one full session should produce session records that
   `curio-sim`'s scoreboard reads without errors.
3. Time it: aim for under 45 minutes per participant including consent and
   questionnaires.
