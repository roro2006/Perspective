# Target 3: measure the collaboration, not the pick

Targets 1 and 2 are scored one request at a time: did the arm reach the right
object? The idea is bigger than that. A robot that uses the person's view
should make *working together* smoother: fewer repeats and corrections, less
waiting, a robot that moves while you're still talking, and a robot that
notices it's wrong before you say so. Target 3 measures that, adds the first
implicit-feedback signal (where the person looks), and turns sessions into a
dataset.

Work breakdown and status: [`tasks/`](../tasks/README.md) (cards T3-xx).
Everything here runs in the sim against simulated people: it tests the loop
and the measurement, not real users. The pilot protocol (T3-10) is the bridge.

## 1. What a task is
A **task** is several requests in one session with one person: "hand me the
blue one", then "the red one on the left", then "that one". The scene
(`collab/tasks.py`) has two same-coloured pairs and a single block, so some
requests need a frame ("on the left") and some don't; an optional box hides
the single block from the robot. The simulated worker (`collab/worker.py`)
picks the goals and words them the way a `SimUser` would: with their frame
habit, sometimes "my"/"your", sometimes just "that one" while looking at it.
The belief carries over between requests, as it would in a real session.

## 2. What we record
A `Session` (`collab/session.py`) is one task: who, which condition, the goals,
which were done, and one `Step` per thing the person said (request,
correction, answer) with the tracer's marks in absolute seconds. Sessions are
JSON lines, so a run can be scored and re-scored without re-running it. The
decision records (target 1) and episode log stay as they are; the session is
the task-level index over them.

## 3. What we score
`collab/metrics.py`, per session, then averaged per condition. The idle and
concurrency measures follow Hoffman's human-robot fluency metrics (IEEE THMS,
2019), computed from the tracer marks:

| Metric | Definition | Better |
|---|---|---|
| success | goals done and accepted / goals | higher |
| completion_s | session end − start | lower |
| turns | things the person had to say | lower |
| corrections | "no, ..." steps | lower |
| questions | steps where the robot asked "your left or mine?" / "which one?" | lower |
| human_idle_s | Σ (arrived − speech_end): the person waiting after speaking | lower |
| robot_idle_frac | 1 − (time the arm is moving / completion) | lower |
| concurrent_frac | time the person is speaking *and* the arm is moving / completion | higher |
| functional_delay_s | median (motion_start − speech_end); negative = moved early | lower |
| reply_s | median (tts_start − speech_end) | lower |

## 4. Conditions
- `view-off`: robot camera only (Curio off).
- `view-on`: robot + person's camera, the target-1 belief, target-2 grounding
  when a grounder is configured.
- `view-on+gaze`: as `view-on`, plus implicit feedback (section 5).

The claim to test: from `view-off` to `view-on` to `view-on+gaze`, success and
concurrency go up and turns, corrections, questions and waiting go down, and
they keep improving across repeated tasks with the same person.

## 5. Implicit feedback: the person looks at what they meant
When the arm heads for the wrong object, the simulated person turns their head
to the one they meant (a real person does this before they find the words).
`collab/gaze.py` finds the object nearest the head camera's line of sight, and
`GazeWatch` fires when the gaze rests on something other than the arm's goal
for `DWELL_S`. The runtime then treats it like a soft "no": re-target to the
gazed object if it's a reading of the current request, record it as a
correction with source "gaze", and learn from it. This is head direction, not
eye tracking; real gaze needs Aria-class glasses (pivot doc, section 2).

## 6. Dataset export
`collab/export.py` writes sessions as episodes in a LeRobot-style layout
(`meta/info.json`, `meta/episodes.jsonl`, `data/episode_000000.jsonl`) with
JSON instead of parquet, so no new dependencies. Frames aren't stored yet;
adding the paired camera frames is the step after.

## 7. Cards
| Wave | Cards | Agent |
|---|---|---|
| 0 | T3-00 contracts, tests, board | done |
| 1 | T3-01 session · T3-02 metrics · T3-03 worker · T3-04 task scene · T3-05 gaze · T3-08 export | Qwen ×6, two at a time |
| 2 | T3-06 task runner (01, 03, 04) · T3-10 pilot protocol (02) | Claude · Claude |
| 3 | T3-07 gaze in the runtime (05, 06) · T3-09 scoreboard CLI (02, 06, 08) | Claude · any |
| 4 | T3-11 results write-up (07, 09) | Claude |

Metrics and export need only the `Session` dataclasses, which are in the
contract already, so they don't wait for T3-01.
