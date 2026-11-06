# How to demo Curio (T5-33)

Curio lets a robot follow "the red one on my left" from **your** side of the table. Face to
face, your left is the robot's right, and most robots get that backwards. This demo shows it
live in simulation, with Curio switched off and then on.

Two front ends run the same decision code (`session.decide`):
- **MuJoCo viewer** (main): one native window. You **talk to it** ("Curio, hand me the red one
  on my left") and it answers out loud. The terminal still works as a backup.
- **Browser console** (backup): the same thing as a web page, plus a measured A/B and replays.

---

## 1. Before you present (10 minutes)

1. Plug the laptop in. Integrated graphics slow down on battery.
2. Open a terminal and start the viewer:
   ```
   cd C:\Users\Harry\Desktop\MoreProjects\Perspective-t5-33
   uv run python -m curio_robot.demo.viewer
   ```
   A MuJoCo window opens on scene `perspective 0`, and the microphone starts listening. The
   top-left text says *Voice: listening*. Keep the terminal visible as a backup.
   (`--no-voice` turns the microphone off.)
3. Say *"Curio, hand me the red one on my left"*, wait for the arm to arrive, then say
   *"Curio, reset"*. That warms everything up and checks the mic. If it doesn't react, see
   section 7.
4. Use the laptop's speakers at a moderate volume, or headphones. Curio ignores what it hears
   while it's talking, but loud speakers next to the mic can still confuse it.
5. Optional: turn on the RTX 4070 in Device Manager (Display adapters) for smoother rendering.
   The demo runs without it.
6. No internet, GPU or model server is needed. Everything is local, including the speech
   recognition (Windows' built-in, offline recognizer).

## 2. What's on screen

| Where | What |
|---|---|
| Main view | The SO-101 arm, the table, the blocks, and the person's head on the far side. Drag to orbit, scroll to zoom. |
| Top-right inset | **Human view**: the person's head camera. It says NOT USED when Curio is off. |
| Inset below it | **Robot view**: the robot's own camera. |
| Rings in the insets | Orange: the robot saw it. Blue: the person saw it. Thick magenta: Curio's choice. |
| Discs on the table | Magenta: Curio's target. Green: the ground truth for your words. |
| Top-left text | Mode, scene, where the person is seated, request, target and why, truth, result, arm status. |
| Bottom-left text | Off vs on results, after a comparison. |

## 3. The 60-second script

Every step can be **spoken** (the "Say to Curio" column) or typed in the terminal. Your
narration to the audience is in italics.

| Step | Say to Curio (or type) | Narration |
|---|---|---|
| 0 | (nothing) | *"This is a simulation. The arm motion is scripted. What's live is Curio working out which block you mean, from your side, on whatever you type."* |
| 1 | (nothing) | *"The person sits across the table. Two identical red blocks. They say: hand me the red one on my left."* |
| 2 | "Curio, off", then "Curio, hand me the red one on my left" (`off`, Enter) | *"Robot camera only. It reads 'left' as its own left..."* The arm picks up the **wrong** red block and hands it to the person. Result: **WRONG** (the magenta disc, where it took the block from, is away from the green one). |
| 3 | "Curio, on", then the same sentence (`on`, Enter) | *"Now Curio adds the person's view, so 'my left' means their left."* The arm picks up the right block and sets it down in front of the person: *"Here you go."* Result: **CORRECT**, magenta on green. Each request starts from the scene's own layout, so the block is back on the table every time. |
| 4 | "Curio, walk around" (`seat 180`) | The person walks round to stand beside the robot. *"Now I'm standing next to the robot. Same words."* |
| 5 | "Curio, hand me the red one on my left" (Enter) | Curio picks the **other** red block: CORRECT. *"My left changed when I moved, and Curio followed me."* |
| 6 | "Curio, compare" (`c`) | Off and on are both CORRECT here. *"Standing beside it, my left is the robot's left, so even the robot alone gets it. Curio is right from every seat; the robot alone is right only from its own."* Then "Curio, go back" (`seat 0`). |
| 7 | "Curio, next scene" twice (`n` twice) | Loads `occlusion 1`. *"The blue block is behind a box. The robot can't see it, but the person can."* |
| 8 | "Curio, compare" (`c`) | Runs off, then on. Off: **NOT FOUND**. On: **CORRECT**. *"Curio puts what the person sees into the robot's map."* |
| 9 | (offer the keyboard) | *"Try your own."* Good ones on `perspective 0` (`scene perspective 0`): "the green one on my right", "the red one closest to me", "the green one on your right". |

The last one is a useful closer: "your right" means the robot's right, and Curio gets it right
with or without the head camera. It isn't just flipping left and right; it reads whose
frame the words are in.

Short version, if someone is walking past: `scene perspective 0`, then `c`.
*"Same words, same scene. Off, it's wrong. On, it's right."*

## 4. Voice and terminal commands

Start every spoken command with **"Curio"**. It only listens for these sentences, which is what
makes it reliable in a noisy room:

| Say "Curio, ..." | Same as typing |
|---|---|
| "hand me / give me / pass me / pick up / get / grab / go to the *red / blue / green / yellow* *one / block* ..." optionally followed by "on my left / on my right / on your left / on your right / closest to me / furthest from me / closest to you / furthest from you" | the sentence itself |
| "on", "turn on" / "off", "turn off" | `on` / `off` |
| "compare", "show the difference" | `c` |
| "next scene" | `n` |
| "reset" | `r` |
| "walk around", "stand beside the robot", "go to the other side" | `seat 180` |
| "go back", "sit across" | `seat 0` |
| "go to the end" / "go to the other end" | `seat 90` / `seat -90` |
| "walk", "keep walking" | `walk` |

Curio answers out loud ("Got it, the red one."; "Robot view only. The red one."; "I can't see a
blue one."). If it heard you but wasn't sure, the Voice line says *not sure I heard ...* and
nothing happens: just say it again.

Terminal commands:

| Type | Does |
|---|---|
| any sentence | runs it with the current mode, e.g. `the red one on my left` |
| Enter alone | the scene's own request |
| `on` / `off` | Curio on (robot + head camera) or off (robot camera only) |
| `c` | compare: the same words with Curio off, then on |
| `n` | next showcase scene: perspective 0, occlusion 1, perspective 1, cluttered 2 |
| `seat D` | walk the person to D degrees round the table: `0` across from the robot, `90` / `-90` the ends, `180` beside the robot |
| `walk` / `walk D` | walk 45 (or D) degrees further round |
| `scene K S` | load scene K (`perspective`, `occlusion`, `cluttered`) with seed S |
| `r` | reset the current scene |
| `h` | help |
| `q` | quit (or close the window) |

Each showcase scene was checked: Curio off is wrong (or finds nothing) and Curio on is correct.
Moving the person was checked on `perspective 0` in 20-degree steps all the way round: Curio
on was correct at every seat. Curio off was correct only from about 80 degrees round to the
robot's side, where the person's left matches the robot's.

The understood words are a colour plus left/right, nearest/closest or farthest/furthest, with
"my" (yours) or "your" (the robot's). Anything else still gets an answer, and Truth shows
"unknown".

## 5. What is live and what is scripted

Say this plainly if anyone asks. Don't claim more.

| Live (computed every time you press Enter) | Scripted or simulated |
|---|---|
| Rendering the head camera and the robot camera from the current scene | The pick and hand-over: scripted inverse-kinematics motion; the block is carried once the jaws close (no contact physics), **not a learned policy** |
| Detecting blocks, fusing both views into one map, working out whose left | Scene layouts: a seeded generator, random but repeatable |
| Choosing the block from whatever you type | The person: a simulated camera, not a real person or camera |
| Checking the choice against the simulator's ground truth | The language reading: keyword rules, not the trained model (which isn't on this laptop). Speech is Windows' offline recognizer, listening for a fixed list of sentences |

## 6. Questions you'll get

- **"Is this a real robot?"** *"No, it's a simulation of the SO-101 arm. The decision is real
  code running live; the arm motion is scripted."*
- **"Is there AI in this?"** *"In this demo the perspective step is geometry and rules. We
  also trained a vision-language model for the same step. On our 350-item benchmark it scores
  0.92, against 0.38-0.40 for the robot camera alone. It isn't running on this laptop."*
  (Source: docs/target5-summary.md.)
- **"Does it work with existing robot policies?"** *"That's the next test. We put Curio in
  front of policies trained by others: the LIBERO evaluation hasn't run yet, and the Go2
  navigation policy we ported failed its own qualification before Curio was tested."*
- **"How fast is it?"** Point at the latency in the terminal: detection and resolution take a
  few milliseconds. The pause before the arm moves is the cameras rendering on a laptop.

## 7. If something goes wrong

Don't debug on stage. Spend five seconds at most, then step down this list.

1. **Wrong result you didn't expect:** type `r`, then run again. Or `scene perspective 0`
   and `c`, which is known good.
2. **Voice doesn't react:** check the Voice line. If it says *not sure I heard*, speak closer
   and say "Curio" clearly. If it says *voice stopped*, or the mic is wrong, check the Windows
   default microphone (Settings > System > Sound > Input), or just type in the terminal.
3. **Typing does nothing:** click the terminal first. Keyboard focus is probably on the MuJoCo
   window.
4. **Window frozen or closed:** `q`, then start it again (step 1.2). It opens in a few seconds.
5. **Viewer won't start at all:** run the browser console instead (below).
6. **No laptop:** describe the beat with a recording from rehearsal, and say it's a replay.

## 8. Backup: the browser console

```
uv run python -m curio_robot.demo        # then open http://127.0.0.1:8765
```
It has the same scenes and the same decisions, plus:
- a **Showcase** button that runs the script above with captions;
- **Off vs on** side by side with ground truth;
- **Measure 50 / scene**, the sim A/B measured live on this laptop (about a minute);
- **Replay** of any earlier run, marked REPLAY.

Keys: Enter runs, Space switches Curio on and off, Esc cancels. Options: `--port`,
`--scene perspective:0`, `--no-record`, `--host 0.0.0.0`. On a headless Linux machine, set
`MUJOCO_GL=egl` and use `ssh -L 8765:localhost:8765`.

## 9. How it works

```
 terminal (you type)         MuJoCo viewer window              browser (backup)
        │                            ▲                                ▲
        ▼                            │ scene, insets, text            │ server-sent events
 ┌──────────────────────────────────┴────────────────────────────────┴──────────┐
 │ session.decide(world, scene, text, curio)                                     │
 │   SimWorld.capture()  -> robot + head camera renders (480 x 360)              │
 │   pipeline.step()     -> detector -> fusion (seen_by) -> perspective -> rule  │
 │   oracle_xy()         -> ground truth from true positions and the person's    │
 │                          true eye and gaze                                    │
 │ then: pick it up and set it down in front of the person (handover.py,         │
 │       scripted kinematic motion: the block is carried once the jaws close)    │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Code: `curio_robot/demo/` (`viewer.py`, `session.py`, `server.py`, `status.py`,
`static/index.html`). Tests: `uv run pytest -q tests/test_t5_33_demo.py`.
