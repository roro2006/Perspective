# How to demo Curio (T5-33)

Curio lets a robot follow "the red one on my left" from **your** side of the table. Face to
face, your left is the robot's right, and most robots get that backwards. This demo shows it
live in simulation, with Curio switched off and then on.

Two front ends run the same decision code (`session.decide`):
- **MuJoCo viewer** (main): one native window. You type in the terminal.
- **Browser console** (backup): the same thing as a web page, plus a measured A/B and replays.

---

## 1. Before you present (10 minutes)

1. Plug the laptop in. Integrated graphics slow down on battery.
2. Open a terminal and start the viewer:
   ```
   cd C:\Users\Harry\Desktop\MoreProjects\Perspective-t5-33
   uv run python -m curio_robot.demo.viewer
   ```
   A MuJoCo window opens on scene `perspective 0`. Keep the terminal visible next to it
   (window left, terminal right, or terminal at the bottom), because that's where you type.
3. Press Enter once in the terminal, wait for the arm to arrive, then type `r` to reset.
   That warms everything up.
4. Optional: turn on the RTX 4070 in Device Manager (Display adapters) for smoother rendering.
   The demo runs without it.
5. No internet, GPU or model server is needed. Everything is local.

## 2. What's on screen

| Where | What |
|---|---|
| Main view | The SO-101 arm, the table, the blocks, and the person's head on the far side. Drag to orbit, scroll to zoom. |
| Top-right inset | **Human view**: the person's head camera. It says NOT USED when Curio is off. |
| Inset below it | **Robot view**: the robot's own camera. |
| Rings in the insets | Orange: the robot saw it. Blue: the person saw it. Thick magenta: Curio's choice. |
| Discs on the table | Magenta: Curio's target. Green: the ground truth for your words. |
| Top-left text | Mode, scene, request, target and why, truth, result, arm status. |
| Bottom-left text | Off vs on results, after a comparison. |

## 3. The 60-second script

Commands go in the terminal. **What to say is in italics.**

| Step | Type | Say |
|---|---|---|
| 0 | (nothing) | *"This is a simulation. The arm motion is scripted. What's live is Curio working out which block you mean, from your side, on whatever you type."* |
| 1 | (nothing) | *"The person sits across the table. Two identical red blocks. They say: hand me the red one on my left."* |
| 2 | `off` then Enter | *"Robot camera only. It reads 'left' as its own left..."* The arm goes to the wrong block. Result: **WRONG** (magenta disc away from the green one). |
| 3 | `on` then Enter | *"Now Curio adds the person's view, so 'my left' means their left."* Result: **CORRECT**, magenta on green. |
| 4 | `n` twice | Loads `occlusion 1`. *"The blue block is behind a box. The robot can't see it, but the person can."* |
| 5 | `c` | Runs off, then on. Off: **NOT FOUND**. On: **CORRECT**. *"Curio puts what the person sees into the robot's map."* |
| 6 | (offer the keyboard) | *"Try your own."* Good ones on `perspective 0` (`scene perspective 0`): "the green one on my right", "the red one closest to me", "the green one on your right". |

The last one is a useful closer: "your right" means the robot's right, and Curio gets it right
with or without the head camera. It isn't just flipping left and right; it reads whose
frame the words are in.

Short version, if someone is walking past: `scene perspective 0`, then `c`.
*"Same words, same scene. Off, it's wrong. On, it's right."*

## 4. Terminal commands

| Type | Does |
|---|---|
| any sentence | runs it with the current mode, e.g. `the red one on my left` |
| Enter alone | the scene's own request |
| `on` / `off` | Curio on (robot + head camera) or off (robot camera only) |
| `c` | compare: the same words with Curio off, then on |
| `n` | next showcase scene: perspective 0, occlusion 1, perspective 1, cluttered 2 |
| `scene K S` | load scene K (`perspective`, `occlusion`, `cluttered`) with seed S |
| `r` | reset the current scene |
| `h` | help |
| `q` | quit (or close the window) |

Each showcase scene was checked: Curio off is wrong (or finds nothing) and Curio on is correct.
The understood words are a colour plus left/right, nearest/closest or farthest/furthest, with
"my" (yours) or "your" (the robot's). Anything else still gets an answer, and Truth shows
"unknown".

## 5. What is live and what is scripted

Say this plainly if anyone asks. Don't claim more.

| Live (computed every time you press Enter) | Scripted or simulated |
|---|---|
| Rendering the head camera and the robot camera from the current scene | The arm motion: inverse kinematics to the chosen block, no grasp, **not a learned policy** |
| Detecting blocks, fusing both views into one map, working out whose left | Scene layouts: a seeded generator, random but repeatable |
| Choosing the block from whatever you type | The person: a simulated camera, not a real person or camera |
| Checking the choice against the simulator's ground truth | The language reading: keyword rules, not the trained model (which isn't on this laptop) |

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
2. **Typing does nothing:** click the terminal first. Keyboard focus is probably on the MuJoCo
   window.
3. **Window frozen or closed:** `q`, then start it again (step 1.2). It opens in a few seconds.
4. **Viewer won't start at all:** run the browser console instead (below).
5. **No laptop:** describe the beat with a recording from rehearsal, and say it's a replay.

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
 │ then: IK steps toward the chosen block (scripted motion)                      │
 └───────────────────────────────────────────────────────────────────────────────┘
```

Code: `curio_robot/demo/` (`viewer.py`, `session.py`, `server.py`, `status.py`,
`static/index.html`). Tests: `uv run pytest -q tests/test_t5_33_demo.py`.
