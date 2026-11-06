# How to demo Curio (T5-33, T5-34)

There are two live demos in the same viewer: **the arm** (sections 1-9, the default) and
**the rescue dog** (section 10, `--robot dog`). Do the arm first. It is the easiest to
follow. The dog then shows that the same idea works when the robot can't see the person.

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
  navigation policy we ported failed its own qualification before Curio was tested. The dog
  demo uses a pretrained walking policy with our own path follower."*
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

---

## 10. The second demo: the rescue dog (T5-34)

A firefighter stands at the end of a smoky corridor, facing down it. A Unitree Go2 comes in
from the other end, facing them, so the firefighter's left is the dog's right. They say
*"Survivor's in the room on my left"* and the dog has to pick the right one of two side rooms.
It can't see the firefighter through the smoke.

**Start it** (same terminal habits as the arm):
```
uv run python -m curio_robot.demo.viewer --robot dog
```
The first time on a new machine, fetch the Go2 model and walking policy and convert the
weights (see third_party/README.md), then:
`uv run python scripts/poc/export_policies.py --go2`. If the files aren't found, the viewer
says so and exits.

**On screen:** the corridor from above, following the dog while it walks. On the floor, a
green disc marks the room the firefighter meant (simulator truth) and a magenta disc marks
the room Curio chose. Top right: the firefighter's head camera and the dog's camera. Top left:
the mode, the request, the room chosen and why, the truth, and where the dog stopped.

**The 45-second script**
1. *"The dog can't see the firefighter. They're at the far end and say which room."* Say *"Curio,
   survivor's in the room on my left"* (or press Enter). The dog walks to the north room,
   about 12 s. **CORRECT**, "in the room the firefighter meant".
2. *"Without Curio the dog uses its own left."* Say *"Curio, compare"*. Curio off sends it
   to the south room (**WRONG**), then Curio on sends it north (**CORRECT**). Do this
   **before** step 3: once they've turned round, their left and the dog's agree, and both are
   right.
3. *"The firefighter turns round. Same words."* Say *"Curio, turn around"*, then the request again. The
   dog goes to the south room, where the survivor now is.

**More buildings (T5-35).** Besides this rescue corridor there are three kinds of building,
each generated from a seed like the arm's scenes: 11-14 m corridors with 3-6 rooms, each door
in a colour (red, blue, green, yellow), the room tinted to match. `n` steps through the
showcase list; `scene doorway 1` loads one directly. Changing building closes the window and
reopens it on the new one (about 2 s). The terminal, the voice and the Curio switch carry over.
You can also start on one: `--robot dog --scene corridor:0`.

| Kind | Set-up | Request | Why the dog alone gets it wrong |
|---|---|---|---|
| `rescue` | T5-34's corridor, two plain rooms | "Survivor's in the room on my left" | Face to face, left flips |
| `corridor` | Firefighter at the west end facing the dog; two same-colour doors face each other across the corridor, plus others | "The green room on my right" | Face to face, left flips |
| `doorway` | Firefighter with their back to one wall, facing across; two same-colour doors on the far wall, one each side of them | "The blue room on my right" | Their left is along the corridor; to the dog both doors are on the same side, so it's a coin toss |
| `closest` | Two or three same-colour doors along the corridor; firefighter at the west end | "The red room closest to me" / "furthest from me" | The dog's nearest is the firefighter's furthest |

**Survivors vary too (T5-36).** In these three kinds each seed also picks the survivors:
one, two or three in the meant room, sometimes one more in another room. Outfits: red jacket,
navy coat, green hoodie, denim, orange overalls, white lab coat, hi-vis vest, or hard hat
with hi-vis. Skin tones vary. Poses: lying, lying and waving, sitting against the wall, or
sitting and waving. The top-left "Survivors" line says who is where. On arrival the dog says
how many it found ("Found two survivors", "Nobody here"); that count comes from the
simulator. Survivors are scenery: Curio never sees them and still picks the room from the
floor map. `turn` moves them to the room the words now mean. The rescue scene keeps its one
survivor.

Showcase order (`n`): rescue 0, corridor 0, doorway 0, closest 0, corridor 1, doorway 1,
closest 1. These were picked by the T5-35 gate's rule: Curio off wrong, Curio on right, and
the walk worked. A good 2-minute run: rescue 0 (script above), then `n` to corridor 0 and
`c`, then doorway 0 and `c`, then closest 0 and `c`.

Voice phrases: *"Curio, "* + *the survivor's in the room / survivor's in the room / search
the room / go to the room / check the room / go check the room* + *on my left / on my right
/ on your left / on your right*; or *the / search the / go to the / check the* + *red / blue
/ green / yellow* + *room* + *on my left / on my right / on your left / on your right /
closest to me / furthest from me / nearest me*; and *compare, on, off, reset, turn around,
next scene*. Typed: any request, `c`, `on`, `off`, `turn`, `n`, `scene KIND SEED`, `r`,
`h`, `q`.

**Live and scripted, for the dog**

| Live | Not Curio, or not live |
|---|---|
| The physics: the Go2 walks under a pretrained walking policy (cagataydev, Isaac Lab, CC-BY-4.0) in MuJoCo, stepped live | The route: **our** path follower through the corridor's door waypoints. Curio only chooses the room |
| The firefighter's head-camera pose and the dog's camera pose, read from the scene | The rooms come from the building's floor map, not from a detector. No VLM on this laptop |
| The choice of room: the same rule resolver as the arm demo, ranking the two rooms from whoever is speaking | The firefighter and survivor are posed figures (they don't move on their own) |
| The check against the simulator's truth | The navigation policy we tried to use instead (T5-32) failed its own qualification and is not used |

Credit the walking policy if you show this publicly: "cagataydev / strands-robots Isaac Lab
showcase (strands-labs/robots#4227)", Go2 model by Unitree Robotics (MuJoCo Menagerie). The
bottom-right overlay already says it.

**Measured** (out/t5_34/gate.md, run once on this laptop, gate set before building): the dog
reaches either room within 0.35 m on 20/20 jittered starts with 0 falls; Curio on picks the
meant room 4/4 (off 2/4: wrong whenever the two face each other); end to end it ends in the
meant room 12/12. The live loop can run 5.5x real time with the insets on. The paced run
viewers see manages 0.97x: each dog-camera refresh costs about 0.2 s, and the dog catches up
after it.

The generated buildings (out/t5_35/gate.md, run once, gate set before building), per kind:
the dog reached every room of seeds 0-9 within 0.35 m (corridor 44/44, doorway 35/35,
closest 34/34, 0 falls). Curio on picked the meant room 50/50 in each kind; Curio off got
0/50 (corridor), 26/50 (doorway, chance) and 0/50 (closest). End to end, 10/10 per kind.

**If something goes wrong:** the dog falls or stops short: `r`, then the request again
(every request starts it from the corridor's end). Window frozen: `q` and restart, about 5 s.
Can't start: present the arm only and describe the dog with the T5-20 video.
