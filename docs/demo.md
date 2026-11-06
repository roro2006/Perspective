# Curio live demo console (T5-33)

A browser console that runs a typed request through the real Curio pipeline on live
renders of the simulated SO-101 table. It shows the person's head camera, the robot's
camera and the scene side by side, with Curio's actual output next to the ground truth.

## Start
```bash
uv sync
uv run python -m curio_robot.demo            # then open http://127.0.0.1:8765
```
Options: `--port 8765`, `--scene perspective:0` (first scene, kind:seed), `--no-record`, and
`--host 0.0.0.0` to expose it on the network (the default is localhost only).

Needs only CPU and MuJoCo: no GPU, no model server and no network. One render takes about
70 ms on a laptop CPU, and the stream runs at about 8 frames per second while the arm moves.
On a headless Linux server, set `MUJOCO_GL=egl` and open it through `ssh -L 8765:localhost:8765`.

## What is live and what isn't
| Part | What runs |
|---|---|
| Cameras | MuJoCo renders of the robot camera, the person's head camera and an overview, every run |
| Curio | `pipeline.step`: segmentation detector, fusion with `seen_by`, perspective, rule resolver. Curio on fuses the head camera; off uses the robot camera only |
| Target | whatever `pipeline.step` returns; nothing is hardcoded |
| Ground truth | true block positions and the person's true eye and gaze, read with the scenarios' own left/right rule. Shown as unknown when the words fall outside that rule |
| Arm | **scripted** position-only IK steps to the chosen block. Not a learned policy |
| Measured A/B | `sim.ab.run` measured on this machine when you press the button |
| Replay | every live run is recorded to `out/demo/recordings/`; ⟲ plays one back under a REPLAY banner |
| Not in this demo | the trained VLM grounder (Qwen3-VL + Curio LoRA) and any pretrained policy. The System card lists the drop-ins as the board records them |

## Architecture
```
 browser (index.html)                   python -m curio_robot.demo
 ┌───────────────────────┐   POST /api/*   ┌──────────────────────────────────────────────┐
 │ command box, toggle   │ ──────────────► │ server.py  validate -> session.submit(job)   │
 │ panes, interpretation │                 │            409 if busy (one arm, one job)    │
 │ compare, A/B, status  │ ◄────────────── │ Hub: server-sent events, per-client queue    │
 └───────────────────────┘ GET /api/events └──────────────┬───────────────────────────────┘
                                                          │ job queue
                                          ┌───────────────▼───────────────────────────────┐
                                          │ session.py worker thread (owns the GL context)│
                                          │  SimWorld.capture -> pipeline.step -> Target  │
                                          │  oracle_xy (ground truth) -> verdict          │
                                          │  IK steps -> frames -> publish + record       │
                                          └───────────────────────────────────────────────┘
```

## One-minute walkthrough
1. **Open it.** Point at the three panes: "the person's head camera, the robot's camera, and
   the table. Everything here is simulation, and the arm motion is scripted. The part that's
   real is Curio deciding which object you mean."
2. **Press ▶ Showcase**, or do it by hand:
   - Scene `perspective`, seed 0: two red blocks. Request "Hand me the red one on my left".
   - Turn Curio off (Space) and press Run. The robot reads "left" as its own left and goes
     to the wrong block (WRONG, red ring vs green truth).
   - Turn Curio on and press Run. The head camera is fused, and "left" is read from the
     person's side (CORRECT).
   - Scene `occlusion`, seed 1: "Hand me the blue one". Off: nothing found, because the box
     hides it from the robot. On: the person can see it, so Curio can.
3. **Let someone type.** Try "the green one on your right" ("your" means the robot's frame)
   or "the red one closest to me".
4. **If asked about policies,** point at the System card: the LIBERO drop-in (T5-31) hasn't
   run, and the Go2 navigation port failed its own qualification (T5-32), so Curio wasn't
   tested on it.

Keys: Enter runs, Space toggles Curio (outside the text box), Esc cancels.

## Fallback ladder
1. A run looks wrong: press Reset and run again. Don't debug on stage.
2. The page shows "disconnected": the server stopped. Restart the command and reload.
3. No laptop or no Python: play a recording (⟲) captured during rehearsal, and say it's a
   replay. The banner says so too.
