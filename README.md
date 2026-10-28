# Curio: a robot that sees what you see

Curio fuses a co-worker's first-person view (a phone on a head strap, later
smart glasses) with a robot's own cameras. The robot can then find objects it
can't see, understand "on my left" from *your* side of the table, and stop when
you react.

This is the successor to Curio for screens. Start with
the product docs. The code exists to make them true.

## Product (read these first)
| Doc | For |
|---|---|
| [product/narrative.md](product/narrative.md) | The story, the moat, and what we can and can't claim. Source of truth. |
| [product/pitch-strategy.md](product/pitch-strategy.md) | 3-minute finals script, live demo and fallbacks, slides, Q&A bank, rehearsal plan |
| [product/teaser-video.md](product/teaser-video.md) | 2:30 teaser storyboard, shot list, production schedule |
| [product/booth-playbook.md](product/booth-playbook.md) | Tabling event: booth layout, demo loops, talk tracks, lead capture, resets |
| [product/results.md](product/results.md) | Simulation A/B numbers and how to present them |
| [docs/pivot-shared-perspective.md](docs/pivot-shared-perspective.md) | Research gap, one-week MVP plan, go/no-go gate for buying the SO-101 |

## Architecture

```
  World (swap point)               Product code (same on sim and real bench)
 ┌───────────────────────┐   frames   ┌─────────────┐   ┌──────────┐   ┌───────────┐
 │ sim/world.py (MuJoCo) │ ─────────► │ perception  │ ► │ fusion   │ ► │ resolver  │ ► target
 │ real/bench.py (TODO)  │            │ (detector)  │   │ (map +   │   │ (rules or │
 └───────────────────────┘            └─────────────┘   │ seen_by) │   │  VLM)     │
          ▲                                             └──────────┘   └───────────┘
          └── move_gripper_to(target) ◄──────────────────────────────────────┘
      monitor.py (reactions) · episode_log.py (dataset) · voice.py · dashboard.py
```

`use_human_view` in `pipeline.step` is the **Curio on/off** switch used in the
demo, the A/B and the booth.

| Module | Status |
|---|---|
| `types`, `geometry`, `fusion`, `perspective`, `resolver`, `pipeline`, `monitor`, `episode_log` | Working, tested |
| `sim/` (SO-101 scene, scenarios, A/B, shot renderer, CLI) | Working |
| `real/bench.py`, `real/calibration.py`, `perception.OpenVocabDetector` | Stubs (Track 2) |
| `voice.py` (bridge to clicky's Whisper/Kokoro/VLM clients), `dashboard.py` (live three-pane UI) | Stubs |

## Run

```bash
uv sync
uv run pytest
uv run curio-sim shot perspective --seed 0 --out out/shots   # off/on frames for video and slides
uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2
```

Renders go to `out/` (git-ignored).

The SO-101 model is vendored from MuJoCo Menagerie
(`curio_robot/assets/so101`, Apache-2.0, see its LICENSE).

## Next (in order)
1. Wrist-camera refinement in the sim (fixes occlusion under calibration error; see results.md).
2. Track 2: real webcam + phone head cam + ArUco → live object map (`real/`).
3. Live dashboard (three panes + on/off switch) for the stage and booth.
4. Voice bridge to clicky's clients; VLM resolver on the lab GPU.
5. Real SO-101 once the go/no-go gate passes.
