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
| `runtime/` (live pipelined runtime: speculative motion, VLM confirm, interrupts, latency tracer) | Working in sim, tested |
| `voice/` (VAD, streaming Whisper, Kokoro TTS with barge-in), `vlm.py`, `perception.OnnxYoloWorldDetector` | Written against local servers / ONNX; not yet run on the 7900 XT |
| `real/bench.py`, `real/calibration.py` | Stubs (Track 2) |
| `dashboard.py` (live three-pane UI) | Stub |
| `conventions/` (readings, belief, noise, decide, profiles) | Working, tested |
| `recorder` (decision log) | Working, tested |
| `sim/users`, `sim/learn` (simulated users, learning curves) | Working, tested |
| `runtime/` target-1 integration (T1-08) | In progress |

## Run

```bash
uv sync
uv run pytest
uv run curio-sim shot perspective --seed 0 --out out/shots   # off/on frames for video and slides
uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2
```

Renders go to `out/` (git-ignored).

### Live runtime

```
 main thread   capture ──► Latest[frames] ──► perception thread (detector, fusion, tracker) ──► Latest[map]
 voice-in      mic ► VAD ► streaming Whisper ──► Heard(start/partial/end/final) ─┐
 perception    hand enters workspace ──► Interrupt ─────────────────────────────┤ one event queue
 motion        small IK steps toward the goal ──► ArmEvent(started/arrived/...) ─┤
 vlm pool      VLM on a map snapshot ──► VlmReply ──────────────────────────────┘
                                                   ▼
                                     orchestrator: Motion.go/stop, VoiceOut.say
```

A partial transcript with a colour starts the arm (toward the midpoint of a
pair until "left"/"right" arrives); the final transcript re-checks with the
rules and asks the VLM, which confirms or redirects mid-motion. "No", "stop" or
a hand in the workspace stops the arm; "no, the other one" switches to the
other block of the pair and is logged as a correction.

```bash
uv run curio-sim live perspective --seed 3              # one episode + latency waterfall
uv run curio-sim live perspective --runs 20 [--off]     # success rate and p50 latencies
```

### On the RX 7900 XT (Windows)

```bash
uv sync --extra gpu --extra voice --extra cameras
```

| Model | Runs as | AMD backend |
|---|---|---|
| Detector (YOLO-World, colour prompts baked in) | in-process, `OnnxYoloWorldDetector` | ONNX Runtime DirectML (`accel.py`) |
| Whisper (base.en / small.en) | `whisper-server --port 8178` (whisper.cpp) | Vulkan build |
| Kokoro TTS | Kokoro-FastAPI on `:8880` | ONNX / DirectML |
| VLM (Qwen3-VL-8B GGUF) | `llama-server --port 8080` (llama.cpp) | Vulkan or HIP build |

All three servers fit in the card's 20 GB together. Try the real VLM
against the sim with `--llm-url http://127.0.0.1:8080/v1`. Use a headset mic:
the VAD has no echo cancellation.

The SO-101 model is vendored from MuJoCo Menagerie
(`curio_robot/assets/so101`, Apache-2.0, see its LICENSE).

## Next (in order)
1. Runtime integration of target-1 learning (T1-08, in progress).
2. Live learning demo (T1-10).
3. Wrist-camera refinement in the sim (fixes occlusion under calibration error; see results.md).
4. Track 2: real webcam + phone head cam + ArUco → live object map (`real/`).
5. Live dashboard (three panes + on/off switch) for the stage and booth.
6. Bring up whisper-server, Kokoro and llama-server on the 7900 XT; measure the real waterfall.
7. Real SO-101 once the go/no-go gate passes.
