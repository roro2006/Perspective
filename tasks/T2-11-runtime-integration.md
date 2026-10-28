---
id: T2-11
title: Wire the grounder into the live runtime
status: done
agent: claude
depends: [T2-01, T2-02, T2-03, T2-04, T2-05, T2-06]
owns: [curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/runtime/perception_stage.py, curio_robot/voice/stt.py, curio_robot/sim/live.py, tests/test_t2_runtime.py]
reads: [docs/target2-grounding.md, curio_robot/grounding/history.py, curio_robot/grounding/anchor.py, curio_robot/grounding/marks.py, curio_robot/grounding/prompt.py, curio_robot/grounding/client.py, curio_robot/grounding/fuse.py, curio_robot/events.py, curio_robot/runtime/orchestrator.py, curio_robot/runtime/__init__.py, curio_robot/runtime/perception_stage.py, tests/test_t1_runtime.py, tests/test_vlm_belief.py]
accept: uv run pytest tests/test_t2_runtime.py tests/test_runtime.py tests/test_t1_runtime.py tests/test_vlm_belief.py --strict-impl
---

## Goal
When a grounder is configured, the final transcript is grounded in the frame
the person saw at the anchor word, and the result is fused with the belief.

## Behaviour
- `RuntimeConfig.grounder: VllmGrounder | None = None`. Without it, nothing
  changes (the text VLM path and every existing test stay as they are).
- Perception pushes every human-camera frame into a `FrameHistory` with
  `time.perf_counter()` at capture.
- `StreamingRecognizer` requests word timestamps when its transcriber
  supports them and puts `parse_words(...)` results in `Heard.words` on the
  final (keep the `Transcriber` protocol backward compatible).
- On a final the belief acts on: anchor = `anchor_time(words)` or the
  speech-end time; frame = `history.at(anchor)` or the latest; marks, render,
  `build_request(..., p_own_frame=belief.p_own_frame())`, `grounder.score` on
  the pool; post a `GroundingReply`.
- On `GroundingReply`: `fuse(belief.posterior(readings), reply.probs)` then
  `decide`; redirect only if the fused top object differs from the current
  target and clears the act threshold. With no readings (rules found
  nothing), the grounder alone may act. Record the reply as stage "vlm" with
  the fused posterior; outcomes judge whichever object the arm ended at.
- Tracer stages: `ground_sent`, `ground_reply`, `ground_redirected`.

## New tests (`tests/test_t2_runtime.py`, yours to write)
Fake grounder returning fixed probs; reuse the `_brain` pattern:
1. A confident grounder can't move an explicit "my left".
2. "Hand me that one" with no colour: no readings, so the grounder's top
   object is acted on.
3. A close belief plus a confident grounder redirects.
4. `FrameHistory.at` is asked for the anchor word's time, not the latest.

## Done when
The accept command passes, `uv run curio-sim live perspective --seed 3` still
succeeds, plus the full suite and ruff.
