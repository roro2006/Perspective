---
id: T2-05
title: vLLM grounding client with letter probabilities
status: todo
agent: qwen
depends: []
owns: [curio_robot/grounding/client.py]
reads: [docs/target2-grounding.md, curio_robot/grounding/prompt.py, curio_robot/vlm.py, tests/test_t2_client.py]
accept: uv run pytest tests/test_t2_client.py --strict-impl
---

## Goal
Implement `letter_probs` and `VllmGrounder` in `curio_robot/grounding/client.py`
exactly as the docstrings specify. Keep `GroundingAnswer` and `DEFAULT_URL`.

## Hints
- `vlm.py`'s `ChatEndpoint.complete` shows the house style for a JSON POST
  with `urllib.request`. No new dependencies.
- Probabilities: `math.exp(logprob)` per allowed token, summed per letter,
  then divided by the total of the allowed ones.
- `score` must never raise: wrap the request and parsing in one try/except
  and return `source="unavailable"` with `error=str(e)`.
- `prompt.py` is a stub for you; only use the `GroundingRequest` dataclass.
- The test runs a fake server on localhost; nothing reaches a real model.

## Done when
The accept command passes, plus the full suite and ruff (see AGENTS.md).
