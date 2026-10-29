---
id: T5-09
title: End-to-end experiment harness: grounder places the ring, SmolVLA picks
status: done
agent: claude
depends: [T5-08]
owns: [scripts/e2e_eval.py, tests/test_t5_e2e.py, docs/target5-perspective-policy.md]
reads: [docs/target5-perspective-policy.md, scripts/smolvla_eval.py, curio_robot/bench/items.py, curio_robot/bench/conditions.py, curio_robot/grounding/client.py, curio_robot/sim/physics.py, curio_robot/sim/demos.py, docs/bringup-7900xt.md, tasks/notes/T5-08.md]
accept: uv run pytest tests/test_t5_e2e.py -q
---

## Goal
The headline target-5 result: does a perspective-aware grounder make the
policy pick the block the person meant? A person asks for a block ("the red
one on my left", "... on your left", "... on the left" with a habit, or a
unique colour); four setups decide where the ring goes, then the fine-tuned
SmolVLA picks in closed loop in PhysWorld.

## Behaviour
- Setups: words only (no ring); base grounder (geometry condition, the live
  Curio request) with the ring on its pick; the same with the target-2 LoRA;
  oracle (ring on the true target).
- scripts/e2e_eval.py, two stages because the grounder server and the policy
  do not share the GPU: `ground` (project env) builds the trials from fresh
  seeds (70000+) and saves one choices file per grounder name plus the oracle
  and words-only choices; `rollout` (lerobot env) reuses smolvla_eval's
  rollout, ring and policy loading and scores per setup and kind: grounding
  accuracy and pick outcome (target / partner / other / nothing), JSON out.
- A fake grounder (first letter) so the ground stage is testable on CPU.
- No bench/ API changes; nothing in curio_robot/ imports torch or lerobot.
