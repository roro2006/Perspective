---
id: T5-24
title: Corrections as a data flywheel; the grounder retrains and is promoted only through a gate
status: todo
agent: claude
budget: 100000
depends: [T5-13, T5-18]
owns: [curio_robot/flywheel/__init__.py, curio_robot/flywheel/log.py, curio_robot/flywheel/request.py, curio_robot/flywheel/scenes.py, curio_robot/flywheel/deploy.py, curio_robot/flywheel/gate.py, curio_robot/flywheel/serve.py, curio_robot/flywheel/registry.json, scripts/flywheel_deploy.py, scripts/flywheel_data.py, scripts/flywheel_gate.py, tests/test_t5_flywheel.py, tasks/T5-24-correction-flywheel.md, tasks/notes/T5-24.md]
reads: [scripts/train_lora.py, scripts/bench_lora_data.py, curio_robot/bench/items.py, curio_robot/bench/conditions.py, curio_robot/bench/score.py, curio_robot/bench/cli.py, curio_robot/sim/ground_eval.py, curio_robot/grounding/client.py, curio_robot/grounding/marks.py, curio_robot/grounding/prompt.py, scripts/poc/scene_base.py, scripts/poc/scenes_a.py, scripts/poc/kit.py, docs/bringup-7900xt.md, tasks/notes/T5-18.md]
accept: uv run pytest -q tests/test_t5_flywheel.py
---

## Goal
When the deployed grounder gets a request wrong, the person corrects it ("no, behind ME").
Each correction becomes a training example, the grounder LoRA retrains, and the new
adapter replaces the old one only if a gate fixed in advance says so. The first turn of the
flywheel targets the known weakness: "behind me / in front of me" (the T5-18 warehouse miss).

1. `curio_robot/flywheel/`: a correction event (utterance, the request and image the
   grounder saw, the object map, its pick and probabilities, the correction, the resolved
   true target, speaker id) stored as JSONL plus PNGs, and converted to rows in the
   grounding_data JSONL format that `scripts/train_lora.py` reads unchanged.
2. Simulated deployment (`scripts/flywheel_deploy.py`): a new kind, `front-behind`
   (pairs of look-alike objects, one in front of and one behind the speaker, where the
   robot's own front/behind gives the other answer). Requests use the T5-18 depth layout
   (`scene_base.depth_request`, reproduced in `flywheel/request.py`). The current
   grounder (v2, from `flywheel/registry.json`) answers; when it is wrong a simulated user
   corrects it and the event is logged; a 20% sample of correct answers is logged as
   confirmed. Episodes run until 450 corrections (cap 3,000 episodes).
3. Retrain (`scripts/flywheel_data.py`, then `train_lora.py` with its defaults, the v2
   recipe): v2's data rebuilt exactly as for v2 (`bench_lora_data.py --seeds 11-19
   --deictic-from out/grounding/data.jsonl`, checked against v2's `data_sha256`) plus the
   flywheel rows. Convert to `curio-ground-r16-v3-f16.gguf` under `out/t5_24/`; never touch
   `curio-ground-r16-v2-f16.gguf`.
4. Gate (`scripts/flywheel_gate.py`): scores both adapters, prints promote or reject with
   reasons, and only on promote copies the GGUF into `llama/models/lora` and adds it to
   `flywheel/registry.json`.

Seeds: bench uses 0, v2 training 11-19, e2e 70000+. Deployment episodes use seeds
524000-526999; the held-out front/behind test uses 624000-624999. The data builder refuses
the held-out seed range and the held-out phrasings.

## GATE (written before any training or evaluation; not to be changed afterwards)
Each adapter is served alone on a fresh llama-server (Qwen3VL-4B Q8_0 + mmproj + `--lora`,
the bring-up flags). A prediction is the argmax of the returned probabilities (as
`bench.score.vlm_predict`); no answer counts as wrong.

Test sets:
- (a) `out/bench5c`, all 350 items, 7 kinds, geometry condition.
- deictic: the deictic rows among the 302 target-2 held-out ids
  (`out/lora/qwen3vl4b-r16/eval_ids.json` in `out/grounding/data.jsonl`), the set that
  showed v2's deictic regression. Treated as one more kind in criterion 3.
- (b) front/behind held-out: 200 items from seed range 624000+, generated before training:
  100 with the training phrasings and 100 with phrasings no training row uses; speaker
  frame 3 to 1 over robot frame.
- (c) the T5-18 warehouse query "Bring the tote from the rack behind me." (speaker frame,
  correct = tote_a), rebuilt from `scripts/poc` scene code. Reported, not gated.

v3 is PROMOTED only if all three hold, on point estimates:
1. front/behind held-out accuracy of v3 >= 0.85, and >= v2's accuracy + 0.25.
2. bench5c overall accuracy of v3 >= v2's - 0.01.
3. No bench5c kind, and not deictic, drops by more than 0.05 (v3 >= v2 - 0.05 per kind).

Otherwise the flywheel REJECTS v3, which is an acceptable reported outcome. Reported with
every number: 95% Wilson intervals per accuracy and a paired bootstrap 95% interval
(10,000 resamples, seed 0) for each v3 - v2 difference; per-kind tables; the train_lora
held-out accuracy. One training run: no data, mix or hyperparameter change after any
test-set score has been seen.
