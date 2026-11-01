# Target 5: where it stands (2026-11-01)

The question: can a robot follow "the red one on my left" when a
perspective-aware grounding layer tells it which block, and a learned policy
does the picking? Everything below is simulation (MuJoCo, SO-101), with
scripted demonstrations, not real people or hardware.

## What is shown
- **Perspective needs the person's view.** On the 350-item benchmark the robot
  camera alone scores 0.38-0.40; blocks hidden from the robot: 0.00.
- **Curio geometry plus a tuned grounder beats both a keyword rule and a VLM
  reasoning from images.** LoRA v2, geometry condition: 0.92 overall (rule
  0.77, same model inferring perspective from both images 0.78); hidden-pair
  1.00; free wording 0.90; unseen phrasing 0.80 (old adapter 0.56). Latency
  unchanged at about 110 ms.
- **The grounder decides, the policy follows.** In every SmolVLA run, a ring on
  the wrong block makes it pick that block and never the named one; with no
  ring it picks the target under 3% of the time.
- **End to end** (200 requests, LoRA v2 grounder then SmolVLA): target picks rose
  from 0.14 (run 1) to 0.42, 0.46 and 0.515 (runs 2-4), within 1 point of the
  true-target ring every time. Wrong-block picks stay at 1-2%.
- **The expert** picks 99.6% of open-scene blocks and 87% of blocks hidden
  behind a box.

## What is not
- **The grasp.** Best policy (run 4): 0.50 ring-on-target on 100 scenes; every
  run since run 2 missed its pre-registered gate (0.40, 0.55, 0.55). The main
  miss: jaws close 2-3 cm high with a jaw resting on the block.
- **No hidden-block policy data or evaluation yet** (expert can now do them).
- **"That one" regressed** in LoRA v2 (0.55 -> 0.45 on the T2 set); v2 is not
  the live adapter yet.
- **Nothing on real people or hardware.**

## Next, when resumed
1. Run 5: noise that reaches the block, so recoveries from a resting jaw are in
   the data; or a grasp-success critic to re-try.
2. LoRA v3 with more "that one" rows before v2 goes live.
3. Policy data and evaluation with blocks hidden behind the box.

Details: product/results.md, docs/target5-perspective-policy.md, tasks/T5-*.
