# Simulation results (Phase 0)

Run 2026-10-28, `uv run curio-sim ab --trials 200` (seed 0). Each trial renders
one scene and resolves the request twice: **Curio on** (robot camera + human
camera) and **off** (robot camera only). A trial succeeds when the chosen point
is within 2 cm of the correct block, which is within the SO-101 gripper's
tolerance. Detection is ground truth from segmentation, so these numbers
isolate **viewpoint and perspective**, not detector quality.

| Scenario | What it is | Off | On (perfect head pose) | On (1 cm / 2° head-pose error) | On (2 cm / 4°) |
|---|---|---|---|---|---|
| Perspective | Two same-colour blocks, "the X one on my left/right" | 21.5% | 98.0% | 80.5% | 61.0% |
| Occlusion | Box placed between the robot cam and the target | 0.0% | 100.0% | 40.0% | 13.5% |
| **Cluttered** | Nothing staged: random box, 4–5 blocks, mixed requests | **50.5%** | **97.5%** | **85.5%** | **75.0%** |

## How to use these on stage
- **Headline:** the cluttered row, with realistic calibration error. *"In random
  cluttered scenes, success goes from 50% to 86% with the human's view."*
- The occlusion row's 0% is **by construction** (we put the box there). Say
  "when the target is hidden from the robot," never "robots fail 100% of the
  time."
- These are simulation numbers. Label them "simulation" on the slide and pair
  them with the real-camera Track 2 numbers once measured.

## What we learned: calibration is the product problem
Objects seen *only* by the human camera are located only as well as we know
where that camera is. At half a metre, a 2° head-pose error moves the estimate
by about 1.7 cm, which is most of the grasp tolerance. That's why occlusion
falls from 100% to 40% with modest pose noise.

The fixes, in order (these are also the engineering-depth story for judges):
1. **Close the loop with the wrist camera.** The human view gets the gripper
   *near* the object; the SO-101's wrist camera then sees it and refines the
   grasp. Next thing to build in the sim.
2. **Use the robot as the calibration target.** The arm is visible in the
   human's view, and its joint angles are known exactly, so solve the head-cam
   pose against it every frame, not just against the ArUco board.
3. **Fuse over time.** Average an object's position across frames as the head
   moves; noise drops roughly with the square root of the number of views.

Re-run: `uv run curio-sim ab --trials 200 --pose-noise 0.01 --rot-noise 2`
