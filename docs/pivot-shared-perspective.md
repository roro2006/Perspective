# Pivot: a robot that sees what you see

Working thesis: a robot working next to a person gets better when it can use
**the person's view** of the shared workspace (what they see, where they look,
how they react) alongside its own cameras. We are *not* training the robot on
human hand motion; the human is a second observer, not a demonstrator.

Pitch line: *"Curio saw what you saw on your screen. Now your robot does too."*

---

## 1. Research: is the gap real?

Research as of 2026-10-28. Paper details come from abstracts and listings;
read the ones we cite on slides before the finals.

### Crowded: human video as robot *training data*
Using first-person human video to teach robots motor skills is a busy field
with funded companies behind it. Stay out of this lane.

- EgoMimic, EMMA, EgoBridge: train on Aria-glasses footage of humans, with the
  same glasses mounted on the robot so both views match.
- EgoVerse (2026): a large egocentric dataset for robot learning (academic and
  industry partners).
- EgoAVFlow (Feb 2026): learns manipulation *and* where to look from human
  egocentric video via 3D flow.
- PhysBrain (Dec 2025): human egocentric data as a bridge from VLMs to robots.
- Companies: Mecka (reported $100M+ revenue run rate), Micro1, Human Archive,
  Vectro Labs.

### Exists in pieces, never combined: the collaborator's view during the task

| Piece | What exists | What's missing |
|---|---|---|
| **Implicit feedback from the human** | Gaze dynamics detect robot failures in collaboration (90% accuracy for execution failures within 5 s, HRI 2025 late-breaking report). Facial reactions and EEG (brain-signal) error responses have also been used as feedback. | These are lab classifiers on eye trackers and EEG. None is wired into a robot policy loop, and none uses the collaborator's first-person video. |
| **Perspective-taking** | VLMs default to the camera's viewpoint when asked about someone else's (VPT benchmarks; LeRF 2026). Synthetic-data training for VPT is at the proof-of-concept stage. | Nothing at the robot-policy (VLA) level, and nothing resolving "my left" from the speaker's *live* head pose. |
| **Extra viewpoints for occlusion** | Active-vision work moves a robot-mounted camera to recover from occlusion (AV-ALOHA, AVR, BAVO-Bench). ActiveUMI feeds operator *head pose* to the policy. | No system uses a co-working human's camera as a live observation for the robot. |
| **Collaborative VLAs** | "Learning to Assist" (Jun 2026) steers a VLA at inference time so it doesn't help before the human is ready. | Robot-view only. |

### The gap we claim
No published system or dataset, as far as we found, uses a **co-working
human's live first-person view** as (a) an extra observation that fills robot
occlusions, (b) a source of implicit attention and feedback signals, and
(c) the reference frame for spatial language, all in one loop.

The paired dataset (human view + robot view + robot actions + human
speech/reactions, recorded while working together) also doesn't exist. Ego
datasets record humans alone, and teleop datasets record robots alone.

Honest framing for judges: *each ingredient has precedent in research; the
combined system and the dataset it produces are new.* Don't claim we invented
perspective-taking or gaze-based failure detection.

### Sources
- [EgoAVFlow (arXiv 2602.22461)](https://arxiv.org/abs/2602.22461)
- [EgoVerse (arXiv 2604.07607)](https://arxiv.org/pdf/2604.07607)
- [EMMA (arXiv 2509.04443)](https://arxiv.org/pdf/2509.04443)
- [EgoBridge (arXiv 2509.19626)](https://arxiv.org/pdf/2509.19626)
- [EgoMimic](https://openreview.net/pdf/c92e0b2d677302fe497d46a89b11df2c5c6b3d3b.pdf)
- [PhysBrain (arXiv 2512.16793)](https://arxiv.org/html/2512.16793v1)
- [Gaze dynamics for robot failure detection (arXiv 2503.07622)](https://arxiv.org/abs/2503.07622)
- [Robot error awareness through human reactions (arXiv 2501.05723)](https://arxiv.org/pdf/2501.05723)
- [RL from implicit human feedback via EEG (arXiv 2507.13171)](https://arxiv.org/html/2507.13171v1)
- [Learning to Assist: Collaborative VLAs (arXiv 2606.12475)](https://arxiv.org/abs/2606.12475)
- [Synthetic worlds for VPT (arXiv 2505.14366)](https://arxiv.org/html/2505.14366)
- [LeRF: reference frames for perspective taking](https://www.alphaxiv.org/abs/2609.36219)
- [Seeing Through Their Eyes: VPT in VLMs (arXiv 2409.12969)](https://arxiv.org/pdf/2409.12969)
- [AV-ALOHA active vision (arXiv 2409.17435)](https://arxiv.org/pdf/2409.17435)
- [ActiveUMI (arXiv 2510.01607)](https://arxiv.org/pdf/2510.01607)
- [BAVO-Bench occlusion recovery (arXiv 2609.37292)](https://arxiv.org/html/2609.37292v1)
- [Mecka raises $60M (Dealroom)](https://dealroom.co/news/160699-mecka-raises-60m-series-b-to-turn-human-motion-into-robot-training-data/)

---

## 2. The MVP: what the demo must prove

Three demo beats, each showing something the robot **cannot do from its own
cameras**:

1. **It sees what I see (occlusion).** An object is hidden from the robot's
   camera (behind a box, or under its own arm) but visible to the human.
   "Grab the blue cube." The robot gets it. With the human feed switched off,
   it can't find it.
2. **It knows what "my left" means (perspective).** Two identical objects.
   "Hand me the one on my left." The robot picks the one on the *human's*
   left, which is the robot's right.
3. **It notices when I react (implicit feedback).** The robot reaches for the
   wrong object, the human's hand moves into view or they say "no", and the
   robot stops and re-targets. Every correction is logged as a labeled example.

Closing slide: the dataset this produced (episodes, hours, corrections) plus
the A/B numbers from beats 1 and 2.

### The one experiment we run
For beats 1 and 2: **10 trials each, with vs. without the human view.** Report
success rate and the latency from the end of speech to the arm starting to
move. A real A/B number beats any claim on stage.

### Deliberately out of scope this week
- **Training a learned grasping policy.** The MVP uses a scripted top-down pick
  at a target point. The learned policy is the roadmap, and the paired data is
  what it will be trained on.
- **Eye tracking.** We use where the human camera points, plus hand entry and
  speech, as the attention and reaction signals. Real gaze needs Aria-class
  glasses later.
- **Robot-as-calibration-target.** We calibrate with a printed ArUco board;
  using the arm itself as the target (from its known joint positions) is a
  stretch goal.

---

## 2b. Phase 0: prove it in software before buying the arm (2–3 days)

The thesis is about perception and perspective, not grasping. So we can test
it without a robot and buy the SO-101 only once it passes.

**Track 1: simulation (MuJoCo, runs on a laptop).**
- Scene: a table, coloured blocks, occluding boxes, and an SO-ARM100/SO-101
  model. The SO-ARM100 is in MuJoCo Menagerie, and SO-101 URDFs are in
  TheRobotStudio's repo; check both before choosing.
- Two cameras: a fixed robot camera, and a "human head" camera that moves
  along scripted paths, looks at targets, and has a capsule body that can block
  its own view.
- Run the same pipeline as the real MVP: detector on both renders → table-frame
  object map → resolver → inverse kinematics pick in sim. Start with sim
  ground-truth segmentation to debug, then switch to the real detector so the
  numbers are honest.
- Benefit: hundreds of randomized trials overnight instead of 10 by hand.

**Track 2: real cameras, no arm (the sim-to-real check for perception).**
- A phone on a head strap plus a webcam, a real table, and an ArUco board.
- Measure the object-map position error against a ruler, and check that
  "my left" resolves correctly as a person walks around the table.
- The arm's intended target is drawn as an overlay on the live feeds.

**Go/no-go gate (buy the arm only if all pass):**

| Test | Pass |
|---|---|
| Occlusion A/B in sim, randomized scenes (N ≥ 200) | Success rate with the human view at least 30 points higher than without |
| Perspective ("my left/right", "closest to me") over random head poses | ≥ 90% correct |
| Real-camera object-map error (Track 2) | < 2 cm, within the SO-101 gripper's tolerance |
| End of speech → target chosen | < 1 s |

If it passes, the only risk left is grasping a block at a known point, which
is the well-solved part. The sim A/B chart also becomes a slide.

---

## 3. Hardware (decide on Day 0)

| Item | Pick | Fallback |
|---|---|---|
| Arm | SO-101 (LeRobot, open source, a few hundred dollars in parts) **only if parts arrive by Day 1** | Any arm the department already has (uFactory, myCobot, a lab arm). Last resort: a pan-tilt laser pointer (keeps beats 1–2, weakens the story). |
| Robot camera | Fixed USB webcam on the robot's side of the table | Depth camera (RealSense / OAK-D) if available; not required |
| Human camera | Phone on a head strap or chest mount, streamed over Wi-Fi (DroidCam / IP Webcam) | GoPro on a head strap |
| Shared frame | Printed ArUco board taped to the table | — |
| Compute | Laptop runs capture, detector, IK and voice. Lab H100/H200 runs the VLM over the existing SSH tunnel (`server/`). | Claude API through the existing worker |
| Props | 6–8 coloured blocks or markers (identical pairs for beat 2), a box for occlusion | — |

**Day 0 blocker:** who has an arm, and when. Everything else is cheap and
on hand.

---

## 4. Software architecture

```
 human cam (phone) ─┐                       ┌─ voice (Curio: Whisper, Kokoro TTS)
                    ├─► capture + timestamps│
 robot cam ─────────┘          │            │
                               ▼            ▼
                      calibration (ArUco: both cameras → table frame,
                               │  human cam re-solved every frame)
                               ▼
                      perception: open-vocab detector on BOTH views
                      → lift detections to table plane → fused object map
                        (each object: position, seen-by-robot?, seen-by-human?)
                               │
              utterance ──────►▼
                      referent resolver (VLM on lab GPU)
                      input: object map + human head pose + utterance
                      output: [TARGET:obj_id]  ← same idea as Curio's POINT tag
                               │
                               ▼
                      speculative motion: arm starts toward the top candidate
                      while the VLM confirms; re-targets on correction
                               │
                               ▼
                      IK pick/place at table coordinates (scripted)

          ── reaction monitor (hand enters view in human cam / "no"/"stop") ─► pause + re-ask
          ── episode logger (everything above) ──► LeRobot dataset format
          ── latency tracer (per stage) ──► numbers for the slide
```

### Components and what we reuse

| Component | Approach | Reuse from Curio |
|---|---|---|
| Voice in/out | As today | `clients/whisper_transcription_client.py`, `clients/kokoro_tts_client.py`, `hotkey.py`, `mic_capture.py` |
| VLM call | Local Qwen3-VL on the lab GPU, Claude as fallback | `clients/openai_llm_client.py`, `server/serve.sh`, `config.py` local backend |
| Structured target output | `[TARGET:id]` tag parsed like `[POINT:x,y:label]` | `point_parser.py` pattern |
| Memory ("where did I put the tape?") | Stretch | `memory_*` modules |
| Dashboard | Live view: both feeds, object map, which camera saw what | `lab_server.py` pattern (stdlib HTTP server) |
| Calibration | OpenCV ArUco board → `solvePnP` per camera | new |
| Detection | OWLv2 or Grounding DINO (open vocabulary) on the laptop GPU; YOLO-World if too slow | new |
| Lift to table | Ray from camera through the detection's base pixel, intersected with the table plane | new |
| Perspective | Deterministic geometry: express object positions in the human head frame, then left/right/near/far is a sign check. The VLM only picks *which relation* was said. | new |
| Reaction monitor | MediaPipe Hands on the human feed (hand enters the workspace) + keyword spotting on the transcript | new |
| Arm control | LeRobot drivers + IK; top-down pick at (x, y), fixed grasp height per object | new |
| Logger | LeRobot dataset format: both video streams, head pose, utterance, target, joint states, correction events | new |

### Inference engineering (what we measure and pitch)
- **Cascade:** the small detector runs every frame on both streams; the 32B
  VLM runs once per utterance on a compact text object map, not on raw frames.
- **Frame gating:** skip detection on human-camera frames that barely changed
  (pixel diff or ArUco pose delta); reuse object tracks instead.
- **Speculative motion:** borrowed from Curio's pointing. The arm starts toward
  the detector's top candidate as soon as the noun is transcribed, and the VLM
  confirms or redirects mid-motion. We pitch "speech end → arm moving".
- **Tracing:** timestamps at every stage. One latency waterfall chart goes on a
  slide.

---

## 5. Day-by-day (7 days, 4 people)

Owners: **A** hardware/arm · **B** cameras/calibration/perception ·
**C** voice, VLM, Curio integration · **D** data, metrics, dashboard, pitch.

| Day | Goal (exit criterion) | A | B | C | D |
|---|---|---|---|---|---|
| 0 | Arm sourced; demo script frozen | Source arm/parts | Order cameras/mounts; print ArUco | Fork Curio voice loop into `robot/` | Write 3-beat demo script and A/B protocol |
| 1 | Arm moves to commanded (x, y); both cameras stream | Assemble + calibrate arm | Two synced streams; ArUco pose for both | VLM endpoint up; prompt for `[TARGET:id]` | Episode logger skeleton |
| 2 | Live object map from both views | Arm ↔ table-frame calibration (touch 4 board corners) | Detector + lift to table + fusion | Resolver on a text object map (unit tests with fake maps) | Dashboard: feeds + map |
| 3 | Voice → pick works end to end (no perspective yet) | Scripted pick/place | Visibility flags per camera | Wire voice → resolver → arm | Latency tracer |
| 4 | **Beats 1 and 2 work** | Grasp reliability | Head-frame transform for "my left" | Perspective relations in the prompt | Run first A/B trials |
| 5 | **Beat 3 works**; speculative motion | Pause/re-target on interrupt | Hand-entry detector | Keyword interrupts + speculative start | Log corrections as labeled events |
| 6 | Robust demo | Lighting, prop layout, failure drills | Fallbacks (re-calibrate in 10 s) | Rehearse voice lines | Finish A/B (10×2×2); record backup video |
| 7 | Pitch ready | Rehearse | Rehearse | Rehearse | Deck: thesis, gap, demo, A/B, latency, dataset, roadmap |

**Cut order if behind:** speculative motion → beat 3 → dashboard polish.
Never cut beat 1, the A/B numbers, or the backup video.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Arm parts don't arrive | Day-0 decision; borrowed arm or the laser-pointer fallback |
| Head-camera pose jitters when the ArUco board is partly out of view | Big board, several boards around the table edge, smooth the pose over a few frames |
| Detector misses small or identical objects | Distinct coloured props; colour in the prompt; check detection on the dashboard before each trial |
| Lab GPU busy at demo time | Claude fallback via the existing worker; cache the prompt |
| Venue Wi-Fi breaks the phone stream | Phone tethered over USB; a travel router |
| Judge asks "isn't this just multi-camera fusion?" | Two answers. The second camera is *attached to the person*, so it's aimed at what matters and moves with the task. And it gives a reference frame ("my left") and a feedback signal (reactions), which a static camera can't. |

---

## 7. Moat story for the deck
1. **Data nobody has:** paired human-view + robot-view collaboration episodes
   with language, targets and corrections. Every deployment produces more.
2. **Systems:** real-time fusion of a *moving* human camera with robot control,
   inside a latency budget, on cheap hardware.
3. **Wedge:** benchtop collaboration (teaching labs, maker spaces, electronics
   assembly), where we already have a foothold through Curio's classroom lab
   mode.
4. **Roadmap:** train a perspective-aware policy on the paired data, and use
   human reactions as a learned reward.
