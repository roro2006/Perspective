# Pitch competition strategy (finals)

Assumed format, from the earlier deck: **3-minute pitch with a live demo**,
then Q&A. Confirm the Q&A length, slide-count limits and whether you get a
table or power on stage. Those change the demo plan.

## What the judges must leave believing
1. **The problem is real:** robots don't share your perspective, and that's why
   working next to them is clumsy.
2. **The insight is non-obvious:** the human's eyes are the best sensor in the
   room, and nobody is using them this way.
3. **It works:** we saw it happen and we saw the A/B numbers.
4. **It compounds:** every use produces data competitors can't get.
5. **This team ships:** we built Curio for screens, then this, fast.

Everything in the 3 minutes serves one of those five. Cut anything that doesn't.

## The live demo (what's actually on stage)

**Digital-twin mode.** This works before we own an arm and survives the arm
breaking.
- A real tabletop with blocks and an ArUco mat, plus a webcam on the "robot"
  side.
- The presenter wears a phone on a head strap (the human camera).
- The screen shows three panes: **robot view | your view | the twin**. The twin
  is the MuJoCo SO-101 acting on the scene that Curio built from the two real
  cameras.
- If the SO-101 has arrived and is reliable, the real arm moves too, and the
  twin becomes the "what Curio thinks" pane.

**The demo beat (about 40 seconds):**
1. Two identical red blocks, one on each side. A box hides a blue block from
   the robot's camera.
2. The presenter says: *"Hand me the red one on my left."*
3. **Curio off:** the twin reaches for the robot's left, the wrong block. The
   audience sees it fail.
4. Flip the switch to **Curio on:** the twin reaches for the presenter's left.
5. *"And the blue one."* With only the robot camera, nothing is found. With
   Curio on, the human view finds it behind the box.

**Fallback ladder (decide on stage, never apologize):**
1. Live demo fails → play the same beat from the teaser video (cued on a hidden
   slide).
2. The camera stream drops → switch the demo laptop to the recorded-replay mode
   (same UI, recorded frames).
3. Never debug on stage. A five-second hesitation is the maximum before
   switching down the ladder.

## The 3-minute script

Speaker A presents and Speaker B operates the demo. Times are cumulative.
About 380 spoken words total, which leaves room for the demo.

| Time | Slide / stage | Spoken (Speaker A) |
|---|---|---|
| 0:00–0:20 | **Cold open, live.** A wears the head cam. No slide yet. | "Hand me the red one on my left." *(The twin picks the wrong one.)* "Every robot today makes that mistake, because it only sees the world through its own eyes. Not yours." |
| 0:20–0:45 | Slide 1, **Problem.** Photo of a person and a cobot at a bench. | "Robots are moving onto workbenches, into labs and into warehouses, right next to people. When two people work together, they share a perspective. They know what you can see, what 'my left' means, and when your face says 'no, not that.' Robots have none of that." |
| 0:45–1:05 | Slide 2, **Insight.** "The best sensor in the room is the person." | "Your eyes are already pointed at exactly what matters, and cameras on people are becoming normal: glasses, headsets, a phone on a strap. So we asked: what if the robot could see through *your* eyes?" |
| 1:05–1:45 | **Live demo** (three panes on screen). B flips Curio on. | "This is Curio. It fuses my view with the robot's, live." *"Hand me the red one on my left."* (Correct.) *"And the blue one."* "The robot's camera can't see it; it's behind the box. But I can, so now Curio can." |
| 1:45–2:10 | Slide 3, **Results.** A/B bar chart (sim plus real; numbers in `results.md`) and a latency waterfall. | "We tested it. In random cluttered scenes, with realistic calibration error, success goes from 50 to 86 percent with the human's view (simulation; update with real-camera numbers). End of speech to arm moving: under a second, because a small model watches every frame and the big model only runs when you speak." |
| 2:10–2:35 | Slide 4, **Moat.** Flywheel: use → paired data → better robot → more use. | "Every time someone works with Curio, we record something nobody else has: what the human saw, what the robot saw, what was asked and every correction. Egocentric-data companies record humans alone. Teleop datasets record robots alone. We record collaboration." |
| 2:35–2:50 | Slide 5, **Team + origin.** Curio-for-screens screenshot. | "We started with Curio for screens, an AI that sees your screen and points. It taught us that an assistant is only useful if it sees what you see. Now we're bringing that to robots." |
| 2:50–3:00 | Slide 6, **Close.** Logo, "Your eyes, its hands," the ask. | "Curio: a robot that sees what you see. *[The ask: what you want from this room.]*" |

Rules for the script:
- Fill in the X/Y numbers from the actual A/B runs. Never round up.
- The ask must be specific: money amount, pilot partners, or an intro to the
  robotics team at a named company.
- Rehearse to 2:50 so there's 10 seconds of buffer.

## Slides (6 + backups)
1. Problem: one photo, one sentence.
2. Insight: one sentence, large.
3. Results: two charts only (A/B, latency).
4. Moat: the flywheel diagram, plus a comparison of whose data covers what
   (ego-data companies: human only; teleop: robot only; Curio: both, together).
5. Team and origin.
6. Close and the ask.

**Backup slides** (after the end slide, for Q&A only):
- Architecture diagram.
- Research landscape table with citations.
- Market and wedge (benchtop → lab automation → platform).
- Business model.
- Roadmap: digital twin → real arm → learned policy on paired data → reactions
  as reward.
- Hidden slide: the demo-beat video (fallback).

## Q&A bank (practice every one out loud)

| Likely question | Answer (keep it under 20 seconds) |
|---|---|
| Isn't this just adding a second camera? | No. A static camera doesn't move with the task or know who's speaking. The human's camera is aimed at what matters, carries the speaker's reference frame ("my left") and captures reactions. A wall camera gives you none of that. |
| Do people really want to wear a camera? | In labs and assembly, many already wear safety glasses, and smart glasses are going mainstream. It also works with a phone on a strap today. And the model learns from the human's view during training, so the long-term goal is a robot that's better even when nobody wears one. |
| Why won't Physical Intelligence or Google just do this? | They build robot-centric foundation models and buy data. We're the collaboration layer and the data source. Best case for them, we're a partner. Best case for us, they're customers. |
| Mecka and others already sell egocentric data. | They record humans alone, to teach motor skills. We record humans and robots working together, with language and corrections. That's a different dataset for a different problem. |
| What's your moat if someone copies the software? | The paired data accumulates with every deployment, and the real-time fusion stack is hard to get right. Software can be copied; months of collaboration data can't. |
| Privacy? A camera on someone's face? | Processing is local; frames used for control never leave the bench. Recording data is opt-in per session, and the system runs without recording. |
| How accurate is the calibration of a moving head camera? | Today it uses ArUco markers on the table, re-solved every frame. We measured a position error of *[X cm]*. Next, the robot's own arm becomes the calibration target, since it knows its exact joint positions. |
| What happens when the human is wrong? | The robot still checks with its own cameras before grasping. The human's view adds evidence; it doesn't override safety. |
| Is the demo real or simulated? | Perception is real: real cameras, real table. The arm in the twin is simulated until our SO-101 arrives. The software path is identical. |
| Who pays? | Wedge: teaching labs and electronics benches (SDK per robot). Scale: robot makers licensing the module, and model labs buying paired data. |
| Why this team? | We built Curio for screens: voice, vision and pointing, self-hosted inference on H100s, a classroom lab mode. This is the same problem in the physical world. |
| What would you do with the money / prize? | *[Specific: arms, glasses, a pilot with N labs, and what milestone it buys.]* |

## Delivery
- **Roles:** A speaks, B runs the demo and fallbacks, C watches the clock and
  holds the backup laptop with the video cued.
- **Eye contact over slides.** Slides are pictures, not notes.
- **Say the numbers slowly.** They're the most credible thing you'll say.
- **End on the ask, then stop talking.**

## Rehearsal plan (finals week)
| When | What |
|---|---|
| Day 1 | Read-through for timing only; cut to 2:50. |
| Day 2 | Full run with the demo, ×3. Log every demo failure and its cause. |
| Day 3 | Fallback drill: B triggers a random failure each run, and A recovers without stopping. |
| Day 4 | Q&A drill: teammates fire the bank plus new questions, with a 20-second limit. |
| Day 5 | Dress rehearsal in front of outsiders (TAs, friends), timed, with feedback. |
| Day of | One run in the room if allowed. Check lighting, Wi-Fi, power and the screen resolution for three panes. |
