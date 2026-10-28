# Teaser video (2:30)

**Purpose:** played before or after the pitch, looped at the booth (muted, with
captions), and linked from the QR code. It must work **with the sound off**:
burned-in captions, big on-screen text.

**Tone:** confident and curious, not hype. Real footage beats rendered
footage. Show the sim only where it explains something real footage can't
(the robot's view, the object map, the A/B).

**Voiceover:** about 330 words at a calm pace. One narrator from the team.

## Storyboard

Sources: **REAL** = filmed at the bench; **POV** = head-cam footage;
**SIM** = rendered by `curio-sim shot`; **SCREEN** = screen recording;
**CARD** = text on black.

| # | Time | Visual | Source | VO / on-screen text |
|---|---|---|---|---|
| 1 | 0:00–0:08 | POV: hands at a workbench, two identical red blocks. A robot arm on the far side. | POV | VO: *"Hand me the red one on my left."* |
| 2 | 0:08–0:14 | The arm reaches for the wrong block (the robot's left). Hold on it. | REAL or SIM | Text: **"Wrong one."** |
| 3 | 0:14–0:20 | CARD | CARD | **Robots don't see what you see.** |
| 4 | 0:20–0:40 | Split screen: robot camera view vs. human POV of the same moment. The blue block is visible only in the POV pane. | SIM + POV | VO: *"Every robot sees the world through its own cameras. When it works next to you, it doesn't know what you can see, what 'my left' means, or when you're about to say no."* |
| 5 | 0:40–0:55 | Quick cuts: people working together at benches; one glances, the other hands something over. | REAL (stock or staged) | VO: *"People work together because they share a perspective. A glance, a point, 'that one.'"* |
| 6 | 0:55–1:10 | Curio-for-screens footage: the pixel companion flies to a button on screen. | SCREEN | VO: *"We built Curio, an AI that sees your screen and points at what it's talking about. It taught us one thing: an assistant is only useful if it sees what you see."* |
| 7 | 1:10–1:18 | CARD | CARD | **What if your robot could see through your eyes?** |
| 8 | 1:18–1:38 | Three-pane dashboard: robot view, human view, twin. The object map fills in from both views, coloured by which camera sees each object. | SCREEN + SIM | VO: *"Curio fuses your first-person view with the robot's cameras, live. Everything you see goes into one shared map of the table."* Text: **1. It sees what you see.** |
| 9 | 1:38–1:50 | Replay of shot 1, now correct: the arm takes the block on *your* left. | REAL or SIM | Text: **2. It understands your perspective.** VO: *"'My left' means your left."* |
| 10 | 1:50–2:02 | The arm reaches toward the wrong block; the hand enters the frame with a "no". The arm stops and re-targets. | REAL or SIM + POV | Text: **3. It notices when you react.** VO: *"And when you react, it stops and corrects itself."* |
| 11 | 2:02–2:15 | Animated flywheel: use → paired data → better robots. Then the A/B chart animating in. | SCREEN (motion graphics) | VO: *"Every collaboration becomes data no one else has: what you saw, what the robot saw, and every correction. With your view, the robot finds hidden objects X times more often."* |
| 12 | 2:15–2:25 | POV: the person takes the handed block and nods. | POV | VO: *"Robots that work with you, not just near you."* |
| 13 | 2:25–2:30 | Logo card | CARD | **Curio: your eyes, its hands.** QR code + URL. |

Replace "X times" in shot 11 with the measured number, or cut the clause.

## Shot list (what to capture)

**At the bench (one 2-hour session, good daylight or two softboxes):**
- [ ] POV head-cam takes of shots 1, 9, 10 and 12 (phone on a head strap at
      60fps, locked exposure). Five takes each.
- [ ] Wide shots of the bench showing the person, head cam and arm (or the
      monitor with the twin if there's no arm yet).
- [ ] Close-ups: a hand placing blocks, the ArUco mat, the head-cam strap
      going on.
- [ ] Human-human b-roll for shot 5: two teammates assembling something and
      handing parts over.

**Rendered (`curio-sim shot …`, 1080p):**
- [ ] Shot 2 (wrong pick) and shot 9 (correct pick) from the same camera angle.
- [ ] Shot 4: robot view and human view of the occluded-block scene.
- [ ] Shot 8: object map build-up, coloured by viewpoint.
- [ ] Shot 10: reach, stop, re-target.

**Screen recordings (OBS at 1080p60):**
- [ ] Shot 6: Curio-for-screens pointing at a UI element (use existing
      footage if the deck already has it).
- [ ] Shot 8: the live dashboard during a real session.
- [ ] Shot 11: the A/B chart (export from the results notebook) and the
      flywheel (slide animation).

## Production schedule

| Day | Work |
|---|---|
| 1 | Lock the script; record the VO scratch track on a phone; rough animatic using cards. |
| 2–3 | Capture the bench footage and sim renders. |
| 4 | Edit in DaVinci Resolve (free): picture lock. |
| 5 | Final VO, music (royalty-free, credited), captions, colour, export 1080p + a square cut for social. |

**Exports:**
- `teaser_1080p.mp4` (pitch/QR)
- `teaser_booth_loop.mp4` (no VO, captions, loops cleanly)
- `demo_beat_40s.mp4` (the stage fallback, shots 1, 9 and 8)
