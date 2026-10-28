# Booth playbook (tabling event)

Visitors include engineers, recruiters and leaders from FAANG-scale companies.
They'll see dozens of tables. We get about **30 seconds** to earn a longer
conversation.

## What we want from the day (rank these before the event)
1. **Warm intros** to robotics, AR/glasses or ML-infra teams (names, emails,
   follow-up dates).
2. **Technical validation:** a senior engineer saying "that's real" is pitch
   material ("engineers from X asked to…").
3. **Hardware and data access:** e.g. research-glasses programs (Meta's Project
   Aria runs one for academic partners; check eligibility) or loaner arms.
4. Recruiting interest in the team: welcome, but not the goal.

Each conversation ends with one specific ask. Write down which one you got.

## The booth

```
            ┌──────────── monitor (three panes) ────────────┐
            │ robot view │   your view   │  twin / arm      │
            └───────────────────────────────────────────────┘
   [robot-side webcam]   ┌──── ArUco mat, blocks, box ────┐   [SO-101 if we have it]
                         └────────────────────────────────┘
   [head-cam strap on a hook]        [QR stand]        [laptop: replay mode backup]
   Sign above: "Your eyes, its hands." + small: "A robot that sees what you see."
```

- **The visitor is the demo:** hand them the head-cam strap. Nothing beats
  them saying "my left" and watching it work from *their* view.
- The teaser loop plays on a second screen or tablet when nobody is at the
  table.
- Keep wipes for the head strap; ask before putting it on anyone.

## Demo loops

**30-second hook (everyone):**
1. "Want to be the robot's eyes?" Hand over the strap.
2. "Say: 'the red one on my left.'" The twin or arm picks correctly.
3. Flip Curio off and repeat: wrong pick. "That's every robot today: it only
   sees from its own cameras."
4. "Curio fuses your view with the robot's. Want to see how?" → 2-minute loop.

**2-minute walkthrough (interested visitors):**
- Hidden-object beat: the box hides the blue block from the robot cam. "Your
  view finds it."
- Reaction beat: let them say "no" mid-reach and watch the re-target.
- One sentence on the data: "Every session records what you saw, what the robot
  saw and every correction. That dataset doesn't exist anywhere else."
- The ask (see below).

**10-minute deep dive (engineers who stay):**
- Architecture on the laptop (`README.md` diagram).
- Calibration: ArUco now, the robot arm as the calibration target next.
- The cascade and latency waterfall: a small detector every frame, the VLM per
  request, speculative motion.
- The research gap: what EgoMimic, EgoVerse and the gaze-failure work do and
  don't do (`docs/pivot-shared-perspective.md` §1).
- Ask what *they'd* measure. Write it down; it's free product feedback.

## Talk tracks by company
Tailor the 2-minute loop's last sentence and the ask. Ask the visitor what team
they're on before you pitch; don't guess from the badge.

| If they work on… | Lead with | Ask |
|---|---|---|
| AR / smart glasses | "Your glasses become a robot's eyes. That's a new reason to wear them." | An intro to a research or partnerships contact; research-kit access |
| Robotics / embodied AI | "Your models are trained robot-centric. Here's the collaborator's view, and paired data you don't have." | Feedback on the dataset format; an intro to a data or research team |
| Warehouse / logistics robotics | "Humans and robots share stations. The person's view covers what the robot can't see." | Which station tasks hurt most; an intro to an applied team |
| ML infra / inference | "A moving camera in a robot control loop under a 1-second budget. Here's our cascade." | Critique of the latency stack; contacts on on-device inference |
| Recruiters / non-technical | The 30-second hook only, then the teaser QR. | Their card; who in the org builds robots or glasses |

## Lead capture
- QR on the stand → a form (name, company, team, "what caught your eye",
  follow-up yes/no). Opt-in only; no auto-adding to anything.
- Back each one up with a quick paper note: name, team, what they asked, our
  ask, follow-up date.
- **Same night:** a follow-up email to every "yes" with the teaser link and one
  line referencing their question.

## Staffing (3–4 people)
| Role | Job |
|---|---|
| Demo driver | Runs the laptop, resets blocks, flips Curio on and off |
| Host | Greets, hands over the strap, runs the 30-second hook |
| Closer | Takes the deep dives and the asks, logs leads |
| (Floater) | Watches the queue, recalibrates, swaps batteries, eats so others can |

Rotate every 45 minutes. The host role is the most tiring.

## Failure modes and resets
| Problem | Fix (under 15 s) |
|---|---|
| Head-cam pose jumps | Ask the visitor to look at the mat for a second (the marker re-acquires); press `R` to recalibrate |
| Lighting changes (windows, spotlights) | Bring a clip light; lock the camera exposure at setup |
| Wi-Fi to the phone dies | USB-tethered phone camera; travel router as backup |
| Lab GPU unreachable | The VLM falls back to the rule-based resolver (handles "left/right/nearest/colour") |
| Arm misbehaves | Unplug the arm; the twin keeps the demo alive |
| Everything dies | Replay mode on the backup laptop + the teaser loop |

## Packing list
- [ ] Monitor + HDMI, laptop + charger, backup laptop with replay mode
- [ ] Robot-side webcam + mount/clamp; phone + head strap + USB cable; power bank
- [ ] ArUco mat (laminated) + spare print; block set (×2 identical reds, a
      blue, a green) + occluding box
- [ ] SO-101 + power supply (if we have it); clip light; power strip; tape
- [ ] Sign, QR stand, business cards or a one-pager; wipes; pens and lead sheet
