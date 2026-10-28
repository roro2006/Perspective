# Curio: the narrative

The source of truth for every pitch, video, booth conversation and slide. If a
claim isn't here, don't say it on stage. Items marked **[VERIFY]** need a real
number or source before they go on a slide.

## One-liner
**Curio is a robot that sees what you see.** It fuses your first-person view
with the robot's own cameras, so it can work alongside you the way a person
would.

Short version for the booth sign: *Your eyes, its hands.*

## The story in five beats

1. **Robots are moving next to people.** Cobots in labs and factories, arms on
   workbenches, humanoids in warehouses. The interface is shifting from code to
   conversation: "hand me that."
2. **But every robot sees the world only through its own cameras.** When you
   work with a person, you rely on shared perspective. They know what you can
   see, what "on my left" means, and when your face says "no, not that one."
   Robots have none of that. They miss objects they can't see, misread
   instructions given from your side of the table, and keep going when you
   flinch.
3. **The best sensor in the room is the person.** Your eyes are already
   pointed at exactly what matters. Cameras on people are becoming normal:
   smart glasses, headsets, a phone on a strap.
4. **Curio connects the two.** Software that fuses a co-worker's live
   first-person view with the robot's cameras in real time:
   - **It sees what you see:** it finds objects its own cameras can't.
   - **It understands your perspective:** "my left" means *your* left.
   - **It notices your reactions:** a reach, a "no", a stare, and it stops and
     re-targets.
5. **Every collaboration becomes training data nobody else has:** paired
   human-view and robot-view episodes, with the request, the target and every
   correction. That's our moat, and the next generation of robot models will
   need it.

## Where we came from (continuity story)
We built **Curio for screens**: a voice AI that watches your screen, answers
questions and points at what it's talking about, with a classroom lab mode for
TAs. Building it taught us one thing: **an assistant is only useful if it sees
what you see.** The screen was the first place we could test that. The physical
world is the bigger one.

What carries over: the voice loop (speech → vision-language model → spoken
answer), grounding language to a point, conversation memory, our self-hosted
inference stack (Qwen3-VL on H100/H200 workstations), and the lab-mode
classroom foothold.

## Why now
- Cameras on people are going mainstream: consumer smart glasses, mixed-reality
  headsets, Meta's Project Aria research glasses. **[VERIFY: a shipment or
  adoption number for smart glasses]**
- Robots now take orders in plain language. Vision-language-action models (π0,
  Gemini Robotics, GR00T) made "hand me that" a real interface.
- Those models are trained **robot-centric**: robot cameras, robot actions. Human
  first-person video is used today to teach robots *motor skills*. Nobody uses
  the **co-worker's** view to make the robot a better *collaborator*. (Research
  summary: `docs/pivot-shared-perspective.md` §1.)

## The technology (the inference story)
Fusing a camera that is *moving on someone's head* with robot control, live,
on cheap hardware:
- **Shared frame:** both cameras are localized to the table every frame (ArUco
  markers now, the robot's own arm as the calibration target next).
- **Cascade:** a small detector runs on every frame from both cameras. The big
  vision-language model runs once per request, on a compact object map instead
  of raw video.
- **Speculative motion:** the arm starts toward the most likely target the
  moment you name it, and the big model confirms or redirects mid-motion.
  Borrowed from Curio's speculative pointing.
- **Latency budget:** from end of speech to arm moving in under a second. We
  show the waterfall chart.

## The moat
1. **Data:** paired collaboration episodes (human view, robot view, actions,
   language, corrections). Ego-data companies record humans alone; teleop
   datasets record robots alone. Every Curio deployment produces this data for
   free.
2. **Systems:** real-time fusion of a moving human camera into a robot's control
   loop. This is hard engineering, and we've built the stack.
3. **Position:** we are the layer *between* glasses and robots. Glasses makers
   want reasons to wear them; robot makers want better collaboration. Curio
   works with both.

## Who it's for
- **Wedge:** benchtop collaboration, meaning teaching labs, maker spaces and
  electronics assembly. Structured, repetitive work, $300 arms can do it, and we
  already have a foothold through Curio's lab mode.
- **Next:** lab automation and light assembly lines, wherever humans and cobots
  share a bench.
- **Platform:** an SDK for robot makers (the "see what the human sees" module)
  and a data partnership for foundation-model labs.

## Business model (pitch-level)
- SDK license per deployed robot.
- Data licensing: paired collaboration episodes for robot foundation-model
  companies. The market exists: ego-data companies like Mecka reportedly run
  at $100M+ annual revenue. **[VERIFY before quoting]**

## What we can honestly claim today
- The research gap (with citations) and why it matters.
- A working simulation of the full loop with the A/B results: with vs. without
  the human view.
- Real-camera perception on a real table (object map accuracy, "my left"
  accuracy). **[fill in once measured]**
- Curio for screens: built and working, with classroom lab mode.

## What we don't claim
- That we invented perspective-taking or gaze-based failure detection. Those
  have research precedent; the combined system and the dataset are new.
- A trained end-to-end policy. The MVP uses scripted picks, and the paired data
  is what we'll train on.
- Customers or pilots we don't have.
