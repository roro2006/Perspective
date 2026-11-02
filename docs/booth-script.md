# Booth script: Curio proof-of-concept video

The video (`out/poc/curio_poc.mp4`, 56 s) carries no on-screen disclaimer. The
presenter says what it is, out loud, every time it's shown. The video is a
simulation. Say so in the first sentence. Never let someone watch it without
hearing that.

## Opening line (always, before or as the video starts)

> "Everything you're about to see is a simulation we built. The robots and their
> movements are scripted. The part that's real is the AI that decides *which*
> object you mean. It runs live on every scene, and we didn't train it on any of
> these settings."

Short version, for someone walking past:

> "This is a sim run. Watch the robot work out whose left you mean."

## Voice-over, timed to the video

| Time | On screen | Say |
|---|---|---|
| 0:00 | "On my left." / "Whose left?" | "Ask a robot for the thing on your left, and it has to work out whose left you mean." |
| 0:04 | Person and robot face to face | "Face to face, your left is its right. Most robots get this backwards." |
| 0:08 | Warehouse (red ✕, red ✕, then ✓) | "In this simulated warehouse, 'behind me' trips our model up. It gets it wrong even after the correction, and we left that in. The next order, 'on my left', it gets right." |
| 0:19 | Rescue dog | "Here the person and the robot are in different corridors. The model still works out the room from the firefighter's side." |
| 0:25 | Delivery drone | "Seen from above, it still picks the side the person meant." |
| 0:31 | Home humanoid | "Directions from the sofa, carried out across the room." |
| 0:37 | Underwater ROV | "The diver is tilted. The model works from their body, not the horizon." |
| 0:43 | Space free-flyer | "Upside down in orbit, it still finds their left." |
| 0:49 | Grid of six | "Five of six right, in settings it never trained on. All in simulation; the robot motion is scripted." |

## What to say if asked

- **"Is this real?"** "It's a simulation. The robots and how they move are
  scripted. The real part is the model that resolves 'my left': it looks at the
  person's head-camera view and picks the object, live, on each scene."
- **"Was it trained on these scenes?"** "No. It was trained on a tabletop task
  with coloured blocks. These six settings are new to it."
- **"Why did the warehouse fail?"** "It was only ever trained on left and right.
  'Behind me' is new, and the object behind you isn't in your camera's view at
  all. That's the next thing we'd train."
- **"Does a real robot do this?"** "Not yet. We have one simulated robot arm
  that picks the right block from the grounding about 57% of the time. The rest
  of the misses are grasping, not understanding. The next step is real hardware."
- **"How fast is it?"** "About a tenth of a second per decision on a desktop
  GPU."

## Rules for whoever is at the table

1. Say "simulation" before or during the first play for every new group.
2. Don't call the robots "ours" or imply they exist as products.
3. Point out the warehouse miss yourself; it builds trust.
4. If you print a still from the video, caption it "simulation".
