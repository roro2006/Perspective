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

---

# Booth script: versatility reel ("Whatever you mean")

Video: `out/poc_reel/curio_reel.mp4` (76 s; 720p copy `curio_reel_720p.mp4`).
Like the first video, it has nothing on screen saying what is real. The
presenter says it. This reel needs one more disclosure than the first: only
**two of the nine scenes** run the real model. The other seven are scripted, to
show where the system is going.

| Scene | Ability | What drives it |
|---|---|---|
| 1 Warehouse | see (what's hidden from you) | **live model** |
| 7 Underwater ROV | your frame (your left) | **live model** |
| 2-6, 8, 9 | reach, look, point, remember, place, move, learn | scripted: roadmap |

## Opening line (always)

> "This is a simulation. Two of these scenes run our real model live: the
> warehouse and the diver. The other seven are scripted. They show where we're
> taking it next."

## Voice-over, timed to the video (approximate)

| Time | On screen | Say |
|---|---|---|
| 0:00 | "The one on my left." ... "Where did I put it?" | "People don't give robots coordinates. They say things like this, from where *they* stand." |
| 0:05 | **See**: warehouse, tote hidden behind boxes | "This one is live. 'The tote I can't see.' The model works out which tote is hidden from the picker." |
| 0:12 | Reach: kitchen | "From here on it's scripted, showing where we're going. 'The one I can't reach': the robot judges reach from the person's body." |
| 0:19 | Look: workshop | "'Bring me that.' The person only looks at it." |
| 0:25 | Point: corridor | "Pointing, read from the firefighter's eyes, not their arm." |
| 0:32 | Remember: living room | "The head camera saw where the keys went, so the robot can answer 'where did I put them?'" |
| 0:39 | Place: assistive table | "The reverse of fetching: put it by *my* right hand." |
| 0:46 | **Face**: diver | "Live again. The diver is tilted, and the model still finds their left." |
| 0:53 | Move: hallway | "Scripted. 'Stay on my left', even through a turn." |
| 1:00 | Learn: lab bench | "Scripted. It's corrected once, and after that 'the usual' means the torque wrench." |
| 1:06 | Grid of nine | "Two working live today, seven on the roadmap. One wearable camera, any robot." |

## What to say if asked

- **"Which parts are real?"** "The warehouse and the diver scenes: the model
  sees the person's head-camera view and picks the object live. The rest is
  scripted to show the roadmap. All of it is simulation."
- **"Did the live scenes ever get it wrong?"** "In this run, the hidden tote was
  a close call: about 53% confidence. The diver's left was certain. In the
  first video it got 'behind me' wrong, and we left that in."
- **"Why does the hidden-tote scene have no red X?"** "Because the robot's-eye
  reading got it right too, in that run. We didn't stage a failure."
- **"Learning habits: is that real?"** "Our system already learns *whose* left a
  person means from their corrections. Learning which object 'the usual' means
  isn't built yet. That scene is roadmap."
- **"What's hardest on the list?"** "Memory and the moving frame. Both need the
  model to keep track over time, not just answer once."

## Rules for whoever is at the table

1. Say "simulation" and "two live, seven scripted" to every new group.
2. Never describe a scripted scene as something it does today.
3. If someone films the screen, repeat the opening line on camera.
