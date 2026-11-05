# Curio layer: one goal interface for any robot policy

Your policy already knows how to pick things up, walk to a point, or follow an instruction.
What it can't know is which object a person means when they say "the red one on my left"
from the other side of the table. Their left is your robot's right. Curio answers that one
question and hands the answer over in the input format your policy already takes. It
doesn't plan or control anything.

## The call

```python
from curio_robot.layer import resolve, ring, pose, robot_words

goal = resolve(
    "Hand me the red one on my left",  # what the person said
    head_frame,                         # CameraFrame from their head camera, or None
    {"robot": robot_frame},             # the robot's cameras, name -> CameraFrame
    object_map,                         # ObjectMap: ids, colours, world positions
    speaker,                            # their Belief (memory tracking), or None
)
```

`resolve` sends one request to a served vision-language model (Qwen3-VL-4B with Curio's
LoRA adapter, behind an OpenAI-compatible `/v1/chat/completions` on
`http://127.0.0.1:8081/v1`). The request is the person's head-camera image with a letter
drawn on every mapped object, and each object's side as both the person and the robot see
it. The model answers with one letter, and the first token's logprobs give a probability
for every letter. Without a head frame, the same request is made on the robot's own view.
Optional keyword arguments: `grounder` (any object with `score(request)`), `base` (the
robot base pose, below), `p_own_frame` (the person's learned habit of meaning their own
left), `movers` (who moved which object), and `tau` (the ask threshold, 0.95 by default).
If the model gives no answer, `resolve` raises `GroundingUnavailable` and doesn't guess.

## The Goal

| field | meaning |
|---|---|
| `target_id` | the object id from your object map that the person means |
| `target_xyz_robot` | its position in the robot base frame: x forward, y left, z up, metres |
| `frame` | `"speaker"` if it was grounded on the person's view, `"robot"` if on the robot's cameras only |
| `confidence` | the model's probability for the chosen object |
| `alternatives` | `(id, probability)` for every other marked object, most likely first |
| `ask` | a yes/no question to ask before acting (e.g. "The one on your left?"), or None |
| `note` | something to say first when the object was moved while the person couldn't see it ("I moved it; it's now on your right, nearer you."), or None |
| `adapter` | the model adapter that answered, read from `curio_robot/flywheel/registry.json` |

`ask` is set when `confidence` is under 0.95 and some question is worth at least 0.1 bit
about the target. In that case, ask the question, update, and call `resolve` again, or
use `curio_robot.clarify.decide` to run the whole exchange. `note` needs `speaker`.
Without a belief tracker there is nothing to compare against.

The base frame comes from `base=Base(position, yaw)`, the robot base pose in the world
frame of your object map. If you don't pass it, Curio puts the base directly below the
robot camera at world z = 0, facing the camera's horizontal heading. `Goal.to_dict()` gives
plain JSON.

## Three adapters

**ring** (`ring(goal, robot_frame)`): the robot image with a magenta ring (radius 9 px,
2 px wide) around the target's pixel. This is the visual goal our SmolVLA pick policies
were trained on. It matches the training and evaluation marker pixel for pixel.

**pose** (`pose(goal)` or `pose(goal, base)`): the target position as a 3-vector, in the
robot base frame or, given the base, in the world frame. Motion controllers start from
this. Our G1 and Go2 walkers, wheeled AMR, drone, ROV and space flyer each turn it into
their own stand-off or approach point.

**words** (`robot_words(goal, object_map)`): an instruction in the robot's own frame for
language-conditioned policies such as pi0, for example `"pick up the red block on your
left"`. The rule is deterministic. It names the colour, adds nothing if the colour is
unique, otherwise adds "on your left" or "on your right" (the robot's left) when that
beats the next look-alike by 2 cm, and otherwise "closest to you" or "furthest from you".
The words never mention the person, so a policy that has never seen the head camera can
follow them. `check(words, object_map)` reads them back with the same rule. It returns
the one object they pick out, or None when they don't single one out. Verb and noun are
parameters (`verb="pick up"`, `noun="block"`).

## Example

Robot camera at (-0.2, 0, 0.55) facing +x, person across the table facing -x, two red
blocks at y = +0.08 (the robot's left) and y = -0.08 (the person's left). The
probabilities shown are illustrative:

```python
goal = resolve("Hand me the red one on my left", head, {"robot": robot}, om, None)
# Goal(target_id='red_2', target_xyz_robot=(0.45, -0.08, 0.0125), frame='speaker',
#      confidence=0.998, alternatives=(('red_1', 0.002), ('blue_1', 0.0)),
#      ask=None, note=None, adapter='r16-v4')
robot_words(goal, om)   # 'pick up the red block on your right'
pose(goal)              # array([ 0.45, -0.08,  0.0125])
ring(goal, robot)       # robot.rgb with the ring on red_2
```

## Measured

Gate results for this layer are in `out/t5_27/gate.md`. Card T5-27 covers equivalence with
the existing code paths, words a camera-free rule can resolve, and a generic reader
following the words, all on the 350-item bench5c benchmark. It also covers the latency of
`resolve` against the live server.
