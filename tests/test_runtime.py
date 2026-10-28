import queue
import time

import numpy as np

from curio_robot.events import ArmEvent, Heard, Interrupt, VlmReply
from curio_robot.fusion import Tracker
from curio_robot.geometry import look_at
from curio_robot.perception import yolo_detections
from curio_robot.runtime.orchestrator import Orchestrator, confirmation
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import ROBOT_CAM, CameraPose, ObjectMap, WorldObject
from curio_robot.vlm import VlmAnswer, VlmResolver
from curio_robot.voice.audio import EnergyVAD
from curio_robot.voice.scripted import PrintedVoice
from curio_robot.voice.stt import StreamingRecognizer


def _pose(eye, look):
    eye = np.asarray(eye, float)
    return CameraPose(eye, look_at(eye, np.asarray(look, float)))


ROBOT = _pose([-0.2, 0, 0.5], [0.25, 0, 0])  # robot's left is +y
HUMAN = _pose([0.7, 0, 0.4], [0.25, 0, 0])  # human's left is -y


def _obj(oid, color, xy):
    return WorldObject(oid, color, np.array([*xy, 0.0125]), {ROBOT_CAM})


class FakeMotion:
    def __init__(self):
        self.goals: list[tuple[np.ndarray | None, str]] = []

    def go(self, goal, tag):
        self.goals.append((np.asarray(goal), tag))

    def stop(self):
        self.goals.append((None, "stop"))


class FixedVlm:
    def __init__(self, object_id):
        self.object_id = object_id

    def resolve(self, text, object_map):
        return VlmAnswer(self.object_id, "", "fake", 1.0)


REDS = [_obj("red_1", "red", (0.25, 0.08)), _obj("red_2", "red", (0.25, -0.08))]


def _brain(vlm=None, objects=REDS):
    maps: Latest[ObjectMap] = Latest()
    maps.put(ObjectMap(objects, ROBOT, HUMAN))
    bus: queue.Queue = queue.Queue()
    motion, voice, tracer = FakeMotion(), PrintedVoice(quiet=True), Tracer()
    brain = Orchestrator(bus, maps, motion, voice, tracer, vlm=vlm)
    return brain, bus, motion, voice, tracer


def _say(brain, text, final=True):
    t = time.perf_counter()
    brain.handle(Heard("start", "", t))
    words = text.split()
    for i in range(1, len(words) + 1):
        brain.handle(Heard("partial", " ".join(words[:i]), t))
    brain.handle(Heard("end", "", t))
    if final:
        brain.handle(Heard("final", text, t))


# ---- tracking -------------------------------------------------------------


def test_tracker_keeps_ids_when_detection_order_changes():
    tracker = Tracker()
    a, b = _obj("red_1", "red", (0.2, 0.1)), _obj("red_2", "red", (0.2, -0.1))
    first = tracker.update(ObjectMap([a, b], ROBOT, HUMAN))
    ids = {o.object_id: o.position[1] for o in first.objects}
    # Same blocks, detected in the other order and nudged: ids must follow the blocks.
    swapped = [_obj("red_1", "red", (0.2, -0.095)), _obj("red_2", "red", (0.2, 0.105))]
    second = tracker.update(ObjectMap(swapped, ROBOT, HUMAN))
    for o in second.objects:
        assert np.sign(o.position[1]) == np.sign(ids[o.object_id])


def test_tracker_coasts_then_drops_unseen_objects():
    tracker = Tracker(max_missed=2)
    tracker.update(ObjectMap([_obj("blue_1", "blue", (0.2, 0))], ROBOT, HUMAN))
    for _ in range(2):
        m = tracker.update(ObjectMap([], ROBOT, HUMAN))
        assert [o.seen_by for o in m.objects] == [set()]
    assert tracker.update(ObjectMap([], ROBOT, HUMAN)).objects == []


# ---- orchestrator: the pipelining -----------------------------------------


def test_arm_starts_before_speech_ends():
    brain, _, motion, _, tracer = _brain()
    _say(brain, "hand me the red one on my left", final=False)
    tags = [tag for _, tag in motion.goals]
    assert tags[0] == "approach"  # "the red one": drift toward both reds
    assert tags[-1] == "speculative"  # "... on my left": commit before speech ends
    assert np.allclose(motion.goals[-1][0][:2], (0.25, -0.08))  # the speaker's left
    assert tracer.has(1, "speculative_target") and not tracer.has(1, "final_transcript")


def test_vlm_redirects_mid_motion():
    brain, bus, motion, voice, tracer = _brain(vlm=FixedVlm("red_1"))
    _say(brain, "hand me the red one on my left")
    assert motion.goals[-1][1] == "speculative"
    brain.handle(bus.get(timeout=2))  # the VLM's reply
    assert motion.goals[-1][1] == "vlm"
    assert np.allclose(motion.goals[-1][0][:2], (0.25, 0.08))
    assert tracer.has(1, "redirected")
    assert voice.lines == ["Got it, the red one on your left."]


def test_unavailable_vlm_keeps_the_rule_target():
    brain, bus, motion, voice, _ = _brain(vlm=VlmResolver([]))
    _say(brain, "the red one on my left")
    reply = bus.get(timeout=2)
    assert isinstance(reply, VlmReply) and reply.source == "unavailable"
    n = len(motion.goals)
    brain.handle(reply)
    assert len(motion.goals) == n
    assert voice.lines == ["Got it, the red one on your left."]


def test_stale_vlm_reply_is_ignored():
    brain, _, motion, _, _ = _brain(vlm=FixedVlm("red_1"))
    _say(brain, "the red one on my left")
    _say(brain, "the red one on my right")
    n = len(motion.goals)
    brain.handle(VlmReply(1, "red_1", "", "fake", time.perf_counter()))
    assert len(motion.goals) == n


def test_no_the_other_one_corrects_and_logs():
    brain, _, motion, voice, _ = _brain()
    _say(brain, "the red one on my left")
    assert np.allclose(motion.goals[-1][0][:2], (0.25, -0.08))
    _say(brain, "no the other one")
    assert any(goal is None for goal, _ in motion.goals)
    assert motion.goals[-1][1] == "correction"
    assert np.allclose(motion.goals[-1][0][:2], (0.25, 0.08))
    assert voice.lines[-1] == "Sorry, the other red one."


def test_hand_in_workspace_stops_the_arm():
    brain, _, motion, voice, _ = _brain()
    _say(brain, "the red one on my left")
    brain.handle(Interrupt("hand", "hand entered workspace", time.perf_counter()))
    assert motion.goals[-1] == (None, "stop")
    assert voice.lines[-1] == "Stopping."


def test_arrival_of_a_replaced_goal_is_ignored():
    brain, _, motion, _, tracer = _brain()
    _say(brain, "the red one on my left")
    brain.handle(ArmEvent("arrived", np.array([9.0, 9.0, 0.0]), time.perf_counter(), "old"))
    assert not tracer.has(1, "arrived") and not brain.done.is_set()
    brain.handle(ArmEvent("arrived", motion.goals[-1][0], time.perf_counter(), "rules"))
    assert tracer.has(1, "arrived") and brain.done.is_set()


def test_missing_colour_is_said_out_loud():
    brain, _, _, voice, _ = _brain()
    _say(brain, "the green one")
    assert voice.lines == ["I don't see a green one."]


def test_confirmation_phrasing():
    assert confirmation("hand me the red one on my left", "red") == "Got it, the red one on your left."
    assert confirmation("the blue one on your right", "blue") == "Got it, the blue one on my right."


# ---- voice ------------------------------------------------------------------


def _tone(seconds, amp, sr=16000, block=480):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return [x[i : i + block] for i in range(0, len(x) - block + 1, block)]


def _audio():
    blocks = _tone(0.6, 0.0) + _tone(1.0, 0.3) + _tone(0.8, 0.0)
    return [(b, i * 0.03) for i, b in enumerate(blocks)]


def test_vad_finds_start_and_end():
    vad = EnergyVAD()
    events = [e for b, t in _audio() if (e := vad.push(b, t))]
    assert [e.kind for e in events] == ["start", "end"]
    assert abs(events[0].t - 0.6) < 0.05 and abs(events[1].t - 1.59) < 0.05


class CountingStt:
    def __init__(self):
        self.calls = 0

    def transcribe(self, pcm, sample_rate):
        self.calls += 1
        return f"words {len(pcm) // 1600}"


def test_streaming_recognizer_emits_partials_then_final():
    heard: list[Heard] = []
    rec = StreamingRecognizer(CountingStt(), iter(_audio()), partial_every=0.2)
    rec.start(heard.append)
    deadline = time.time() + 3
    while time.time() < deadline and not any(h.kind == "final" for h in heard):
        time.sleep(0.01)
    kinds = [h.kind for h in heard]
    assert kinds[0] == "start" and kinds[-1] == "final" and "end" in kinds
    assert "partial" in kinds
    rec.close()


# ---- detector decode ----------------------------------------------------------


def test_yolo_decode_undoes_letterbox_and_suppresses_duplicates():
    # Two classes, three anchors: two overlapping red boxes and one blue box.
    raw = np.array(
        [
            [100, 102, 300],  # cx
            [200, 201, 300],  # cy
            [20, 20, 30],  # w
            [20, 20, 30],  # h
            [0.9, 0.8, 0.0],  # red
            [0.0, 0.1, 0.7],  # blue
        ],
        dtype=np.float32,
    )
    dets = yolo_detections(raw, (0.5, 0, 80), ROBOT_CAM, ["red", "blue"], conf=0.25, iou=0.5)
    by_color = {d.color: d for d in dets}
    assert len(dets) == 2
    assert by_color["red"].pixel == (200.0, 240.0)
    assert by_color["blue"].pixel == (600.0, 440.0)
