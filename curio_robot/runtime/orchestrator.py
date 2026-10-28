"""The brain: one thread that handles every event in order and decides what
the arm and the voice do. This is where the pipelining happens:

  partial transcript  -> the rule resolver picks a target on the live map and
                         the arm starts (or, with "the red one" and two reds,
                         drifts toward both until "left" arrives)
  final transcript    -> rules again on the full sentence, then the VLM is
                         asked on a snapshot of the map, off this thread
  VlmReply            -> confirm (keep going) or redirect mid-motion, and speak
  Interrupt / "no"    -> stop the arm, stop talking, log a correction
  ArmEvent            -> waterfall marks, "can't reach that" """

from __future__ import annotations

import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from curio_robot.episode_log import EpisodeLog
from curio_robot.events import ArmEvent, Event, Heard, Interrupt, VlmReply
from curio_robot.monitor import check_speech
from curio_robot.resolver import parse_request, resolve
from curio_robot.runtime.motion import Motion
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import ObjectMap
from curio_robot.vlm import VlmResolver
from curio_robot.voice import VoiceOut


def confirmation(text: str, color: str) -> str:
    req = parse_request(text)
    where = ""
    if req.relation in ("left", "right"):
        where = f" on {'your' if req.speaker_frame else 'my'} {req.relation}"
    elif req.relation:
        where = f", the {req.relation} one"
    return f"Got it, the {color} one{where}."


class Orchestrator:
    def __init__(
        self,
        bus: queue.Queue[Event],
        maps: Latest[ObjectMap],
        motion: Motion,
        voice_out: VoiceOut,
        tracer: Tracer,
        *,
        vlm: VlmResolver | None = None,
        log: EpisodeLog | None = None,
    ) -> None:
        self._bus, self._maps, self._motion, self._voice = bus, maps, motion, voice_out
        self._tracer, self._vlm, self._log = tracer, vlm, log
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm")
        self.utterance = 0
        self.goal_id: str | None = None  # object the arm is heading to (None: idle/approach)
        self.goal_tag = ""
        self.last_target: str | None = None  # last object actually committed to
        self.arm_active = False
        self._goal_pos: np.ndarray | None = None
        self._vlm_pending = False
        self.done = threading.Event()  # set when an utterance has been fully served
        self._interrupted = False
        self._spoken = False
        self._running = False
        self._thread = threading.Thread(target=self._loop, name="orchestrator", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while self._running:
            try:
                event = self._bus.get(timeout=0.1)
            except queue.Empty:
                continue
            self.handle(event)

    # ---- event handlers ----------------------------------------------------

    def handle(self, event: Event) -> None:
        if isinstance(event, Heard):
            self._on_heard(event)
        elif isinstance(event, Interrupt):
            self._on_interrupt(event)
        elif isinstance(event, VlmReply):
            self._on_vlm(event)
        elif isinstance(event, ArmEvent):
            self._on_arm(event)

    def _on_heard(self, e: Heard) -> None:
        if e.kind == "start":
            self.utterance += 1
            self._interrupted, self._spoken = False, False
            self.done.clear()
            self._voice.interrupt()  # barge-in: the human talks, the robot stops talking
            self._tracer.mark(self.utterance, "speech_start", e.t)
            return
        u = self.utterance
        if e.kind == "end":
            self._tracer.mark(u, "speech_end", e.t)
            return
        stop = check_speech(e.text)
        if stop and not self._interrupted:
            self._on_interrupt(Interrupt("speech", stop.detail, e.t))
        if e.kind == "partial":
            self._tracer.mark(u, "first_partial")
            if not self._interrupted:
                self._speculate(e.text)
            return
        self._tracer.mark(u, "final_transcript")
        if self._log:
            self._log.event("utterance", utterance=u, text=e.text)
        if not e.text.strip():
            return
        if self._interrupted:
            self._correct(e.text)
        else:
            self._commit(e.text)

    def _speculate(self, text: str) -> None:
        """Act on a partial transcript, before the speaker has finished."""
        _, object_map = self._maps.get()
        req = parse_request(text)
        if object_map is None or req.color is None:
            return
        candidates = object_map.by_color(req.color)
        if not candidates:
            return
        if len(candidates) > 1 and req.relation is None:
            if self.goal_tag != "approach":
                centroid = np.mean([o.position for o in candidates], axis=0)
                self._go(None, centroid, "approach")
                self._tracer.mark(self.utterance, "approach")
            return
        target = resolve(text, object_map)
        if target.object_id and target.object_id != self.goal_id:
            self._go(target.object_id, target.position, "speculative")
            self._tracer.mark(self.utterance, "speculative_target")

    def _commit(self, text: str) -> None:
        """Final transcript: rules now, VLM in the background."""
        u = self.utterance
        _, object_map = self._maps.get()
        if object_map is None:
            self._say("I can't see the table yet.")
            return
        target = resolve(text, object_map)
        self._tracer.mark(u, "rules_target")
        if target.object_id is None:
            self._motion.stop()
            req = parse_request(text)
            self._say(f"I don't see a {req.color} one." if req.color else "Which one do you mean?")
            self.done.set()
            return
        if target.object_id != self.goal_id:
            self._go(target.object_id, target.position, "rules")
        self.last_target = target.object_id
        color = target.object_id.split("_")[0]
        if self._log:
            self._log.event("target", utterance=u, source="rules", object_id=target.object_id,
                            reason=target.reason)
        if self._vlm is None:
            self._say(confirmation(text, color), confirm=True)
            return
        self._tracer.mark(u, "vlm_sent")
        self._vlm_pending = True
        self._pool.submit(self._ask_vlm, u, text, object_map)

    def _ask_vlm(self, u: int, text: str, object_map: ObjectMap) -> None:
        answer = self._vlm.resolve(text, object_map)
        self._bus.put(
            VlmReply(u, answer.object_id, answer.spoken, answer.source, time.perf_counter(),
                     {"latency_ms": round(answer.latency_ms, 1), "errors": list(answer.errors),
                      "text": text})
        )

    def _on_vlm(self, r: VlmReply) -> None:
        if r.utterance != self.utterance or self._interrupted:
            return  # the human has moved on
        self._tracer.mark(r.utterance, "vlm_reply", r.t)
        self._vlm_pending = False
        if self._log:
            self._log.event("target", utterance=r.utterance, source=r.source,
                            object_id=r.object_id, spoken=r.spoken, **r.detail)
        _, object_map = self._maps.get()
        chosen = self.last_target
        if r.object_id and r.object_id != self.last_target and object_map is not None:
            obj = next((o for o in object_map.objects if o.object_id == r.object_id), None)
            if obj is not None:
                self._go(obj.object_id, obj.position, "vlm")
                self._tracer.mark(r.utterance, "redirected")
                self.last_target = chosen = obj.object_id
        if r.spoken and r.source != "unavailable":
            self._say(r.spoken, confirm=True)
        elif chosen:
            self._say(confirmation(r.detail.get("text", ""), chosen.split("_")[0]), confirm=True)
        if not self.arm_active:
            self.done.set()

    def _correct(self, text: str) -> None:
        """The human said "no ..." mid-motion. A colour means a new request;
        "the other one" with a pair means the other of the pair."""
        req = parse_request(text)
        _, object_map = self._maps.get()
        if req.color or object_map is None:
            self._interrupted = False
            self._commit(text)
            return
        words = set(re.findall(r"[a-z]+", text.lower()))
        if "other" in words and self.last_target:
            color = self.last_target.split("_")[0]
            others = [o for o in object_map.by_color(color) if o.object_id != self.last_target]
            if len(others) == 1:
                self._interrupted = False
                self._go(others[0].object_id, others[0].position, "correction")
                if self._log:
                    self._log.event("correction", utterance=self.utterance, text=text,
                                    wrong=self.last_target, right=others[0].object_id)
                self.last_target = others[0].object_id
                self._say(f"Sorry, the other {color} one.")
                return
        self._say("Okay. Which one?")
        self.done.set()

    def _on_interrupt(self, i: Interrupt) -> None:
        self._interrupted = True
        was_active = self.arm_active
        self._motion.stop()
        self._voice.interrupt()
        self.arm_active, self.goal_id, self.goal_tag, self._goal_pos = False, None, "", None
        if self._log:
            self._log.event("interrupt", utterance=self.utterance, source=i.source,
                            detail=i.detail, target=self.last_target, arm_was_moving=was_active)
        if i.source == "hand" and was_active:
            self._say("Stopping.")

    def _on_arm(self, e: ArmEvent) -> None:
        u = self.utterance
        if e.kind == "started":
            self._tracer.mark(u, "motion_start", e.t)
            return
        if e.tag == "approach" or self._goal_pos is None or not np.allclose(e.goal, self._goal_pos):
            return  # the midpoint between candidates, or a goal already replaced
        self.arm_active = False
        if e.kind == "arrived":
            self._tracer.mark(u, "arrived", e.t)
            if self._log:
                self._log.event("arrived", utterance=u, object_id=self.goal_id, tag=e.tag,
                                waterfall_ms=self._tracer.waterfall(u))
            if not self._vlm_pending:
                self.done.set()
        elif e.kind == "unreachable":
            self._say("I can't reach that one.")
            self.done.set()

    # ---- helpers -----------------------------------------------------------

    def _go(self, object_id: str | None, position: np.ndarray, tag: str) -> None:
        self.goal_id, self.goal_tag, self.arm_active = object_id, tag, True
        self._goal_pos = np.asarray(position, float).copy()
        self._motion.go(self._goal_pos, tag)

    def _say(self, text: str, *, confirm: bool = False) -> None:
        if confirm:
            if self._spoken:
                return  # one confirmation per utterance, whoever answers first
            self._spoken = True
        self._voice.say(text)
