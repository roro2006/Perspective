"""The brain: one thread that handles every event in order and decides what
the arm and the voice do. This is where the pipelining happens:

  partial transcript  -> readings -> belief posterior -> decide on the live map:
                         act starts the arm, hedge drifts toward the weighted
                         centre ("the red one" with two reds), ask waits
  final transcript    -> decide again on the full sentence; act goes and asks the
                         VLM off this thread, hedge/ask stops and asks "your left
                         or mine?", and the next answer resolves it
  VlmReply            -> confirm (keep going) or redirect mid-motion, and speak
  Interrupt / "no"    -> stop the arm, stop talking, learn from the correction
  ArmEvent            -> waterfall marks, "can't reach that", accept window
  GroundingReply      -> target 2: the grounder's view of the frame at the key word,
                         fused with the belief; it may redirect or settle "that one"

Every decision goes to the DecisionRecorder and every outcome (accepted,
corrected, answered) updates the person's ConventionBelief (docs/target1-conventions.md)."""

from __future__ import annotations

import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from curio_robot.conventions.belief import ConventionBelief, Outcome, learn_from_outcome
from curio_robot.conventions.decide import Decision, decide
from curio_robot.conventions.noise import flip_probability
from curio_robot.conventions.profiles import ProfileStore
from curio_robot.conventions.readings import Frame, Reading, enumerate_readings
from curio_robot.episode_log import EpisodeLog
from curio_robot.events import ArmEvent, Event, GroundingReply, Heard, Interrupt, VlmReply
from curio_robot.grounding.anchor import anchor_time
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.fuse import fuse
from curio_robot.grounding.history import FrameHistory
from curio_robot.grounding.marks import place_marks, render_marks
from curio_robot.grounding.prompt import GroundingRequest, build_request
from curio_robot.monitor import check_speech
from curio_robot.recorder import DecisionRecord, DecisionRecorder, Stage
from curio_robot.resolver import parse_request
from curio_robot.runtime.motion import Motion
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import CameraFrame, ObjectMap
from curio_robot.vlm import VlmResolver
from curio_robot.voice import VoiceOut

ACCEPT_WINDOW_S = 2.0  # an arrival nobody complains about within this long counts as accepted
# The VLM's pick counts as this many votes against the belief's posterior; a
# tie keeps the belief. Whose "left" is the belief's job, not the VLM's: with
# the act threshold at 0.7, a 2x vote can't overturn a confident pick, an
# explicit "my left" or a learned habit. Target 2 replaces this with the VLM's
# own probabilities.
VLM_VOTE = 2.0


def confirmation(text: str, color: str) -> str:
    req = parse_request(text)
    where = ""
    if req.relation in ("left", "right"):
        where = f" on {'your' if req.speaker_frame else 'my'} {req.relation}"
    elif req.relation:
        where = f", the {req.relation} one"
    return f"Got it, the {color} one{where}."


def _framed_confirmation(text: str, readings: list[Reading], object_id: str) -> str:
    """confirmation() for the frame the robot actually chose, so a learned
    robot-frame "left" is confirmed as "my left" rather than "your left"."""
    color = object_id.split("_")[0]
    reading = next((r for r in readings if r.object_id == object_id and r.frame), None)
    if reading is None or reading.explicit or reading.relation not in ("left", "right"):
        return confirmation(text, color)
    marker = "my" if reading.frame == "speaker" else "your"
    return confirmation(f"the {color} one on {marker} {reading.relation}", color)


def answer_frame(text: str) -> Frame | None:
    """The frame named in a reply to "your left or mine?": mine/my -> speaker,
    yours/your -> robot; the last one said wins."""
    frame: Frame | None = None
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in ("mine", "my"):
            frame = "speaker"
        elif word in ("yours", "your"):
            frame = "robot"
    return frame


@dataclass(slots=True)
class _Pending:
    """A committed (or asked) decision still waiting for its outcome."""

    utterance: int
    text: str
    readings: list[Reading]
    chosen_id: str | None


class Grounder(Protocol):
    """What the orchestrator needs from grounding.client.VllmGrounder (fakes in tests)."""

    def score(self, request: GroundingRequest) -> GroundingAnswer: ...


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
        person: str = "anonymous",
        profiles: ProfileStore | None = None,
        recorder: DecisionRecorder | None = None,
        belief: ConventionBelief | None = None,
        grounder: Grounder | None = None,
        history: FrameHistory | None = None,
    ) -> None:
        self._bus, self._maps, self._motion, self._voice = bus, maps, motion, voice_out
        self._tracer, self._vlm, self._log = tracer, vlm, log
        self.person, self._profiles, self._recorder = person, profiles, recorder
        if belief is not None:
            self.belief = belief
        elif profiles is not None:
            self.belief = profiles.load(person)
        else:
            self.belief = ConventionBelief()
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
        self._partial_key: tuple[str, str | None] | None = None
        self._committed: _Pending | None = None  # last act, for corrections and accepts
        self._asked: _Pending | None = None  # waiting for "yours" / "mine"
        self._accept: tuple[_Pending, float] | None = None  # arrived; deadline for "accepted"
        self._accept_closing = False  # the next utterance started: accept unless it's a "no"
        # Target 2: with a grounder, the final is grounded in the frame at the key word.
        self._grounder, self._history = grounder, history
        self._words: tuple = ()  # word timings of the last final transcript
        self._speech_end = 0.0  # its speech-end time, the anchor when there are no words
        self._ground_job: _Pending | None = None  # the decision a GroundingReply judges
        self._running = False
        self._thread = threading.Thread(target=self._loop, name="orchestrator", daemon=True)

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def close(self) -> None:
        """Stop the loop. An arrival nobody objected to before shutdown counts as accepted."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._accept is not None:
            self._learn_accepted()

    def _loop(self) -> None:
        while self._running:
            try:
                event = self._bus.get(timeout=0.1)
            except queue.Empty:
                self._expire_accept(time.perf_counter())
                continue
            self.handle(event)

    # ---- event handlers ----------------------------------------------------

    def handle(self, event: Event) -> None:
        self._expire_accept(time.perf_counter())
        if isinstance(event, Heard):
            self._on_heard(event)
        elif isinstance(event, Interrupt):
            self._on_interrupt(event)
        elif isinstance(event, VlmReply):
            self._on_vlm(event)
        elif isinstance(event, ArmEvent):
            self._on_arm(event)
        elif isinstance(event, GroundingReply):
            self._on_grounding(event)

    def _on_heard(self, e: Heard) -> None:
        if e.kind == "start":
            self.utterance += 1
            self._interrupted, self._spoken = False, False
            self._partial_key = None
            self._accept_closing = self._accept is not None
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
        if self._accept_closing and not self._interrupted:
            self._learn_accepted()  # the next utterance is not a complaint
        if e.kind == "partial":
            self._tracer.mark(u, "first_partial")
            if not self._interrupted:
                self._speculate(e.text)
            return
        self._tracer.mark(u, "final_transcript")
        self._words, self._speech_end = e.words, e.t
        if self._log:
            self._log.event("utterance", utterance=u, text=e.text)
        if not e.text.strip():
            return
        if self._asked is not None and self._is_answer(e.text):
            self._answer(e.text)
        elif self._interrupted:
            self._correct(e.text)
        else:
            self._commit(e.text)

    def _decide(
        self, text: str, object_map: ObjectMap
    ) -> tuple[list[Reading], dict[str, float], Decision]:
        readings = enumerate_readings(text, object_map)
        posterior = self.belief.posterior(readings)
        return readings, posterior, decide(posterior, object_map, text)

    def _speculate(self, text: str) -> None:
        """Act on a partial transcript, before the speaker has finished."""
        _, object_map = self._maps.get()
        if object_map is None or parse_request(text).color is None:
            return  # "hand me the ..." says nothing yet; don't drift toward everything
        readings, posterior, decision = self._decide(text, object_map)
        key = (decision.kind, decision.object_id)
        if key != self._partial_key:
            self._partial_key = key
            self._record("partial", text, readings, posterior, decision)
        if decision.kind == "act" and decision.object_id != self.goal_id:
            self._go(decision.object_id, decision.position, "speculative")
            self._tracer.mark(self.utterance, "speculative_target")
        elif decision.kind == "hedge" and self.goal_tag != "approach":
            self._go(None, decision.position, "approach")
            self._tracer.mark(self.utterance, "approach")

    def _commit(self, text: str) -> None:
        """Final transcript: the belief decides now, the VLM checks in the background."""
        u = self.utterance
        self._asked = None
        _, object_map = self._maps.get()
        if object_map is None:
            self._say("I can't see the table yet.")
            return
        readings, posterior, decision = self._decide(text, object_map)
        self._tracer.mark(u, "rules_target")
        self._record("final", text, readings, posterior, decision)
        if decision.kind == "none":
            self._halt()
            req = parse_request(text)
            if not readings and req.color is None and self._ground(u, text, [], None, object_map):
                return  # "that one": the rules found nothing, the grounder may settle it
            self._say(f"I don't see a {req.color} one." if req.color else "Which one do you mean?")
            self.done.set()
            return
        if decision.kind in ("hedge", "ask"):
            if self._ground(u, text, readings, None, object_map):
                return  # "that one" / an unsure frame: let the grounder look before asking
            self._halt()
            self._asked = _Pending(u, text, readings, None)
            self._say(decision.question or _question(text))
            self.done.set()
            return
        object_id = decision.object_id
        if object_id != self.goal_id:
            self._go(object_id, decision.position, "belief")
        self.last_target = object_id
        self._committed = _Pending(u, text, readings, object_id)
        if self._log:
            self._log.event("target", utterance=u, source="belief", object_id=object_id,
                            reason=f"{decision.kind} p={decision.confidence:.2f}")
        if self._ground(u, text, readings, object_id, object_map):
            return
        if self._vlm is None:
            self._say(_framed_confirmation(text, readings, object_id), confirm=True)
            return
        self._tracer.mark(u, "vlm_sent")
        self._vlm_pending = True
        self._pool.submit(self._ask_vlm, u, text, object_map, self.belief.p_own_frame())

    def _is_answer(self, text: str) -> bool:
        """A reply naming a frame, and not a new request for another colour."""
        if answer_frame(text) is None:
            return False
        color = parse_request(text).color
        return color is None or color == parse_request(self._asked.text).color

    def _answer(self, text: str) -> None:
        """'Yours' / 'mine' after we asked: learn it, confirm, go."""
        asked, self._asked = self._asked, None
        frame = answer_frame(text)
        reading = next((r for r in asked.readings if r.frame == frame), None)
        _, object_map = self._maps.get()
        obj = None
        if reading is not None and object_map is not None:
            obj = next((o for o in object_map.objects if o.object_id == reading.object_id), None)
        if obj is None:
            self._say("Okay. Which one?")
            self.done.set()
            return
        decision = Decision("act", obj.object_id, np.asarray(obj.position, float), 1.0)
        self._record("final", text, [reading], {obj.object_id: 1.0}, decision)
        self._learn(asked, "answered", obj.object_id, obj.object_id, object_map)
        self._go(obj.object_id, obj.position, "answered")
        self.last_target = obj.object_id
        self._committed = None  # already learned from; its arrival teaches nothing more
        self._say(_framed_confirmation(asked.text, asked.readings, obj.object_id), confirm=True)

    def _ask_vlm(self, u: int, text: str, object_map: ObjectMap, p_own_frame: float) -> None:
        answer = self._vlm.resolve(text, object_map, p_own_frame=p_own_frame)
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
        obj = None
        if r.object_id and object_map is not None:
            obj = next((o for o in object_map.objects if o.object_id == r.object_id), None)
        text = r.detail.get("text", "")
        readings = self._committed.readings if self._committed else []
        if obj is not None:
            vlm_decision = Decision("act", obj.object_id, np.asarray(obj.position, float), 1.0)
            self._record("vlm", text, readings, {obj.object_id: 1.0}, vlm_decision)
        else:
            self._record("vlm", text, readings, {}, Decision("none", None, None, 0.0))
        overruled = False
        if obj is not None and obj.object_id != self.last_target:
            if self._vlm_tips(readings, obj.object_id):
                self._go(obj.object_id, obj.position, "vlm")
                self._tracer.mark(r.utterance, "redirected")
                self.last_target = chosen = obj.object_id
                if self._committed is not None:
                    self._committed.chosen_id = chosen  # outcomes now judge the VLM's pick
            else:
                overruled = True
                self._tracer.mark(r.utterance, "vlm_overruled")
                if self._log:
                    self._log.event("vlm_overruled", utterance=r.utterance, vlm=obj.object_id,
                                    kept=self.last_target)
        if r.spoken and r.source != "unavailable" and not overruled:
            self._say(r.spoken, confirm=True)
        elif chosen:
            self._say(_framed_confirmation(text, readings, chosen), confirm=True)
        if not self.arm_active:
            self.done.set()

    def _ground(
        self,
        u: int,
        text: str,
        readings: list[Reading],
        chosen_id: str | None,
        object_map: ObjectMap,
    ) -> bool:
        """Send the frame the person saw at the key word to the grounder, off this
        thread. False (nothing sent) without a grounder or without a frame."""
        if self._grounder is None or self._history is None:
            return False
        anchor = anchor_time(list(self._words))
        frame = self._history.at(self._speech_end if anchor is None else anchor)
        frame = frame or self._history.latest()
        if frame is None:
            return False
        self._ground_job = _Pending(u, text, readings, chosen_id)
        self._tracer.mark(u, "ground_sent")
        self._pool.submit(self._ask_grounder, u, text, object_map, frame,
                          self.belief.p_own_frame())
        return True

    def _ask_grounder(
        self, u: int, text: str, object_map: ObjectMap, frame: CameraFrame, p_own_frame: float
    ) -> None:
        """Marks, render, request, score; always posts a reply so the utterance finishes."""
        try:
            marks = place_marks(object_map, frame)
            image, _ = render_marks(frame, marks)
            request = build_request(text, object_map, marks, image, p_own_frame=p_own_frame)
            answer = self._grounder.score(request)
            reply = GroundingReply(u, dict(answer.probs), answer.letter, answer.source,
                                   answer.latency_ms, time.perf_counter())
        except Exception as e:  # noqa: BLE001 - a broken request degrades to "unavailable"
            reply = GroundingReply(u, {}, None, f"error: {e}", 0.0, time.perf_counter())
        self._bus.put(reply)

    def _on_grounding(self, r: GroundingReply) -> None:
        """Fuse the grounder's probabilities with the belief, then decide: it may
        redirect a committed pick, or act alone when the rules found nothing."""
        job = self._ground_job
        if job is None or job.utterance != r.utterance or r.utterance != self.utterance:
            return  # the human has moved on
        self._ground_job = None
        if self._interrupted:
            return
        self._tracer.mark(r.utterance, "ground_reply", r.t)
        if self._log:
            self._log.event("grounding", utterance=r.utterance, source=r.source,
                            letter=r.letter, latency_ms=round(r.latency_ms, 1),
                            probs={k: round(p, 3) for k, p in r.probs.items()})
        _, object_map = self._maps.get()
        fused = fuse(self.belief.posterior(job.readings), r.probs)
        decision = Decision("none", None, None, 0.0)
        if object_map is not None:
            decision = decide(fused, object_map, job.text)
        self._record("vlm", job.text, job.readings, fused, decision)
        if job.chosen_id is None:  # the belief couldn't act: the fused posterior decides
            if decision.kind != "act":
                self._halt()
                if job.readings:
                    self._asked = _Pending(r.utterance, job.text, job.readings, None)
                self._say(_question(job.text))
                self.done.set()
                return
            self._go(decision.object_id, decision.position, "vlm")
            self.last_target = decision.object_id
            self._committed = _Pending(r.utterance, job.text, job.readings, decision.object_id)
        elif decision.kind == "act" and decision.object_id != self.last_target:
            self._go(decision.object_id, decision.position, "vlm")
            self._tracer.mark(r.utterance, "ground_redirected")
            self.last_target = decision.object_id
            if self._committed is not None and self._committed.utterance == r.utterance:
                self._committed.chosen_id = decision.object_id  # outcomes judge the arm's pick
        if self._log:
            self._log.event("target", utterance=r.utterance, source="grounder",
                            object_id=self.last_target, reason=f"fused {decision.kind}")
        if self.last_target:
            self._say(_framed_confirmation(job.text, job.readings, self.last_target),
                      confirm=True)
        if not self.arm_active:
            self.done.set()

    def _vlm_tips(self, readings: list[Reading], vlm_id: str) -> bool:
        """Would the VLM's vote change the belief's top object to vlm_id?"""
        posterior = self.belief.posterior(readings)
        if vlm_id not in posterior:
            return not readings  # nothing to weigh it against: trust the VLM
        mine = posterior[vlm_id] * VLM_VOTE
        best_other = max((p for k, p in posterior.items() if k != vlm_id), default=0.0)
        return mine > best_other + 1e-9  # a tie (within float error) keeps the belief

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
                self._correct_to(text, others[0].object_id, others[0].position, object_map)
                self._say(f"Sorry, the other {color} one.")
                return
        self._say("Okay. Which one?")
        self.done.set()

    def _correct_to(
        self, text: str, right_id: str, position: np.ndarray, object_map: ObjectMap
    ) -> None:
        """Go to the other object, record the correction and learn from it."""
        wrong_id = self.last_target
        self._interrupted = False
        self._go(right_id, position, "correction")
        if self._log:
            self._log.event("correction", utterance=self.utterance, text=text,
                            wrong=wrong_id, right=right_id)
        committed, self._committed = self._committed, None
        readings = committed.readings if committed else []
        decision = Decision("act", right_id, np.asarray(position, float), 1.0)
        self._record("correction", text, readings, {right_id: 1.0}, decision)
        if committed is not None and committed.chosen_id == wrong_id:
            self._learn(committed, "corrected", wrong_id, right_id, object_map)
        self.last_target = right_id

    def _on_interrupt(self, i: Interrupt) -> None:
        self._interrupted = True
        self._accept, self._accept_closing = None, False  # a complaint is not an acceptance
        was_active = self.arm_active
        self._halt()
        self._voice.interrupt()
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
            committed = self._committed
            if committed is not None and committed.chosen_id == self.goal_id:
                self._accept = (committed, time.perf_counter() + ACCEPT_WINDOW_S)
                self._accept_closing = False
            if not self._vlm_pending and self._ground_job is None:
                self.done.set()
        elif e.kind == "unreachable":
            self._say("I can't reach that one.")
            self.done.set()

    # ---- learning ----------------------------------------------------------

    def _expire_accept(self, now: float) -> None:
        if self._accept is not None and now >= self._accept[1]:
            self._learn_accepted()

    def _learn_accepted(self) -> None:
        arrived = self._accept[0]
        self._accept, self._accept_closing = None, False
        if self._committed is arrived:
            self._committed = None  # learned from; a later "no" teaches nothing twice
        _, object_map = self._maps.get()
        self._learn(arrived, "accepted", arrived.chosen_id, None, object_map)

    def _learn(
        self,
        pending: _Pending,
        outcome: Outcome,
        chosen_id: str | None,
        right_id: str | None,
        object_map: ObjectMap | None,
    ) -> float:
        """Update the belief, save the profile, log the outcome against the decision's utterance."""
        p_flip = _flip(pending.readings, object_map) if outcome == "corrected" else 0.0
        weight = learn_from_outcome(self.belief, pending.readings, chosen_id or "", outcome,
                                    right_id=right_id, p_flip=p_flip)
        if self._profiles is not None:
            self._profiles.save(self.person, self.belief)
        if self._recorder is not None:
            self._recorder.outcome(pending.utterance, outcome, right_id, time.perf_counter())
        if self._log:
            self._log.event("learned", utterance=pending.utterance, outcome=outcome,
                            weight=round(weight, 3), p_flip=round(p_flip, 3),
                            p_own_frame=round(self.belief.p_own_frame(), 3))
        return weight

    def _record(
        self,
        stage: Stage,
        text: str,
        readings: list[Reading],
        posterior: dict[str, float],
        decision: Decision,
    ) -> None:
        if self._recorder is None:
            return
        self._recorder.record(DecisionRecord(
            self.utterance, time.perf_counter(), text, stage, list(readings), dict(posterior),
            decision, self.belief.to_dict(), self.person,
        ))

    # ---- helpers -----------------------------------------------------------

    def _go(self, object_id: str | None, position: np.ndarray, tag: str) -> None:
        self.goal_id, self.goal_tag, self.arm_active = object_id, tag, True
        self._goal_pos = np.asarray(position, float).copy()
        self._motion.go(self._goal_pos, tag)

    def _halt(self) -> None:
        self._motion.stop()
        self.arm_active, self.goal_id, self.goal_tag, self._goal_pos = False, None, "", None

    def _say(self, text: str, *, confirm: bool = False) -> None:
        if confirm:
            if self._spoken:
                return  # one confirmation per utterance, whoever answers first
            self._spoken = True
        self._voice.say(text)


def _question(text: str) -> str:
    """What to ask after a hedge (decide only words a question for "ask")."""
    relation = parse_request(text).relation
    return f"Your {relation} or mine?" if relation in ("left", "right") else "Which one do you mean?"


def _flip(readings: list[Reading], object_map: ObjectMap | None) -> float:
    """How likely head-pose noise swapped the two framed objects, from the speaker's view."""
    if object_map is None or object_map.human_pose is None:
        return 0.0
    framed = {r.frame: r.object_id for r in readings if r.frame}
    speaker_id, robot_id = framed.get("speaker"), framed.get("robot")
    if speaker_id is None or robot_id is None or speaker_id == robot_id:
        return 0.0
    positions = {o.object_id: np.asarray(o.position, float) for o in object_map.objects}
    if speaker_id not in positions or robot_id not in positions:
        return 0.0
    return flip_probability(positions[speaker_id], positions[robot_id], object_map.human_pose)
