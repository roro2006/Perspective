"""One simulated person does a whole fetch task with the live runtime; the run
comes back as a Session. Task T3-06; the "view-on+gaze" condition, T3-07.

One Runtime serves the whole task, so the belief learned on one request is
used on the next, as in a real session. The person's lines come from one
ReactiveVoice whose react() returns the next thing to say: an answer to "your
left or mine?", a correction when the arm commits to the wrong object, or the
next goal once the arm has arrived at this one."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from curio_robot.collab.session import Session, Step, StepKind
from curio_robot.collab.tasks import apply_scene, fetch_scene, true_map
from curio_robot.collab.worker import Goal, describe, plan_task
from curio_robot.grounding.client import VllmGrounder
from curio_robot.perception import SegmentationDetector
from curio_robot.runtime import Runtime, RuntimeConfig
from curio_robot.sim.live import ReactiveVoice
from curio_robot.sim.users import SimUser
from curio_robot.sim.world import SimWorld, ensure_world
from curio_robot.types import ObjectMap
from curio_robot.voice.scripted import PrintedVoice

CONDITIONS = ("view-off", "view-on", "view-on+gaze")
REACT_S = 0.3  # how long the person takes to see where the arm is going
# How long after finishing a line the person takes to say "no" to a wrong pick.
# They look at the one they meant first (they look at each goal as they ask for
# it, in every condition), so with gaze feedback the robot can fix it in time.
CORRECT_S = 0.9
NEXT_S = 0.4  # pause after an accepted arrival before the next request
MAX_CORRECTIONS = 2  # per goal; after that the person gives up on it
GOAL_TIMEOUT_S = 20.0  # per goal; a stuck goal is given up rather than hung on
SETTLE_S = 0.6  # quiet time after the last line before the task is over


@dataclass(slots=True)
class _Said:
    kind: StepKind
    text: str
    goal_id: str
    robot_lines: int  # len(voice_out.lines) when this was said
    target_before: str | None  # the arm's target (live id) when this was said
    chosen_id: str | None = None  # where the arm was (true id) when the next line came


class _Person:
    """What the simulated person says next. Polled by ReactiveVoice between lines."""

    def __init__(
        self, goals: list[Goal], user: SimUser, truth: ObjectMap, rng: np.random.Generator
    ) -> None:
        self.goals, self.user, self.truth, self.rng = goals, user, truth, rng
        self.rt: Runtime | None = None  # set once the Runtime exists
        self.said = [_Said("request", goals[0].text, goals[0].object_id, 0, None)]
        self.done: list[str] = []
        self.finished = False
        self.look: np.ndarray | None = None  # a head target for the world's thread to apply
        self._i = 0
        self._corrections = 0
        self._seen = 0
        self._wait_from: float | None = None
        self._arrived_at: float | None = None
        self._goal_t0 = time.perf_counter()
        self._matched: dict[str, str] = {}  # live id -> true id, once seen

    def react(self) -> str | None:
        now = time.perf_counter()
        if self.finished or self.rt is None:
            return None
        self._wait_from = self._wait_from or now
        goal = self.goals[self._i]
        if now - self._goal_t0 > GOAL_TIMEOUT_S:
            return self._next(done=False)
        if now - self._wait_from < REACT_S:
            return None  # give the robot a moment to commit before judging it
        lines = self.rt.voice_out.lines
        new, self._seen = lines[self._seen:], len(lines)
        if any("or mine?" in line for line in new):
            return self._say("answer", "mine" if goal.frame != "robot" else "yours")
        if any(line.endswith(("Which one do you mean?", "Which one?")) for line in new):
            again = describe(goal.object_id, self.truth, self.user, self.rng)
            return self._say("answer", again.text)  # say it again, without "that one"
        if any(line.startswith("I don't see") for line in new):
            return self._next(done=False)
        brain = self.rt.brain
        if brain.last_target == self.said[-1].target_before and not brain.done.is_set():
            return None  # still deciding (or the grounder is looking)
        chosen = self.chosen_now()
        if chosen != goal.object_id:
            if now - self._wait_from < CORRECT_S:
                return None  # still looking at the one they meant
            if self._corrections >= MAX_CORRECTIONS:
                return self._next(done=False)
            self._corrections += 1
            return self._say("correction", _correction(chosen, goal))
        if brain.arm_active or not brain.done.is_set():
            return None
        self._arrived_at = self._arrived_at or now
        return self._next(done=True) if now - self._arrived_at >= NEXT_S else None

    def chosen_now(self) -> str | None:
        """The true id of the object the arm is committed to. The live map's ids
        are the tracker's own, so match by position, and remember the match: an
        object the arm is reaching for can drop out of the robot camera's view."""
        live_id = self.rt.brain.last_target
        if live_id is None:
            return None
        _, live = self.rt.maps.get()
        obj = next((o for o in live.objects if o.object_id == live_id), None) if live else None
        if obj is not None:
            self._matched[live_id] = min(
                self.truth.objects,
                key=lambda t: float(np.linalg.norm(t.position[:2] - obj.position[:2])),
            ).object_id
        return self._matched.get(live_id)

    def _say(self, kind: StepKind, text: str, goal: Goal | None = None) -> str:
        goal = goal or self.goals[self._i]
        self.said[-1].chosen_id = self.chosen_now()
        self.said.append(_Said(kind, text, goal.object_id, len(self.rt.voice_out.lines),
                               self.rt.brain.last_target))
        self._wait_from, self._arrived_at = None, None
        return text

    def _next(self, *, done: bool) -> str | None:
        if done:
            self.done.append(self.goals[self._i].object_id)
        self._i += 1
        self._corrections, self._goal_t0 = 0, time.perf_counter()
        if self._i == len(self.goals):
            self.finished = True
            self.said[-1].chosen_id = self.chosen_now()
            return None
        goal = self.goals[self._i]
        self.look = _position(self.truth, goal.object_id)  # people look at what they ask for
        return self._say("request", goal.text, goal)


def _correction(chosen: str | None, goal: Goal) -> str:
    """ "no, the other one" within a same-coloured pair, else name the colour."""
    color = goal.object_id.split("_")[0]
    if chosen is not None and chosen.split("_")[0] == color:
        return "no, the other one"
    return f"no, the {color} one"


def _position(truth: ObjectMap, object_id: str) -> np.ndarray:
    return next(o.position for o in truth.objects if o.object_id == object_id).copy()


def run_task(
    user: SimUser,
    condition: str,
    seed: int,
    *,
    n_goals: int = 4,
    p_deictic: float = 0.0,
    occlude: bool = False,
    world: SimWorld | None = None,
    profiles_dir: Path | None = None,
    grounder_url: str | None = None,
    quiet: bool = True,
) -> Session:
    """Behaviour: tasks/T3-06-task-runner.md. ValueError for a condition not in
    CONDITIONS."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; one of {CONDITIONS}")
    rng = np.random.default_rng(seed)
    scene = fetch_scene(rng, occlude=occlude)
    truth = true_map(scene)
    goals = plan_task(truth, user, n_goals, rng, p_deictic)
    with ensure_world(world) as owned:
        world = owned
        world.home()
        apply_scene(world, scene)
        world.set_human(scene.human_eye, _position(truth, goals[0].object_id))

        person = _Person(goals, user, truth, rng)
        on_say = None if quiet else (lambda text: print(f"  human: {text}"))
        voice_in = ReactiveVoice(goals[0].text, person.react, on_say=on_say)
        mark_tts: list = []  # filled once the Runtime (and its tracer) exists, as in sim.live
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(
                world,
                SegmentationDetector(world.block_colors),
                voice_in=voice_in,
                voice_out=PrintedVoice(on_start=lambda _: mark_tts and mark_tts[0](), quiet=quiet),
                config=RuntimeConfig(
                    use_human_view=condition != "view-off",
                    person=user.name,
                    profiles_dir=profiles_dir or Path(tmp),
                    grounder=VllmGrounder(grounder_url) if grounder_url else None,
                    implicit_feedback=condition == "view-on+gaze",
                ),
            )
            person.rt = rt
            mark_tts.append(lambda: rt.tracer.mark(rt.brain.utterance, "tts_start"))
            t_start, t_end = _drive(rt, person, voice_in, world, scene.human_eye,
                                    deadline_s=GOAL_TIMEOUT_S * (n_goals + 1))
        return _session(person, rt, user, condition, scene.kind, seed, t_start, t_end)


def _drive(rt: Runtime, person: _Person, voice_in: ReactiveVoice, world: SimWorld,
           eye: np.ndarray, deadline_s: float) -> tuple[float, float]:
    """Run until the person is done and the robot has settled; (start, end) times."""
    rt.start()
    t_start = time.perf_counter()
    try:
        while time.perf_counter() < t_start + deadline_s:
            rt.run(0.05)
            person.chosen_now()  # match the arm's target while it's still in view
            if person.look is not None:  # the world belongs to this thread
                with world.lock:
                    world.set_human(eye, person.look)
                person.look = None
            brain = rt.brain
            if (person.finished and voice_in.idle_for(SETTLE_S) and brain.done.is_set()
                    and not brain.arm_active):
                break
        rt.run(0.2)  # let the last events land
    finally:
        t_end = time.perf_counter()
        rt.close()  # an arrival nobody objected to is learned as accepted here
    return t_start, t_end


def _session(person: _Person, rt: Runtime, user: SimUser, condition: str, task: str,
             seed: int, t_start: float, t_end: float) -> Session:
    lines = rt.voice_out.lines
    steps = []
    for k, said in enumerate(person.said):
        stop = person.said[k + 1].robot_lines if k + 1 < len(person.said) else len(lines)
        steps.append(Step(
            utterance=k + 1, kind=said.kind, text=said.text, goal_id=said.goal_id,
            chosen_id=said.chosen_id,
            asked=any(line.endswith("?") for line in lines[said.robot_lines:stop]),
            marks=rt.tracer.marks(k + 1),
        ))
    return Session(
        session_id=f"{user.name}-{condition}-{seed}", person=user.name, condition=condition,
        task=task, seed=seed, t_start=t_start, t_end=t_end,
        goals=tuple(g.object_id for g in person.goals), done=tuple(person.done),
        steps=tuple(steps),
    )
