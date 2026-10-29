"""Learning curves for target 1: simulated users, many requests each, error
rate against request number. Task T1-09 (fast mode); T1-10 (live mode, the full runtime for one person).

    uv run curio-sim learn --requests 15 --users 50 [--noise] [--baseline]
    uv run curio-sim learn --live --user robot-frame --requests 6"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from curio_robot.conventions.belief import ConventionBelief, learn_from_outcome
from curio_robot.conventions.decide import decide
from curio_robot.conventions.noise import flip_probability
from curio_robot.conventions.readings import enumerate_readings
from curio_robot.geometry import OBJECT_PLANE_Z, look_at, perturb_pose
from curio_robot.perception import SegmentationDetector
from curio_robot.resolver import parse_request
from curio_robot.runtime import Runtime, RuntimeConfig
from curio_robot.sim.live import ReactiveVoice
from curio_robot.sim.scenarios import (
    COLORS,
    _in_workspace,
    _random_xy,
    _sample_human,
    perspective,
)
from curio_robot.sim.users import SimUser, Utterance
from curio_robot.sim.world import ROBOT_CAM_LOOK, ROBOT_CAM_POS, SimWorld, ensure_world
from curio_robot.types import CameraPose, ObjectMap, WorldObject
from curio_robot.voice.scripted import PrintedVoice

HALF_LEN_MIN = 0.045
HALF_LEN_MAX = 0.07
FLIP_SAMPLES = 100
REACT_S = 0.3  # how long the simulated person takes to notice the arm going the wrong way
SETTLE_S = 0.6  # quiet time after the person's last line before a request is over
LIVE_TIMEOUT_S = 20.0  # per request; a stuck request ends rather than hangs the demo


def _sample_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two block centres on the table, at a random angle, both inside the workspace."""
    while True:
        mid = _random_xy(rng)
        angle = rng.uniform(0, np.pi)
        half_len = rng.uniform(HALF_LEN_MIN, HALF_LEN_MAX)
        half = half_len * np.array([np.cos(angle), np.sin(angle)])
        a, b = mid - half, mid + half
        if _in_workspace(a) and _in_workspace(b):
            return a, b


def learning_curve(
    user: SimUser,
    *,
    requests: int = 15,
    users: int = 50,
    seed: int = 0,
    pos_noise: float = 0.0,
    rot_noise_deg: float = 0.0,
    learn: bool = True,
) -> dict:
    """Fast mode, no runtime or arm: for each of `users` fresh copies of
    `user` (fresh belief each), run `requests` perspective scenes. Per scene:
    utter -> readings -> posterior -> decide; the user reacts to the chosen
    object; learn_from_outcome (unless learn=False, the baseline).
    The decision is the final one, so only "act" moves the arm: "hedge" and
    "ask" both mean the robot asks "your left or mine?", the user answers
    (SimUser.answer), and the robot learns with outcome "answered" and then
    takes the intended object (an ask is never an error).
    Scenes: sim.scenarios.perspective geometry without rendering is fine;
    two same-colour blocks, a random side, a random human pose.
    p_flip: noise.flip_probability for the pair when pos/rot noise > 0, and
    the human pose given to the robot is perturbed with geometry.perturb_pose.

    Returns {"user", "requests", "users", "learn", "error_rate": [per request
    index, fraction of users wrong], "ask_rate": [...], "final_p_own_frame":
    mean over users}. "ask" counts as not wrong and not right; track it apart."""
    rng = np.random.default_rng(seed)
    robot_pose = CameraPose(ROBOT_CAM_POS, look_at(ROBOT_CAM_POS, ROBOT_CAM_LOOK))
    noisy = pos_noise > 0 or rot_noise_deg > 0

    errors = [0] * requests
    asks = [0] * requests
    finals = []

    for _ in range(users):
        belief = ConventionBelief()
        for i in range(requests):
            eye, look = _sample_human(rng)
            true_human = CameraPose(eye, look_at(eye, look))
            color = str(rng.choice(COLORS))

            a, b = _sample_pair(rng)
            pa = np.array([*a, OBJECT_PLANE_Z])
            pb = np.array([*b, OBJECT_PLANE_Z])
            objects = [
                WorldObject(f"{color}_1", color, pa),
                WorldObject(f"{color}_2", color, pb),
            ]

            side = str(rng.choice(["left", "right"]))
            utt = user.utter(color, side, rng)
            # Ground truth uses the true pose; the robot only sees the perturbed one.
            intended = user.intended(utt, ObjectMap(objects, robot_pose, true_human))
            seen = perturb_pose(true_human, rng, pos_noise, rot_noise_deg) if noisy else true_human

            omap = ObjectMap(objects, robot_pose, seen)
            readings = enumerate_readings(utt.text, omap)
            d = decide(belief.posterior(readings), omap, utt.text)

            p_flip = 0.0
            if noisy:
                p_flip = flip_probability(
                    pa, pb, seen,
                    pos_sigma=pos_noise,
                    rot_sigma_deg=rot_noise_deg,
                    samples=FLIP_SAMPLES,
                    seed=int(rng.integers(2**31)),
                )

            if d.kind == "act":
                if d.object_id == intended:
                    outcome, chosen = "accepted", d.object_id
                else:
                    errors[i] += 1
                    outcome, chosen = "corrected", d.object_id
                right = None
            else:
                # hedge / ask / none: robot asks, user answers, robot takes intended
                asks[i] += 1
                user.answer(utt)
                outcome, chosen, right = "answered", intended, intended

            if learn:
                learn_from_outcome(belief, readings, chosen, outcome, right_id=right, p_flip=p_flip)

        finals.append(belief.p_own_frame())

    return {
        "user": user.name,
        "requests": requests,
        "users": users,
        "learn": learn,
        "error_rate": [e / users for e in errors],
        "ask_rate": [n / users for n in asks],
        "final_p_own_frame": float(np.mean(finals)),
    }


class _Reactor:
    """What the simulated person says next in live mode, from what the robot
    has committed to and said so far. One per request."""

    def __init__(self, user: SimUser, utt: Utterance) -> None:
        self.user, self.utt = user, utt
        self.rt: Runtime | None = None  # set once the Runtime exists
        self.outcome = "accepted"
        self.intended: str | None = None
        self._seen = 0
        self._reacted = False
        self._first_poll: float | None = None

    def react(self) -> str | None:
        """Polled by ReactiveVoice once the request has been said."""
        now = time.perf_counter()
        self._first_poll = self._first_poll or now
        if self._reacted or now - self._first_poll < REACT_S:
            return None  # give the robot a moment to commit before judging it
        if self.intended is None:
            _, omap = self.rt.maps.get()
            if omap is None:
                return None
            self.intended = self.user.intended(self.utt, omap)
        lines = self.rt.voice_out.lines
        new, self._seen = lines[self._seen:], len(lines)
        if any("or mine?" in line for line in new):
            self._reacted, self.outcome = True, "answered"
            return self.user.answer(self.utt)
        chosen = self.rt.brain.last_target
        if chosen is not None and chosen != self.intended:
            self._reacted, self.outcome = True, "corrected"
            return self.user.react(self.intended, chosen)
        return None


def live_learning(
    user: SimUser,
    *,
    requests: int = 6,
    seed: int = 0,
    quiet: bool = False,
    world: SimWorld | None = None,
    profiles_dir: Path | None = None,
) -> dict:
    """Live mode: one simulated person, `requests` marker-free requests through
    the full runtime (rendering, perception, motion, voice). The person reacts
    to the arm (sim.live.ReactiveVoice); the belief carries over between
    requests through the person's profile, as it would across a session.

    No VLM: the mock one is the rule resolver, which always takes the speaker's
    left and would overrule what the robot has learned (feeding the VLM into
    the posterior is target 2).

    Returns {"user", "requests", "outcomes": [accepted|corrected|answered],
    "p_own_frame": [after each request], "waterfall_ms": [per request],
    "final_p_own_frame"}."""
    rng = np.random.default_rng(seed)
    with ensure_world(world) as owned:
        world = owned
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            profiles = profiles_dir or Path(tmp)
            for i in range(requests):
                if not quiet:
                    print(f"request {i + 1}/{requests}")
                rows.append(_live_request(user, rng, world, profiles, quiet))
        return {
            "user": user.name,
            "requests": requests,
            "outcomes": [r["outcome"] for r in rows],
            "p_own_frame": [r["p_own_frame"] for r in rows],
            "waterfall_ms": [r["waterfall_ms"] for r in rows],
            "final_p_own_frame": rows[-1]["p_own_frame"] if rows else ConventionBelief().p_own_frame(),
        }


def _live_request(
    user: SimUser, rng: np.random.Generator, world: SimWorld, profiles: Path, quiet: bool
) -> dict:
    """One request on a fresh block pair and a fresh Runtime (fresh tracker and waterfall)."""
    world.home()
    sc = perspective(world, rng)
    color = parse_request(sc.request).color
    utt = user.utter(color, str(rng.choice(["left", "right"])), rng)
    reactor = _Reactor(user, utt)

    on_say = None if quiet else (lambda text: print(f"  human: {text}"))
    voice_in = ReactiveVoice(utt.text, reactor.react, on_say=on_say)
    mark_tts: list = []
    voice_out = PrintedVoice(on_start=lambda _: mark_tts and mark_tts[0](), quiet=quiet)
    rt = Runtime(
        world,
        SegmentationDetector(world.block_colors),
        voice_in=voice_in,
        voice_out=voice_out,
        config=RuntimeConfig(person=user.name, profiles_dir=profiles),
    )
    reactor.rt = rt
    mark_tts.append(lambda: rt.tracer.mark(rt.brain.utterance, "tts_start"))
    rt.start()
    try:
        end = time.perf_counter() + LIVE_TIMEOUT_S
        while time.perf_counter() < end:
            rt.run(0.1)
            brain = rt.brain
            if voice_in.idle_for(SETTLE_S) and brain.done.is_set() and not brain.arm_active:
                break
        rt.run(0.2)  # let the last events land
    finally:
        rt.close()  # an arrival nobody objected to is learned as accepted here
    p_own = rt.brain.belief.p_own_frame()
    waterfall = rt.tracer.waterfall(1)
    if not quiet:
        b = rt.brain.belief
        print(f"  -> {reactor.outcome}; belief p_own_frame={p_own:.2f} "
              f"(a={b.own_frame_a:.2f}, b={b.own_frame_b:.2f})")
        stages = ", ".join(f"{k} {v:.0f}" for k, v in waterfall.items())
        print(f"  waterfall ms from speech end: {stages}")
    return {"outcome": reactor.outcome, "p_own_frame": p_own, "waterfall_ms": waterfall}
