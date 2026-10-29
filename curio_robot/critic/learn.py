"""Phase 2a: is the critic a better training signal than spoken corrections?

A target-choice policy learns, per person, whose left/right they mean (as in
target 1). Every source drives the same learner on the same people, scenes and
words; only the feedback differs:
- none: never learns (keeps the speaker-frame default);
- spoken: "no" means the other candidate, silence means the reach was right;
- critic: the critic's verdict at the end of the reach, when it is confident;
- critic+spoken: a heard "no" wins, else the critic;
- oracle: the true wanted block (upper bound).
The reactions are simulated, and the critic was trained on the same reaction
model, so this measures the signal under our assumptions, not on people.
Task T4-10."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curio_robot import perspective
from curio_robot.collab.tasks import fetch_scene, true_map
from curio_robot.conventions.readings import Frame
from curio_robot.critic.episodes import HOME, HOVER, Episode, arm_path, frame_times
from curio_robot.critic.model import Critic
from curio_robot.critic.reactions import FAMILIES, Style, sample_style, simulate_reaction

SOURCES = ("none", "spoken", "critic", "critic+spoken", "oracle")
PERSON_TYPES = {"own-frame": 0.95, "robot-frame": 0.05, "mixed": 0.5}  # p_own_frame, as target 1
FRAMES: tuple[Frame, Frame] = ("speaker", "robot")


@dataclass(slots=True)
class FrameBelief:
    """Beta evidence for "this person means their own left" (speaker) against
    "they mean the robot's left" (robot). Ties go to the speaker frame, the
    view-on default."""

    speaker: float = 1.0
    robot: float = 1.0

    def choose(self) -> Frame:
        return "speaker" if self.speaker >= self.robot else "robot"

    def update(self, frame: Frame, weight: float = 1.0) -> None:
        if frame == "speaker":
            self.speaker += weight
        else:
            self.robot += weight


@dataclass(frozen=True, slots=True)
class Interaction:
    episode: Episode
    picks: dict[str, int]  # frame -> the block that frame's reading of the words picks
    said: Frame  # the frame the person meant


@dataclass(frozen=True, slots=True)
class Step:
    person: int
    person_type: str
    family: str
    source: str
    k: int  # interaction number for this person, from 0
    ambiguous: bool  # the two frames pick different blocks
    wrong: bool
    label: int | None
    label_correct: bool | None


def interaction(
    seed: int, person: int, k: int, p_own: float, style: Style, choose: Frame
) -> Interaction:
    """One request and reach. The scene, the wanted block and the words depend
    only on (seed, person, k), so every source sees the same ones; the reach
    goes where `choose` reads the words, and the reaction follows from that."""
    rng = np.random.default_rng([seed, person, k])
    scene = fetch_scene(rng)
    truth = true_map(scene)
    obj_pos = np.array([o.position for o in truth.objects])
    colors = tuple(o.color for o in truth.objects)
    intended = int(rng.integers(0, 4))
    said: Frame = "speaker" if rng.random() < p_own else "robot"

    viewers = {"speaker": truth.human_pose, "robot": truth.robot_pose}
    candidates = truth.by_color(colors[intended])
    wanted_id = truth.objects[intended].object_id
    left = perspective.pick(candidates, "left", viewers[said]).object_id
    side = "left" if left == wanted_id else "right"
    ids = [o.object_id for o in truth.objects]
    picks = {f: ids.index(perspective.pick(candidates, side, viewers[f]).object_id) for f in FRAMES}

    arm_target = picks[choose]
    t = frame_times()
    gripper = arm_path(t, HOME, obj_pos[arm_target] + [0.0, 0.0, HOVER])
    reaction = simulate_reaction(t, scene.human_eye, obj_pos, intended, arm_target, gripper,
                                 style, rng)
    ep = Episode(
        t=t, obj_pos=obj_pos, colors=colors, request_color=colors[intended], intended=intended,
        arm_target=arm_target, gripper=gripper, head_pos=scene.human_eye,
        head_fwd=reaction.head_fwd, speech=reaction.speech, t_no=reaction.t_no,
        family=style.family, seed=seed,
    )
    return Interaction(ep, picks, said)


def _other_candidate(ep: Episode) -> int:
    others = [j for j in np.flatnonzero(ep.candidates) if j != ep.arm_target]
    return int(others[0]) if others else ep.arm_target


def label_for(source: str, ep: Episode, critic: Critic | None, threshold: float) -> int | None:
    """The block this source says the person wanted, or None for no label."""
    if source == "none":
        return None
    if source == "oracle":
        return ep.intended
    heard_no = any(text.startswith("no") for _, text in ep.speech)
    if source == "spoken":
        return _other_candidate(ep) if heard_no else ep.arm_target
    if source == "critic+spoken" and heard_no:
        return _other_candidate(ep)
    if source in ("critic", "critic+spoken"):
        if critic is None:
            raise ValueError(f"source {source!r} needs a critic")
        p_wrong, intended = critic.predict(ep)
        if p_wrong[-1] >= threshold:
            return int(np.argmax(intended[-1]))
        if p_wrong[-1] <= 1.0 - threshold:
            return ep.arm_target
        return None
    raise ValueError(f"unknown source: {source}")


def run_person(
    seed: int, person: int, person_type: str, family: str, source: str, n: int,
    critic: Critic | None, threshold: float,
) -> list[Step]:
    """n interactions for one simulated person, learning from one source."""
    style = sample_style(np.random.default_rng([seed, person]), family)
    p_own = PERSON_TYPES[person_type]
    belief = FrameBelief()
    steps = []
    for k in range(n):
        it = interaction(seed, person, k, p_own, style, belief.choose())
        ep = it.episode
        label = label_for(source, ep, critic, threshold)
        ambiguous = it.picks["speaker"] != it.picks["robot"]
        if label is not None and ambiguous:
            for f in FRAMES:
                if it.picks[f] == label:
                    belief.update(f)
        steps.append(Step(
            person, person_type, family, source, k, ambiguous, ep.wrong, label,
            None if label is None else label == ep.intended,
        ))
    return steps


def run(
    people_per_cell: int, n: int, seed: int, critic: Critic | None, threshold: float,
    sources: tuple[str, ...] = SOURCES,
) -> list[Step]:
    """Every source on the same people: people_per_cell for each person type and
    reaction family."""
    steps: list[Step] = []
    person = 0
    for person_type in PERSON_TYPES:
        for family in FAMILIES:
            for _ in range(people_per_cell):
                for source in sources:
                    steps += run_person(seed, person, person_type, family, source, n, critic,
                                        threshold)
                person += 1
    return steps


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(steps: list[Step], last: int = 5) -> dict:
    """Per source: wrong-reach rate overall and over each person's last `last`
    interactions, label coverage (labels per interaction), label accuracy, and
    the same rates split by person type and by reaction family."""

    def block(sel: list[Step]) -> dict:
        n_max = max((s.k for s in sel), default=-1) + 1
        labelled = [s for s in sel if s.label is not None]
        return {
            "n": len(sel),
            "wrong": _mean([float(s.wrong) for s in sel]),
            "wrong_last": _mean([float(s.wrong) for s in sel if s.k >= n_max - last]),
            "coverage": len(labelled) / len(sel) if sel else None,
            "label_acc": _mean([float(bool(s.label_correct)) for s in labelled]),
        }

    out: dict = {}
    for source in dict.fromkeys(s.source for s in steps):
        mine = [s for s in steps if s.source == source]
        out[source] = {
            "overall": block(mine),
            "by_type": {t: block([s for s in mine if s.person_type == t])
                        for t in dict.fromkeys(s.person_type for s in mine)},
            "by_family": {f: block([s for s in mine if s.family == f])
                          for f in dict.fromkeys(s.family for s in mine)},
            "curve": [_mean([float(s.wrong) for s in mine if s.k == k])
                      for k in range(max(s.k for s in mine) + 1)],
        }
    return out
