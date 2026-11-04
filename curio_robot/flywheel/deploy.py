"""Simulated deployment: the current grounder answers front/behind requests; a simulated
person corrects it when it is wrong. Only corrections are logged, plus a small random sample
of the answers the person let stand (so retraining sees some confirmed cases too and does
not learn "always switch")."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from curio_robot.flywheel.log import CorrectionEvent, EventStore, split_request
from curio_robot.flywheel.scenes import KIND, FBItem, objects_dict, pose_dict, request_for
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.prompt import GroundingRequest

CONFIRM_RATE = 0.2  # share of correct answers logged as confirmed
N_SPEAKERS = 24

CORRECTIONS = {  # (frame, relation) -> what a person says after a wrong pick
    ("speaker", "behind"): ("No, behind ME.", "No, the one behind me.", "Not that one, at my back."),
    ("speaker", "front"): ("No, in front of ME.", "No, the one in front of me.",
                           "Not that one, the one I'm looking at."),
    ("robot", "behind"): ("No, behind YOU.", "No, the one behind you."),
    ("robot", "front"): ("No, in front of YOU.", "No, the one in front of you."),
}

Score = Callable[[GroundingRequest], GroundingAnswer]


@dataclass(frozen=True, slots=True)
class Outcome:
    item: FBItem
    pick: str | None
    correct: bool
    event: CorrectionEvent | None  # None: correct and not sampled


def speaker_of(seed: int) -> str:
    return f"spk-{seed % N_SPEAKERS:02d}"


def resolve(item: FBItem, request: GroundingRequest, answer: GroundingAnswer,
            pick: str | None) -> tuple[str, str]:
    """(true target, how it was settled) after a correction. The robot drops the rejected
    object and takes its next most likely same-label one; the person accepts it if right,
    else points at the right one."""
    label = next(o.color for o in item.object_map.objects if o.object_id == item.target_id)
    same = [o.object_id for o in item.object_map.objects
            if o.color == label and o.object_id != pick and o.object_id in request.letters.values()]
    if same:
        retry = max(same, key=lambda i: answer.probs.get(i, 0.0))
        if retry == item.target_id:
            return item.target_id, "retry"
    return item.target_id, "pointing"


def run_episode(item: FBItem, episode: int, score: Score, grounder: str,
                rng: np.random.Generator, store: EventStore | None,
                confirm_rate: float = CONFIRM_RATE) -> Outcome:
    """Ask, compare with what the person meant, log a correction (or a sampled confirm)."""
    request = request_for(item)
    answer = score(request)
    pick = max(answer.probs, key=answer.probs.get) if answer.probs else None
    correct = pick == item.target_id
    if correct and rng.random() >= confirm_rate:
        return Outcome(item, pick, True, None)
    event_id = f"e{episode:05d}-{item.seed}"
    image_rel = f"images/{event_id}.png"
    messages, png = split_request(request, image_rel)
    if correct:
        outcome, correction, (target, how) = "confirmed", None, (item.target_id, "confirmed")
    else:
        options = CORRECTIONS[(item.frame, item.relation)]
        correction = options[int(rng.integers(0, len(options)))]
        outcome, (target, how) = "corrected", resolve(item, request, answer, pick)
    om = item.object_map
    event = CorrectionEvent(
        event_id=event_id, episode=episode, seed=item.seed, speaker_id=speaker_of(item.seed),
        kind=KIND, utterance=item.text, messages=messages, letters=dict(request.letters),
        prompt_version=request.prompt_version, image=image_rel,
        objects=objects_dict(om),
        cameras={"human": pose_dict(om.human_pose), "robot": pose_dict(om.robot_pose)},
        pick=pick, pick_probs={k: float(v) for k, v in answer.probs.items()},
        outcome=outcome, correction=correction, resolved_target=target, resolved_by=how,
        grounder=grounder,
        meta={"frame": item.frame, "relation": item.relation, "pool": item.pool,
              "latency_ms": round(answer.latency_ms, 1), "error": answer.error},
    )
    if store is not None:
        store.append(event, png)
    return Outcome(item, pick, correct, event)

