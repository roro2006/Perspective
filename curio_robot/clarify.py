"""Ask back only when it's worth it. When the grounder's letter probabilities leave the target
unclear, the robot asks one short yes/no question in the speaker's frame ("The one on your
left?") instead of guessing: the one with the highest expected information gain under the
grounder's posterior, then a Bayesian update on the answer. Task T5-22.

Side and distance questions are relative ("the one on your left" = the leftmost of the
contenders of one colour), which is how people read them and how the benchmark's answers are
defined (perspective.pick). The answer model allows a fraction `eps` of wrong answers, so
one "yes" never makes the robot certain and a second question can still overturn it."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPS = 0.1  # answer-noise model: P(wrong answer)
MIN_GAIN_BITS = 0.1  # a question worth less than this isn't asked
MIN_CONTENDER_P = 0.02  # objects below this posterior aren't talked about
MAX_QUESTIONS = 2
MIN_SIDE_GAP = 0.01  # m: "on your left" needs the two this far apart sideways
MIN_DISTANCE_GAP = 0.02  # m: "nearer you" needs this much difference in distance
LANDMARK_NEAR = 0.5  # m: "by the X" / "in front of the X" reach


@dataclass(frozen=True, slots=True)
class Landmark:
    name: str  # as said: "box", "boxes", "window"
    xy: tuple[float, float]  # world, m
    half: float  # half its width, m: how far off the sight line it still blocks
    near: float = LANDMARK_NEAR  # m: how close counts as "by" / "in front of" it


@dataclass(frozen=True, slots=True)
class Candidate:
    object_id: str
    color: str
    lateral: float  # m to the speaker's right (negative: speaker's left)
    distance: float  # horizontal m from the speaker
    landmarks: tuple[str, ...] = ()  # relations as the speaker sees them: "behind the boxes"


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    yes: frozenset[str]  # object ids for which the truthful answer is yes
    kind: str  # landmark, color, side, distance


@dataclass(frozen=True, slots=True)
class Decision:
    choice: str | None
    asked: tuple[tuple[Question, bool], ...]
    prior: dict[str, float]
    posterior: dict[str, float]


Answerer = Callable[[Question], bool]


# ---- geometry ------------------------------------------------------------------------------


def landmark_relation(obj_xy, speaker_xy, landmark: Landmark) -> str | None:
    """How the speaker would place the object against the landmark, horizontal plane:
    "behind the X" when the landmark sits on their sight line before the object, "in front of
    the X" when it sits just beyond it, "by the X" when merely close; None otherwise."""
    near = landmark.near
    obj, eye, lm = (np.asarray(p, float)[:2] for p in (obj_xy, speaker_xy, landmark.xy))
    sight = obj - eye
    length = float(np.linalg.norm(sight))
    if length < 1e-9:
        return None
    u = sight / length
    along = float((lm - eye) @ u)
    off = float(np.linalg.norm(lm - eye - along * u))
    if off <= landmark.half:
        if 0.0 < along < length:
            return f"behind the {landmark.name}"
        if length <= along <= length + near:
            return f"in front of the {landmark.name}"
    if float(np.linalg.norm(obj - lm)) <= near:
        return f"by the {landmark.name}"
    return None


# ---- questions -----------------------------------------------------------------------------


def normalize(probs: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, p) for p in probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs} if probs else {}
    return {k: max(0.0, p) / total for k, p in probs.items()}


def contenders(posterior: dict[str, float], min_p: float = MIN_CONTENDER_P) -> list[str]:
    """Ids worth talking about, most likely first (always at least the top one)."""
    ranked = sorted(posterior, key=lambda k: -posterior[k])
    return [k for i, k in enumerate(ranked) if i == 0 or posterior[k] >= min_p]


def _phrase(relation: str) -> str:
    """Speaker-frame wording: behind / in front of are said "from you"."""
    return relation if relation.startswith("by ") else f"{relation} from you"


def questions(candidates: list[Candidate], posterior: dict[str, float]) -> list[Question]:
    """Every templated question over the contenders, in tie-break order: landmark, colour,
    side, distance. Questions that split the candidates the same way are listed once."""
    by_id = {c.object_id: c for c in candidates}
    live = [by_id[k] for k in contenders(posterior) if k in by_id]
    colours = list(dict.fromkeys(c.color for c in live))
    mixed = len(colours) > 1
    out: list[Question] = []

    relations = list(dict.fromkeys(r for c in live for r in c.landmarks))
    for rel in relations:
        yes = frozenset(c.object_id for c in live if rel in c.landmarks)
        out.append(Question(f"The one {_phrase(rel)}?", yes, "landmark"))
    if mixed:
        for colour in colours:
            yes = frozenset(c.object_id for c in candidates if c.color == colour)
            out.append(Question(f"The {colour} one?", yes, "color"))
    for colour in colours:
        group = [c for c in live if c.color == colour]
        if len(group) < 2:
            continue
        noun = f"{colour} one" if mixed else "one"
        out += _relative(group, noun, "side", lambda c: c.lateral, MIN_SIDE_GAP,
                         ("on your left", "on your right"))
        out += _relative(group, noun, "distance", lambda c: c.distance, MIN_DISTANCE_GAP,
                         ("nearer you", "farther from you"))
    return _dedupe(out, set(posterior))


def _relative(group, noun, kind, key, gap, words) -> list[Question]:
    ranked = sorted(group, key=key)
    found = []
    if key(ranked[1]) - key(ranked[0]) >= gap:
        found.append(Question(f"The {noun} {words[0]}?", frozenset({ranked[0].object_id}), kind))
    if key(ranked[-1]) - key(ranked[-2]) >= gap:
        found.append(Question(f"The {noun} {words[1]}?", frozenset({ranked[-1].object_id}),
                              kind))
    return found


def split_key(question: Question, universe: set[str]) -> frozenset[str]:
    """The same split asked either way round ("left?" / "right?" of two) gets one key."""
    yes = frozenset(question.yes & universe)
    no = frozenset(universe - yes)
    return min(yes, no, key=lambda s: (len(s), sorted(s)))


def _dedupe(found: list[Question], universe: set[str]) -> list[Question]:
    seen: set[frozenset[str]] = set()
    out = []
    for q in found:
        key = split_key(q, universe)
        if key in seen or not (q.yes & universe) or not (universe - q.yes):
            continue
        seen.add(key)
        out.append(q)
    return out


def _h(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def gain(question: Question, posterior: dict[str, float], eps: float = EPS) -> float:
    """Expected information gain about the target, bits, of asking under answer noise eps:
    H(answer) - H(answer | target) = h(q(1-eps) + (1-q)eps) - h(eps), q = P(yes set)."""
    q = sum(p for k, p in posterior.items() if k in question.yes)
    return _h(q * (1 - eps) + (1 - q) * eps) - _h(eps)


def best_question(candidates: list[Candidate], posterior: dict[str, float],
                  asked: tuple[Question, ...] = (), eps: float = EPS,
                  min_gain: float = MIN_GAIN_BITS) -> Question | None:
    """The question with the highest gain (ties: template order), skipping splits already
    asked; None when nothing is worth min_gain bits."""
    universe = set(posterior)
    done = {split_key(q, universe) for q in asked}
    best, best_gain = None, min_gain - 1e-12
    for q in questions(candidates, posterior):
        if split_key(q, universe) in done:
            continue
        g = gain(q, posterior, eps)
        if g > best_gain + 1e-9:
            best, best_gain = q, g
    return best


def update(posterior: dict[str, float], question: Question, answer: bool,
           eps: float = EPS) -> dict[str, float]:
    """Bayes: P(answer | target) = 1 - eps when the truthful answer matches, else eps."""
    like = {k: (1 - eps) if ((k in question.yes) == answer) else eps for k in posterior}
    return normalize({k: p * like[k] for k, p in posterior.items()})


# ---- policy --------------------------------------------------------------------------------


def argmax(posterior: dict[str, float]) -> str | None:
    return max(posterior, key=posterior.get) if posterior else None


def decide(candidates: list[Candidate], probs: dict[str, float], answer: Answerer, tau: float,
           k: int = MAX_QUESTIONS, eps: float = EPS, min_gain: float = MIN_GAIN_BITS
           ) -> Decision:
    """Act if max p >= tau; otherwise ask up to k questions (stopping once max p >= tau or no
    question is worth min_gain), then act on the most likely object."""
    prior = normalize(probs)
    post = dict(prior)
    asked: list[tuple[Question, bool]] = []
    while post and max(post.values()) < tau and len(asked) < k:
        q = best_question(candidates, post, tuple(a[0] for a in asked), eps, min_gain)
        if q is None:
            break
        a = bool(answer(q))
        asked.append((q, a))
        post = update(post, q, a, eps)
    return Decision(argmax(post), tuple(asked), prior, post)


def truthful(target: str) -> Answerer:
    return lambda q: target in q.yes


def noisy(target: str, rng: np.random.Generator, flip: float = EPS) -> Answerer:
    """Truthful, except each answer is flipped with probability `flip`."""
    honest = truthful(target)
    return lambda q: honest(q) != bool(rng.random() < flip)
