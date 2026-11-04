"""Task families for teach-by-doing (task T5-23): random demonstrations, their ground-truth
plans, a new layout of the same blocks for the robot, and the goal check.

(a) sort: two colours into "my left" / "my right" zones; (b) stack 2-3 colours in order;
(c) put X to the left or right of Y (the person's); (d) two steps mixing those.

A demonstration is kept only if each placement reads back as what was meant (ego_plan.classify
on the true positions): the person makes their intent clear, as a teacher would. A new layout
is kept only if both Curio and the baseline could place every step (reachable and free), so the
baseline fails, when it fails, because of the frame and not the reach."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from curio_robot.collab.tasks import TaskScene
from curio_robot.ego_plan import (
    BLOCK_HALF,
    FREE,
    ROBOT_FRAME,
    TABLE_POINT,
    PersonFrame,
    Step,
    classify,
    goal_for,
    name_objects,
    reachable,
)
from curio_robot.sim.ego_world import HandMove, Seat
from curio_robot.sim.scenarios import COLORS, WORKSPACE_X, WORKSPACE_Y
from curio_robot.sim.world import BlockSpec

FAMILIES = ("a_sort", "b_stack", "c_beside", "d_mixed")
SEAT_DEG = 40.0  # the person sits within this of straight across the table
SEAT_R = 0.48
EYE_Z = (0.35, 0.41)
START_GAP = 0.055  # m between block centres at the start
SAME_COLOUR_S = 0.04  # m apart along the person's left axis: ids by left-to-right are stable
SORT_BAND = 0.03  # m: sort blocks start this close to the middle line
MAX_TRIES = 400
CLEAR_NEAR = 0.10  # m: blocks other than the reference stay this far from a placement...
CLEAR_ANGLE = np.radians(40)  # ...when they are this close to sideways of it
CLEAR_MOVE = 0.05  # m a demonstrated block travels at least
DEMO_BESIDE = [(0.06, 0.0)]  # the person puts a block squarely beside another
FEASIBLE_MARGIN = 0.01  # m more room the new layout leaves than the robot will need

# The goal check (the card's definitions)
ZONE_EDGE = 0.05
LEFT_RANGE = (0.03, 0.10)
LEFT_ANGLE = np.radians(45)
ON_TABLE_Z = 0.02
ON_XY = 0.015
ON_DZ = (0.018, 0.032)
MOVED = 0.02
TABLE = ((-0.25, 0.75), (-0.45, 0.45))


@dataclass(frozen=True, slots=True)
class Goal:
    var: int  # block index (a variable: any block of its colour may play it)
    relation: str  # "on" | "left_of" | "right_of" | "zone"
    ref: int | str  # block index, or "left" / "right" for a zone


@dataclass(frozen=True, slots=True)
class Episode:
    family: str
    seed: int
    seat: Seat
    colors: tuple[str, ...]
    demo_xy: tuple[tuple[float, float], ...]
    moves: tuple[HandMove, ...]
    goals: tuple[Goal, ...]
    plan: tuple[Step, ...]  # ground truth, in the demo's ids
    new_xy: tuple[tuple[float, float], ...]

    def scene(self, new: bool) -> TaskScene:
        xy = self.new_xy if new else self.demo_xy
        blocks = tuple(BlockSpec(c, p) for c, p in zip(self.colors, xy))
        return TaskScene(self.family, blocks, self.seat.eye, self.seat.look, None, 0.0)


class Reject(Exception):
    """This draw doesn't make a clear demonstration or a usable new layout; draw again."""


def _seat(rng: np.random.Generator) -> Seat:
    th = np.radians(rng.uniform(-SEAT_DEG, SEAT_DEG))
    eye = np.array([*(TABLE_POINT + SEAT_R * np.array([np.cos(th), np.sin(th)])),
                    rng.uniform(*EYE_Z)])
    look = np.array([*(TABLE_POINT + rng.normal(0, 0.02, 2)), 0.0])
    return Seat(eye, look)


def _xyz(xy: tuple[float, float]) -> np.ndarray:
    return np.array([xy[0], xy[1], BLOCK_HALF])


def _layout(rng: np.random.Generator, colors: list[str], frame: PersonFrame, band: list[bool]
            ) -> list[tuple[float, float]]:
    """Start positions: in the workspace, apart, same colours apart sideways; band[i]: near
    the middle line (sort blocks, which must start outside both zones)."""
    out: list[np.ndarray] = []
    for i, c in enumerate(colors):
        for _ in range(200):
            if band[i]:
                p = frame.point(rng.uniform(-SORT_BAND, SORT_BAND), rng.uniform(-0.12, 0.12))
            else:
                p = np.array([rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y), BLOCK_HALF])
            if not reachable(p) or any(np.linalg.norm((p - q)[:2]) < START_GAP for q in out):
                continue
            s = frame.coords(p)[0]
            if any(colors[j] == c and abs(frame.coords(q)[0] - s) < SAME_COLOUR_S
                   for j, q in enumerate(out)):
                continue
            out.append(p)
            break
        else:
            raise Reject("layout")
    return [(float(p[0]), float(p[1])) for p in out]


def _intent(family: str, rng: np.random.Generator) -> tuple[list[str], list[Goal], list[bool]]:
    """Colours, goals in block indices, and which blocks start in the middle band."""
    cols = [str(c) for c in rng.permutation(COLORS)]
    side = str(rng.choice(["left", "right"]))
    other = "right" if side == "left" else "left"

    def extra(n: int) -> list[str]:
        return [str(rng.choice(COLORS)) for _ in range(n)]

    if family == "a_sort":
        na, nb = [(2, 1), (1, 2), (2, 2)][rng.integers(3)]
        colors = [cols[0]] * na + [cols[1]] * nb
        if na + nb == 3 and rng.random() < 0.5:
            colors.append(cols[2])
        goals = [Goal(i, "zone", side if c == cols[0] else other)
                 for i, c in enumerate(colors) if c in cols[:2]]
        order = rng.permutation(len(goals))
        return colors, [goals[k] for k in order], [True] * len(colors)
    if family == "b_stack":
        k = int(rng.integers(2, 4))
        colors = cols[:k] + extra(int(rng.integers(1, 3)) if k == 2 else 1)
        return colors, [Goal(j, "on", j - 1) for j in range(1, k)], [False] * len(colors)
    if family == "c_beside":
        colors = cols[:2] + extra(int(rng.integers(1, 3)))
        return colors, [Goal(0, f"{side}_of", 1)], [False] * len(colors)
    template = int(rng.integers(4))
    colors = cols[:3] + extra(1)
    if template == 0:  # X beside Y, then Z on X
        goals = [Goal(0, f"{side}_of", 1), Goal(2, "on", 0)]
    elif template == 1:  # X into a zone, then Z beside X
        goals = [Goal(0, "zone", side), Goal(2, f"{other}_of", 0)]
    elif template == 2:  # X on Y, then Z beside Y
        goals = [Goal(0, "on", 1), Goal(2, f"{side}_of", 1)]
    else:  # X into one zone, Z into the other
        goals = [Goal(0, "zone", side), Goal(2, "zone", other)]
    band = [template in (1, 3) and i in (0, 2) for i in range(len(colors))]
    return colors, goals, band


def _step(goal: Goal, names: list[str]) -> Step:
    ref = goal.ref if isinstance(goal.ref, str) else names[goal.ref]
    return Step(names[goal.var], goal.relation, ref)


def _demo_moves(goals: list[Goal], pos: list[np.ndarray], names: list[str], frame: PersonFrame
                ) -> list[HandMove]:
    """Where the person puts each block; Reject unless each reads back as intended."""
    pos = [p.copy() for p in pos]
    moves = []
    for g in goals:
        step = _step(g, names)
        current = dict(zip(names, pos))
        target = goal_for(step, current, frame, beside=DEMO_BESIDE)
        if target is None:
            raise Reject("demo goal")
        others = {n: p for n, p in current.items() if n != step.obj}
        if classify(target, others, frame) != Step("", step.relation, step.ref):
            raise Reject("ambiguous")
        if not _clear(target, others, step, frame) or np.linalg.norm(target - pos[g.var]) < CLEAR_MOVE:
            raise Reject("unclear")
        pos[g.var] = target
        moves.append(HandMove(g.var, target))
    return moves


def _clear(target: np.ndarray, others: dict[str, np.ndarray], step: Step,
           frame: PersonFrame) -> bool:
    """No block but the reference is near enough to the placement, sideways, to be read as
    "beside" it, even with a centimetre of error: the demonstration has a margin."""
    for name, q in others.items():
        d = (target - q)[:2]
        dist = float(np.linalg.norm(d))
        if name == step.ref or abs(target[2] - q[2]) >= 0.02 or dist >= CLEAR_NEAR:
            continue
        if abs(d @ frame.left[:2]) / dist >= np.cos(CLEAR_ANGLE):
            return False
    return True


def _feasible(plan: list[Step], pos: list[np.ndarray], names: list[str], frame: PersonFrame
              ) -> bool:
    current = dict(zip(names, (p.copy() for p in pos)))
    for step in plan:
        g = goal_for(step, current, frame, FREE + FEASIBLE_MARGIN)
        if g is None:
            return False
        current[step.obj] = g
    return True


def make_episode(family: str, seed: int) -> Episode:
    rng = np.random.default_rng([FAMILIES.index(family), seed])
    for _ in range(MAX_TRIES):
        try:
            return _draw(family, seed, rng)
        except Reject:
            continue
    raise RuntimeError(f"no episode for {family} seed {seed}")


def _draw(family: str, seed: int, rng: np.random.Generator) -> Episode:
    seat = _seat(rng)
    frame = seat.frame
    colors, goals, band = _intent(family, rng)
    demo_xy = _layout(rng, colors, frame, band)
    pos = [_xyz(p) for p in demo_xy]
    names = name_objects(list(zip(colors, pos)), frame)
    moves = _demo_moves(goals, pos, names, frame)
    plan = [_step(g, names) for g in goals]
    for _ in range(50):
        new_xy = _layout(rng, colors, frame, band)
        new_pos = [_xyz(p) for p in new_xy]
        new_names = name_objects(list(zip(colors, new_pos)), frame)
        if (_feasible(plan, new_pos, new_names, frame)
                and _feasible(plan, new_pos, new_names, ROBOT_FRAME)
                and not satisfied(goals, colors, new_pos, new_pos, frame, moved_check=False)):
            return Episode(family, seed, seat, tuple(colors), tuple(demo_xy), tuple(moves),
                           tuple(goals), tuple(plan), tuple(new_xy))
    raise Reject("new layout")


# ---- the goal check ------------------------------------------------------------------------


def holds(rel: str, p: np.ndarray, q: np.ndarray | str, frame: PersonFrame) -> bool:
    """One goal relation on true block centres (the card's definitions)."""
    if rel == "zone":
        s = frame.coords(p)[0]
        return s >= ZONE_EDGE if q == "left" else s <= -ZONE_EDGE
    assert not isinstance(q, str)
    d = (p - q)[:2]
    if rel == "on":
        return bool(np.linalg.norm(d) <= ON_XY and ON_DZ[0] <= p[2] - q[2] <= ON_DZ[1])
    dist = float(np.linalg.norm(d))
    if not LEFT_RANGE[0] <= dist <= LEFT_RANGE[1] or max(p[2], q[2]) >= ON_TABLE_Z:
        return False
    side = frame.left[:2] if rel == "left_of" else -frame.left[:2]
    return bool(d @ side / dist >= np.cos(LEFT_ANGLE))


def satisfied(goals: tuple[Goal, ...] | list[Goal], colors: tuple[str, ...] | list[str],
              start: list[np.ndarray], final: list[np.ndarray], frame: PersonFrame,
              moved_check: bool = True) -> bool:
    """Some one-to-one assignment of the goals' variables to blocks of their colours makes
    every relation true, with every moved variable's block displaced >= MOVED; and no block
    has left the table."""
    for p in final:
        if not (TABLE[0][0] <= p[0] <= TABLE[0][1] and TABLE[1][0] <= p[1] <= TABLE[1][1]
                and p[2] > 0):
            return False
    vars_ = sorted({g.var for g in goals} | {g.ref for g in goals if not isinstance(g.ref, str)})
    moved = {g.var for g in goals}
    options = [[j for j, c in enumerate(colors) if c == colors[v]] for v in vars_]
    for pick in itertools.product(*options):
        if len(set(pick)) < len(pick):
            continue
        at = dict(zip(vars_, pick))
        if moved_check and any(np.linalg.norm(final[at[v]] - start[at[v]]) < MOVED
                               for v in moved):
            continue
        if all(holds(g.relation, final[at[g.var]],
                     g.ref if isinstance(g.ref, str) else final[at[g.ref]], frame)
               for g in goals):
            return True
    return False
