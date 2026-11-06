"""Building layouts for the dog demo (T5-35), seeded like the arm's `sim.scenarios`.

A straight corridor along +x with rooms on both sides in 3 m slots. Each room's door has a
coloured frame and a stripe of paint in front of it, so "the red room" means something to both
cameras. The dog enters at the east end facing west; the firefighter is somewhere it can't see
(smoke in between). Kinds:

- corridor: the firefighter at the west end, facing the dog. Two same-colour rooms face each
  other across the corridor: "the <c> room on my <left|right>". Face to face, it flips.
- doorway: the firefighter stands with their back to one wall, facing across. Two same-colour
  rooms on the far wall, one each side of them: their left is along the corridor, while for
  the dog both rooms are on the same side.
- closest: two or three same-colour rooms along the corridor, the firefighter at the west end:
  "the <c> room closest to me / furthest from me". The dog's nearest is the other end.
- rescue: T5-20's fixed scene (T5-34), wrapped so the demo treats it like the others.

A layout knows its rooms (the floor map), the people's poses, the dog's start, the route to
each room (corridor lane, then through the door as T5-20's rescue route) and how to build its
MuJoCo spec with the Go2 attached."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from curio_robot.sim.world import COLOR_RGBA

KINDS = ("corridor", "doorway", "closest", "rescue")
NEW_KINDS = ("corridor", "doorway", "closest")
COLORS = ("red", "blue", "green", "yellow")
SLOT = 3.0  # m along the corridor per room
WEST = -0.4  # x of the corridor's west end wall
FIRST = 0.5  # x where the first slot starts
EAST_LOBBY = 2.6  # m of corridor past the last slot, where the dog starts
HALF_W = 0.9  # corridor half width (wall centre lines)
ROOM_D = 3.0  # room depth beyond the corridor wall
DOOR_HALF = 0.55
LANE = 0.35  # the dog keeps this far off the centre line, away from a firefighter by a wall
FF_Z = 0.94
FF_ANGLES = {"neck_pitch": 0.05, "l_elbow": -0.15, "r_elbow": -0.15}
FF_RESCUE = {"neck_pitch": 0.05, "l_sh_pitch": -0.5, "l_sh_abd": 1.45, "l_elbow": -0.1}
SURVIVOR_ANGLES = {"l_sh_abd": 0.5, "r_sh_abd": -0.9, "l_knee": 0.4}


@dataclass(frozen=True)
class Room:
    color: str
    door_x: float
    side: int  # +1 north (y > 0), -1 south
    slot: int

    @property
    def center(self) -> np.ndarray:
        return np.array([self.door_x, self.side * (HALF_W + ROOM_D / 2), 1.0])

    @property
    def goal(self) -> np.ndarray:
        """Where the dog stops: just inside, as T5-20's rescue goal (door x + 0.25, 2.3)."""
        return np.array([self.door_x + 0.25, 2.3 * self.side])


@dataclass
class Layout:
    kind: str
    seed: int
    request: str
    rooms: dict[str, Room]
    ff: tuple[float, float, float]  # firefighter x, y, yaw (facing)
    start: tuple[float, float, float]  # dog x, y, yaw
    length: float
    lane: float = 0.0
    smoke: list[float] = field(default_factory=list)
    ff_angles: dict = field(default_factory=lambda: dict(FF_ANGLES))

    # ---- what DogWorld asks of a layout --------------------------------------------------

    def room_xyz(self) -> dict[str, np.ndarray]:
        return {rid: r.center for rid, r in self.rooms.items()}

    def color_of(self, rid: str) -> str:
        return self.rooms[rid].color

    def route(self, rid: str) -> tuple[list, np.ndarray]:
        r = self.rooms[rid]
        x, s = r.door_x, r.side
        pts = [(x + 0.8, self.lane), (x, 0.6 * s), (x + 0.05, 1.7 * s), tuple(r.goal)]
        return pts, r.goal

    def survivor(self, rid: str) -> tuple[tuple[float, float, float], float]:
        r = self.rooms[rid]
        return (r.door_x + 0.4, r.side * (HALF_W + ROOM_D * 0.66), 0.15), -r.side * np.pi * 0.15

    def overview(self) -> dict:
        mid = (WEST + self.length) / 2
        return {"lookat": (mid, 0.0, 0.0), "distance": 0.95 * self.length + 3.0,
                "azimuth": 100.0, "elevation": -62.0}

    def spec(self) -> mujoco.MjSpec:
        from kit import PERSON, Palette, humanoid, shell
        from phys_core import attach
        from phys_go2 import go2_spec
        from phys_scenes import RESCUE_YELLOW, with_gravity
        from scenes_a import box, wall

        w = [wall(WEST, -HALF_W, WEST, HALF_W), wall(self.length, -HALF_W, self.length, HALF_W)]
        paint = 'contype="0" conaffinity="0"'
        for s in (1, -1):
            y = HALF_W * s
            gaps = sorted(r.door_x for r in self.rooms.values() if r.side == s)
            x0 = WEST
            for gx in gaps:
                w.append(wall(x0, y, gx - DOOR_HALF, y))
                x0 = gx + DOOR_HALF
            w.append(wall(x0, y, self.length, y))
        for r in self.rooms.values():
            s, a = r.side, FIRST + SLOT * r.slot
            b, y0, y1 = a + SLOT, HALF_W * r.side, (HALF_W + ROOM_D) * r.side
            w += [wall(a, y0, a, y1), wall(b, y0, b, y1), wall(a, y1, b, y1)]
            rgba = " ".join(f"{c:.2f}" for c in COLOR_RGBA[r.color])
            for jx in (r.door_x - DOOR_HALF - 0.04, r.door_x + DOOR_HALF + 0.04):  # door frame
                w.append(box((jx, y0, 0.8), (0.05, 0.1, 0.8), rgba, paint))
            w.append(box((r.door_x, y0, 1.55), (DOOR_HALF + 0.09, 0.1, 0.07), rgba, paint))
            w.append(box((r.door_x, y0 - 0.3 * s, 0.003), (DOOR_HALF, 0.22, 0.003), rgba, paint))
            # a tinted floor inside, so the overview shows which room is which
            w.append(box(((a + b) / 2, (y0 + y1) / 2, 0.002), (SLOT / 2 - 0.08, ROOM_D / 2 - 0.08,
                                                              0.002),
                         " ".join(f"{0.55 + 0.35 * c:.2f}" for c in COLOR_RGBA[r.color][:3])
                         + " 1", paint))
        smoke = "".join(box((x, 0, 0.9), (0.5, 0.85, 0.9), "0.55 0.55 0.57 0.25", paint)
                        for x in self.smoke)
        survivor = humanoid("s_", Palette((0.55, 0.15, 0.15), (0.8, 0.62, 0.5),
                                          (0.8, 0.62, 0.5), (0.9, 0.9, 0.9)), head_cam=False)
        ff = humanoid("p_", Palette((0.45, 0.36, 0.2), PERSON.skin, (0.1, 0.1, 0.12),
                                    (0.95, 0.95, 0.2)))
        xml = shell("".join(w) + smoke + survivor + ff, floor="concrete",
                    sky=((0.32, 0.3, 0.3), (0.12, 0.11, 0.11)), ambient=(0.3, 0.29, 0.28))
        spec = mujoco.MjSpec.from_string(with_gravity(xml))
        attach(spec, go2_spec(color=RESCUE_YELLOW), "base", "go2_")
        return spec


class RescueLayout(Layout):
    """T5-34's scene: T5-20's PhysDog geometry, rooms room_n / room_s (no door colours)."""

    def __init__(self) -> None:
        from phys_scenes import PhysDog

        self._scene = PhysDog()
        super().__init__("rescue", 0, "Survivor's in the room on my left.", {},
                         (-1.2, 0.0, 0.0), self._scene.start()[:2] + (np.pi,), 9.9,
                         ff_angles=dict(FF_RESCUE))
        self._rooms = {c.oid: np.asarray(c.pos, float) for c in self._scene.candidates}

    def room_xyz(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._rooms.items()}

    def color_of(self, rid: str) -> str:
        return "grey"

    def route(self, rid: str) -> tuple[list, np.ndarray]:
        (leg,) = self._scene.legs({"geo": rid})
        return leg.route, np.asarray(leg.goal, float)

    def survivor(self, rid: str) -> tuple[tuple[float, float, float], float]:
        s = 1 if self._rooms[rid][1] > 0 else -1
        return (2.4, 2.9 * s, 0.15), np.pi * 0.85 * s

    def overview(self) -> dict:
        return {"lookat": (3.2, 0.0, 0.0), "distance": 12.5, "azimuth": 112.0,
                "elevation": -58.0}

    def spec(self) -> mujoco.MjSpec:
        return self._scene.spec()


# ---- generators ----------------------------------------------------------------------------


def _frame(rng: np.random.Generator) -> tuple[int, float]:
    n = int(rng.integers(3, 5))  # slots per side
    return n, FIRST + SLOT * n + EAST_LOBBY


def _door(rng: np.random.Generator, slot: int) -> float:
    return round(FIRST + SLOT * (slot + 0.5) + float(rng.uniform(-0.3, 0.3)), 3)


def _add(rooms: dict[str, Room], rng, color: str, slot: int, side: int) -> None:
    n = sum(r.color == color for r in rooms.values()) + 1
    rooms[f"{color}_{n}"] = Room(color, _door(rng, slot), side, slot)


def _others(rooms: dict[str, Room], rng, n_slots: int, avoid: str, k: int,
            sides=(1, -1)) -> None:
    """k more rooms in free slots, in colours other than `avoid`."""
    free = [(s, d) for s in range(n_slots) for d in sides
            if not any(r.slot == s and r.side == d for r in rooms.values())]
    others = [c for c in COLORS if c != avoid]
    for i in rng.permutation(len(free))[:k]:
        _add(rooms, rng, str(rng.choice(others)), *free[i])


def _smoke(rng, length: float) -> list[float]:
    return [round(float(x), 2) for x in rng.uniform(0.45 * length, 0.75 * length, 2)]


def corridor(seed: int) -> Layout:
    rng = np.random.default_rng([1, seed])
    n, length = _frame(rng)
    c = str(rng.choice(COLORS))
    rooms: dict[str, Room] = {}
    k = int(rng.integers(0, n))
    for side in (1, -1):
        _add(rooms, rng, c, k, side)
    _others(rooms, rng, n, c, int(rng.integers(1, 4)))
    side = str(rng.choice(["left", "right"]))
    return Layout("corridor", seed, f"The {c} room on my {side}.", rooms, (0.6, 0.0, 0.0),
                  (length - 1.2, 0.0, np.pi), length, smoke=_smoke(rng, length))


def doorway(seed: int) -> Layout:
    rng = np.random.default_rng([2, seed])
    n, length = _frame(rng)
    c = str(rng.choice(COLORS))
    wall_side = int(rng.choice([1, -1]))  # the firefighter's back is to this wall
    k1, k2 = sorted(int(v) for v in rng.choice(n, 2, replace=False))
    rooms: dict[str, Room] = {}
    _add(rooms, rng, c, k1, -wall_side)
    _add(rooms, rng, c, k2, -wall_side)
    _others(rooms, rng, n, c, int(rng.integers(1, 3)))
    x1, x2 = (r.door_x for r in rooms.values() if r.color == c)
    xf = round((x1 + x2) / 2, 3)
    side = str(rng.choice(["left", "right"]))
    ff = (xf, 0.5 * wall_side, -wall_side * np.pi / 2)  # facing across, away from their wall
    return Layout("doorway", seed, f"The {c} room on my {side}.", rooms, ff,
                  (length - 1.2, 0.0, np.pi), length, lane=-LANE * wall_side,
                  smoke=[x for x in _smoke(rng, length) if abs(x - xf) > 1.0])


def closest(seed: int) -> Layout:
    rng = np.random.default_rng([3, seed])
    n, length = _frame(rng)
    c = str(rng.choice(COLORS))
    m = int(rng.integers(2, 4))
    rooms: dict[str, Room] = {}
    for k in sorted(int(v) for v in rng.choice(n, m, replace=False)):
        _add(rooms, rng, c, k, int(rng.choice([1, -1])))
    _others(rooms, rng, n, c, int(rng.integers(0, 3)))
    rel = str(rng.choice(["closest to me", "furthest from me"]))
    yaw = float(rng.choice([0.0, np.pi]))
    return Layout("closest", seed, f"The {c} room {rel}.", rooms, (0.6, 0.0, yaw),
                  (length - 1.2, 0.0, np.pi), length, smoke=_smoke(rng, length))


def make(kind: str, seed: int = 0) -> Layout:
    if kind == "rescue":
        return RescueLayout()
    return {"corridor": corridor, "doorway": doorway, "closest": closest}[kind](seed)
