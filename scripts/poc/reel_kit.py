"""Shared pieces for the T5-19 versatility reel, on top of the T5-18 kit: the reel's scene
interface (beats with world points instead of grounder chips), the maths the tests check
(reach radius, points in the person's frame, the eye-to-fingertip ray), a few new props and
the reel's overlays (cold open, payoff grid). T5-18 files are imported, never changed, so
curio_poc.mp4 renders exactly as before."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import compose as C
import numpy as np
from kit import Path, min_jerk
from PIL import Image, ImageDraw
from scene_base import Scene, window

REACH = 0.72  # m from the shoulder: a standing adult's comfortable reach (arm ~0.6 + lean)


# ---- beats ---------------------------------------------------------------------------------


@dataclass
class Beat:
    t0: float
    t1: float
    kind: str  # say, x, ok, ray, pulse, flash
    text: str = ""
    pos: tuple[float, float, float] | None = None
    pos2: tuple[float, float, float] | None = None


@dataclass
class ReelScene(Scene):
    word: str = ""  # the payoff-grid label
    real: tuple[str, ...] = ()  # query keys sent to the live grounder
    good: bool = True  # filled in after grounding: did Curio's beat land on the right object
    outcome: str = ""
    trail: list = field(default_factory=list)  # world points for the ghost trail

    def beats(self, answers: dict[str, str | None]) -> list[Beat]:
        raise NotImplementedError

    def events(self, answers, info):  # T5-18 interface, unused by the reel
        return []

    def inset_cam(self) -> str:
        return self.person + "head_cam"


def answer_or(answers: dict, key: str, default: str) -> str:
    a = answers.get(key)
    return default if a is None else a


# ---- maths the tests check -----------------------------------------------------------------


def in_reach(shoulder, point, radius: float = REACH) -> bool:
    return float(np.linalg.norm(np.asarray(point, float) - np.asarray(shoulder, float))) <= radius


def person_point(pos, yaw: float, local) -> np.ndarray:
    """A point given in the person's body frame (x forward, y left, z up) in world
    coordinates, for a person at pos turned yaw about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly, lz = local
    return np.asarray(pos, float) + np.array([c * lx - s * ly, s * lx + c * ly, lz])


def ray_hit_plane_y(origin, through, y: float) -> np.ndarray | None:
    """Where the ray from origin through `through` meets the plane y = const (None if it
    runs parallel or away)."""
    o, d = np.asarray(origin, float), np.asarray(through, float) - np.asarray(origin, float)
    if abs(d[1]) < 1e-9:
        return None
    k = (y - o[1]) / d[1]
    return o + k * d if k > 0 else None


def pointed_door(origin, through, doors: dict[str, float], wall_y: float) -> str | None:
    """The door (id -> x of its centre on the wall y = wall_y) nearest where the ray lands."""
    hit = ray_hit_plane_y(origin, through, wall_y)
    if hit is None:
        return None
    return min(doors, key=lambda k: abs(doors[k] - hit[0]))


# ---- props ---------------------------------------------------------------------------------


def ghost(fragment: str, alpha: float = 0.28, tint: str | None = None) -> str:
    """The same MJCF fragment with every rgba made translucent (and optionally one colour),
    cameras and lights removed, collisions off: a ghost copy for the robot-only beat."""
    def sub(m):
        vals = m.group(1).split()
        rgb = tint.split()[:3] if tint else vals[:3]
        return f'rgba="{" ".join(rgb)} {alpha}" contype="0" conaffinity="0"'
    out = re.sub(r'rgba="([^"]+)"', sub, fragment)
    out = re.sub(r"<camera[^>]*/>", "", out)
    out = re.sub(r"<light[^>]*/>", "", out)
    return out.replace('material="gloss"', "")


def follow_bot(p: str = "fb_") -> str:
    """A small two-wheeled follow robot: round body, mast with a status light, front camera."""
    return f"""
  <body name="{p}root" pos="0 0 0">
    <freejoint name="{p}free"/>
    <geom type="cylinder" pos="0 0 0.17" size="0.2 0.09" rgba="0.92 0.93 0.94 1"
          material="gloss"/>
    <geom type="cylinder" pos="0 0 0.27" size="0.17 0.012" rgba="0.1 0.62 0.68 1"/>
    <geom type="cylinder" pos="0 0.2 0.1" euler="1.5708 0 0" size="0.1 0.03"
          rgba="0.14 0.15 0.17 1"/>
    <geom type="cylinder" pos="0 -0.2 0.1" euler="1.5708 0 0" size="0.1 0.03"
          rgba="0.14 0.15 0.17 1"/>
    <geom type="capsule" fromto="-0.05 0 0.28 -0.05 0 0.55" size="0.018" rgba="0.3 0.3 0.33 1"/>
    <geom type="box" pos="0.0 0 0.58" size="0.06 0.09 0.04" rgba="0.14 0.15 0.17 1"/>
    <geom type="box" pos="0.062 0 0.58" size="0.004 0.06 0.02" rgba="0.2 0.8 0.95 1"/>
    <camera name="{p}cam" pos="0.07 0 0.58" xyaxes="0 -1 0 0 0 1" fovy="75"/>
  </body>"""


def coral_tree(x: float, y: float, seed: int = 11) -> str:
    """A branching coral cluster: a low rock, then trunks that fork twice into thinner
    tapered capsules, plus a flat fan and a few short tube polyps. No round caps, so it
    reads as coral rather than a patch of mushrooms. Same seed, same cluster."""
    rng = np.random.default_rng(seed)
    g = [(f'<geom type="ellipsoid" pos="{x} {y} 0.04" size="0.34 0.3 0.12" '
          'rgba="0.40 0.36 0.32 1"/>'),
         (f'<geom type="ellipsoid" pos="{x + 0.2} {y - 0.18} 0.03" size="0.2 0.16 0.09" '
          'rgba="0.36 0.34 0.3 1"/>')]
    cols = ("0.95 0.45 0.5 1", "0.98 0.62 0.32 1", "0.78 0.38 0.72 1")

    def branch(p0, d, length, r, depth, col):
        p1 = p0 + d * length
        g.append(f'<geom type="capsule" fromto="{p0[0]:.3f} {p0[1]:.3f} {p0[2]:.3f} '
                 f'{p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}" size="{r:.3f}" rgba="{col}"/>')
        if depth == 0:
            return
        for _ in range(2):
            nd = d + rng.normal(0, 0.55, 3)
            nd[2] = abs(nd[2]) + 0.6
            nd /= np.linalg.norm(nd)
            branch(p1, nd, length * 0.72, r * 0.68, depth - 1, col)

    for k in range(7):
        a = 2 * np.pi * k / 7 + rng.uniform(-0.3, 0.3)
        rad = 0.08 + 0.12 * (k % 2)
        base = np.array([x + rad * np.cos(a), y + rad * np.sin(a), 0.1])
        d = np.array([0.35 * np.cos(a), 0.35 * np.sin(a), 1.0])
        branch(base, d / np.linalg.norm(d), rng.uniform(0.16, 0.22), 0.028, 3, cols[k % 3])
    g.append(f'<geom type="ellipsoid" pos="{x - 0.22} {y + 0.05} 0.3" size="0.015 0.18 0.17" '
             'rgba="0.62 0.3 0.62 0.9"/>')
    for k in range(6):
        px, py = x + rng.uniform(-0.35, 0.35), y + rng.uniform(-0.3, 0.3)
        g.append(f'<geom type="cylinder" pos="{px:.3f} {py:.3f} 0.2" size="0.025 0.07" '
                 'rgba="0.95 0.85 0.4 1"/>')
    return "".join(g)


# ---- a reaching arm ------------------------------------------------------------------------

L1, L2 = 0.5, 0.5  # upper arm and forearm of reach_arm, m
SH_H = 0.2  # shoulder height above the arm's base, m
GRIP = 0.12  # wrist to grip point (the gripper points straight down), m


def reach_arm(p: str, pos, col: str = "0.95 0.45 0.12 1") -> str:
    """A fixed-base arm with yaw, shoulder, elbow and wrist hinges; set_arm solves it so the
    gripper hangs straight down over a grip point."""
    dark = "0.14 0.15 0.17 1"
    return f"""
  <body name="{p}base" pos="{pos[0]} {pos[1]} {pos[2]}">
    <geom type="cylinder" pos="0 0 0.03" size="0.12 0.03" rgba="{dark}"/>
    <body name="{p}turn" pos="0 0 0.06">
      <joint name="{p}yaw" axis="0 0 1"/>
      <geom type="cylinder" pos="0 0 0.06" size="0.075 0.07" rgba="{col}" material="gloss"/>
      <body name="{p}upper" pos="0 0 {SH_H - 0.06}">
        <joint name="{p}sh" axis="0 1 0"/>
        <geom type="sphere" size="0.065" rgba="{dark}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {L1}" size="0.045" rgba="{col}"
              material="gloss"/>
        <body name="{p}fore" pos="0 0 {L1}">
          <joint name="{p}el" axis="0 1 0"/>
          <geom type="sphere" size="0.052" rgba="{dark}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {L2}" size="0.036" rgba="{col}"
                material="gloss"/>
          <body name="{p}hand" pos="0 0 {L2}">
            <joint name="{p}wr" axis="0 1 0"/>
            <geom type="sphere" size="0.042" rgba="{dark}"/>
            <geom type="box" pos="0 0 0.04" size="0.03 0.055 0.02" rgba="{dark}"/>
            <geom type="box" pos="0 0.04 0.09" size="0.012 0.008 0.04" rgba="{dark}"/>
            <geom type="box" pos="0 -0.04 0.09" size="0.012 0.008 0.04" rgba="{dark}"/>
          </body>
        </body>
      </body>
    </body>
  </body>"""


def arm_angles(base, grip) -> dict[str, float]:
    """yaw, sh, el, wr putting reach_arm's grip point at `grip` (clamped to its reach)."""
    wrist = np.asarray(grip, float) + [0, 0, GRIP]
    d = wrist - (np.asarray(base, float) + [0, 0, SH_H])
    yaw = float(np.arctan2(d[1], d[0]))
    r, h = float(np.hypot(d[0], d[1])), float(d[2])
    dist = float(np.clip(np.hypot(r, h), 0.1, L1 + L2 - 1e-3))
    b = float(np.arccos(np.clip((dist ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)))
    a = float(np.arctan2(r, h) - np.arctan2(L2 * np.sin(b), L1 + L2 * np.cos(b)))
    return {"yaw": yaw, "sh": a, "el": b, "wr": float(np.pi - a - b)}


def set_arm(model, data, p: str, base, grip) -> None:
    for k, v in arm_angles(base, grip).items():
        data.qpos[model.joint(p + k).qposadr[0]] = v


def track(t: float, keys: list[tuple[float, tuple]]) -> np.ndarray:
    """Piecewise min-jerk between (time, point) keyframes; held before the first and after
    the last."""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], float)
    for (t0, p0), (t1, p1) in pairwise(keys):
        if t <= t1:
            f = min_jerk((t - t0) / max(1e-6, t1 - t0))
            return np.asarray(p0, float) + f * (np.asarray(p1, float) - np.asarray(p0, float))
    return np.asarray(keys[-1][1], float)


def two_stage(cache: dict, start, wrong, right, t: float, times, frac: float = 0.42,
              mid=None):
    """Ground motion for a naive-then-corrected beat: head a fraction `frac` of the way toward
    the wrong goal over times[0:2], then reroute to the right goal over times[2:4] (through
    `mid` if given). Returns (path, tau) for kit's drive(); caches the two paths."""
    n0, n1, r0, r1 = times
    start, wrong, right = (np.asarray(v, float) for v in (start, wrong, right))
    stop = start + frac * (wrong - start)
    if "n" not in cache:
        cache["n"] = Path([start, stop])
        pts = [stop, right] if mid is None else [stop, np.asarray(mid, float), right]
        cache["r"] = Path(pts)
    if t < r0:
        return cache["n"], window(t, n0, n1)
    return cache["r"], window(t, r0, r1)


# ---- overlays ------------------------------------------------------------------------------


def ray(d, p0, p1, col, width, a) -> None:
    d.line((*p0, *p1), fill=(*col, int(255 * a)), width=width)
    d.ellipse((p1[0] - 8, p1[1] - 8, p1[0] + 8, p1[1] + 8), fill=(*col, int(255 * a)))


def pulse(d, c, t: float, text: str, a: float) -> None:
    """An expanding ring plus a small tag: something was remembered."""
    x, y = c
    f = (t % 0.8) / 0.8
    r = 30 + 70 * f
    d.ellipse((x - r, y - r, x + r, y + r), outline=(120, 200, 255, int(220 * a * (1 - f))),
              width=6)
    fnt = C.font(30, True)
    tw = d.textlength(text, font=fnt)
    C.pill(d, (x - tw / 2 - 18, y - 120, x + tw / 2 + 18, y - 72), a, fill=(20, 60, 100))
    d.text((x, y - 96), text, font=fnt, fill=(210, 235, 255, int(255 * a)), anchor="mm")


def cold_open(t: float) -> np.ndarray:
    """Black, four lines stacking up about 0.5 s apart, then the title."""
    img = Image.new("RGB", (C.W, C.H), (0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    lines = ("The one on my left.", "That one.", "The one I can't reach.", "Where did I put it?")
    for i, text in enumerate(lines):
        a = C.alpha_of(t, 0.3 + 0.55 * i, 2.95, 0.3)
        if a > 0:
            d.text((C.W / 2, 330 + 110 * i), f"“{text}”", font=C.font(70, True),
                   fill=(255, 255, 255, int(255 * a)), anchor="mm")
    a = C.alpha_of(t, 3.1, 5.2, 0.35)
    if a > 0:
        d.text((C.W / 2, C.H / 2), "Robots should understand you", font=C.font(88, True),
               fill=(255, 255, 255, int(255 * a)), anchor="mm")
        d.text((C.W / 2, C.H / 2 + 100), "from where you stand.", font=C.font(88, True),
               fill=(255, 255, 255, int(255 * a)), anchor="mm")
    return np.asarray(img)


def payoff(tiles: list[tuple[np.ndarray, str, bool, str]], t: float) -> np.ndarray:
    """3x3 grid of final frames, one word each, a check (or the real miss) on each, the
    title on top and the line at the bottom."""
    img = Image.new("RGBA", (C.W, C.H), (10, 11, 14, 255))
    tw, th, gx, gy = 448, 252, 22, 48
    x0 = (C.W - 3 * tw - 2 * gx) // 2
    y0 = 118
    for i, (frame, word, good, outcome) in enumerate(tiles):
        a = C.alpha_of(t, 0.12 * i, 99, 0.4)
        pic = Image.fromarray(frame).resize((tw, th), Image.LANCZOS).convert("RGBA")
        pic.putalpha(int(255 * a))
        x, y = x0 + (i % 3) * (tw + gx), y0 + (i // 3) * (th + gy)
        img.alpha_composite(pic, (x, y))
        d = ImageDraw.Draw(img)
        C.check_mark(d, (x + tw - 34, y + 34), 24, good, a)
        d.text((x, y + th + 6), word, font=C.font(30, True), fill=(255, 255, 255, int(255 * a)))
        if outcome:
            d.text((x + tw, y + th + 12), outcome, font=C.font(22),
                   fill=(*(C.GREEN if good else C.RED), int(255 * a)), anchor="ra")
    d = ImageDraw.Draw(img)
    a = C.alpha_of(t, 1.4, 99, 0.5)
    d.text((C.W / 2, 62), "Curio: robots that see it your way.", font=C.font(64, True),
           fill=(255, 255, 255, int(255 * a)), anchor="mm")
    a = C.alpha_of(t, 3.0, 99, 0.5)
    d.text((C.W / 2, C.H - 42), "“One wearable camera. Any robot.”",
           font=C.font(44, True), fill=(255, 255, 255, int(255 * a)), anchor="mm")
    return np.asarray(img.convert("RGB"))
