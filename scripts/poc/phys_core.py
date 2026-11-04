"""T5-20 shared physics pieces: where the downloaded robots live, attaching them into a scene
with MjSpec, the pure-pursuit layer that turns a route into velocity commands, the diff-drive
wheel mapping and the force/torque PD that flies the free bodies.

Everything here is plain maths on numpy arrays so the tests can check it without the robots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np

TP = Path("C:/Users/Rohan/dev/Perspective/third_party")
G1_DIR = TP / "unitree_rl_gym"
G1_XML = G1_DIR / "resources/robots/g1_description/g1_12dof.xml"
G1_YAML = G1_DIR / "deploy/deploy_mujoco/configs/g1.yaml"
GO2_XML = TP / "mujoco_menagerie/unitree_go2/go2.xml"
GO2_ENV = TP / "cagataydev__strands-isaaclab-go2-rough-policy/params/env.yaml"
G1_NPZ = TP / "derived/g1_motion.npz"
GO2_NPZ = TP / "derived/go2_rough.npz"
GRAVITY = 9.81


def have_assets() -> bool:
    return all(p.exists() for p in (G1_XML, G1_YAML, GO2_XML, GO2_ENV, G1_NPZ, GO2_NPZ))


def read_flat_yaml(path: Path) -> dict:
    """The deploy configs are flat `key: value` files with inline lists; parse just that
    (the project env has no PyYAML)."""
    out: dict = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        out[key] = val
    text = Path(path).read_text()
    for key, val in list(out.items()):
        if val.startswith("["):  # lists may continue on the next lines
            start = text.index(key + ":")
            body = text[text.index("[", start) + 1:text.index("]", start)]
            out[key] = [float(v) for v in body.replace("\n", " ").split(",") if v.strip()]
        elif val.startswith('"'):
            out[key] = val.strip('"')
        else:
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = float(val)
    return out


# ---- MjSpec helpers ------------------------------------------------------------------------


def load_child(path: Path) -> mujoco.MjSpec:
    """A robot spec with absolute mesh paths, so it can be attached into a scene built from
    a string."""
    child = mujoco.MjSpec.from_file(str(path))
    mdir = Path(path).parent / child.meshdir
    for m in child.meshes:
        m.file = str((mdir / m.file).resolve()).replace("\\", "/")
    child.meshdir = ""
    return child


def attach(parent: mujoco.MjSpec, child: mujoco.MjSpec, root: str, prefix: str,
           pos=(0.0, 0.0, 0.0)) -> None:
    frame = parent.worldbody.add_frame(pos=list(pos))
    frame.attach_body(child.body(root), prefix, "")


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ---- pure pursuit --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pursuit:
    """Tuning for one robot: lookahead, top speeds, when to slow down and stop, and how much
    heading error makes it turn on the spot first."""

    lookahead: float = 0.5
    v_max: float = 0.6
    w_max: float = 0.8
    stop: float = 0.08
    slow: float = 0.5
    turn_first: float = 1.0
    k_heading: float = 1.5
    heading_tol: float = 0.12
    turn_speed: float = 0.0  # forward speed kept while turning toward the route
    creep: float = 0.0  # least forward speed on the way in (so it doesn't crawl to a stop)


def closest_on_polyline(points: np.ndarray, p: np.ndarray, seg0: int = 0) -> tuple[int, float]:
    """(segment index, fraction along it) of the point of the polyline nearest p, searching
    from segment seg0 on so progress never runs backwards."""
    best, bi, bf = np.inf, seg0, 0.0
    for i in range(seg0, len(points) - 1):
        a, b = points[i], points[i + 1]
        d = b - a
        f = float(np.clip((p - a) @ d / max(1e-12, d @ d), 0.0, 1.0))
        dist = float(np.linalg.norm(a + f * d - p))
        if dist < best - 1e-9:
            best, bi, bf = dist, i, f
    return bi, bf


def lookahead_point(points: np.ndarray, seg: int, frac: float, dist: float) -> np.ndarray:
    """The point `dist` further along the polyline from (seg, frac); the end if it runs out."""
    a, b = points[seg], points[seg + 1]
    here = a + frac * (b - a)
    left = dist
    while True:
        rest = float(np.linalg.norm(points[seg + 1] - here))
        if rest >= left or seg + 1 == len(points) - 1:
            if rest < 1e-9:
                return points[seg + 1].copy()
            return here + min(left, rest) * (points[seg + 1] - here) / rest
        left -= rest
        seg += 1
        here = points[seg].copy()


def pursuit_command(x: float, y: float, yaw: float, target, end, heading: float | None,
                    prm: Pursuit) -> tuple[float, float, bool]:
    """(forward speed, yaw rate, done) steering a unicycle at (x, y, yaw) toward the lookahead
    point `target`, slowing near the route's `end` and finally turning to `heading`."""
    to_end = np.asarray(end, float)[:2] - (x, y)
    d_end = float(np.linalg.norm(to_end))
    if d_end < prm.stop:
        if heading is None:
            return 0.0, 0.0, True
        err = wrap(heading - yaw)
        if abs(err) < prm.heading_tol:
            return 0.0, 0.0, True
        return 0.0, float(np.clip(prm.k_heading * err, -prm.w_max, prm.w_max)), False
    to = np.asarray(target, float)[:2] - (x, y)
    alpha = wrap(float(np.arctan2(to[1], to[0])) - yaw)
    ld = max(1e-6, float(np.linalg.norm(to)))
    if abs(alpha) > prm.turn_first:
        w = float(np.clip(prm.k_heading * alpha, -prm.w_max, prm.w_max))
        return prm.turn_speed, w, False
    v = prm.v_max * min(1.0, d_end / prm.slow) * max(0.0, np.cos(alpha))
    v = max(v, prm.creep * max(0.0, np.cos(alpha)))
    w = float(np.clip(2.0 * v * np.sin(alpha) / ld, -prm.w_max, prm.w_max))
    if v < 1e-3:  # near the end and off-line: steer on the spot
        w = float(np.clip(prm.k_heading * alpha, -prm.w_max, prm.w_max))
    return float(v), w, False


class RouteFollower:
    """Pure pursuit along a 2-D route that ends at a goal, optionally with a final heading."""

    def __init__(self, points, heading: float | None = None, prm: Pursuit | None = None) -> None:
        pts = np.asarray(points, float)[:, :2]
        keep = [pts[0]] + [p for a, p in pairwise(pts) if np.linalg.norm(p - a) > 1e-6]
        self.points = np.asarray(keep if len(keep) > 1 else [pts[0], pts[0] + 1e-6])
        self.heading = heading
        self.prm = prm or Pursuit()
        self.seg = 0
        self.done = False
        self.at_end = False

    @property
    def goal(self) -> np.ndarray:
        return self.points[-1]

    def command(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        if self.done:
            return 0.0, 0.0
        to_end = self.goal - (x, y)
        d_end = float(np.linalg.norm(to_end))
        behind = abs(wrap(float(np.arctan2(to_end[1], to_end[0])) - yaw)) > np.pi / 2
        if d_end < self.prm.stop or (d_end < 2 * self.prm.stop and behind):
            self.at_end = True  # latch: from here on only turn to the final heading
        if self.at_end:
            v, w, self.done = pursuit_command(x, y, yaw, self.goal, (x, y), self.heading,
                                              self.prm)
            return v, w
        self.seg, frac = closest_on_polyline(self.points, np.array([x, y]), self.seg)
        tgt = lookahead_point(self.points, self.seg, frac, self.prm.lookahead)
        v, w, _ = pursuit_command(x, y, yaw, tgt, self.goal, self.heading, self.prm)
        return v, w


# ---- wheels and free bodies ----------------------------------------------------------------


def diff_drive(v: float, w: float, track: float, radius: float) -> tuple[float, float]:
    """(left, right) wheel speeds in rad/s for body speed v (m/s) and yaw rate w (rad/s)."""
    return (v - w * track / 2) / radius, (v + w * track / 2) / radius


def force_pd(pos, vel, target, kp: float, kd: float, fmax: float) -> np.ndarray:
    """Spring-damper force toward target, clipped to fmax in magnitude."""
    f = kp * (np.asarray(target, float) - pos) - kd * np.asarray(vel, float)
    n = float(np.linalg.norm(f))
    return f if n <= fmax else f * (fmax / n)


def drag(vel, lin: float, quad: float) -> np.ndarray:
    v = np.asarray(vel, float)
    return -lin * v - quad * np.linalg.norm(v) * v


class Carrot:
    """A setpoint that slides along a 3-D route at a capped speed with smooth start and stop,
    for the force PD to chase."""

    def __init__(self, points, speed: float, t0: float) -> None:
        self.points = np.asarray(points, float)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self.t0 = t0
        self.duration = max(0.5, 1.875 * self.length / speed)  # min-jerk peak = 1.875 mean

    @property
    def goal(self) -> np.ndarray:
        return self.points[-1]

    def at(self, t: float) -> np.ndarray:
        tau = float(np.clip((t - self.t0) / self.duration, 0.0, 1.0))
        s = tau ** 3 * (10 - 15 * tau + 6 * tau * tau) * self.length
        i = int(np.clip(np.searchsorted(self.s, s) - 1, 0, len(self.points) - 2))
        f = (s - self.s[i]) / max(1e-9, self.s[i + 1] - self.s[i])
        return self.points[i] + f * (self.points[i + 1] - self.points[i])


def body_state(model, data, body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(COM position, linear velocity, rotation matrix, angular velocity) in world axes."""
    b = model.body(body).id
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    return data.xipos[b].copy(), vel[3:].copy(), data.xmat[b].reshape(3, 3).copy(), vel[:3].copy()


def attitude_torque(rot: np.ndarray, omega: np.ndarray, z_des: np.ndarray, yaw_des: float,
                    kp: float, kd: float) -> np.ndarray:
    """Torque (world axes) turning the body's z axis onto z_des and its heading onto yaw_des."""
    z = rot[:, 2]
    tilt = np.cross(z, z_des / np.linalg.norm(z_des))
    yaw_err = wrap(yaw_des - float(np.arctan2(rot[1, 0], rot[0, 0])))
    return kp * (tilt + yaw_err * z) - kd * omega
