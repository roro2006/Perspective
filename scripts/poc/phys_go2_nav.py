"""T5-32: the Menagerie Go2 driven by kouylty's pretrained Isaac Lab navigation policy
(Isaac-Go2-Navigation-Rough-v0, MIT) on top of the walking policy it was trained with, in
place of T5-20's hand-written path follower. Both are rsl_rl MLPs run here in numpy.

Walking policy (locomotion/policy.pt, 235 -> 512 -> 256 -> 128 -> 12, ELU, Identity
normaliser): the same 235-d layout as Isaac Lab's Go2 rough velocity task, so phys_go2's port
is reused unchanged (Isaac breadth-first joint order, default pose, DCMotor PD 25 / 0.5, action
scale 0.25, 50 Hz, flat height scan).

Navigation policy (navigation/policy.pt, 203 -> 256 -> 128 -> 64 -> 3, ELU, empirical
normaliser (x - mean) / (std + 0.01)). The 203-d observation, worked out from the normaliser
statistics, the first-layer weights and probes (see tasks/notes/T5-32.md):
  [0:3]    base linear velocity, base frame         (mean vx 0.45 m/s)
  [3:6]    base angular velocity, base frame
  [6:9]    projected gravity                        (mean z -0.98)
  [9:13]   pose command: goal x, y, z in the yaw-aligned base frame and the heading error,
           as Isaac Lab's UniformPose2dCommand (z = goal z - base z, goal z = 0.4 m above the
           ground on flat terrain)
  [13:200] 17 x 11 height scan, as the walking policy's
  [200:203] the navigation policy's previous action
Its action is the (vx, vy, wz) velocity command for the walking policy, at 5 Hz (every 10th
walking step, Isaac Lab's navigation decimation), clipped to the walking policy's command
range. Waypoints are handed over one at a time; the last one is the goal."""

from __future__ import annotations

from collections import deque

import numpy as np
from phys_core import TP, wrap
from phys_go2 import (
    ACTION_SCALE,
    DEFAULT_ISAAC,
    N_SCAN,
    POLICY_HZ,
    Go2Walker,
)
from phys_policy import MlpPolicy

NAV_SRC = TP / "kouylty__Isaac-Go2-Navigation-Rough-v0"
LOCO_NPZ = TP / "derived/go2_nav_loco.npz"
NAV_NPZ = TP / "derived/go2_nav_nav.npz"
SAMPLES_NPZ = TP / "derived/go2_nav_samples.npz"

NAV_OBS_DIM = 3 + 3 + 3 + 4 + N_SCAN + 3  # 203
NAV_HZ = 5
GOAL_Z = 0.4  # Isaac Lab: goal z = terrain origin + the Go2's default root height (0.4 m)
CMD_LIMIT = np.array([1.0, 1.0, 1.0], np.float32)  # walking policy's training range
WP_RADIUS = 0.5  # m: an intermediate waypoint is passed within this
STOP_SPEED, STOP_WINDOW_S, MIN_FINAL_S = 0.10, 1.0, 2.0  # "stopped" = arrived


class NavPolicy(MlpPolicy):
    """An rsl_rl actor with its EmpiricalNormalization folded in front."""

    def __init__(self, path) -> None:
        super().__init__(path)
        w = np.load(path)
        self.mean, self.std, self.eps = w["mean"], w["std"], float(w["eps"])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs.astype(np.float32) - self.mean) / (self.std + np.float32(self.eps))
        return super().__call__(x)


def pose_command(x: float, y: float, z: float, yaw: float, goal) -> np.ndarray:
    """Isaac Lab's pose command for goal (gx, gy, heading): position in the yaw-aligned base
    frame (z = GOAL_Z - base z, flat ground) and the wrapped heading error."""
    gx, gy, gh = goal
    dx, dy = gx - x, gy - y
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy, GOAL_Z - z, wrap(gh - yaw)],
                    np.float32)


def nav_observation(lin_b, ang_b, grav_b, cmd4, scan, last) -> np.ndarray:
    return np.concatenate([lin_b, ang_b, grav_b, cmd4, scan, last]).astype(np.float32)


def waypoints(start, points, heading: float | None = None) -> list[tuple[float, float, float]]:
    """(x, y, heading) per point: the heading of travel from the previous point (Isaac Lab's
    simple heading), or `heading` for the last point when given."""
    out, prev = [], start
    for k, (px, py) in enumerate(points):
        h = float(np.arctan2(py - prev[1], px - prev[0]))
        if heading is not None and k == len(points) - 1:
            h = float(heading)
        out.append((float(px), float(py), h))
        prev = (px, py)
    return out


class Go2NavWalker(Go2Walker):
    """Go2Walker with kouylty's walking weights and the navigation policy choosing its
    velocity command. Without a route, `cmd` is left as set (for the walking check)."""

    def __init__(self, model, data, p: str = "go2_") -> None:
        super().__init__(model, data, p)
        self.policy = MlpPolicy(LOCO_NPZ)
        self.nav = NavPolicy(NAV_NPZ)
        assert self.nav.obs_dim == NAV_OBS_DIM
        self.nav_every = max(1, round(POLICY_HZ / NAV_HZ))

    def reset(self) -> None:
        super().reset()
        self.nav_action = np.zeros(3, np.float32)
        self.wps: list[tuple[float, float, float]] = []
        self.wp_i = 0
        self.steps = 0
        self.final_steps = 0
        self.speeds: deque = deque(maxlen=round(STOP_WINDOW_S * POLICY_HZ))

    def go(self, points, heading: float | None = None) -> None:
        x, y, _ = self.pose()
        self.wps = waypoints((x, y), points, heading)
        self.wp_i, self.final_steps = 0, 0
        self.speeds.clear()

    @property
    def goal(self) -> tuple[float, float, float] | None:
        return self.wps[self.wp_i] if self.wps else None

    @property
    def arrived(self) -> bool:
        """Stopped on the last waypoint: mean speed under STOP_SPEED over STOP_WINDOW_S,
        at least MIN_FINAL_S after the last waypoint was handed over."""
        if not self.wps:
            return True
        if self.wp_i < len(self.wps) - 1 or self.final_steps < MIN_FINAL_S * POLICY_HZ:
            return False
        return len(self.speeds) == self.speeds.maxlen and float(np.mean(self.speeds)) < STOP_SPEED

    def nav_obs(self) -> np.ndarray:
        d = self.d
        v = self.free.dofadr[0]
        rot = d.xmat[self.base].reshape(3, 3)
        x, y, yaw = self.pose()
        z = self.height()
        return nav_observation(rot.T @ d.qvel[v:v + 3], d.qvel[v + 3:v + 6],
                               rot.T @ np.array([0.0, 0.0, -1.0]),
                               pose_command(x, y, z, yaw, self.goal),
                               self._scan(z), self.nav_action)

    def _advance(self) -> None:
        x, y, _ = self.pose()
        while self.wp_i < len(self.wps) - 1:
            gx, gy, _ = self.wps[self.wp_i]
            if np.hypot(gx - x, gy - y) > WP_RADIUS:
                break
            self.wp_i += 1

    def after_step(self) -> None:
        self.counter += 1
        if self.counter % self.decim:
            return
        if self.wps:
            self._advance()
            if self.steps % self.nav_every == 0:
                self.nav_action = self.nav(self.nav_obs()).astype(np.float32)
                self.cmd = np.clip(self.nav_action, -CMD_LIMIT, CMD_LIMIT)
            if self.wp_i == len(self.wps) - 1:
                self.final_steps += 1
            v = self.free.dofadr[0]
            self.speeds.append(float(np.hypot(*self.d.qvel[v:v + 2])))
        self.steps += 1
        self.action = self.policy(self.obs()).astype(np.float32)
        self.target = DEFAULT_ISAAC + ACTION_SCALE * self.action
