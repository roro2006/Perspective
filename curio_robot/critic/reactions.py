"""How a simulated person reacts while the arm reaches: where their head
points each frame and what they say. Assumed numbers, widely spread; phase 3
measures real ones. Task T4-01."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RATE_HZ = 10.0  # frames per second for everything in the critic
HEAD_LAG_S = 0.15  # the head follows the eyes with this time constant
SETTLED_M = 0.01  # gripper this close to its goal counts as arrived
FAMILIES = ("attentive", "slow", "distracted", "silent")

# Uniform ranges per family, in Style field order (notice_s .. glance_s).
# "silent" draws attentive's ranges, then speak_delay_s becomes inf.
RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "attentive": ((0.2, 0.5), (0.8, 0.95), (0.1, 0.3), (0.4, 0.8), (1.0, 3.0), (0.05, 0.2),
                  (0.2, 0.4)),
    "slow": ((0.5, 1.0), (0.6, 0.9), (0.2, 0.5), (0.8, 1.5), (2.0, 4.0), (0.1, 0.3),
             (0.3, 0.6)),
    "distracted": ((0.3, 0.8), (0.4, 0.7), (0.1, 0.4), (0.5, 1.2), (3.0, 6.0), (0.4, 1.0),
                   (0.3, 0.8)),
}


@dataclass(frozen=True, slots=True)
class Style:
    family: str
    notice_s: float  # after the reach starts, when a wrong target becomes clear to them
    p_look_intended: float  # after noticing, chance their eyes go to the block they want
    look_delay_s: float  # noticing -> eyes on the wanted block
    speak_delay_s: float  # noticing -> "no" starts; inf: never speaks
    head_noise_deg: float  # per-frame jitter of the head direction
    glance_rate_hz: float  # random glances at any block, per second
    glance_s: float  # how long one glance lasts


@dataclass(frozen=True, slots=True)
class Reaction:
    head_fwd: np.ndarray  # (F, 3) unit vectors
    speech: tuple[tuple[float, str], ...]  # (time, transcript so far), time-sorted
    t_no: float | None  # when "no" starts; None if they never speak (or nothing was wrong)


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise vectors along the last axis."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sample_style(rng: np.random.Generator, family: str) -> Style:
    """Seven rng.uniform(lo, hi) draws, one per range in Style field order,
    from RANGES[family] ("silent" uses "attentive"); for "silent" set
    speak_delay_s = inf after drawing. Unknown family -> ValueError."""
    if family not in RANGES and family != "silent":
        raise ValueError(f"unknown reaction family: {family!r}")
    ranges = RANGES["attentive" if family == "silent" else family]
    v = [float(rng.uniform(lo, hi)) for lo, hi in ranges]
    speak_delay_s = math.inf if family == "silent" else v[3]
    return Style(
        family=family,
        notice_s=v[0],
        p_look_intended=v[1],
        look_delay_s=v[2],
        speak_delay_s=speak_delay_s,
        head_noise_deg=v[4],
        glance_rate_hz=v[5],
        glance_s=v[6],
    )


def _apply_glances(look: np.ndarray, t: np.ndarray, obj_pos: np.ndarray, style: Style,
                   rng: np.random.Generator) -> None:
    """Overwrite look points in place with random glances; later glances win."""
    n = rng.poisson(style.glance_rate_hz * t[-1])
    starts = np.sort(rng.uniform(t[0], t[-1], n))
    which = rng.integers(0, obj_pos.shape[0], n)
    for start, block in zip(starts, which, strict=True):
        during = (t >= start) & (t < start + style.glance_s)
        look[during] = obj_pos[block]


def _look_points(t: np.ndarray, obj_pos: np.ndarray, intended: int, arm_target: int,
                 gripper: np.ndarray, style: Style, rng: np.random.Generator) -> np.ndarray:
    """Where the eyes point in each frame, shape (F, 3)."""
    look = gripper.copy()  # watching the arm
    settled = np.linalg.norm(gripper[:, :2] - obj_pos[arm_target, :2], axis=1) <= SETTLED_M
    arrived = np.logical_or.accumulate(settled)
    look[arrived] = obj_pos[arm_target]
    _apply_glances(look, t, obj_pos, style, rng)
    # Drawn on every reach; only a wrong reach can use it.
    look_draw = rng.random()
    if arm_target != intended and look_draw < style.p_look_intended:
        look[t >= style.notice_s + style.look_delay_s] = obj_pos[intended]
    return look


def _head_direction(t: np.ndarray, head_pos: np.ndarray, look: np.ndarray) -> np.ndarray:
    """Head unit vectors lagging the eye direction by HEAD_LAG_S, shape (F, 3)."""
    eye = _unit(look - head_pos)
    alpha = 1.0 - math.exp(-(t[1] - t[0]) / HEAD_LAG_S)
    head = np.empty_like(eye)
    head[0] = eye[0]
    for f in range(1, len(t)):
        head[f] = _unit(head[f - 1] + alpha * (eye[f] - head[f - 1]))
    return head


def simulate_reaction(
    t: np.ndarray,
    head_pos: np.ndarray,
    obj_pos: np.ndarray,
    intended: int,
    arm_target: int,
    gripper: np.ndarray,
    style: Style,
    rng: np.random.Generator,
) -> Reaction:
    """Head direction per frame and speech for one reach.

    t: (F,) seconds from the reach start, evenly spaced; obj_pos: (N, 3);
    gripper: (F, 3). wrong = arm_target != intended.

    1. Look point per frame (where the eyes are):
       - default: gripper[f] (watching the arm);
       - once the arm has arrived, norm(gripper[f, :2] - obj_pos[arm_target,
         :2]) <= SETTLED_M (xy only; the gripper hovers above the block):
         look at obj_pos[arm_target];
       - glances: n = rng.poisson(style.glance_rate_hz * t[-1]); starts =
         sorted rng.uniform(t[0], t[-1], n); which = rng.integers(0, N, n);
         during [start, start + glance_s) look at obj_pos[which]; a later
         glance overrides an earlier one where they overlap;
       - wrong only: look_draw = rng.random() (drawn always, after the
         glances, wrong or not); if wrong and look_draw < p_look_intended,
         from notice_s + look_delay_s on look at obj_pos[intended],
         overriding everything above (glances included).
    2. Head direction: eye direction e_f = unit(look_f - head_pos). h_0 = e_0;
       h_f = unit(h_{f-1} + a * (e_f - h_{f-1})), a = 1 - exp(-dt / HEAD_LAG_S),
       dt = t[1] - t[0].
    3. Noise: z = rng.normal(size=(F, 3)) (drawn last); head_fwd[f] =
       unit(h_f + radians(head_noise_deg) * z[f]).
    4. Speech, wrong and speak_delay_s finite only: t_no = notice_s +
       speak_delay_s; speech = ((t_no + 0.2, "no"), (t_no + 0.7, "no, the
       other one")). Otherwise speech = () and t_no = None. Events after
       t[-1] are kept (the transcript arrives late; consumers filter by time).
    """
    look = _look_points(t, obj_pos, intended, arm_target, gripper, style, rng)
    head = _head_direction(t, head_pos, look)
    head_fwd = _unit(head + math.radians(style.head_noise_deg) * rng.normal(size=(len(t), 3)))

    if arm_target != intended and math.isfinite(style.speak_delay_s):
        t_no = style.notice_s + style.speak_delay_s
        speech = ((t_no + 0.2, "no"), (t_no + 0.7, "no, the other one"))
        return Reaction(head_fwd=head_fwd, speech=speech, t_no=t_no)
    return Reaction(head_fwd=head_fwd, speech=(), t_no=None)
