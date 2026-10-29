"""How a simulated person reacts while the arm reaches: where their head
points each frame and what they say. Assumed numbers, widely spread; phase 3
measures real ones. Task T4-01."""

from __future__ import annotations

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


def sample_style(rng: np.random.Generator, family: str) -> Style:
    """Seven rng.uniform(lo, hi) draws, one per range in Style field order,
    from RANGES[family] ("silent" uses "attentive"); for "silent" set
    speak_delay_s = inf after drawing. Unknown family -> ValueError."""
    raise NotImplementedError("T4-01")


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
    raise NotImplementedError("T4-01")
