"""Acceptance tests for T1-03 (conventions/noise.py)."""

import time

import numpy as np
from scene import HUMAN

from curio_robot.conventions.noise import flip_probability

ACROSS = (np.array([0.25, -0.08, 0.0125]), np.array([0.25, 0.08, 0.0125]))  # well apart, sideways
IN_LINE = (np.array([0.30, 0.0005, 0.0125]), np.array([0.20, -0.0005, 0.0125]))  # along the gaze


def test_well_separated_pair_never_flips():
    assert flip_probability(*ACROSS, HUMAN) < 0.01


def test_pair_in_line_with_the_gaze_often_flips():
    p = flip_probability(*IN_LINE, HUMAN)
    assert 0.1 < p <= 0.5


def test_more_noise_more_flips():
    low = flip_probability(*IN_LINE, HUMAN, rot_sigma_deg=0.5)
    high = flip_probability(*IN_LINE, HUMAN, rot_sigma_deg=4.0)
    assert low < high


def test_zero_noise_never_flips():
    assert flip_probability(*IN_LINE, HUMAN, pos_sigma=0.0, rot_sigma_deg=0.0) == 0.0


def test_exact_tie_is_a_coin_flip():
    a = np.array([0.3, 0.0, 0.0125])
    b = np.array([0.2, 0.0, 0.0125])
    assert flip_probability(a, b, HUMAN) == 0.5


def test_deterministic_and_symmetric():
    p1 = flip_probability(*IN_LINE, HUMAN, seed=7)
    assert flip_probability(*IN_LINE, HUMAN, seed=7) == p1
    assert flip_probability(IN_LINE[1], IN_LINE[0], HUMAN, seed=7) == p1


def test_fast_enough_for_the_live_loop():
    flip_probability(*IN_LINE, HUMAN)  # warm up
    t0 = time.perf_counter()
    flip_probability(*IN_LINE, HUMAN, samples=400)
    assert time.perf_counter() - t0 < 0.1  # target 20 ms; loose bound for slow machines
