"""The learned critic: a small numpy model, no new dependencies. One shared
hidden layer scores each block; a softmax over the blocks of the requested
colour says which one the person wants; p_wrong is the chance it isn't the
arm's target. Task T4-04."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curio_robot.critic.episodes import Episode
from curio_robot.critic.features import D, features  # noqa: F401 (used by the implementation)

HIDDEN = 16


class Critic:
    def __init__(self, d: int = D, hidden: int = HIDDEN, seed: int = 0) -> None:
        """rng = np.random.default_rng(seed); W1 = rng.normal(0, 1/sqrt(d),
        (d, hidden)); b1 = zeros(hidden); w2 = rng.normal(0, 1/sqrt(hidden),
        hidden). Feature scaling: mean = zeros(d), std = ones(d) until fit."""
        raise NotImplementedError("T4-04")

    def scores(self, x: np.ndarray) -> np.ndarray:
        """x: (..., N, d) -> (..., N): tanh(((x - mean) / std) @ W1 + b1) @ w2."""
        raise NotImplementedError("T4-04")

    def probs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Softmax of scores over the last axis where mask (broadcast to the
        scores' shape) is True; exactly 0 where it is False. Subtract the max
        over masked entries for stability. A row with no True entries -> all 0."""
        raise NotImplementedError("T4-04")

    def predict(self, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
        """(p_wrong (F,), intended (F, N)): intended = probs(features(ep),
        ep.candidates); p_wrong = 1 - intended[:, ep.arm_target]."""
        raise NotImplementedError("T4-04")

    def fit(
        self,
        episodes: list[Episode],
        epochs: int = 30,
        lr: float = 0.01,
        batch: int = 64,
        seed: int = 0,
    ) -> list[float]:
        """Train on every frame of every episode; returns the mean loss per epoch.

        - X = stacked features (M, N, d) over all frames of all episodes (every
          episode has the same N), mask (M, N) = candidates, y (M,) = intended.
        - Set mean / std per feature over X (std 0 -> 1) before training.
        - Loss: mean over frames of -log(probs[y]) (add 1e-12 inside the log).
        - Gradient by hand: with h = tanh(xn @ W1 + b1), g = probs - onehot(y)
          (0 off the mask) divided by the batch size: dw2 = sum g * h;
          dh = g[..., None] * w2 * (1 - h**2); dW1 = sum xn outer dh;
          db1 = sum dh.
        - Adam (beta1 0.9, beta2 0.999, eps 1e-8) on W1, b1, w2; each epoch
          shuffles frames with np.random.default_rng(seed + epoch)."""
        raise NotImplementedError("T4-04")

    def save(self, path: Path) -> None:
        """np.savez with W1, b1, w2, mean, std; creates parent folders."""
        raise NotImplementedError("T4-04")

    @classmethod
    def load(cls, path: Path) -> Critic:
        """Inverse of save: shapes from the file."""
        raise NotImplementedError("T4-04")

