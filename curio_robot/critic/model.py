"""The learned critic: a small numpy model, no new dependencies. One shared
hidden layer scores each block; a softmax over the blocks of the requested
colour says which one the person wants; p_wrong is the chance it isn't the
arm's target. Task T4-04."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from curio_robot.critic.episodes import Episode
from curio_robot.critic.features import D, features

HIDDEN = 16
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
LOG_EPS = 1e-12


def _masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over the last axis of the True entries of mask; exactly 0 elsewhere.
    Rows with no True entry come out all 0."""
    masked = np.where(mask, scores, -np.inf)
    has_any = mask.any(axis=-1, keepdims=True)
    top = np.where(has_any, masked.max(axis=-1, keepdims=True), 0.0)
    e = np.where(mask, np.exp(masked - top), 0.0)
    total = e.sum(axis=-1, keepdims=True)
    return np.divide(e, total, out=np.zeros_like(e), where=total > 0)


def _stack_episodes(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features (M, N, d), candidate masks (M, N) and intended indices (M,) over every frame."""
    xs, masks, ys = [], [], []
    for ep in episodes:
        x = features(ep)
        frames = x.shape[0]
        xs.append(x)
        masks.append(np.repeat(ep.candidates[None, :], frames, axis=0))
        ys.append(np.full(frames, ep.intended, dtype=int))
    return np.concatenate(xs), np.concatenate(masks), np.concatenate(ys)


def _batch_loss_grads(
    params: dict[str, np.ndarray], xn: np.ndarray, mask: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    """Summed cross-entropy over the batch, and gradients of the batch-mean loss."""
    W1, b1, w2 = params["W1"], params["b1"], params["w2"]
    batch = xn.shape[0]
    rows = np.arange(batch)
    h = np.tanh(xn @ W1 + b1)  # (B, N, H)
    p = _masked_softmax(h @ w2, mask)  # (B, N)
    loss = float(-np.log(p[rows, y] + LOG_EPS).sum())
    g = p.copy()
    g[rows, y] -= 1.0
    g = np.where(mask, g, 0.0) / batch  # (B, N)
    dh = g[..., None] * w2 * (1.0 - h**2)  # (B, N, H)
    grads = {
        "W1": np.einsum("bnd,bnh->dh", xn, dh),
        "b1": dh.sum(axis=(0, 1)),
        "w2": np.einsum("bn,bnh->h", g, h),
    }
    return loss, grads


def _adam_step(
    param: np.ndarray,
    grad: np.ndarray,
    state: tuple[np.ndarray, np.ndarray],
    step: int,
    lr: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """One bias-corrected Adam update; returns the new parameter and its moments."""
    m, v = state
    m = BETA1 * m + (1.0 - BETA1) * grad
    v = BETA2 * v + (1.0 - BETA2) * grad**2
    m_hat = m / (1.0 - BETA1**step)
    v_hat = v / (1.0 - BETA2**step)
    return param - lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS), (m, v)


class Critic:
    def __init__(self, d: int = D, hidden: int = HIDDEN, seed: int = 0) -> None:
        """rng = np.random.default_rng(seed); W1 = rng.normal(0, 1/sqrt(d),
        (d, hidden)); b1 = zeros(hidden); w2 = rng.normal(0, 1/sqrt(hidden),
        hidden). Feature scaling: mean = zeros(d), std = ones(d) until fit."""
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.mean = np.zeros(d)
        self.std = np.ones(d)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """x: (..., N, d) -> (..., N): tanh(((x - mean) / std) @ W1 + b1) @ w2."""
        xn = (x - self.mean) / self.std
        return np.tanh(xn @ self.W1 + self.b1) @ self.w2

    def probs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Softmax of scores over the last axis where mask (broadcast to the
        scores' shape) is True; exactly 0 where it is False. Subtract the max
        over masked entries for stability. A row with no True entries -> all 0."""
        s = self.scores(x)
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), s.shape)
        return _masked_softmax(s, mask)

    def predict(self, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
        """(p_wrong (F,), intended (F, N)): intended = probs(features(ep),
        ep.candidates); p_wrong = 1 - intended[:, ep.arm_target]."""
        intended = self.probs(features(ep), ep.candidates)
        return 1.0 - intended[:, ep.arm_target], intended

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
        X, mask, y = _stack_episodes(episodes)
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)
        Xn = (X - self.mean) / self.std

        params = {"W1": self.W1, "b1": self.b1, "w2": self.w2}
        moments = {key: (np.zeros_like(p), np.zeros_like(p)) for key, p in params.items()}
        total = Xn.shape[0]
        losses: list[float] = []
        step = 0
        for epoch in range(epochs):
            order = np.random.default_rng(seed + epoch).permutation(total)
            loss_sum = 0.0
            for start in range(0, total, batch):
                idx = order[start : start + batch]
                loss, grads = _batch_loss_grads(params, Xn[idx], mask[idx], y[idx])
                loss_sum += loss
                step += 1
                for key, grad in grads.items():
                    params[key], moments[key] = _adam_step(
                        params[key], grad, moments[key], step, lr
                    )
            losses.append(loss_sum / total)

        self.W1, self.b1, self.w2 = params["W1"], params["b1"], params["w2"]
        return losses

    def save(self, path: Path) -> None:
        """np.savez with W1, b1, w2, mean, std; creates parent folders."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, w2=self.w2, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> Critic:
        """Inverse of save: shapes from the file."""
        with np.load(path) as f:
            W1, b1, w2 = f["W1"], f["b1"], f["w2"]
            mean, std = f["mean"], f["std"]
        critic = cls(d=W1.shape[0], hidden=W1.shape[1])
        critic.W1, critic.b1, critic.w2 = W1, b1, w2
        critic.mean, critic.std = mean, std
        return critic
