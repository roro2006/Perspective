"""T5-20: the two downloaded locomotion policies as small numpy networks. The weights come
from export_policies.py (run once in the lerobot-rocm venv); here they need numpy only, so the
project env can run them without torch or onnxruntime."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class G1Policy:
    """unitree_rl_gym's G1 motion.pt: a one-layer LSTM (47 -> 64) whose state persists between
    calls, then Linear 64->32, ELU, Linear 32->12. PyTorch gate order is i, f, g, o."""

    def __init__(self, path: Path | str) -> None:
        w = np.load(path)
        self.w_ih, self.w_hh, self.b = w["w_ih"], w["w_hh"], w["b"]
        self.w0, self.b0, self.w1, self.b1 = w["w0"], w["b0"], w["w1"], w["b1"]
        self.hidden = self.w_hh.shape[1]
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros(self.hidden, np.float32)
        self.c = np.zeros(self.hidden, np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        z = self.w_ih @ obs.astype(np.float32) + self.w_hh @ self.h + self.b
        i, f, g, o = np.split(z, 4)
        self.c = _sigmoid(f) * self.c + _sigmoid(i) * np.tanh(g)
        self.h = (_sigmoid(o) * np.tanh(self.c)).astype(np.float32)
        return self.w1 @ _elu(self.w0 @ self.h + self.b0) + self.b1


class MlpPolicy:
    """An rsl_rl actor exported to ONNX: Gemm layers with ELU between them (none after the
    last). The Go2 export has no observation normaliser in its graph."""

    def __init__(self, path: Path | str) -> None:
        w = np.load(path)
        n = sum(1 for k in w.files if k.startswith("w"))
        self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in range(n)]

    @property
    def obs_dim(self) -> int:
        return int(self.layers[0][0].shape[1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for k, (a, b) in enumerate(self.layers):
            x = a @ x + b
            if k < len(self.layers) - 1:
                x = _elu(x)
        return x
