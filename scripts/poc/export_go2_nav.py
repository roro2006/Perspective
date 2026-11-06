"""T5-32: turn kouylty's two Isaac Lab Go2 policies (navigation and its walking policy,
TorchScript `policy.pt` from isaaclab_rl's exporter) into plain numpy weight files, so the
project env can run them without torch, and store input/output samples so the CPU tests can
check the numpy forward pass against torch without torch installed.

Run once in the lerobot-rocm venv (CPU), which has torch:

    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe -I scripts/poc/export_go2_nav.py

Writes third_party/derived/go2_nav_loco.npz, go2_nav_nav.npz and go2_nav_samples.npz
(gitignored) and prints the largest numpy-vs-torch difference for each policy. It also checks
that each policy.pt actor equals the actor in the training checkpoint next to it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TP = Path("C:/Users/Rohan/dev/Perspective/third_party")
SRC = TP / "kouylty__Isaac-Go2-Navigation-Rough-v0"
OUT = TP / "derived"
CKPT = {"locomotion": "model_1998.pt", "navigation": "model_2499.pt"}
N_SAMPLES = 64

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phys_go2_nav import LOCO_NPZ, NAV_NPZ, SAMPLES_NPZ, NavPolicy
from phys_policy import MlpPolicy


def _actor(sd: dict) -> dict[str, np.ndarray]:
    """rsl_rl Sequential Linear/ELU actor: Linear layers at indices 0, 2, 4, 6."""
    w = {}
    for k, i in enumerate((0, 2, 4, 6)):
        w[f"w{k}"] = sd[f"actor.{i}.weight"].detach().numpy().astype(np.float32)
        w[f"b{k}"] = sd[f"actor.{i}.bias"].detach().numpy().astype(np.float32)
    return w


def export(name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    m = torch.jit.load(str(SRC / name / "policy.pt"), map_location="cpu").eval()
    sd = m.state_dict()
    w = _actor(sd)
    ck = torch.load(str(SRC / name / CKPT[name]), map_location="cpu", weights_only=True)
    same = all(np.array_equal(w[f"w{k}"], ck["model_state_dict"][f"actor.{i}.weight"].numpy())
               for k, i in enumerate((0, 2, 4, 6)))
    print(f"{name}: policy.pt actor == {CKPT[name]} actor: {same}")
    n_in = w["w0"].shape[1]
    if "normalizer._mean" in sd:  # EmpiricalNormalization: (x - mean) / (std + eps)
        w["mean"] = sd["normalizer._mean"].numpy().astype(np.float32).reshape(-1)
        w["std"] = sd["normalizer._std"].numpy().astype(np.float32).reshape(-1)
        w["eps"] = np.float32(m.normalizer.eps)
        x = (w["mean"] + w["std"] * rng.normal(0, 1, (N_SAMPLES, n_in))).astype(np.float32)
    else:  # the walking policy's normaliser is Identity
        x = rng.normal(0, 1, (N_SAMPLES, n_in)).astype(np.float32)
    path = NAV_NPZ if "mean" in w else LOCO_NPZ
    np.savez(path, **w)
    pol = NavPolicy(path) if "mean" in w else MlpPolicy(path)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = np.stack([pol(r) for r in x])
    worst = float(np.abs(got - ref).max())
    sizes = [w[f"w{k}"].shape for k in range(4)]
    print(f"{name}: layers {sizes}; max |numpy - torch| over {N_SAMPLES} inputs = {worst:.2e}")
    return x, ref, worst


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(32)
    lx, ly, lw = export("locomotion", rng)
    nx, ny, nw = export("navigation", rng)
    np.savez(SAMPLES_NPZ, loco_x=lx, loco_y=ly, nav_x=nx, nav_y=ny,
             worst=np.array([lw, nw], np.float32))
    print("wrote", LOCO_NPZ.name, NAV_NPZ.name, SAMPLES_NPZ.name)
