"""T5-20: turn the two downloaded locomotion policies into plain numpy weight files, so the
project env (no torch, no onnxruntime) can run them, and check the numpy forward pass against
the reference runtime on random inputs.

Run in the lerobot-rocm venv, which has torch and onnxruntime:

    C:/Users/Rohan/dev/lerobot-rocm/.venv/Scripts/python.exe -I scripts/poc/export_policies.py

Writes third_party/derived/g1_motion.npz and third_party/derived/go2_rough.npz (gitignored)
and prints the largest numpy-vs-reference difference for each."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phys_core import TP
from phys_policy import G1Policy, MlpPolicy

G1_PT = TP / "unitree_rl_gym/deploy/pre_train/g1/motion.pt"
GO2_ONNX = TP / "cagataydev__strands-isaaclab-go2-rough-policy/exported/policy.onnx"
OUT = TP / "derived"

# ---- a minimal protobuf reader, enough for ONNX initializers and nodes ---------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    out, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf: bytes):
    """(field number, wire type, value) for each field; length-delimited values are bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        yield num, wt, v


def _tensor(buf: bytes, base: Path) -> tuple[str, np.ndarray]:
    dims, name, raw, ext, dtype = [], "", None, {}, 1
    for num, wt, v in _fields(buf):
        if num == 1:
            if wt == 2:  # packed
                j = 0
                while j < len(v):
                    d, j = _varint(v, j)
                    dims.append(d)
            else:
                dims.append(v)
        elif num == 2:
            dtype = v
        elif num == 8:
            name = v.decode()
        elif num == 9:
            raw = v
        elif num == 13:
            kv = {k: val for k, _, val in _fields(v)}
            ext[kv[1].decode()] = kv[2].decode()
    assert dtype == 1, f"{name}: only float32 is handled"
    if raw is None:
        data = (base / ext["location"]).read_bytes()
        off, n = int(ext.get("offset", 0)), int(ext["length"])
        raw = data[off:off + n]
    return name, np.frombuffer(raw, "<f4").reshape(dims).copy()


def onnx_graph(path: Path) -> tuple[dict[str, np.ndarray], list[tuple[str, list[str], list[str]]]]:
    model = path.read_bytes()
    graph = next(v for num, _, v in _fields(model) if num == 7)
    inits, nodes = {}, []
    for num, _, v in _fields(graph):
        if num == 5:
            k, arr = _tensor(v, path.parent)
            inits[k] = arr
        elif num == 1:
            ins, outs, op = [], [], ""
            for n2, _, v2 in _fields(v):
                if n2 == 1:
                    ins.append(v2.decode())
                elif n2 == 2:
                    outs.append(v2.decode())
                elif n2 == 4:
                    op = v2.decode()
            nodes.append((op, ins, outs))
    return inits, nodes


# ---- exports -------------------------------------------------------------------------------


def export_g1() -> None:
    import torch
    m = torch.jit.load(str(G1_PT), map_location="cpu")
    p = {k: v.detach().numpy().astype(np.float32) for k, v in m.named_parameters()}
    w = {"w_ih": p["memory.weight_ih_l0"], "w_hh": p["memory.weight_hh_l0"],
         "b": p["memory.bias_ih_l0"] + p["memory.bias_hh_l0"],
         "w0": p["actor.0.weight"], "b0": p["actor.0.bias"],
         "w1": p["actor.2.weight"], "b1": p["actor.2.bias"]}
    np.savez(OUT / "g1_motion.npz", **w)
    pol = G1Policy(OUT / "g1_motion.npz")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        x = rng.normal(0, 1, 47).astype(np.float32)
        ref = m(torch.from_numpy(x).unsqueeze(0)).detach().numpy().squeeze()
        worst = max(worst, float(np.abs(pol(x) - ref).max()))
    print(f"g1: LSTM 47->64, MLP 64->32->12; max |numpy - torch| over 50 steps = {worst:.2e}")


def export_go2() -> None:
    inits, nodes = onnx_graph(GO2_ONNX)
    print("go2 ops:", [op for op, _, _ in nodes])
    print("go2 initializers:", {k: v.shape for k, v in inits.items()})
    # The graph is Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm (torch Linear: B is (out, in)).
    ops = [op for op, _, _ in nodes]
    assert set(ops) == {"Gemm", "Elu"}, f"unexpected ops {ops}"
    w: dict[str, np.ndarray] = {}
    for layer, (_, ins, _) in enumerate(n for n in nodes if n[0] == "Gemm"):
        w[f"w{layer}"], w[f"b{layer}"] = inits[ins[1]], inits[ins[2]]
    np.savez(OUT / "go2_rough.npz", **w)
    pol = MlpPolicy(OUT / "go2_rough.npz")
    try:  # the check needs onnxruntime (T5-20 ran it in the lerobot-rocm venv)
        import onnxruntime as ort
    except ImportError:
        print(f"go2: {len(w) // 2} layers written; onnxruntime not installed, check skipped")
        return
    sess = ort.InferenceSession(str(GO2_ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("go2 input:", inp.name, inp.shape, "output:", [o.shape for o in sess.get_outputs()])
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (64, int(inp.shape[1]))).astype(np.float32)
    ref = np.concatenate([sess.run(None, {inp.name: r[None]})[0] for r in x])
    got = np.stack([pol(r) for r in x])
    print(f"go2: {len(w) // 2} layers; max |numpy - onnxruntime| over 64 inputs = "
          f"{float(np.abs(got - ref).max()):.2e}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--go2" not in sys.argv:  # --go2: the dog only (the G1 export needs torch)
        export_g1()
    export_go2()
