"""GPU selection for in-process models. The target box is an AMD RX 7900 XT
(RDNA3, gfx1100, 20 GB) on Windows, where ONNX Runtime reaches the GPU through
DirectML (pip: onnxruntime-directml). The same code picks ROCm/MIGraphX on a
Linux ROCm install and falls back to CPU anywhere else; DirectML also covers
Intel Arc and NVIDIA, so a laptop runs the same build."""

from __future__ import annotations

from pathlib import Path

PREFERRED = (
    "DmlExecutionProvider",  # Windows: AMD / Intel / NVIDIA via DirectML
    "MIGraphXExecutionProvider",  # Linux ROCm
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
)


def onnx_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [p for p in PREFERRED if p in available]


def onnx_session(model: Path | str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # DirectML needs these off: it does its own memory planning and is single-stream.
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model), sess_options=opts, providers=onnx_providers())
