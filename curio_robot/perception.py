"""Object detection. The sim uses segmentation ground truth; real cameras will
use an open-vocabulary detector (YOLO-World on ONNX Runtime) behind the
same interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from curio_robot.types import CameraFrame, Detection

MIN_VISIBLE_PIXELS = 12


class Detector(Protocol):
    def detect(self, frame: CameraFrame) -> list[Detection]: ...


class SegmentationDetector:
    """Ground-truth detector for the sim: an object counts as seen when enough
    of its pixels are visible. Occlusion therefore comes from real rendering."""

    def __init__(self, id_to_color: dict[int, str]) -> None:
        self._id_to_color = id_to_color

    def detect(self, frame: CameraFrame) -> list[Detection]:
        if frame.segmentation is None:
            raise ValueError("SegmentationDetector needs a frame with segmentation")
        found = []
        for obj_id, color in self._id_to_color.items():
            vs, us = np.nonzero(frame.segmentation == obj_id)
            if len(us) < MIN_VISIBLE_PIXELS:
                continue
            found.append(
                Detection(
                    camera=frame.camera,
                    color=color,
                    pixel=(float(us.mean()), float(vs.mean())),
                    area=len(us),
                )
            )
        return found


class OnnxYoloWorldDetector:
    """Real-camera detector: YOLO-World exported to ONNX with the colour
    prompts baked in, run through ONNX Runtime (DirectML on the RX 7900 XT).

    Export once, on any machine with ultralytics:
        from ultralytics import YOLOWorld
        m = YOLOWorld("yolov8s-worldv2.pt")
        m.set_classes(["red block", "blue block", "green block", "yellow block"])
        m.export(format="onnx", imgsz=640, batch=2)   # batch=2: both cameras in one call

    `colors` names what each class index means, in the same order."""

    def __init__(
        self,
        model_path: str,
        colors: list[str],
        *,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        from curio_robot.accel import onnx_session

        self._session = onnx_session(model_path)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self._batch = inp.shape[0] if isinstance(inp.shape[0], int) else None  # None: dynamic
        self._colors, self._conf, self._iou = colors, conf, iou

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    def detect(self, frame: CameraFrame) -> list[Detection]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[CameraFrame]) -> list[list[Detection]]:
        """Both cameras in one GPU call when the model was exported with that batch."""
        if not frames:
            return []
        if self._batch is not None and self._batch != len(frames):
            out: list[list[Detection]] = []
            for i in range(0, len(frames), self._batch):
                chunk = frames[i : i + self._batch]
                pad = self._batch - len(chunk)
                out += self._run(chunk + [chunk[-1]] * pad)[: len(chunk)]
            return out
        return self._run(frames)

    def _run(self, frames: list[CameraFrame]) -> list[list[Detection]]:
        tensors, letterboxes = zip(*(letterbox(f.rgb, self._size) for f in frames))
        raw = self._session.run(None, {self._input: np.stack(tensors)})[0]
        return [
            yolo_detections(r, lb, f.camera, self._colors, self._conf, self._iou)
            for r, lb, f in zip(raw, letterboxes, frames)
        ]


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, float, float]]:
    """(3, size, size) float32 in [0, 1], plus (scale, pad_x, pad_y) to undo it."""
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py : py + nh, px : px + nw] = np.asarray(
        Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
    )
    return canvas.transpose(2, 0, 1).astype(np.float32) / 255.0, (scale, px, py)


def yolo_detections(
    raw: np.ndarray,
    lb: tuple[float, float, float],
    camera: str,
    colors: list[str],
    conf: float,
    iou: float,
) -> list[Detection]:
    """Decode one image of YOLOv8-style output, shape (4 + classes, anchors):
    centre-x, centre-y, width, height in letterboxed pixels, then class scores."""
    scale, px, py = lb
    boxes, scores = raw[:4].T, raw[4:].T
    cls = scores.argmax(axis=1)
    best = scores[np.arange(len(cls)), cls]
    keep = best >= conf
    boxes, cls, best = boxes[keep], cls[keep], best[keep]
    found = []
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        for i in _nms(boxes[idx], best[idx], iou):
            cx, cy, w, h = boxes[idx[i]]
            found.append(
                Detection(
                    camera=camera,
                    color=colors[int(c)],
                    pixel=(float((cx - px) / scale), float((cy - py) / scale)),
                    area=int(w * h / scale**2),
                )
            )
    return found


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    x1, y1 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
    x2, y2 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    area = (x2 - x1) * (y2 - y1)
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        rest = np.array(order, dtype=int)
        if not len(rest):
            break
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        overlap = inter / (area[i] + area[rest] - inter + 1e-9)
        order = [int(j) for j, o in zip(rest, overlap) if o < iou]
    return keep
