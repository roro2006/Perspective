"""Object detection. The sim uses segmentation ground truth; real cameras will
use an open-vocabulary detector (OWLv2 / Grounding DINO) behind the same
interface."""

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


class OpenVocabDetector:
    """Real-camera detector. TODO(Track 2): OWLv2 or Grounding DINO with colour
    prompts ("red block", ...); run on the laptop GPU (ROCm or ONNX/DirectML)."""

    def detect(self, frame: CameraFrame) -> list[Detection]:
        raise NotImplementedError("Real-camera detection lands with Track 2")
