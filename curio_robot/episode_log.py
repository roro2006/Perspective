"""Episode logger: the dataset is the moat, so every session is recorded as
paired human-view / robot-view events with the request, target and
corrections. JSONL now; convert to the LeRobot dataset format for training."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.round(4).tolist()
    if isinstance(value, set):
        return sorted(value)
    return value


class EpisodeLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        record = {"t": time.time(), "kind": kind} | {k: _plain(v) for k, v in fields.items()}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
