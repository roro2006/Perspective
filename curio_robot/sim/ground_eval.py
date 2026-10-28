"""Score a grounder on the sim eval set and gate a new model against a baseline.

The CLI only parses arguments; this module does the work so tests can pass a
fake grounder instead of a served model."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol

from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.evaluate import Example, Prediction, evaluate, passes_gate
from curio_robot.grounding.prompt import GroundingRequest
from curio_robot.sim.grounding_data import load_dataset

DEFAULT_MAX_P95_MS = 800.0


class Grounder(Protocol):
    """Anything with .score(request) -> GroundingAnswer."""

    def score(self, request: GroundingRequest) -> GroundingAnswer: ...


def _build_request(row: dict, data_path: Path) -> GroundingRequest:
    """Build a GroundingRequest from a dataset row, embedding the image as a data URL."""
    image_path = data_path.parent / row["image"]
    image_bytes = image_path.read_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()

    messages = []
    for msg in row["messages"][:-1]:
        new_msg = dict(msg)
        if "content" in new_msg and isinstance(new_msg["content"], list):
            new_content = []
            for part in new_msg["content"]:
                new_part = dict(part)
                if new_part.get("type") == "image_url":
                    new_part["image_url"] = {"url": data_url}
                new_content.append(new_part)
            new_msg["content"] = new_content
        messages.append(new_msg)

    return GroundingRequest(
        messages=messages,
        letters=dict(row["letters"]),
        prompt_version=row["prompt_version"],
    )


def load_requests(
    data_path: Path,
    ids: list[str] | None = None,
) -> list[tuple[Example, GroundingRequest]]:
    """Load dataset rows and build (Example, GroundingRequest) pairs."""
    rows = load_dataset(data_path)

    if ids is not None:
        id_set = set(ids)
        dataset_ids = {row["id"] for row in rows}
        missing = sorted(id_set - dataset_ids)
        if missing:
            raise ValueError(f"ids not in dataset: {', '.join(missing)}")
        rows = [row for row in rows if row["id"] in id_set]

    pairs = []
    for row in rows:
        example = Example(id=row["id"], kind=row["kind"], answer=row["answer"])
        pairs.append((example, _build_request(row, data_path)))
    return pairs


def run_ground_eval(
    data_path: Path,
    grounder: Grounder,
    *,
    ids: list[str] | None = None,
    current: dict | None = None,
    max_p95_ms: float = DEFAULT_MAX_P95_MS,
) -> tuple[dict, tuple[bool, str] | None]:
    """Score a grounder on the eval set and optionally gate against a current result.

    Returns (result, verdict): result is evaluate()'s dict; verdict is
    passes_gate()'s (ok, message), or None when there is no current result."""
    pairs = load_requests(data_path, ids)

    answers: dict[str, GroundingAnswer] = {}
    for example, request in pairs:
        answers[example.id] = grounder.score(request)

    def predict(example: Example) -> Prediction:
        answer = answers[example.id]
        return Prediction(letter=answer.letter, latency_ms=answer.latency_ms)

    result = evaluate(predict, [example for example, _ in pairs])

    if current is None:
        return result, None
    return result, passes_gate(result, current, max_p95_ms=max_p95_ms)
