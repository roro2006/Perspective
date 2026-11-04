"""Correction log: one JSON line per event plus the PNG the grounder saw.

An event is a request the deployed grounder answered and the person then either corrected
("no, behind ME") or let stand. It keeps what training needs (the exact request, with the
image as a file, and the resolved true target) and what an audit needs (speaker, pick,
probabilities, the correction words, the object map). `to_rows` turns events into the
grounding_data JSONL rows that scripts/train_lora.py reads unchanged."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from curio_robot.grounding.prompt import GroundingRequest

DATA_URL_PREFIX = "data:image/png;base64,"
OUTCOMES = ("corrected", "confirmed")


@dataclass(frozen=True, slots=True)
class CorrectionEvent:
    event_id: str
    episode: int
    seed: int
    speaker_id: str
    kind: str
    utterance: str
    messages: list[dict]  # the request as sent, its image replaced by `image`
    letters: dict[str, str]  # letter -> object_id
    prompt_version: str
    image: str  # path of the PNG the grounder saw, relative to the store root
    objects: list[dict]  # [{"id", "label", "xyz"}], the map the request was built from
    cameras: dict[str, dict]  # {"human": pose, "robot": pose}
    pick: str | None  # the grounder's argmax object (None: no answer)
    pick_probs: dict[str, float]
    outcome: str  # "corrected" or "confirmed"
    correction: str | None  # the person's words; None when confirmed
    resolved_target: str  # what the person really meant
    resolved_by: str  # how the target was settled ("confirmed", "retry", "pointing")
    grounder: str  # the adapter that answered
    meta: dict = field(default_factory=dict)  # frame, relation, ... (not used by training)

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"{self.event_id}: outcome {self.outcome!r} not in {OUTCOMES}")
        if self.resolved_target not in self.letters.values():
            raise ValueError(f"{self.event_id}: resolved target has no letter")
        if self.outcome == "corrected" and self.correction is None:
            raise ValueError(f"{self.event_id}: a correction needs its words")


def split_request(request: GroundingRequest, image_rel: str) -> tuple[list[dict], bytes]:
    """(messages with the image as image_rel, the PNG bytes). One image per request."""
    messages = json.loads(json.dumps(request.messages))
    parts = [p for p in messages[0]["content"] if p["type"] == "image_url"]
    if len(parts) != 1:
        raise ValueError(f"expected one image, got {len(parts)}")
    url = parts[0]["image_url"]["url"]
    if not url.startswith(DATA_URL_PREFIX):
        raise ValueError("image is not a PNG data URL")
    parts[0]["image_url"]["url"] = image_rel
    return messages, base64.b64decode(url[len(DATA_URL_PREFIX):])


class EventStore:
    """root/events.jsonl and root/images/<event_id>.png. Append-only."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "events.jsonl"

    def image_rel(self, event_id: str) -> str:
        return f"images/{event_id}.png"

    def append(self, event: CorrectionEvent, png: bytes) -> None:
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / event.image).write_bytes(png)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def load(self) -> list[CorrectionEvent]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [CorrectionEvent(**json.loads(line)) for line in lines if line.strip()]


def to_row(event: CorrectionEvent) -> dict:
    """The event as a grounding_data row: the request as the grounder saw it, answered with
    the letter of the resolved target (the person's word, not the grounder's)."""
    answer = next(k for k, v in event.letters.items() if v == event.resolved_target)
    messages = json.loads(json.dumps(event.messages))
    messages.append({"role": "assistant", "content": answer})
    return {
        "id": f"fw-{event.event_id}",
        "kind": event.kind,
        "text": event.utterance,
        "answer": answer,
        "target_id": event.resolved_target,
        "letters": dict(event.letters),
        "prompt_version": event.prompt_version,
        "image": event.image,
        "messages": messages,
        "source": f"flywheel:{event.outcome}",
    }


def to_rows(events: list[CorrectionEvent]) -> list[dict]:
    return [to_row(e) for e in events]
