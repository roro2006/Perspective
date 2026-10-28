"""Beliefs saved per person, so a returning person starts where they left off.
Task T1-05."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from curio_robot.conventions.belief import ConventionBelief

_log = logging.getLogger(__name__)
_PERSON_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _check_person(person: str) -> None:
    if not _PERSON_RE.fullmatch(person):
        raise ValueError(f"Invalid person name: {person!r}")


class ProfileStore:
    """One JSON file per person: <root>/<person>.json holding
    {"person": ..., "belief": ConventionBelief.to_dict(), "updated": unix time}."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, person: str) -> ConventionBelief:
        """The saved belief, or a fresh ConventionBelief() if none (or the file
        is unreadable: log a warning, don't raise)."""
        _check_person(person)
        path = self.root / f"{person}.json"
        if not path.exists():
            return ConventionBelief()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "belief" not in data:
                raise ValueError("Missing 'belief' key")
            return ConventionBelief.from_dict(data["belief"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
            _log.warning("Failed to load profile for %s: %s", person, e)
            return ConventionBelief()

    def save(self, person: str, belief: ConventionBelief) -> None:
        """Write atomically (temp file + replace). Create root if missing.
        Reject person names that aren't [A-Za-z0-9_-]{1,64} with ValueError."""
        _check_person(person)
        self.root.mkdir(parents=True, exist_ok=True)

        payload = {
            "person": person,
            "belief": belief.to_dict(),
            "updated": time.time(),
        }

        # Temp file in the same directory so os.replace is an atomic rename.
        fd, tmp_path = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.root / f"{person}.json")
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def reset(self, person: str) -> None:
        """Forget a person (no error if unknown)."""
        _check_person(person)
        path = self.root / f"{person}.json"
        path.unlink(missing_ok=True)
