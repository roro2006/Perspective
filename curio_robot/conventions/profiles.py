"""Beliefs saved per person, so a returning person starts where they left off.
Task T1-05."""

from __future__ import annotations

from pathlib import Path

from curio_robot.conventions.belief import ConventionBelief


class ProfileStore:
    """One JSON file per person: <root>/<person>.json holding
    {"person": ..., "belief": ConventionBelief.to_dict(), "updated": unix time}."""

    def __init__(self, root: Path) -> None:
        raise NotImplementedError("T1-05")

    def load(self, person: str) -> ConventionBelief:
        """The saved belief, or a fresh ConventionBelief() if none (or the file
        is unreadable: log a warning, don't raise)."""
        raise NotImplementedError("T1-05")

    def save(self, person: str, belief: ConventionBelief) -> None:
        """Write atomically (temp file + replace). Create root if missing.
        Reject person names that aren't [A-Za-z0-9_-]{1,64} with ValueError."""
        raise NotImplementedError("T1-05")

    def reset(self, person: str) -> None:
        """Forget a person (no error if unknown)."""
        raise NotImplementedError("T1-05")
