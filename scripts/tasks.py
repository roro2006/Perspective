"""Task board tooling for parallel agents (see tasks/README.md).

    uv run python scripts/tasks.py list
    uv run python scripts/tasks.py ready
    uv run python scripts/tasks.py check
    uv run python scripts/tasks.py pack T1-01 [--out out/context]

`pack` bundles AGENTS.md, the card, and every file in `reads` and `owns` into
one markdown file sized for a small context window, and repeats the
instructions at the end where a local model will still have them in view."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
CHARS_PER_TOKEN = 3.2  # conservative for code with Qwen/Claude tokenizers
DEFAULT_BUDGET = {"qwen": 16000, "any": 24000, "claude": 60000}
STATUSES = {"todo", "doing", "review", "done", "blocked"}


@dataclass(slots=True)
class Card:
    path: Path
    id: str
    title: str
    status: str
    agent: str
    depends: list[str]
    owns: list[str]
    reads: list[str]
    accept: str
    budget: int
    body: str = field(repr=False, default="")


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [x.strip() for x in raw[1:-1].split(",") if x.strip()]
    return raw


def load_cards() -> list[Card]:
    cards = []
    for path in sorted(TASKS.glob("T*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not m:
            raise SystemExit(f"{path.name}: missing frontmatter")
        meta = {}
        for line in m.group(1).splitlines():
            if line.strip():
                key, _, value = line.partition(":")
                meta[key.strip()] = _parse_value(value)
        agent = meta.get("agent", "any")
        cards.append(Card(
            path=path, id=meta["id"], title=meta.get("title", ""), status=meta.get("status", "todo"),
            agent=agent, depends=meta.get("depends", []) or [], owns=meta.get("owns", []) or [],
            reads=meta.get("reads", []) or [], accept=meta.get("accept", ""),
            budget=int(meta.get("budget", DEFAULT_BUDGET.get(agent, 24000))), body=m.group(2),
        ))
    return cards


def _ancestors(card: Card, by_id: dict[str, Card]) -> set[str]:
    seen: set[str] = set()
    stack = list(card.depends)
    while stack:
        d = stack.pop()
        if d in seen or d not in by_id:
            continue
        seen.add(d)
        stack.extend(by_id[d].depends)
    return seen


def _pack_text(card: Card) -> str:
    files = list(dict.fromkeys([*card.reads, *card.owns]))
    parts = [
        f"# Context pack for {card.id}: {card.title}\n",
        (
            "You are a coding agent working on exactly one task card. Everything you "
            "need is below: the agent rules, the card, then the files. Edit only the "
            "files listed under `owns`.\n"
        ),
        "## AGENTS.md\n", (ROOT / "AGENTS.md").read_text(encoding="utf-8"),
        f"\n## Task card ({card.path.relative_to(ROOT).as_posix()})\n",
        card.path.read_text(encoding="utf-8"),
    ]
    for rel in files:
        p = ROOT / rel
        role = "OWNED (you edit this)" if rel in card.owns else "read-only"
        if p.exists():
            lang = "python" if p.suffix == ".py" else ""
            parts.append(f"\n## {rel} ({role})\n```{lang}\n{p.read_text(encoding='utf-8')}```\n")
        else:
            parts.append(f"\n## {rel} ({role})\n(does not exist yet: you create it)\n")
    parts.append(
        f"\n## Your task, again\nImplement card {card.id} ({card.title}).\n"
        f"- Edit only: {', '.join(card.owns)} and tasks/notes/{card.id}.md\n"
        f"- Done when this passes: `{card.accept}`\n"
        "- Then also: `uv run pytest -q` and `uv run ruff check curio_robot tests`\n"
        f"- Write the handoff note tasks/notes/{card.id}.md and commit on branch "
        f"task/{card.id.lower()}.\n"
        "- Don't edit tests you don't own; if one looks wrong, explain in the note and stop.\n"
    )
    return "".join(parts)


def tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def cmd_list(cards: list[Card]) -> None:
    for c in cards:
        print(f"{c.id:6} {c.status:8} {c.agent:6} deps={','.join(c.depends) or '-':24} {c.title}")


def cmd_ready(cards: list[Card]) -> None:
    done = {c.id for c in cards if c.status == "done"}
    ready = [c for c in cards if c.status == "todo" and set(c.depends) <= done]
    for c in ready:
        print(f"{c.id:6} {c.agent:6} {c.title}")
    if not ready:
        print("(nothing ready)")


def cmd_check(cards: list[Card]) -> int:
    errors, warnings = [], []
    by_id = {c.id: c for c in cards}
    if len(by_id) != len(cards):
        errors.append("duplicate card ids")
    for c in cards:
        if c.status not in STATUSES:
            errors.append(f"{c.id}: unknown status {c.status!r}")
        if c.agent not in DEFAULT_BUDGET:
            errors.append(f"{c.id}: unknown agent {c.agent!r}")
        for d in c.depends:
            if d not in by_id:
                errors.append(f"{c.id}: depends on unknown {d}")
        if c.id in _ancestors(c, by_id):
            errors.append(f"{c.id}: dependency cycle")
        for rel in c.reads:
            if not (ROOT / rel).exists():
                errors.append(f"{c.id}: reads missing file {rel}")
        if not c.accept:
            errors.append(f"{c.id}: no accept command")
        if c.status != "done":
            n = tokens(_pack_text(c))
            if n > c.budget:
                warnings.append(f"{c.id}: context pack ~{n} tokens > budget {c.budget}")
    open_cards = [c for c in cards if c.status != "done"]
    for i, a in enumerate(open_cards):
        for b in open_cards[i + 1 :]:
            shared = set(a.owns) & set(b.owns)
            ordered = a.id in _ancestors(b, by_id) or b.id in _ancestors(a, by_id)
            if shared and not ordered:
                errors.append(f"{a.id} and {b.id} both own {sorted(shared)} and can run in parallel")
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    print(f"{len(cards)} cards, {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


def cmd_pack(cards: list[Card], card_id: str, out: Path) -> None:
    card = next((c for c in cards if c.id.lower() == card_id.lower()), None)
    if card is None:
        raise SystemExit(f"no card {card_id}")
    text = _pack_text(card)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{card.id}.md"
    path.write_text(text, encoding="utf-8")
    n = tokens(text)
    flag = "" if n <= card.budget else f"  (over budget {card.budget}!)"
    print(f"{path.relative_to(ROOT).as_posix()}  ~{n} tokens{flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("ready")
    sub.add_parser("check")
    p = sub.add_parser("pack")
    p.add_argument("id")
    p.add_argument("--out", type=Path, default=ROOT / "out" / "context")
    args = parser.parse_args(argv)
    cards = load_cards()
    if args.cmd == "list":
        cmd_list(cards)
    elif args.cmd == "ready":
        cmd_ready(cards)
    elif args.cmd == "check":
        return cmd_check(cards)
    else:
        cmd_pack(cards, args.id, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
