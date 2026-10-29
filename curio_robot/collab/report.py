"""The scoreboard as a table: one row per fluency metric, one column per
condition. Task T3-09."""

from __future__ import annotations

from dataclasses import fields

from curio_robot.collab.metrics import Fluency

HIGHER_IS_BETTER = ("success", "concurrent_frac")


def format_table(summary: dict[str, dict[str, float | None]]) -> str:
    """Behaviour: tasks/T3-09-scoreboard-cli.md. Input as metrics.summarize returns it.

    Renders a Markdown table with one row per metric and one column per condition.
    The 'better' column indicates whether higher or lower values are preferred.
    """
    if not summary:
        return "(no sessions)"

    conditions = list(summary.keys())
    header = "| metric | " + " | ".join(conditions) + " | better |"
    separator = "| " + " | ".join(["---"] * (len(conditions) + 2)) + " |"

    lines = [header, separator]

    metric_names = [f.name for f in fields(Fluency)] + ["n"]

    for name in metric_names:
        cells = [_format_cell(name, summary[cond].get(name)) for cond in conditions]

        if name == "n":
            better = ""
        elif name in HIGHER_IS_BETTER:
            better = "higher"
        else:
            better = "lower"

        row = "| " + name + " | " + " | ".join(cells) + f" | {better} |"
        lines.append(row)

    return "\n".join(lines)


def _format_cell(name: str, value: float | None) -> str:
    """None is "-", n is an integer, every other number has two decimals."""
    if value is None:
        return "-"
    if name == "n":
        return str(int(value))
    return f"{value:.2f}"
