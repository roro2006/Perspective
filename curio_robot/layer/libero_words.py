"""The words adapter for LIBERO-Spatial policies: a chosen black bowl to the task string that
names it. Task T5-31.

Policies trained on LIBERO-Spatial (SmolVLA, pi0.5) only know ten instructions, one per scene
layout, and each names the bowl to move by where it sits: "pick up the black bowl next to the
ramekin and place it on the plate". Both bowls in every layout sit on a region that one of the
ten instructions names, so once Curio knows which bowl the person means, the instruction for
that bowl's region is the words the policy was trained to follow. No new phrasing is invented.

The tables are the LIBERO-Spatial BDDL files (`:language` is spelled differently there; the
strings below are LIBERO's `task.language`): PLACEMENTS is each file's `(:init ...)` line for
the two bowls, and REGION_TASK says which task's instruction names each region.
`placements_from_bddl` reads the same table back from the files; the T5-31 harness checks the
two agree before it runs."""

from __future__ import annotations

import re

BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")

# LIBERO-Spatial's task order (benchmark.get_benchmark_dict()["libero_spatial"]().tasks).
TASKS = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
)

# The region (the BDDL `(:init (On|In bowl <region>))` target) each task's words name.
REGION_TASK = {
    "main_table_between_plate_ramekin_region": 0,
    "main_table_next_to_ramekin_region": 1,
    "main_table_table_center": 2,
    "cookies_1": 3,
    "wooden_cabinet_1_top_region": 4,
    "glazed_rim_porcelain_ramekin_1": 5,
    "main_table_next_to_box_region": 6,
    "flat_stove_1_cook_region": 7,
    "main_table_next_to_plate_region": 8,
    "wooden_cabinet_1_top_side": 9,
}

# task id -> (region of akita_black_bowl_1, region of akita_black_bowl_2). In every task the
# goal is (On akita_black_bowl_1 plate_1): bowl 1 is the target, bowl 2 its look-alike.
PLACEMENTS = (
    ("main_table_between_plate_ramekin_region", "main_table_next_to_ramekin_region"),
    ("main_table_next_to_ramekin_region", "main_table_next_to_box_region"),
    ("main_table_table_center", "main_table_next_to_plate_region"),
    ("cookies_1", "wooden_cabinet_1_top_side"),
    ("wooden_cabinet_1_top_region", "wooden_cabinet_1_top_side"),
    ("glazed_rim_porcelain_ramekin_1", "cookies_1"),
    ("main_table_next_to_box_region", "flat_stove_1_cook_region"),
    ("flat_stove_1_cook_region", "wooden_cabinet_1_top_side"),
    ("main_table_next_to_plate_region", "main_table_next_to_ramekin_region"),
    ("wooden_cabinet_1_top_side", "flat_stove_1_cook_region"),
)
TARGET = BOWLS[0]  # LIBERO-Spatial's goal bowl in every task


def bowl_task(scene_task: int, bowl: str) -> int:
    """The task id whose instruction names `bowl` where it sits in scene `scene_task`."""
    if not 0 <= scene_task < len(TASKS):
        raise ValueError(f"LIBERO-Spatial has tasks 0-{len(TASKS) - 1}, not {scene_task}")
    if bowl not in BOWLS:
        raise ValueError(f"not a LIBERO-Spatial bowl: {bowl!r}")
    return REGION_TASK[PLACEMENTS[scene_task][BOWLS.index(bowl)]]


def libero_words(scene_task: int, bowl: str) -> str:
    """The LIBERO-Spatial task string for `bowl` in the layout of task `scene_task`. For the
    target bowl this is the scene's own instruction; for its look-alike it is the instruction
    whose region the look-alike sits on, so a wrong grounding sends the policy to the wrong
    bowl rather than to an instruction it never saw."""
    return TASKS[bowl_task(scene_task, bowl)]


_INIT = re.compile(r"\((On|In)\s+(akita_black_bowl_[12])\s+(\w+)\)")


def placements_from_bddl(text: str) -> tuple[str, str]:
    """(region of bowl 1, region of bowl 2) from one BDDL file's `(:init ...)` block."""
    init = text[text.index("(:init"):text.index("(:goal")]
    found = {bowl: region for _, bowl, region in _INIT.findall(init)}
    missing = [b for b in BOWLS if b not in found]
    if missing:
        raise ValueError(f"no init placement for {missing}")
    return found[BOWLS[0]], found[BOWLS[1]]
