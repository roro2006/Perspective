"""T5-25: split-label false-belief items, the tied-label set, and the eval's new numbers
(CPU only; the grounder is a stand-in that reads the prompt's side labels)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from curio_robot import belief_bench as bb
from curio_robot.grounding.client import GroundingAnswer
from curio_robot.grounding.prompt import GroundingRequest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import belief_eval

SMALL = {"false-belief": 6, "control": 6, "present": 6}
# Fingerprint of T5-21's generator (before T5-25) for SMALL with seeds 2101-2103.
T5_21_SMALL_FINGERPRINT = "6cbdca529409db8caec6215ceb871bff9303bd66349ea416188617f0ff30e7ef"


def fingerprint(items: list[bb.BeliefItem]) -> str:
    h = hashlib.sha256()
    for it in items:
        pos = [[o.object_id, o.color, *np.round(o.position, 9).tolist()] for o in it.truth.objects]
        rec = [it.id, it.kind, it.text, it.marked_text, it.frame, it.target_id, it.moved_id,
               it.mover, it.away, pos, np.round(it.current.pose.position, 9).tolist(),
               [list(o.visible) for o in it.observations]]
        h.update(json.dumps(rec).encode())
    return h.hexdigest()


_LINE = re.compile(r"^([A-Z]): (\w+) \(speaker's (left|right), robot's (left|right)\)$")


class LabelGrounder:
    """Answers from the prompt's own labels: colour and side from the words, frame from
    "my"/"your"; the first matching letter."""

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        text = request.messages[0]["content"][1]["text"]
        said = re.search(r'The person said: "(.*)"', text).group(1)
        side = re.findall(r"\b(left|right)\b", said)[-1]
        col = 4 if " your " in f" {said} " else 3
        for line in text.splitlines():
            m = _LINE.match(line)
            if m and m.group(2) in said and m.group(col) == side:
                return GroundingAnswer({request.letters[m.group(1)]: 1.0}, m.group(1), 1.0, "fake")
        return GroundingAnswer({}, None, 1.0, "unavailable", "no match")


@pytest.fixture(scope="module")
def sets():
    from curio_robot.sim.world import SimWorld

    with SimWorld(320, 240) as world:
        return {
            "t5_21": bb.generate(world, SMALL),
            "split": bb.generate(world, SMALL, split_labels=True),
            "control_2112": bb.generate(world, {"control": 6}, {"control": 2112}),
            "tied": bb.generate_tied(world, 6),
        }


def test_split_items_have_distinct_labels_and_the_words_pick_the_partner(sets):
    moved = [i for i in sets["split"] if i.kind != "control"]
    assert len(moved) == 12
    for item in moved:
        target, partner = bb.pair_ids(item)
        labels = bb.side_labels(item)
        assert labels[target] != labels[partner], item.id
        assert labels[bb.words_pick(item)] == item.side
        if item.kind == "false-belief":
            assert bb.words_pick(item) == partner  # the current view points elsewhere
        else:
            # present: the target is what the words pick now, not the block they picked before
            assert bb.words_pick(item) == target and item.moved_id is not None
    assert sum(i.label_redraws for i in moved) > 0


def test_split_controls_equal_t5_21_generator_for_the_same_seed(sets):
    controls = [i for i in sets["split"] if i.kind == "control"]
    assert fingerprint(controls) == fingerprint(sets["control_2112"])
    assert all(i.label_redraws == 0 for i in controls)


def test_default_reproduces_t5_21_items(sets):
    assert bb.SEEDS == {"false-belief": 2101, "control": 2102, "present": 2103}
    assert fingerprint(sets["t5_21"]) == T5_21_SMALL_FINGERPRINT
    assert all(i.label_redraws == 0 for i in sets["t5_21"])


def test_tied_set_has_tied_labels(sets):
    tied = sets["tied"]
    assert [i.id for i in tied] == [f"tied-{k:04d}" for k in range(6)]
    for item in tied:
        assert item.kind == "false-belief" and bb.labels_split(item) is False
        assert bb.words_pick(item) != item.target_id  # the words still flip, as in T5-21


def test_seeds_are_fresh_and_disjoint():
    new = set(bb.SPLIT_SEEDS.values()) | {bb.TIED_SEED}
    assert new == {2111, 2112, 2113, 2114}
    assert not new & set(bb.SEEDS.values())
    with pytest.raises(ValueError):
        bb.make_item(None, np.random.default_rng(0), "false-belief", 0, "sideways")


def test_eval_reports_tied_set_and_current_following_words(sets):
    items = sets["split"] + sets["tied"]
    rows = belief_eval.evaluate(items, LabelGrounder())
    assert {r["set"] for r in rows} == {"false-belief", "control", "present", "tied"}
    fb = [r for r in rows if r["set"] == "false-belief"]
    # On split labels a label reader on the current view takes the block the words point to.
    assert all(r["current"]["chosen"] == r["words_pick"] for r in fb)
    assert not any(r["current"]["ok"] for r in fb)
    s = belief_eval.summarise(rows)
    assert s["false_belief"]["belief"]["n"] == 6 and s["tied"]["current"]["n"] == 6
    assert s["current_follows_words"]["false_belief"]["acc"] == 1.0
    assert s["label_redraws"]["control"] == 0
    assert s["flag"]["n"] == 12  # tied items are not in the gated flag line
    out = belief_eval.table(s)
    assert "Tied-label set" in out and "4_flag>=0.95" in out
