"""Acceptance tests for T1-05 (conventions/profiles.py)."""

import json

import pytest

from curio_robot.conventions.belief import ConventionBelief
from curio_robot.conventions.profiles import ProfileStore


def test_unknown_person_gets_the_prior(tmp_path):
    assert ProfileStore(tmp_path).load("ana") == ConventionBelief()


def test_save_then_load(tmp_path):
    store = ProfileStore(tmp_path / "profiles")  # root doesn't exist yet
    b = ConventionBelief(own_frame_a=2.0, own_frame_b=5.0)
    store.save("ana", b)
    assert ProfileStore(tmp_path / "profiles").load("ana") == b
    data = json.loads((tmp_path / "profiles" / "ana.json").read_text())
    assert data["person"] == "ana" and data["belief"] == b.to_dict() and "updated" in data


@pytest.mark.parametrize("name", ["", "../x", "a b", "x" * 65, "ana.json"])
def test_bad_names_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        ProfileStore(tmp_path).save(name, ConventionBelief())


def test_unreadable_file_gives_the_prior(tmp_path):
    (tmp_path / "ana.json").write_text("{not json")
    assert ProfileStore(tmp_path).load("ana") == ConventionBelief()


def test_reset(tmp_path):
    store = ProfileStore(tmp_path)
    store.save("ana", ConventionBelief(own_frame_a=9.0))
    store.reset("ana")
    store.reset("nobody")
    assert store.load("ana") == ConventionBelief()


def test_no_temp_files_left_behind(tmp_path):
    store = ProfileStore(tmp_path)
    store.save("ana", ConventionBelief())
    store.save("ana", ConventionBelief(own_frame_a=3.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ana.json"]
