"""Acceptance tests for T2-02 (grounding/anchor.py)."""

import pytest

from curio_robot.grounding.anchor import Word, anchor_time, parse_words


def _words(*spec):
    return [Word(text, start, start + 0.2) for text, start in spec]


def test_parse_top_level_words_with_offset():
    data = {"text": "hand me that", "words": [
        {"word": " hand", "start": 0.0, "end": 0.2},
        {"word": " me", "start": 0.2, "end": 0.3},
        {"word": " that", "start": 0.4, "end": 0.6},
    ]}
    words = parse_words(data, offset=10.0)
    assert [w.text for w in words] == [" hand", " me", " that"]
    assert words[2].start == pytest.approx(10.4) and words[2].end == pytest.approx(10.6)


def test_parse_segments():
    data = {"segments": [
        {"words": [{"word": "hand", "start": 0.0, "end": 0.2}]},
        {"words": [{"word": "  ", "start": 0.2, "end": 0.3}, {"word": "that", "start": 0.4, "end": 0.5}]},
    ]}
    assert [w.text for w in parse_words(data)] == ["hand", "that"]


def test_parse_nothing():
    assert parse_words({"text": "hi"}) == []


def test_anchor_on_the_deictic_word():
    words = _words(("Hand", 0.0), (" me", 0.3), (" That", 0.5), (" one.", 0.8))
    assert anchor_time(words) == pytest.approx(0.5)


def test_last_deictic_wins():
    words = _words(("that", 0.1), ("red", 0.3), ("no,", 0.6), ("THIS", 0.9), ("one", 1.1))
    assert anchor_time(words) == pytest.approx(0.9)


def test_colour_when_no_deictic():
    words = _words(("the", 0.0), ("Red,", 0.4), ("one", 0.7))
    assert anchor_time(words) == pytest.approx(0.4)


def test_end_of_speech_otherwise():
    words = _words(("hand", 0.0), ("it", 0.5), ("over", 1.0))
    assert anchor_time(words) == pytest.approx(1.2)


def test_no_words():
    assert anchor_time([]) is None
