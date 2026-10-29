"""T2-10: anchor.parse_words on a real whisper-server reply (whisper.cpp b5454,
base.en, verbose_json, Windows speech saying "Hand me that blue one.").
Trimmed to the fields that matter; punctuation arrives as its own word."""

from __future__ import annotations

from curio_robot.grounding.anchor import anchor_time, parse_words

REPLY = {
    "task": "transcribe",
    "language": "english",
    "duration": 2.02,
    "text": "Hand me that blue one.",
    "segments": [{
        "id": 0, "text": " Hand me that blue one.", "start": 0.0, "end": 2.0,
        "words": [
            {"word": " Hand", "start": 0.14, "end": 0.47, "t_dtw": -1, "probability": 0.61},
            {"word": " me", "start": 0.47, "end": 0.66, "t_dtw": -1, "probability": 0.99},
            {"word": " that", "start": 0.75, "end": 1.17, "t_dtw": -1, "probability": 0.99},
            {"word": " blue", "start": 1.17, "end": 1.21, "t_dtw": -1, "probability": 0.98},
            {"word": " one", "start": 1.64, "end": 2.0, "t_dtw": -1, "probability": 0.99},
            {"word": ".", "start": 2.0, "end": 2.0, "t_dtw": -1, "probability": 0.95},
        ],
    }],
}


def test_real_reply_parses_in_order_with_offset():
    words = parse_words(REPLY, offset=10.0)
    assert [w.text for w in words] == [" Hand", " me", " that", " blue", " one", "."]
    assert words[2].start == 10.75 and words[-1].end == 12.0


def test_real_reply_anchors_on_that():
    assert anchor_time(parse_words(REPLY)) == 0.75  # "that" beats the later "blue"
