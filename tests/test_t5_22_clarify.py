"""T5-22: the ask-back question generator and policy (CPU only)."""

from __future__ import annotations

import numpy as np
import pytest

from curio_robot import clarify
from curio_robot.clarify import Candidate, Landmark


def pair(extra: tuple[Candidate, ...] = ()) -> list[Candidate]:
    return [Candidate("red_1", "red", -0.10, 0.50), Candidate("red_2", "red", 0.10, 0.45),
            *extra]


def test_side_question_is_relative_and_in_the_speaker_frame():
    qs = clarify.questions(pair(), {"red_1": 0.5, "red_2": 0.5})
    assert qs[0].text == "The one on your left?"
    assert qs[0].yes == frozenset({"red_1"})
    # right/left and farther/nearer split the pair the same way: listed once
    assert len(qs) == 1


def test_distance_question_when_side_gap_too_small():
    cands = [Candidate("a", "red", 0.0, 0.40), Candidate("b", "red", 0.005, 0.60)]
    qs = clarify.questions(cands, {"a": 0.5, "b": 0.5})
    assert [q.text for q in qs] == ["The one nearer you?"]


def test_colour_named_when_contenders_mix_colours():
    cands = pair((Candidate("blue_1", "blue", 0.0, 0.5),))
    qs = clarify.questions(cands, {"red_1": 0.4, "red_2": 0.3, "blue_1": 0.3})
    texts = [q.text for q in qs]
    assert "The red one?" in texts
    assert "The red one on your left?" in texts


def test_landmark_relations_from_the_speaker():
    boxes = Landmark("boxes", (1.0, 0.0), 0.3)
    assert clarify.landmark_relation((2.0, 0.0), (0.0, 0.0), boxes) == "behind the boxes"
    assert clarify.landmark_relation((0.8, 0.0), (0.0, 0.0), boxes) == "in front of the boxes"
    assert clarify.landmark_relation((1.0, 0.45), (0.0, 0.0), boxes) == "by the boxes"
    assert clarify.landmark_relation((0.0, 2.0), (0.0, 0.0), boxes) is None


def test_landmark_question_wins_ties_and_says_from_you():
    cands = [Candidate("h", "tote", -0.3, 2.2, ("behind the boxes",)),
             Candidate("v", "tote", 1.1, 1.8)]
    q = clarify.best_question(cands, {"h": 0.53, "v": 0.47})
    assert q is not None and q.text == "The one behind the boxes from you?"


def test_gain_is_largest_for_an_even_split_and_zero_for_a_sure_one():
    q = clarify.Question("q", frozenset({"a"}), "side")
    even = clarify.gain(q, {"a": 0.5, "b": 0.5})
    skew = clarify.gain(q, {"a": 0.9, "b": 0.1})
    assert even > skew > 0
    assert clarify.gain(q, {"a": 1.0, "b": 0.0}) == pytest.approx(0.0, abs=1e-9)
    assert clarify.gain(q, {"a": 0.5, "b": 0.5}, eps=0.0) == pytest.approx(1.0)


def test_update_is_bayes_with_answer_noise():
    q = clarify.Question("q", frozenset({"a"}), "side")
    post = clarify.update({"a": 0.5, "b": 0.5}, q, True, eps=0.1)
    assert post["a"] == pytest.approx(0.9)
    assert sum(post.values()) == pytest.approx(1.0)


def test_confident_grounder_acts_without_asking():
    d = clarify.decide(pair(), {"red_1": 0.97, "red_2": 0.03}, clarify.truthful("red_2"), 0.95)
    assert d.asked == () and d.choice == "red_1"


def test_unsure_grounder_asks_and_follows_the_answer():
    d = clarify.decide(pair(), {"red_1": 0.7, "red_2": 0.3}, clarify.truthful("red_2"), 0.95)
    assert 1 <= len(d.asked) <= clarify.MAX_QUESTIONS
    assert d.asked[0][1] is False  # "on your left?" -> no
    assert d.choice == "red_2"


def test_never_more_than_k_questions_and_no_repeated_split():
    cands = pair((Candidate("red_3", "red", 0.0, 0.60), Candidate("red_4", "red", 0.05, 0.30)))
    probs = {"red_1": 0.25, "red_2": 0.25, "red_3": 0.25, "red_4": 0.25}
    d = clarify.decide(cands, probs, clarify.truthful("red_3"), 0.99, k=2)
    assert len(d.asked) <= 2
    keys = [clarify.split_key(q, set(probs)) for q, _ in d.asked]
    assert len(set(keys)) == len(keys)


def test_nothing_worth_asking_acts_on_the_argmax():
    cands = [Candidate("a", "red", 0.0, 0.5), Candidate("b", "red", 0.001, 0.505)]
    d = clarify.decide(cands, {"a": 0.6, "b": 0.4}, clarify.truthful("b"), 0.95)
    assert d.asked == () and d.choice == "a"


def test_noisy_answerer_flips_about_the_given_fraction():
    q = clarify.Question("q", frozenset({"a"}), "side")
    ans = clarify.noisy("a", np.random.default_rng(0), 0.1)
    wrong = sum(not ans(q) for _ in range(5000))
    assert 0.08 < wrong / 5000 < 0.12


def test_empty_posterior_gives_no_choice():
    d = clarify.decide([], {}, clarify.truthful("a"), 0.9)
    assert d.choice is None and d.asked == ()
