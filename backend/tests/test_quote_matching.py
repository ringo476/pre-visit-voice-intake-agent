"""The quote match tolerates the small slips a model makes when copying a
patient's words, without letting an invented, paraphrased or meaning-changed
quote through. The labelled examples below are also how the threshold is
justified: it has to sit in the gap between the two groups."""

import time

import pytest

from app.state_engine import (
    MIN_FUZZY_WORDS,
    QUOTE_MATCH_THRESHOLD,
    evidence_in_text,
    evidence_match_score,
)

PATIENT = "Um, I do take my albuterol inhaler sometimes, mostly at night. I had a bit of a cough and it gets worse at night."

# Genuine statements with the small slips a model makes when copying: these must be accepted.
ACCEPT = [
    ("copied exactly", PATIENT, "I do take my albuterol inhaler sometimes"),
    ("filler word dropped", PATIENT, "I do take my albuterol inhaler sometimes mostly at night"),
    ("one word changed (take -> took)", PATIENT, "I do took my albuterol inhaler sometimes"),
    ("a small word dropped", PATIENT, "I take my albuterol inhaler sometimes"),
    ("an article dropped", PATIENT, "I had a bit of cough and it gets worse at night"),
    ("a small word added", PATIENT, "I do take my albuterol inhaler sometimes and mostly at night"),
]

# Things that must NOT be accepted: invented, paraphrased, or changed in meaning.
REJECT = [
    ("invented, plausible", PATIENT, "I am allergic to penicillin"),
    ("invented, similar words", PATIENT, "I take my penicillin inhaler every night"),
    ("paraphrase with new words", PATIENT, "I use my albuterol puffer occasionally"),
    ("negation dropped", "honestly I have no fever at all today", "honestly I have a fever at all today"),
    ("negation added", "honestly I have a fever at all times", "honestly I have no fever at all times"),
    ("contraction dropped", "honestly I can't breathe properly right now", "honestly I can breathe properly right now"),
    ("number changed", "my temperature was 101 on Tuesday evening", "my temperature was 100 on Tuesday evening"),
    ("number written differently", "my temperature was one oh one on Tuesday", "my temperature was 101 on Tuesday"),
    ("unrelated sentence", "I've had a cough for a while now and it is worse", "I take penicillin for my allergy every day"),
]


@pytest.mark.parametrize("name, text, quote", ACCEPT, ids=[c[0] for c in ACCEPT])
def test_a_small_slip_in_copying_is_accepted(name, text, quote):
    assert evidence_in_text(text, quote) is True, evidence_match_score(text, quote)


@pytest.mark.parametrize("name, text, quote", REJECT, ids=[c[0] for c in REJECT])
def test_an_invented_paraphrased_or_meaning_changed_quote_is_rejected(name, text, quote):
    assert evidence_in_text(text, quote) is False, evidence_match_score(text, quote)


def test_the_threshold_sits_in_the_gap_between_the_two_groups():
    worst_accepted = min(evidence_match_score(t, q) for _, t, q in ACCEPT)
    best_rejected = max(evidence_match_score(t, q) for _, t, q in REJECT)
    assert best_rejected < QUOTE_MATCH_THRESHOLD <= worst_accepted


def test_an_exact_copy_scores_one_whatever_the_case_or_punctuation():
    assert evidence_match_score("No, no FEVER!", "no no fever") == 1.0


def test_short_quotes_must_match_exactly():
    assert MIN_FUZZY_WORDS >= 3
    assert evidence_in_text("I know it started Monday", "no") is False          # not a fragment of another word
    assert evidence_in_text("it is now Monday", "no") is False                   # one letter off a different word
    assert evidence_in_text("I did not sleep well", "I do not") is False          # three words, so no near-match


def test_a_near_match_can_never_change_a_negation_or_a_number():
    # a high similarity score alone would have accepted both of these
    text = "I have had no fever for 3 days now at all"
    assert evidence_in_text(text, "I have had a fever for 3 days now at all") is False
    assert evidence_in_text(text, "I have had no fever for 4 days now at all") is False
    assert evidence_in_text(text, "I have had no fever for 3 days now at all") is True


def test_an_empty_quote_or_text_never_matches():
    assert evidence_match_score("anything", "   ") == 0.0
    assert evidence_match_score("", "a quote with enough words here") == 0.0


def test_matching_a_long_conversation_is_fast_enough_for_a_live_call():
    long_text = " ".join(f"word{i} and some filler about the cough and the weather" for i in range(1500))
    start = time.perf_counter()
    evidence_in_text(long_text, "this sentence was never actually said by the patient today")
    assert time.perf_counter() - start < 3.0
