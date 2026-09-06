from app.protocol.classifier import classify_complaint


def test_greeting_matches_nothing():
    assert classify_complaint("hi") is None
    assert classify_complaint("hello there") is None
    assert classify_complaint("") is None


def test_respiratory_complaint_classifies_correctly():
    assert classify_complaint("I've had a cough for around a week and a half") == "respiratory-intake"
    assert classify_complaint("I think I have a cold, my chest feels congested") == "respiratory-intake"


def test_leg_injury_complaint_classifies_correctly():
    assert classify_complaint("I twisted my ankle yesterday playing basketball") == "musculoskeletal-leg-injury"
    assert classify_complaint("my knee is really swollen after I fell") == "musculoskeletal-leg-injury"


def test_unrelated_smalltalk_matches_nothing():
    assert classify_complaint("how are you doing today") is None
