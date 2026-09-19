import time

from app.identity import issue_access_token, verify_access_token


def test_a_freshly_issued_token_verifies_to_the_same_session_id():
    token = issue_access_token("session-123")
    assert verify_access_token(token) == "session-123"


def test_a_tampered_token_fails_verification():
    token = issue_access_token("session-123")
    tampered = token[:-1] + ("x" if token[-1] != "x" else "y")
    assert verify_access_token(tampered) is None


def test_an_empty_or_missing_token_fails_verification():
    assert verify_access_token("") is None
    assert verify_access_token(None) is None


def test_a_completely_made_up_token_fails_verification():
    assert verify_access_token("not-a-real-token-at-all") is None


def test_an_expired_token_fails_verification():
    token = issue_access_token("session-123")
    # itsdangerous timestamps at second granularity, so the margin here
    # needs to clear a full second of rounding, not just graze max_age.
    time.sleep(2.2)
    assert verify_access_token(token, max_age_seconds=1) is None


def test_a_token_for_one_session_cannot_be_reused_to_claim_a_different_one():
    token = issue_access_token("session-A")
    assert verify_access_token(token) == "session-A"
    assert verify_access_token(token) != "session-B"
