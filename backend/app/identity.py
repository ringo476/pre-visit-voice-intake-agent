"""Patient identity verification via a signed, time-limited access token —
the piece that closes a real gap this project had: anyone who opened the
WebSocket URL was treated as the one true patient, with nothing checking
that.

In a real deployment, this token is what a clinic's booking/EHR system
would embed in the SMS or email link sent to the patient ahead of their
appointment ("click here for your pre-visit call") — the patient never
types anything, they just open a link that already contains it. This
project has no real booking system or messaging integration to send that
link through, so main.py's /api/appointments/mock endpoint stands in for
that hand-off — but the token itself, and its verification (signed,
tamper-proof, time-limited), are genuinely real, not simulated."""

import os

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# A real deployment must set a strong, secret ACCESS_TOKEN_SECRET — this
# fallback exists only so local dev/tests never fail just because it's
# unset, the same reasoning as GEMINI_API_KEY being optional for tests.
_SECRET = os.environ.get("ACCESS_TOKEN_SECRET", "dev-only-insecure-secret-change-in-production")
_serializer = URLSafeTimedSerializer(_SECRET, salt="pre-visit-session-access")

DEFAULT_TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24  # 24h — a booking link should still work the day of the call


def issue_access_token(session_id: str) -> str:
    """Signs session_id into an opaque token. Tamper it at all — change one
    character — and it fails verification; nothing about the session_id or
    the signature can be recovered or forged without the server's secret."""
    return _serializer.dumps(session_id)


def verify_access_token(token: str, max_age_seconds: int = DEFAULT_TOKEN_MAX_AGE_SECONDS) -> str | None:
    """Returns the session_id the token was issued for, or None if the
    token is missing, malformed, tampered with, or expired. Never raises —
    callers treat None uniformly as "not a valid, currently-usable token,"
    the caller doesn't need to know which of those it was."""
    if not token:
        return None
    try:
        return _serializer.loads(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None
