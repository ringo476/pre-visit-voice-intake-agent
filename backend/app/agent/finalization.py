"""Finalizing the intake is the one irreversible step, so the graph pauses there
(a LangGraph interrupt) until the patient has heard the read-back and said yes.
This module holds the plain-code rules for that pause. No model decides whether
the patient agreed: the check is deliberately strict, and when it is unsure the
answer is "not confirmed" and the conversation simply carries on."""

import re

CLOSING_TEXT = "Thank you, that is everything I need. I have passed your answers on to your care team."

# A reply this long is a statement, not a yes.
MAX_WORDS = 8

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "correct", "right", "sure", "okay", "ok", "exactly", "perfect",
    "good", "fine", "absolutely", "definitely",
}
# Any of these means the patient is adding, correcting or doubting something. "t" is what
# is left of "don't", "isn't" and "can't" once punctuation is removed.
_BLOCKERS = {
    "no", "nope", "not", "t", "never", "wrong", "incorrect", "but", "except", "actually",
    "however", "wait", "hold", "change", "also", "missing", "forgot", "mistake",
}


def is_clear_yes(text: str) -> bool:
    """True only for a short reply that agrees and adds nothing: "yes", "yeah that's
    correct", "sounds good". Anything longer, or containing a correction, a doubt
    or a negation, is not a clear yes."""
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    if not words or len(words) > MAX_WORDS:
        return False
    if any(w in _BLOCKERS for w in words):
        return False
    return any(w in _AFFIRMATIVE for w in words)
