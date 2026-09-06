"""Deterministic chief-complaint classification — decides which protocol
checklist applies from what the patient actually says. This is a
structural decision (it determines which fields get tracked and which
safety rules are even reachable for the rest of the session), so it's kept
out of the LLM's hands entirely, same principle as everything else in this
system: plain keyword matching against each registered protocol, not a
model's free judgment call.

A greeting like "hi" or "hello" matches nothing and returns None — the
agent should keep asking what brings the patient in rather than forcing a
premature classification.
"""

import re

from app.protocol.registry import PROTOCOLS


def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def classify_complaint(text: str) -> str | None:
    """Returns the best-matching protocol_id, or None if nothing matches
    (e.g. small talk / a greeting with no complaint content yet)."""
    tokens = _tokenize(text)
    if not tokens:
        return None

    best_id: str | None = None
    best_score = 0

    for protocol_id, protocol in PROTOCOLS.items():
        score = 0
        for keyword in protocol.keywords:
            keyword_tokens = keyword.lower().split()
            if len(keyword_tokens) == 1:
                if keyword_tokens[0] in tokens:
                    score += 1
            elif keyword.lower() in text.lower():
                score += 1
        if score > best_score:
            best_score = score
            best_id = protocol_id

    return best_id if best_score > 0 else None
