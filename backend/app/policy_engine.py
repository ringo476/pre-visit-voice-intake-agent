"""Deterministic safety policy engine. Runs on every fact write,
unconditionally — not only when the model calls check_safety_protocol.
Plain keyword matching, not an LLM judgment call, so the model cannot talk
its way past it."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.intake_record import Fact, SafetyEvaluation

_RULES_PATH = Path(__file__).parent / "rules.json"
with open(_RULES_PATH, "r", encoding="utf-8") as f:
    _RULES = json.load(f)


@dataclass
class SafetyResult:
    triggered: bool
    rule_id: Optional[str] = None
    action: Optional[str] = None
    scripted_message: Optional[str] = None


_NOT_TRIGGERED = SafetyResult(triggered=False)


_NEGATION_WORDS = {"no", "not", "n't", "denies", "denied", "without", "never", "none", "negative", "isn't", "wasn't", "doesn't", "didn't", "hasn't"}
_NEGATION_WINDOW = 8  # words to look back from a match before treating it as a real (non-negated) hit


def _matches_keyword(text: str, keywords: list[str]) -> bool:
    """Plain substring matching, but skips a hit that's immediately negated
    (e.g. "no swelling on my face", "denies any dizziness") — otherwise a
    patient explicitly ruling a symptom out would trip the same rule as
    actually reporting it, since the keyword text is identical either way."""
    lower = text.lower()
    for keyword in keywords:
        kw = keyword.lower()
        start = 0
        while True:
            idx = lower.find(kw, start)
            if idx == -1:
                break
            preceding_words = lower[:idx].split()[-_NEGATION_WINDOW:]
            if not any(w.strip(".,!?") in _NEGATION_WORDS for w in preceding_words):
                return True
            start = idx + 1
    return False


def _to_result(rule: dict) -> SafetyResult:
    return SafetyResult(
        triggered=True,
        rule_id=rule["id"],
        action=rule["action"],
        scripted_message=rule["scripted_message"],
    )


def evaluate_fact(fact: Fact) -> SafetyResult:
    text = f"{fact.value} {fact.evidence_span or ''}"

    for rule in _RULES["field_rules"]:
        if rule["field"] == fact.field and _matches_keyword(text, rule["keywords"]):
            return _to_result(rule)

    for rule in _RULES["global_keywords"]:
        if _matches_keyword(text, rule["keywords"]):
            return _to_result(rule)

    return _NOT_TRIGGERED


def evaluate_statement(statement: str) -> SafetyResult:
    """Evaluates a free-text statement passed via the check_safety_protocol
    tool (used when the model notices something concerning that hasn't
    necessarily been written as a structured fact yet)."""
    for rule in _RULES["global_keywords"]:
        if _matches_keyword(statement, rule["keywords"]):
            return _to_result(rule)

    for rule in _RULES["field_rules"]:
        if _matches_keyword(statement, rule["keywords"]):
            return _to_result(rule)

    return _NOT_TRIGGERED


def build_evaluation_log_entry(fact_id: str, result: SafetyResult) -> SafetyEvaluation:
    """Every evaluation is logged, triggered or not, for audit and eval metrics."""
    return SafetyEvaluation(
        id=str(uuid.uuid4()),
        fact_id=fact_id,
        triggered=result.triggered,
        rule_id=result.rule_id,
        action=result.action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
