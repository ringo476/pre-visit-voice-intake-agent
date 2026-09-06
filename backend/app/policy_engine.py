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


def _matches_keyword(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


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
