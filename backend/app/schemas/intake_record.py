"""Pydantic models for the intake record and its supporting log types.

`Source.NOT_ASKED` is the implicit default for every protocol field until
something explicit changes it — a field must never be silently promoted to
ASKED_AND_DENIED without a matching logged question event (enforced in
state_engine.py, not here).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    PATIENT_REPORTED = "patient_reported"
    ASKED_AND_DENIED = "asked_and_denied"
    DOCUMENT_SOURCED = "document_sourced"
    INFERRED = "inferred"
    NOT_ASKED = "not_asked"


class FactStatus(str, Enum):
    UNCONFIRMED = "unconfirmed"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"


class Fact(BaseModel):
    """One versioned entry in a field's history. Facts are never mutated in
    place — a correction appends a new Fact whose `supersedes` points at the
    prior one, so the full audit trail survives."""

    id: str
    field: str
    value: str
    source: Source
    # Verbatim quote from the canonical STT transcript (or extracted
    # document text) backing this fact. None only when source is
    # NOT_ASKED or INFERRED.
    evidence_span: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: FactStatus
    timestamp: str
    supersedes: Optional[str] = None
    # Required when source == ASKED_AND_DENIED: the transcript question
    # event that establishes this was actually asked.
    question_event_id: Optional[str] = None


class IntakeRecord(BaseModel):
    session_id: str
    protocol_id: str
    facts: list[Fact] = Field(default_factory=list)
    created_at: str
    updated_at: str


class QuestionEvent(BaseModel):
    """One agent question logged during the conversation, keyed to a protocol field."""

    id: str
    field: str
    question_text: str
    timestamp: str


class TranscriptTurn(BaseModel):
    """One turn of the canonical transcript. This is the only source of truth for 'what was said'."""

    id: str
    speaker: str  # "patient" | "agent"
    text: str
    timestamp: str


class SafetyEvaluation(BaseModel):
    """One safety-policy evaluation log entry — written on every fact write, trigger or not."""

    id: str
    fact_id: str
    triggered: bool
    rule_id: Optional[str] = None
    action: Optional[str] = None
    timestamp: str
