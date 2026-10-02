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
    # The patient was asked and said they don't know / don't remember. Kept
    # distinct from ASKED_AND_DENIED so "I don't know" can never be read as "no".
    UNCERTAIN = "uncertain"
    DOCUMENT_SOURCED = "document_sourced"
    INFERRED = "inferred"
    NOT_ASKED = "not_asked"


class Polarity(str, Enum):
    """The direction of the patient's answer about a topic. Stored on every
    fact so nothing has to guess "is this a no?" from the wording of the value."""

    PRESENT = "present"  # yes, it exists / here is the detail
    ABSENT = "absent"    # no, it does not
    UNKNOWN = "unknown"  # the patient does not know / does not remember


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
    polarity: Polarity = Polarity.PRESENT
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
    # Index in session.transcript of the agent turn that actually spoke this
    # question, and that turn's text. Both stay None until the reply is
    # finalized: an event the tool layer logged but the agent never said
    # (an abandoned or superseded turn) must not count as "asked".
    asked_in_turn: Optional[int] = None
    spoken_text: Optional[str] = None


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
