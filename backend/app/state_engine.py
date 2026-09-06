"""The clinical state engine — pure functions, no I/O. Enforces the core
provenance invariant: a fact can only claim ASKED_AND_DENIED if a matching
question was actually logged first. This is what the tool layer calls;
the reasoning model never touches this directly."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from app.schemas.intake_record import Fact, FactStatus, IntakeRecord, QuestionEvent, Source
from app.schemas.protocol_config import ProtocolConfig

AFFIRMATIVE_SOURCES = {Source.PATIENT_REPORTED, Source.DOCUMENT_SOURCED, Source.INFERRED}


class ProvenanceViolationError(Exception):
    """Raised whenever a write would violate a provenance invariant. Callers
    (the tool layer) are expected to catch this and report a tool error
    back to the model rather than let state silently corrupt."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_empty_record(session_id: str, protocol_id: str) -> IntakeRecord:
    now = _now()
    return IntakeRecord(session_id=session_id, protocol_id=protocol_id, facts=[], created_at=now, updated_at=now)


def apply_fact(
    record: IntakeRecord,
    field: str,
    value: str,
    source: Source,
    evidence_span: Optional[str],
    confidence: float,
    question_events: list[QuestionEvent],
    question_event_id: Optional[str] = None,
) -> IntakeRecord:
    """Appends a new fact to the record. Never mutates or removes existing facts."""
    if source == Source.ASKED_AND_DENIED:
        if not question_event_id:
            raise ProvenanceViolationError(
                f'Cannot record "{field}" as asked_and_denied without a question_event_id'
            )
        event = next((e for e in question_events if e.id == question_event_id), None)
        if event is None:
            raise ProvenanceViolationError(
                f'question_event_id "{question_event_id}" does not match any logged question event'
            )
        if event.field != field:
            raise ProvenanceViolationError(
                f'question_event_id "{question_event_id}" logged a question about '
                f'"{event.field}", not "{field}"'
            )

    timestamp = _now()
    fact = Fact(
        id=str(uuid.uuid4()),
        field=field,
        value=value,
        source=source,
        evidence_span=evidence_span,
        confidence=confidence,
        status=FactStatus.UNCONFIRMED,
        timestamp=timestamp,
        question_event_id=question_event_id,
    )
    return record.model_copy(update={"facts": [*record.facts, fact], "updated_at": timestamp})


def record_correction(
    record: IntakeRecord,
    fact_id: str,
    field: str,
    new_value: str,
    evidence_span: str,
    confidence: float,
) -> IntakeRecord:
    """Appends a corrected version of a fact. The prior fact is left
    untouched; the new fact's `supersedes` links back to it."""
    prior = next((f for f in record.facts if f.id == fact_id), None)
    if prior is None:
        raise ProvenanceViolationError(f'Cannot correct unknown fact id "{fact_id}"')
    if prior.field != field:
        raise ProvenanceViolationError(
            f'Correction targets field "{field}" but fact "{fact_id}" belongs to field "{prior.field}"'
        )

    timestamp = _now()
    corrected = Fact(
        id=str(uuid.uuid4()),
        field=field,
        value=new_value,
        source=Source.PATIENT_REPORTED if prior.source == Source.NOT_ASKED else prior.source,
        evidence_span=evidence_span,
        confidence=confidence,
        status=FactStatus.CORRECTED,
        timestamp=timestamp,
        supersedes=prior.id,
        question_event_id=prior.question_event_id,
    )
    return record.model_copy(update={"facts": [*record.facts, corrected], "updated_at": timestamp})


def get_current_facts(record: IntakeRecord) -> dict[str, Fact]:
    """The current (non-superseded) fact for every field that has one, keyed by field name."""
    superseded = {f.supersedes for f in record.facts if f.supersedes}
    current: dict[str, Fact] = {}
    for fact in record.facts:
        if fact.id in superseded:
            continue
        existing = current.get(fact.field)
        if existing is None or fact.timestamp > existing.timestamp:
            current[fact.field] = fact
    return current


def get_current_fact(record: IntakeRecord, field: str) -> Optional[Fact]:
    return get_current_facts(record).get(field)


class MissingField(BaseModel):
    field: str
    label: str
    category: str


def get_missing_fields(record: IntakeRecord, protocol: ProtocolConfig) -> list[MissingField]:
    """Fields the protocol still needs. A conditional field (`required_if`)
    only becomes required once the referenced field has an affirmative
    current fact — a denial or an unasked field never triggers its
    dependents."""
    current = get_current_facts(record)

    def is_covered(field: str) -> bool:
        fact = current.get(field)
        return fact is not None and fact.source != Source.NOT_ASKED

    def is_affirmed(field: str) -> bool:
        fact = current.get(field)
        return fact is not None and fact.source in AFFIRMATIVE_SOURCES

    missing: list[MissingField] = []
    for pf in protocol.fields:
        required_now = pf.required or (pf.required_if is not None and is_affirmed(pf.required_if))
        if required_now and not is_covered(pf.field):
            missing.append(MissingField(field=pf.field, label=pf.label, category=pf.category))
    return missing


def is_record_complete(record: IntakeRecord, protocol: ProtocolConfig) -> bool:
    return len(get_missing_fields(record, protocol)) == 0
