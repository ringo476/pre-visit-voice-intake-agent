"""The clinical state engine — pure functions, no I/O. Enforces the core
provenance invariant: a fact can only claim ASKED_AND_DENIED if a matching
question was actually logged first. This is what the tool layer calls;
the reasoning model never touches this directly."""

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from app.schemas.intake_record import Fact, FactStatus, IntakeRecord, QuestionEvent, Source, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig

AFFIRMATIVE_SOURCES = {Source.PATIENT_REPORTED, Source.DOCUMENT_SOURCED, Source.INFERRED}
# Both of these are answers to a question the agent must really have asked:
# "no" and "I don't know" each close a field without the patient volunteering
# anything, so each needs proof the question was spoken first.
SPOKEN_QUESTION_SOURCES = {Source.ASKED_AND_DENIED, Source.UNCERTAIN}


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
    source = Source(source)
    if source in SPOKEN_QUESTION_SOURCES:
        if not question_event_id:
            raise ProvenanceViolationError(
                f'Cannot record "{field}" as {source.value} without a question_event_id'
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


def find_asked_event(
    question_events: list[QuestionEvent], field: str, before_turn: int
) -> Optional[QuestionEvent]:
    """The most recent question about `field` that was actually spoken in an
    agent turn earlier than `before_turn` (an index into the transcript).
    The server resolves this itself rather than trusting an id from the
    model: the model only ever sees an event id inside the tool result of
    the turn that created it, so it has nothing valid to hand back on the
    turn where the patient actually answers."""
    candidates = [
        e
        for e in question_events
        if e.field == field and e.asked_in_turn is not None and e.asked_in_turn < before_turn
    ]
    return max(candidates, key=lambda e: e.asked_in_turn, default=None)


def _normalize_for_match(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def patient_reply_after(transcript: list[TranscriptTurn], event: QuestionEvent) -> str:
    """Everything the patient said after the agent spoke `event`'s question."""
    if event.asked_in_turn is None:
        return ""
    return " ".join(t.text for t in transcript[event.asked_in_turn + 1 :] if t.speaker == "patient")


def evidence_follows_question(transcript: list[TranscriptTurn], event: QuestionEvent, evidence: str) -> bool:
    """True if `evidence` is a quote from something the patient said AFTER
    the agent spoke `event`'s question. Punctuation and case are ignored,
    since the model's quote and the STT transcript routinely differ in
    those alone."""
    needle = _normalize_for_match(evidence)
    if not needle or event.asked_in_turn is None:
        return False
    patient_text_after = _normalize_for_match(patient_reply_after(transcript, event))
    # Pad both sides so the quote must match whole words: a bare "no" must not
    # count as found inside "I know".
    return f" {needle} " in f" {patient_text_after} "


def record_correction(
    record: IntakeRecord,
    field: str,
    new_value: str,
    evidence_span: str,
    confidence: float,
) -> IntakeRecord:
    """Appends a corrected version of the current fact for `field`. The
    prior fact is left untouched; the new fact's `supersedes` links back to
    it. Looks up the current fact by field name rather than taking a prior
    fact id from the caller — an id from an earlier update_intake_record
    result is only ever visible within that same turn's own tool-call
    history, never in later turns (each turn rebuilds its message list from
    session.transcript alone), so a caller several turns later has no
    reliable id to supply. The field name, by contrast, is always known."""
    prior = get_current_fact(record, field)
    if prior is None:
        raise ProvenanceViolationError(
            f'Cannot correct "{field}" — nothing has been recorded for it yet; use update_intake_record instead'
        )

    # A correction is always the patient's own new statement. A prior that was
    # never asked, "unsure", or a denial must not leak its source onto the new
    # value: "I did have a fever after all" filed as asked_and_denied would
    # render in the brief as "Patient denies fever".
    new_source = (
        Source.PATIENT_REPORTED
        if prior.source in (Source.NOT_ASKED, Source.UNCERTAIN, Source.ASKED_AND_DENIED)
        else prior.source
    )

    timestamp = _now()
    corrected = Fact(
        id=str(uuid.uuid4()),
        field=field,
        value=new_value,
        source=new_source,
        evidence_span=evidence_span,
        confidence=confidence,
        status=FactStatus.CORRECTED,
        timestamp=timestamp,
        supersedes=prior.id,
        # A changed source means the new value was not given in answer to the
        # logged question, so it must not inherit that question's proof.
        question_event_id=prior.question_event_id if new_source == prior.source else None,
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


_DENIAL_WORDS = {"no", "none", "false", "negative", "denied", "n/a", "na"}


def _looks_like_denial(value: str) -> bool:
    """A fact's source (e.g. patient_reported) only says how the value was
    obtained, not what it says — a fact can be patient_reported AND a
    denial ("no medication allergies") at once. This is a best-effort check
    of the value text itself, since the model phrases denials consistently
    as leading "no"/"none" in this codebase's own tool-call data."""
    first_word = value.strip().lower().split(" ")[0].strip(".,!?") if value.strip() else ""
    return value.strip() == "" or first_word in _DENIAL_WORDS


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
        return fact is not None and fact.source in AFFIRMATIVE_SOURCES and not _looks_like_denial(fact.value)

    missing: list[MissingField] = []
    for pf in protocol.fields:
        required_now = pf.required or (pf.required_if is not None and is_affirmed(pf.required_if))
        if required_now and not is_covered(pf.field):
            missing.append(MissingField(field=pf.field, label=pf.label, category=pf.category))
    return missing


def is_record_complete(record: IntakeRecord, protocol: ProtocolConfig) -> bool:
    return len(get_missing_fields(record, protocol)) == 0
