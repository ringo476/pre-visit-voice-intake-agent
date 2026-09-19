"""Durable storage for SessionState, backed by the database configured in
app/db.py. Deliberately a checkpoint pattern rather than a rewrite of every
call site that touches SessionState: the in-memory dataclass stays exactly
as it is everywhere else in the codebase (tools.py, state_engine.py,
turn_controller.py are all untouched), and main.py just calls save_session()
after each turn completes and load_session() when a connection asks to
resume an existing session_id. This keeps the change low-risk — the actual
business logic that reads and mutates SessionState never has to know the
database exists at all.

Child tables (facts, transcript turns, etc.) are fully replaced on every
save rather than incrementally diffed. That's the right tradeoff at this
project's scale: each turn adds at most a handful of rows, so a delete-then-
reinsert is cheap, and it's impossible for it to drift out of sync with the
in-memory state the way a hand-written incremental merge could."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, delete, select
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agent.session import AssistanceRequest, SessionState
from app.db import Base, get_session_factory
from app.protocol.registry import get_protocol
from app.schemas.document import UploadedDocument
from app.schemas.intake_record import Fact, FactStatus, IntakeRecord, QuestionEvent, SafetyEvaluation, Source, TranscriptTurn

UNCLASSIFIED_PROTOCOL_ID = "unclassified"


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    protocol_id: Mapped[str] = mapped_column(String, default=UNCLASSIFIED_PROTOCOL_ID)
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)
    brief_finalized: Mapped[bool] = mapped_column(Boolean, default=False)
    turn_generation: Mapped[int] = mapped_column(Integer, default=0)
    consent_given_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    appointment_reason_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    appointment_when_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class AccessLogRow(Base):
    """Who (or what) accessed a session's clinical data, and when — distinct
    from the clinical safety_log, which is about escalations, not access
    control. A compliance requirement, not just an engineering nicety: real
    healthcare systems must be able to answer "who looked at this patient's
    record, and when" after the fact."""

    __tablename__ = "access_log"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    action: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)


class FactRow(Base):
    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    field: Mapped[str] = mapped_column(String)
    value: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)
    evidence_span: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)
    supersedes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    question_event_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class TranscriptTurnRow(Base):
    __tablename__ = "transcript_turns"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    speaker: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[str] = mapped_column(String)


class SafetyEvaluationRow(Base):
    __tablename__ = "safety_evaluations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    fact_id: Mapped[str] = mapped_column(String)
    triggered: Mapped[bool] = mapped_column(Boolean)
    rule_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[str] = mapped_column(String)


class QuestionEventRow(Base):
    __tablename__ = "question_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    field: Mapped[str] = mapped_column(String)
    question_text: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[str] = mapped_column(String)


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    mime_type: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)
    uploaded_at: Mapped[str] = mapped_column(String)


class AssistanceRequestRow(Base):
    __tablename__ = "assistance_requests"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[str] = mapped_column(String)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def save_session(session: SessionState) -> None:
    protocol_id = session.protocol.protocol_id if session.protocol else UNCLASSIFIED_PROTOCOL_ID
    factory = get_session_factory()
    async with factory() as db:
        existing = await db.get(SessionRow, session.session_id)
        created_at = existing.created_at if existing else _now()
        row = SessionRow(
            session_id=session.session_id,
            protocol_id=protocol_id,
            created_at=created_at,
            updated_at=_now(),
            brief_finalized=session.brief_finalized,
            turn_generation=session.turn_generation,
            consent_given_at=session.consent_given_at,
            appointment_reason_text=session.appointment_reason_text,
            appointment_when_text=session.appointment_when_text,
        )
        await db.merge(row)

        for model, table in [
            (FactRow, FactRow),
            (TranscriptTurnRow, TranscriptTurnRow),
            (SafetyEvaluationRow, SafetyEvaluationRow),
            (QuestionEventRow, QuestionEventRow),
            (DocumentRow, DocumentRow),
            (AssistanceRequestRow, AssistanceRequestRow),
        ]:
            await db.execute(delete(table).where(table.session_id == session.session_id))

        db.add_all(
            FactRow(
                id=f.id,
                session_id=session.session_id,
                field=f.field,
                value=f.value,
                source=f.source.value,
                evidence_span=f.evidence_span,
                confidence=f.confidence,
                status=f.status.value,
                timestamp=f.timestamp,
                supersedes=f.supersedes,
                question_event_id=f.question_event_id,
            )
            for f in session.record.facts
        )
        db.add_all(
            TranscriptTurnRow(id=t.id, session_id=session.session_id, speaker=t.speaker, text=t.text, timestamp=t.timestamp)
            for t in session.transcript
        )
        db.add_all(
            SafetyEvaluationRow(
                id=e.id,
                session_id=session.session_id,
                fact_id=e.fact_id,
                triggered=e.triggered,
                rule_id=e.rule_id,
                action=e.action,
                timestamp=e.timestamp,
            )
            for e in session.safety_log
        )
        db.add_all(
            QuestionEventRow(id=q.id, session_id=session.session_id, field=q.field, question_text=q.question_text, timestamp=q.timestamp)
            for q in session.question_events
        )
        db.add_all(
            DocumentRow(
                id=d.id,
                session_id=session.session_id,
                filename=d.filename,
                mime_type=d.mime_type,
                text=d.text,
                uploaded_at=d.uploaded_at,
            )
            for d in session.documents
        )
        db.add_all(
            AssistanceRequestRow(id=a.id, session_id=session.session_id, reason=a.reason, timestamp=a.timestamp)
            for a in session.assistance_requests
        )
        await db.commit()


async def load_session(session_id: str) -> Optional[SessionState]:
    factory = get_session_factory()
    async with factory() as db:
        row = await db.get(SessionRow, session_id)
        if row is None:
            return None

        facts = (await db.execute(select(FactRow).where(FactRow.session_id == session_id).order_by(FactRow.timestamp))).scalars().all()
        turns = (
            (await db.execute(select(TranscriptTurnRow).where(TranscriptTurnRow.session_id == session_id).order_by(TranscriptTurnRow.timestamp)))
            .scalars()
            .all()
        )
        evaluations = (
            (await db.execute(select(SafetyEvaluationRow).where(SafetyEvaluationRow.session_id == session_id).order_by(SafetyEvaluationRow.timestamp)))
            .scalars()
            .all()
        )
        questions = (
            (await db.execute(select(QuestionEventRow).where(QuestionEventRow.session_id == session_id).order_by(QuestionEventRow.timestamp)))
            .scalars()
            .all()
        )
        docs = (await db.execute(select(DocumentRow).where(DocumentRow.session_id == session_id).order_by(DocumentRow.uploaded_at))).scalars().all()
        assists = (
            (await db.execute(select(AssistanceRequestRow).where(AssistanceRequestRow.session_id == session_id).order_by(AssistanceRequestRow.timestamp)))
            .scalars()
            .all()
        )

    protocol = get_protocol(row.protocol_id) if row.protocol_id != UNCLASSIFIED_PROTOCOL_ID else None

    record = IntakeRecord(
        session_id=session_id,
        protocol_id=row.protocol_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        facts=[
            Fact(
                id=f.id,
                field=f.field,
                value=f.value,
                source=Source(f.source),
                evidence_span=f.evidence_span,
                confidence=f.confidence,
                status=FactStatus(f.status),
                timestamp=f.timestamp,
                supersedes=f.supersedes,
                question_event_id=f.question_event_id,
            )
            for f in facts
        ],
    )

    return SessionState(
        session_id=session_id,
        record=record,
        protocol=protocol,
        question_events=[QuestionEvent(id=q.id, field=q.field, question_text=q.question_text, timestamp=q.timestamp) for q in questions],
        transcript=[TranscriptTurn(id=t.id, speaker=t.speaker, text=t.text, timestamp=t.timestamp) for t in turns],
        safety_log=[
            SafetyEvaluation(id=e.id, fact_id=e.fact_id, triggered=e.triggered, rule_id=e.rule_id, action=e.action, timestamp=e.timestamp)
            for e in evaluations
        ],
        assistance_requests=[AssistanceRequest(id=a.id, reason=a.reason, timestamp=a.timestamp) for a in assists],
        documents=[UploadedDocument(id=d.id, filename=d.filename, mime_type=d.mime_type, text=d.text, uploaded_at=d.uploaded_at) for d in docs],
        brief_finalized=row.brief_finalized,
        turn_generation=row.turn_generation,
        consent_given_at=row.consent_given_at,
        appointment_reason_text=row.appointment_reason_text,
        appointment_when_text=row.appointment_when_text,
    )


async def record_access(session_id: str, action: str) -> None:
    """Appends one entry to the access-audit log — never overwritten,
    never replaced on the next save_session (unlike facts/transcript,
    which are fully replaced each save, this is genuinely append-only,
    since an audit log that could be rewritten wouldn't be much of one)."""
    factory = get_session_factory()
    async with factory() as db:
        db.add(AccessLogRow(id=str(uuid.uuid4()), session_id=session_id, action=action, timestamp=_now()))
        await db.commit()


async def get_access_log(session_id: str) -> list[dict]:
    factory = get_session_factory()
    async with factory() as db:
        rows = (
            (await db.execute(select(AccessLogRow).where(AccessLogRow.session_id == session_id).order_by(AccessLogRow.timestamp)))
            .scalars()
            .all()
        )
    return [{"action": r.action, "timestamp": r.timestamp} for r in rows]


async def list_sessions(limit: int = 50) -> list[dict]:
    """Backs a simple clinician-facing lookup of past intakes — the feature
    that's only possible once sessions actually survive past one process's
    lifetime."""
    factory = get_session_factory()
    async with factory() as db:
        rows = (await db.execute(select(SessionRow).order_by(SessionRow.updated_at.desc()).limit(limit))).scalars().all()
    return [
        {
            "session_id": r.session_id,
            "protocol_id": r.protocol_id,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "brief_finalized": r.brief_finalized,
        }
        for r in rows
    ]
