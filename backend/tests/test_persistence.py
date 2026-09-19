"""Verifies a SessionState survives a full save/load round trip through a
real (temp-file) SQLite database — not mocked, the actual persistence path
main.py uses. Uses a fresh engine per test, pointed at a temp file, so tests
never touch the real app_data.db or each other."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db as db_module
import app.persistence as persistence
from app.agent.session import AssistanceRequest, create_session
from app.protocol.registry import get_protocol
from app.schemas.document import UploadedDocument
from app.schemas.intake_record import QuestionEvent, SafetyEvaluation, Source, TranscriptTurn
from app.state_engine import apply_fact, record_correction


@pytest.fixture
async def temp_db(tmp_path, monkeypatch):
    """Points the persistence layer's engine at a throwaway temp-file
    database for the duration of one test, then creates its tables."""
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(persistence, "get_session_factory", lambda: factory)
    async with engine.begin() as conn:
        await conn.run_sync(db_module.Base.metadata.create_all)
    yield
    await engine.dispose()


@pytest.mark.asyncio
async def test_round_trip_preserves_facts_transcript_and_protocol(temp_db):
    session = create_session(str(uuid.uuid4()), protocol=get_protocol("respiratory-intake"))
    session.record = apply_fact(session.record, "onset", "5 days ago", Source.PATIENT_REPORTED, "started 5 days ago", 0.9, [])
    session.record = record_correction(session.record, "onset", "1 week ago", "actually a week ago", 0.9)
    session.transcript.append(TranscriptTurn(id=str(uuid.uuid4()), speaker="patient", text="hello", timestamp="t1"))
    session.transcript.append(TranscriptTurn(id=str(uuid.uuid4()), speaker="agent", text="hi there", timestamp="t2"))
    session.turn_generation = 3

    await persistence.save_session(session)
    restored = await persistence.load_session(session.session_id)

    assert restored is not None
    assert restored.session_id == session.session_id
    assert restored.protocol.protocol_id == "respiratory-intake"
    assert restored.turn_generation == 3
    assert [t.text for t in restored.transcript] == ["hello", "hi there"]

    current = {f.field: f for f in restored.record.facts if not any(o.supersedes == f.id for o in restored.record.facts)}
    assert current["onset"].value == "1 week ago"
    assert len(restored.record.facts) == 2  # original + correction, both preserved


@pytest.mark.asyncio
async def test_round_trip_preserves_safety_log_questions_documents_and_assistance(temp_db):
    session = create_session(str(uuid.uuid4()), protocol=get_protocol("respiratory-intake"))
    session.question_events.append(QuestionEvent(id="q1", field="fever", question_text="Any fever?", timestamp="t"))
    session.safety_log.append(SafetyEvaluation(id="s1", fact_id="f1", triggered=True, rule_id="severe_breathing_difficulty", action="emergency_escalation", timestamp="t"))
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="albuterol", uploaded_at="t"))
    session.assistance_requests.append(AssistanceRequest(id="a1", reason="patient asked for a human", timestamp="t"))
    session.brief_finalized = True

    await persistence.save_session(session)
    restored = await persistence.load_session(session.session_id)

    assert restored.question_events[0].field == "fever"
    assert restored.safety_log[0].triggered is True
    assert restored.documents[0].filename == "rx.pdf"
    assert restored.assistance_requests[0].reason == "patient asked for a human"
    assert restored.brief_finalized is True


@pytest.mark.asyncio
async def test_load_unknown_session_returns_none(temp_db):
    assert await persistence.load_session("does-not-exist") is None


@pytest.mark.asyncio
async def test_resaving_a_session_replaces_child_rows_instead_of_duplicating(temp_db):
    """save_session gets called after every turn — the second save must not
    leave stale duplicate rows for facts/turns that already existed."""
    session = create_session(str(uuid.uuid4()), protocol=get_protocol("respiratory-intake"))
    session.transcript.append(TranscriptTurn(id=str(uuid.uuid4()), speaker="patient", text="turn one", timestamp="t1"))
    await persistence.save_session(session)

    session.transcript.append(TranscriptTurn(id=str(uuid.uuid4()), speaker="agent", text="turn two", timestamp="t2"))
    await persistence.save_session(session)

    restored = await persistence.load_session(session.session_id)
    assert [t.text for t in restored.transcript] == ["turn one", "turn two"]


@pytest.mark.asyncio
async def test_unclassified_session_has_no_protocol_after_reload(temp_db):
    session = create_session(str(uuid.uuid4()))  # no protocol yet
    await persistence.save_session(session)
    restored = await persistence.load_session(session.session_id)
    assert restored.protocol is None
