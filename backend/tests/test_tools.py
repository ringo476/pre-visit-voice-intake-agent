import json
import os
from pathlib import Path

from app.agent.session import create_session
from app.agent.tools import create_tool_handlers
from app.schemas.document import UploadedDocument
from app.schemas.protocol_config import ProtocolConfig

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def test_update_intake_record_records_patient_reported_fact_no_safety_trigger():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "onset", "value": "approximately 10 days ago", "source": "patient_reported", "evidence": "It started last Monday", "confidence": 0.9}
    )
    assert result.ok is True
    assert len(session.record.facts) == 1
    assert result.data["safety"] is None


def test_update_intake_record_rejects_patient_reported_without_evidence():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "onset", "value": "10 days ago", "source": "patient_reported", "confidence": 0.9}
    )
    assert result.ok is False
    assert len(session.record.facts) == 0


def test_update_intake_record_rejects_denied_without_prior_question():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "fever", "value": "false", "source": "asked_and_denied", "evidence": "No fever", "confidence": 0.9}
    )
    assert result.ok is False
    assert "question_event_id" in result.error


def test_update_intake_record_accepts_denied_once_question_logged():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    suggested = None
    for _ in range(len(PROTOCOL.fields)):
        next_q = handlers["get_next_intake_question"]({})
        if next_q.data.get("suggested", {}).get("field") == "fever":
            suggested = next_q.data["suggested"]
            break
        if next_q.data.get("suggested"):
            handlers["update_intake_record"](
                {"field": next_q.data["suggested"]["field"], "value": "n/a", "source": "patient_reported", "evidence": "n/a", "confidence": 0.5}
            )
    assert suggested is not None

    result = handlers["update_intake_record"](
        {
            "field": "fever",
            "value": "false",
            "source": "asked_and_denied",
            "evidence": "No fever",
            "confidence": 0.9,
            "question_event_id": suggested["question_event_id"],
        }
    )
    assert result.ok is True


def test_update_intake_record_surfaces_safety_trigger():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {
            "field": "breathing_difficulty",
            "value": "severe, can't breathe",
            "source": "patient_reported",
            "evidence": "I honestly can't breathe properly right now",
            "confidence": 0.9,
        }
    )
    assert result.ok is True
    assert result.data["safety"]["triggered"] is True
    assert any(e.triggered for e in session.safety_log)


def test_record_patient_correction_supersedes_original():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    first = handlers["update_intake_record"](
        {"field": "onset", "value": "10 days ago", "source": "patient_reported", "evidence": "It started last Monday", "confidence": 0.9}
    )
    fact_id = first.data["fact_id"]

    corrected = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "2 weeks ago", "evidence": "Actually, two weeks ago", "confidence": 0.92}
    )
    assert corrected.ok is True
    assert len(session.record.facts) == 2
    assert session.record.facts[1].supersedes == fact_id


def test_record_patient_correction_works_without_the_model_ever_seeing_a_fact_id():
    """The model only ever needs the field name — the prior fact is looked
    up server-side, since an id from an earlier turn's tool result isn't
    visible to the model in a later turn (each turn's message list is
    rebuilt from session.transcript alone, not past tool-call history)."""
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    handlers["update_intake_record"](
        {"field": "onset", "value": "10 days ago", "source": "patient_reported", "evidence": "It started last Monday", "confidence": 0.9}
    )

    corrected = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "2 weeks ago", "evidence": "Actually, two weeks ago", "confidence": 0.92}
    )
    assert corrected.ok is True
    current = next(f for f in session.record.facts if f.status.value == "corrected")
    assert current.value == "2 weeks ago"


def test_record_patient_correction_rejects_field_never_recorded():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    result = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "2 weeks ago", "evidence": "it was two weeks ago", "confidence": 0.9}
    )
    assert result.ok is False


def test_check_safety_protocol_triggers_on_statement():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["check_safety_protocol"]({"statement": "I passed out yesterday"})
    assert result.data["triggered"] is True


def test_check_safety_protocol_scans_existing_facts_when_no_statement():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"](
        {"field": "chest_discomfort", "value": "crushing chest pain", "source": "patient_reported", "evidence": "It's a crushing pressure in my chest", "confidence": 0.9}
    )
    result = handlers["check_safety_protocol"]({})
    assert result.data["triggered"] is True


def test_get_next_intake_question_suggests_field_and_logs_question_event():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["get_next_intake_question"]({})
    assert result.data["done"] is False
    assert result.data["suggested"]["field"]
    assert len(session.question_events) == 1
    assert session.question_events[0].id == result.data["suggested"]["question_event_id"]


def test_get_next_intake_question_reports_done_once_complete():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)

    for _ in range(len(PROTOCOL.fields) + 5):
        next_q = handlers["get_next_intake_question"]({})
        if next_q.data["done"]:
            break
        handlers["update_intake_record"](
            {"field": next_q.data["suggested"]["field"], "value": "n/a", "source": "patient_reported", "evidence": "n/a", "confidence": 0.5}
        )

    final = handlers["get_next_intake_question"]({})
    assert final.data["done"] is True


def test_retrieve_existing_patient_context_returns_relevant_snippet():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["retrieve_existing_patient_context"]({"query": "prior inhaler prescription"})
    assert result.ok is True
    assert len(result.data["results"]) > 0
    assert "inhaler" in result.data["results"][0]["text"].lower()


def test_retrieve_uploaded_document_no_documents():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["retrieve_uploaded_document"]({"query": "medication dose"})
    assert result.ok is True
    assert result.data["results"] == []


def test_retrieve_uploaded_document_with_seeded_document():
    session = create_session("s1", PROTOCOL)
    session.documents.append(
        UploadedDocument(id="doc1", filename="prescription.pdf", mime_type="application/pdf", text="Amoxicillin 500mg twice daily", uploaded_at="t")
    )
    from app.documents.document_store import add_document

    add_document(session.session_id, session.documents[0])

    handlers = create_tool_handlers(session)
    result = handlers["retrieve_uploaded_document"]({"query": "Amoxicillin dose"})
    assert result.ok is True
    assert result.data["results"][0]["filename"] == "prescription.pdf"


def test_generate_clinician_brief_refuses_when_incomplete():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["generate_clinician_brief"]({})
    assert result.ok is False
    assert session.brief_finalized is False


def test_generate_clinician_brief_allows_early_termination():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["generate_clinician_brief"]({"early_termination_reason": "Patient had to leave"})
    assert result.ok is True
    assert session.brief_finalized is True


def test_request_human_assistance_logs_without_touching_record():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session)
    result = handlers["request_human_assistance"]({"reason": "Patient wants to speak to a nurse"})
    assert result.ok is True
    assert len(session.assistance_requests) == 1
    assert len(session.record.facts) == 0
