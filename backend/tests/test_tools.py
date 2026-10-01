import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage

from app.agent.session import create_session
from app.agent.tools import create_tool_handlers
from app.schemas.document import UploadedDocument
from app.schemas.intake_record import QuestionEvent, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def _turn(i: int, speaker: str, text: str) -> TranscriptTurn:
    return TranscriptTurn(id=f"t{i}", speaker=speaker, text=text, timestamp=f"2026-01-01T00:00:{i:02d}")


def _session_asked_about(field: str, patient_reply: str, spoken: bool = True, verifier=None):
    """A session where the agent has (or, with spoken=False, has only
    logged but not said) a question about `field`, and the patient has
    since replied — the state update_intake_record sees on the turn where
    a denial is recorded."""
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "agent", f"Have you had any {field}?"))
    session.question_events.append(
        QuestionEvent(
            id="q1",
            field=field,
            question_text=field,
            timestamp="t",
            asked_in_turn=0 if spoken else None,
            spoken_text=f"Have you had any {field}?" if spoken else None,
        )
    )
    session.transcript.append(_turn(1, "patient", patient_reply))
    return session, create_tool_handlers(session, verifier_llm=verifier)


def _uncertain(field: str, evidence: str, **extra) -> dict:
    return {"field": field, "value": "patient does not know", "source": "uncertain", "evidence": evidence, "confidence": 0.8, **extra}


def _denial(field: str, evidence: str, **extra) -> dict:
    return {"field": field, "value": "false", "source": "asked_and_denied", "evidence": evidence, "confidence": 0.9, **extra}


class FakeRankingLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, messages):
        return AIMessage(content=self.reply)


def test_get_next_intake_question_honors_ranking_llm_when_given():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session, llm=FakeRankingLLM("chief_complaint"))
    # respiratory_intake.json's fixed order starts with chief_complaint
    # anyway, so pick a field that is NOT first in that order to prove the
    # ranking result — not the default — is what actually got surfaced.
    result = handlers["get_next_intake_question"]({})
    first_missing_field = result.data["missing_fields"][0]["field"]
    handlers2 = create_tool_handlers(create_session("s2", PROTOCOL), llm=FakeRankingLLM("fever"))
    result2 = handlers2["get_next_intake_question"]({})
    assert result2.data["suggested"]["field"] == "fever"
    assert first_missing_field != "fever"  # sanity: fever isn't first in fixed order


def test_get_next_intake_question_falls_back_when_ranking_llm_hallucinates():
    session = create_session("s1", PROTOCOL)
    handlers = create_tool_handlers(session, llm=FakeRankingLLM("not_a_real_field"))
    result = handlers["get_next_intake_question"]({})
    # Falls back to the deterministic fixed order (first missing field).
    assert result.data["suggested"]["field"] == result.data["missing_fields"][0]["field"]


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
    session.transcript.append(_turn(0, "patient", "No fever"))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("fever", "No fever"))
    assert result.ok is False
    assert "no question about it has been asked" in result.error
    assert len(session.record.facts) == 0


def test_denial_accepted_when_question_was_spoken_and_patient_answered_after():
    session, handlers = _session_asked_about("fever", "No, no fever at all.")

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "asked_and_denied"
    assert fact.question_event_id == "q1"  # resolved by the server, not supplied by the model


def test_denial_rejected_when_question_was_logged_but_never_spoken():
    """get_next_intake_question logs an event the moment the tool runs. If
    the turn never produced a reply that asked it, that event must not be
    usable to back a denial."""
    session, handlers = _session_asked_about("fever", "No, no fever at all.", spoken=False)

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is False
    assert "no question about it has been asked" in result.error
    assert session.record.facts == []


def test_denial_rejected_when_quote_is_not_in_the_patients_reply():
    session, handlers = _session_asked_about("fever", "Hmm, I'm not sure about that.")

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is False
    assert "evidence quote was not found" in result.error
    assert session.record.facts == []


def test_denial_rejected_when_quote_only_appears_before_the_question_was_asked():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "No fever."))  # said earlier, in answer to nothing
    session.transcript.append(_turn(1, "agent", "Have you had any fever?"))
    session.question_events.append(
        QuestionEvent(id="q1", field="fever", question_text="fever", timestamp="t", asked_in_turn=1, spoken_text="Have you had any fever?")
    )
    session.transcript.append(_turn(2, "patient", "Let me think about that."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("fever", "No fever"))

    assert result.ok is False


def test_denial_ignores_a_question_event_id_supplied_by_the_model():
    """An id the model makes up (or carries over wrongly) must not matter:
    the server resolves the question itself."""
    session, handlers = _session_asked_about("fever", "No fever.")

    result = handlers["update_intake_record"](_denial("fever", "No fever", question_event_id="made-up-id"))

    assert result.ok is True
    assert session.record.facts[-1].question_event_id == "q1"


def test_denial_cannot_borrow_a_question_asked_about_a_different_field():
    session, handlers = _session_asked_about("fever", "No, none.")

    result = handlers["update_intake_record"](_denial("wheezing", "No, none"))

    assert result.ok is False


def test_patient_volunteered_facts_need_no_question_event():
    """The patient answers the smoking question and also volunteers other
    facts in the same breath — those are patient_reported and are recorded
    without any logged question."""
    session, handlers = _session_asked_about(
        "smoking_history", "No, I don't smoke. I've had a fever and a cough for a week, and I have asthma."
    )

    denial = handlers["update_intake_record"](_denial("smoking_history", "No, I don't smoke"))
    fever = handlers["update_intake_record"](
        {"field": "fever", "value": "yes", "source": "patient_reported", "evidence": "I've had a fever", "confidence": 0.9}
    )
    asthma = handlers["update_intake_record"](
        {"field": "respiratory_history", "value": "asthma", "source": "patient_reported", "evidence": "I have asthma", "confidence": 0.9}
    )

    assert denial.ok and fever.ok and asthma.ok
    sources = {f.field: f.source.value for f in session.record.facts}
    assert sources == {"smoking_history": "asked_and_denied", "fever": "patient_reported", "respiratory_history": "patient_reported"}


def test_uncertain_accepted_when_question_was_spoken_and_patient_said_they_dont_know():
    session, handlers = _session_asked_about("fever", "Honestly I don't remember if I had one.")

    result = handlers["update_intake_record"](_uncertain("fever", "I don't remember if I had one"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "uncertain"
    assert fact.question_event_id == "q1"


def test_uncertain_rejected_when_no_question_was_asked():
    """A model must not be able to close a required field as 'unsure' without
    ever asking the patient — that would skip the question entirely."""
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I don't remember."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_uncertain("fever", "I don't remember"))

    assert result.ok is False
    assert session.record.facts == []


def test_uncertain_rejected_when_quote_is_not_in_the_patients_reply():
    session, handlers = _session_asked_about("fever", "Yes, quite high.")

    result = handlers["update_intake_record"](_uncertain("fever", "I don't remember"))

    assert result.ok is False
    assert "evidence quote was not found" in result.error


def test_uncertain_requires_an_evidence_quote():
    session, handlers = _session_asked_about("fever", "I don't remember.")

    result = handlers["update_intake_record"]({"field": "fever", "value": "unsure", "source": "uncertain", "confidence": 0.8})

    assert result.ok is False
    assert "evidence" in result.error


class _FakeVerifier:
    """Stands in for the separate verifier model. Records what it was asked."""

    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        if self.error:
            raise self.error
        return AIMessage(content=self.verdict)


def test_verifier_blocks_a_yes_recorded_as_no_even_with_a_real_quote():
    """The hole the text check alone cannot close: the quote IS in the
    transcript, but it is a 'yes', not a denial."""
    verifier = _FakeVerifier("OTHER")
    session, handlers = _session_asked_about("fever", "Yes, I felt feverish on Tuesday.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "Yes, I felt feverish on Tuesday"))

    assert result.ok is False
    assert "independent reading" in result.error
    assert "neither a clear" in result.error
    assert session.record.facts == []


def test_verifier_accepts_a_genuine_denial():
    verifier = _FakeVerifier("NEGATIVE")
    session, handlers = _session_asked_about("fever", "No, no fever at all.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "asked_and_denied"
    assert len(verifier.prompts) == 1


def test_verifier_tells_the_model_to_use_uncertain_when_the_reply_was_an_i_dont_know():
    verifier = _FakeVerifier("UNSURE")
    session, handlers = _session_asked_about("fever", "I'm not sure, maybe.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "I'm not sure, maybe"))

    assert result.ok is False
    assert "record it as uncertain" in result.error
    assert session.record.facts == []


def test_verifier_tells_the_model_to_use_asked_and_denied_when_an_unsure_label_was_a_clear_no():
    verifier = _FakeVerifier("NEGATIVE")
    session, handlers = _session_asked_about("fever", "No, definitely not.", verifier=verifier)

    result = handlers["update_intake_record"](_uncertain("fever", "No, definitely not"))

    assert result.ok is False
    assert "record it as asked_and_denied" in result.error


def test_the_model_can_correct_itself_after_a_rejection():
    """The rejection is the correction path: the model retries with the label
    the verifier described, and that retry passes."""
    session, handlers = _session_asked_about("fever", "I'm not sure, maybe.", verifier=_FakeVerifier("UNSURE"))

    first = handlers["update_intake_record"](_denial("fever", "I'm not sure, maybe"))
    retry = handlers["update_intake_record"](_uncertain("fever", "I'm not sure, maybe"))

    assert first.ok is False
    assert retry.ok is True
    assert session.record.facts[-1].source.value == "uncertain"


def test_verifier_failure_refuses_the_save_and_leaves_the_field_open(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda s: None)
    verifier = _FakeVerifier(error=RuntimeError("503 unavailable"))
    session, handlers = _session_asked_about("fever", "No, no fever at all.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is False
    assert "Could not verify" in result.error
    assert session.record.facts == []


def test_verifier_gibberish_refuses_the_save():
    session, handlers = _session_asked_about("fever", "No fever.", verifier=_FakeVerifier("probably a no"))

    result = handlers["update_intake_record"](_denial("fever", "No fever"))

    assert result.ok is False
    assert session.record.facts == []


def test_verifier_is_not_called_for_facts_the_patient_volunteered():
    verifier = _FakeVerifier("NEGATIVE")
    session, handlers = _session_asked_about("fever", "I had a fever and a cough.", verifier=verifier)

    result = handlers["update_intake_record"](
        {"field": "fever", "value": "yes", "source": "patient_reported", "evidence": "I had a fever", "confidence": 0.9}
    )

    assert result.ok is True
    assert verifier.prompts == []


def test_verifier_is_not_called_when_the_cheaper_checks_already_failed():
    verifier = _FakeVerifier("NEGATIVE")
    session, handlers = _session_asked_about("fever", "Hmm, not sure.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "an invented quote"))

    assert result.ok is False
    assert verifier.prompts == []  # rejected by the quote check, no model cost


def test_verifier_sees_the_topic_the_spoken_question_and_the_patients_reply():
    verifier = _FakeVerifier("NEGATIVE")
    session, handlers = _session_asked_about("fever", "No, no fever at all.", verifier=verifier)

    handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    prompt = verifier.prompts[0]
    assert "Fever or chills" in prompt  # the checklist label for the field
    assert "Have you had any fever?" in prompt  # what Ava actually said
    assert "No, no fever at all." in prompt  # what the patient replied


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
    assert "question_event_id" not in result.data["suggested"]  # the model never carries ids
    assert session.question_events[0].asked_in_turn is None  # logged, but not spoken yet


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
