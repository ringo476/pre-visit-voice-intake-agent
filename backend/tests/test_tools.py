import json
import re
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
    return {"field": field, "polarity": "unknown", "evidence": evidence, "confidence": 0.8, **extra}


def _denial(field: str, evidence: str, **extra) -> dict:
    return {"field": field, "polarity": "absent", "evidence": evidence, "confidence": 0.9, **extra}


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
    session.transcript.append(_turn(0, "patient", "It started last Monday."))
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


def test_denial_accepted_when_question_was_spoken_and_patient_answered_after():
    session, handlers = _session_asked_about("fever", "No, no fever at all.")

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "asked_and_denied"
    assert fact.question_event_id == "q1"  # resolved by the server, not supplied by the model


def test_denial_rejected_when_quote_is_not_in_the_patients_reply():
    session, handlers = _session_asked_about("fever", "Hmm, I'm not sure about that.")

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is False
    assert "evidence quote was not found" in result.error
    assert session.record.facts == []


def test_denial_ignores_a_question_event_id_supplied_by_the_model():
    """An id the model makes up (or carries over wrongly) must not matter:
    the server resolves the question itself."""
    session, handlers = _session_asked_about("fever", "No fever.")

    result = handlers["update_intake_record"](_denial("fever", "No fever", question_event_id="made-up-id"))

    assert result.ok is True
    assert session.record.facts[-1].question_event_id == "q1"


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
    """Stands in for the separate verifier model. Records what it was asked, and
    answers every claim in the prompt with the given verdict (one 'N: WORD' line
    per claim). Anything that is not a verdict word is returned untouched, to
    exercise the unusable-answer path."""

    def __init__(self, verdict=None, error=None):
        self.verdict = verdict
        self.error = error
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if self.verdict in ("SUPPORTED", "CONTRADICTED", "UNRELATED"):
            n = len(re.findall(r"(?m)^Claim \d+$", prompt))
            return AIMessage(content="\n".join(f"{i}: {self.verdict}" for i in range(1, n + 1)))
        return AIMessage(content=self.verdict)


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
    session.transcript.append(_turn(0, "patient", "I honestly can't breathe properly right now."))
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
    session.transcript.append(_turn(0, "patient", "It started last Monday. Actually, two weeks ago."))
    handlers = create_tool_handlers(session)

    first = handlers["update_intake_record"](
        {"field": "onset", "value": "last Monday", "source": "patient_reported", "evidence": "It started last Monday", "confidence": 0.9}
    )
    fact_id = first.data["fact_id"]

    # Adding a detail to the same answer replaces it without a question. (A value that
    # disagrees is a different case; see test_conflicts.py.)
    corrected = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "last Monday, about two weeks ago", "evidence": "Actually, two weeks ago", "confidence": 0.92}
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
    session.transcript.append(_turn(0, "patient", "It started last Monday. Actually, two weeks ago."))
    handlers = create_tool_handlers(session)

    handlers["update_intake_record"](
        {"field": "onset", "value": "last Monday", "source": "patient_reported", "evidence": "It started last Monday", "confidence": 0.9}
    )

    corrected = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "last Monday, about two weeks ago", "evidence": "Actually, two weeks ago", "confidence": 0.92}
    )
    assert corrected.ok is True
    current = next(f for f in session.record.facts if f.status.value == "corrected")
    assert current.value == "last Monday, about two weeks ago"


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
    session.transcript.append(_turn(0, "patient", "It's a crushing pressure in my chest."))
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
    session.transcript.append(_turn(0, "patient", "n/a"))
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


# ---------------------------------------------------------------------------
# Every label is checked against the origin it claims
# ---------------------------------------------------------------------------

def _patient_reported(field: str, value: str, evidence: str) -> dict:
    return {"field": field, "value": value, "source": "patient_reported", "evidence": evidence, "confidence": 0.9}


def test_patient_reported_rejected_when_the_quote_is_not_something_the_patient_said():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I have had a cough for a while."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_patient_reported("medication_allergies", "none", "I have no allergies"))

    assert result.ok is False
    assert "not found in anything the patient has said" in result.error
    assert session.record.facts == []


def test_patient_reported_accepted_when_the_quote_is_in_the_patients_words():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I'm allergic to penicillin, it gives me hives."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_patient_reported("medication_allergies", "penicillin", "allergic to penicillin"))

    assert result.ok is True


def test_a_quote_the_agent_said_does_not_count_as_something_the_patient_said():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "agent", "Do you have any penicillin allergy?"))
    session.transcript.append(_turn(1, "patient", "Hmm."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_patient_reported("medication_allergies", "penicillin", "penicillin allergy"))

    assert result.ok is False


def test_document_sourced_accepted_when_the_quote_is_in_an_uploaded_document():
    session = create_session("s1", PROTOCOL)
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="Albuterol inhaler 90mcg, 2 puffs as needed", uploaded_at="t"))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "medications_tried", "value": "albuterol", "source": "document_sourced", "evidence": "Albuterol inhaler 90mcg", "confidence": 0.9}
    )

    assert result.ok is True


def test_inferred_chief_complaint_needs_a_booking_reason_to_exist():
    session = create_session("s1", PROTOCOL)  # no booking reason on record
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "chief_complaint", "value": "cough", "source": "inferred", "confidence": 0.5}
    )

    assert result.ok is False


def test_inferred_chief_complaint_is_accepted_when_it_comes_from_the_booking():
    session = create_session("s1", PROTOCOL, appointment_reason_text="a persistent cough")
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](
        {"field": "chief_complaint", "value": "persistent cough", "source": "inferred", "confidence": 0.9}
    )

    assert result.ok is True
    assert session.record.facts[-1].source.value == "inferred"


def test_a_correction_must_quote_something_the_patient_really_said():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "It started last Monday."))
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"](_patient_reported("onset", "10 days ago", "It started last Monday"))

    result = handlers["record_patient_correction"](
        {"field": "onset", "new_value": "2 weeks ago", "evidence": "actually it was two weeks ago", "confidence": 0.9}
    )

    assert result.ok is False
    assert "not found in anything the patient has said" in result.error


# ---------------------------------------------------------------------------
# The model proposes a fact; the SERVER works out the label it earns
# ---------------------------------------------------------------------------

def _present(field: str, value: str, evidence: str, **extra) -> dict:
    return {"field": field, "polarity": "present", "value": value, "evidence": evidence, "confidence": 0.9, **extra}


def test_a_no_after_a_spoken_question_earns_asked_and_denied():
    session, handlers = _session_asked_about("fever", "No, no fever at all.")

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "asked_and_denied"
    assert fact.polarity.value == "absent"
    assert fact.question_event_id == "q1"
    assert result.data["source"] == "asked_and_denied"


def test_a_dont_know_after_a_spoken_question_earns_uncertain():
    session, handlers = _session_asked_about("fever", "Honestly I don't remember if I had one.")

    result = handlers["update_intake_record"](_uncertain("fever", "I don't remember if I had one"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "uncertain"
    assert fact.polarity.value == "unknown"
    assert fact.question_event_id == "q1"


def test_a_no_nobody_asked_about_is_saved_honestly_as_patient_reported():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "There's no fever, but I've been wheezing."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("fever", "There's no fever"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "patient_reported"  # volunteered, so never labelled asked_and_denied
    assert fact.polarity.value == "absent"
    assert fact.question_event_id is None


def test_a_question_that_was_logged_but_never_spoken_does_not_earn_asked_and_denied():
    session, handlers = _session_asked_about("fever", "No, no fever at all.", spoken=False)

    result = handlers["update_intake_record"](_denial("fever", "No, no fever at all"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "patient_reported"


def test_a_quote_said_before_the_question_does_not_earn_asked_and_denied():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "No fever."))  # said earlier, in answer to nothing
    session.transcript.append(_turn(1, "agent", "Have you had any fever?"))
    session.question_events.append(
        QuestionEvent(id="q1", field="fever", question_text="fever", timestamp="t", asked_in_turn=1, spoken_text="Have you had any fever?")
    )
    session.transcript.append(_turn(2, "patient", "Let me think about that."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("fever", "No fever"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "patient_reported"


def test_a_question_about_another_field_does_not_earn_asked_and_denied():
    session, handlers = _session_asked_about("fever", "No, none.")

    result = handlers["update_intake_record"](_denial("wheezing", "No, none"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "patient_reported"


def test_the_model_cannot_choose_the_label():
    """Claiming asked_and_denied changes nothing: the label is worked out from
    what actually happened, and nobody asked."""
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "No fever."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("fever", "No fever", source="asked_and_denied", question_event_id="made-up-id"))

    assert result.ok is True
    fact = session.record.facts[-1]
    assert fact.source.value == "patient_reported"
    assert fact.question_event_id is None


def test_a_present_answer_to_a_spoken_question_is_patient_reported():
    session, handlers = _session_asked_about("fever", "Yes, I felt feverish on Tuesday.")

    result = handlers["update_intake_record"](_present("fever", "felt feverish on Tuesday", "I felt feverish on Tuesday"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "patient_reported"
    assert session.record.facts[-1].polarity.value == "present"


def test_a_quote_found_nowhere_is_rejected():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I have had a cough for a while."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_present("medication_allergies", "penicillin", "I am allergic to penicillin"))

    assert result.ok is False
    assert "not found in anything the patient has said" in result.error
    assert session.record.facts == []


def test_a_present_value_that_reads_as_a_no_is_rejected_as_inconsistent():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "No, I don't smoke."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_present("smoking_history", "none", "I don't smoke"))

    assert result.ok is False
    assert 'polarity is "present"' in result.error


def test_polarity_left_out_is_derived_and_the_claim_is_still_checked():
    verifier = _FakeVerifier("CONTRADICTED")
    session, handlers = _session_asked_about("fever", "Yes, I felt feverish on Tuesday.", verifier=verifier)

    # a call with no polarity and a value that reads as a no (a model that leaves the field out)
    result = handlers["update_intake_record"]({"field": "fever", "value": "no", "evidence": "I felt feverish on Tuesday", "confidence": 0.9})

    assert result.ok is False
    assert len(verifier.prompts) == 1


def test_the_visit_reason_from_the_booking_may_be_recorded_without_a_quote_as_inferred():
    session = create_session("s1", PROTOCOL, appointment_reason_text="a persistent cough")
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"]({"field": "chief_complaint", "polarity": "present", "value": "persistent cough", "confidence": 0.9})

    assert result.ok is True
    assert session.record.facts[-1].source.value == "inferred"


def test_nothing_else_may_be_recorded_without_a_quote():
    session = create_session("s1", PROTOCOL, appointment_reason_text="a persistent cough")
    handlers = create_tool_handlers(session)

    other_field = handlers["update_intake_record"]({"field": "medication_allergies", "polarity": "absent", "confidence": 0.5})
    no_booking = create_tool_handlers(create_session("s2", PROTOCOL))["update_intake_record"](
        {"field": "chief_complaint", "polarity": "present", "value": "cough", "confidence": 0.9}
    )

    assert other_field.ok is False and "evidence quote" in other_field.error
    assert no_booking.ok is False


def test_a_quote_found_in_an_uploaded_document_earns_document_sourced():
    session = create_session("s1", PROTOCOL)
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="Albuterol inhaler 90mcg, 2 puffs as needed", uploaded_at="t"))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_present("medications_tried", "albuterol", "Albuterol inhaler 90mcg"))

    assert result.ok is True
    assert session.record.facts[-1].source.value == "document_sourced"


def test_a_document_quote_with_no_such_text_in_any_document_is_rejected():
    session = create_session("s1", PROTOCOL)
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="Albuterol inhaler 90mcg", uploaded_at="t"))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_denial("medication_allergies", "Allergies: none"))

    assert result.ok is False
    assert "any uploaded document" in result.error


def test_a_quote_the_patient_said_wins_over_a_document():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I take albuterol inhaler 90mcg when needed."))
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="Albuterol inhaler 90mcg", uploaded_at="t"))
    handlers = create_tool_handlers(session)

    handlers["update_intake_record"](_present("medications_tried", "albuterol", "albuterol inhaler 90mcg"))

    assert session.record.facts[-1].source.value == "patient_reported"


# ---------------------------------------------------------------------------
# Every fact is verified by the separate model, whatever its direction
# ---------------------------------------------------------------------------

def _with_patient_words(text: str, verifier):
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", text))
    return session, create_tool_handlers(session, verifier_llm=verifier)


def test_a_yes_is_verified_too_not_only_a_no():
    verifier = _FakeVerifier("SUPPORTED")
    session, handlers = _with_patient_words("It started about two weeks ago.", verifier)

    result = handlers["update_intake_record"](_present("onset", "two weeks ago", "started about two weeks ago"))

    assert result.ok is True
    assert len(verifier.prompts) == 1


def test_a_real_but_unrelated_quote_filed_under_another_field_is_rejected():
    """The gap this closes: the quote is genuinely something the patient said,
    but it is not about allergies."""
    verifier = _FakeVerifier("UNRELATED")
    session, handlers = _with_patient_words("I've had a cough for a while now.", verifier)

    result = handlers["update_intake_record"](_present("medication_allergies", "penicillin", "I've had a cough for a while now"))

    assert result.ok is False
    assert "not about this topic" in result.error
    assert session.record.facts == []


def test_a_yes_filed_as_a_no_with_a_real_quote_is_rejected():
    verifier = _FakeVerifier("CONTRADICTED")
    session, handlers = _session_asked_about("fever", "Yes, I felt feverish on Tuesday.", verifier=verifier)

    result = handlers["update_intake_record"](_denial("fever", "Yes, I felt feverish on Tuesday"))

    assert result.ok is False
    assert "independent reading" in result.error
    assert session.record.facts == []


def test_a_no_phrased_without_any_no_word_is_still_verified():
    """The old first-word guess let 'not at all' skip the check. Polarity is stated
    outright, so every claim is checked whatever words the value uses."""
    verifier = _FakeVerifier("CONTRADICTED")
    session, handlers = _session_asked_about("fever", "Yes, I felt feverish on Tuesday.", verifier=verifier)

    result = handlers["update_intake_record"]({"field": "fever", "polarity": "absent", "value": "not at all", "evidence": "I felt feverish on Tuesday", "confidence": 0.9})

    assert result.ok is False


def test_the_model_can_correct_itself_after_a_rejection():
    class Sequenced:
        """First reading: the claim is wrong. Second reading, after the model retries: fine."""

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="1: CONTRADICTED" if self.calls == 1 else "1: SUPPORTED")

    session, handlers = _session_asked_about("fever", "I'm not sure, maybe.", verifier=Sequenced())

    first = handlers["update_intake_record"](_denial("fever", "I'm not sure, maybe"))
    retry = handlers["update_intake_record"](_uncertain("fever", "I'm not sure, maybe"))

    assert first.ok is False
    assert retry.ok is True
    assert session.record.facts[-1].source.value == "uncertain"


def test_the_visit_reason_from_the_booking_is_not_sent_to_the_verifier():
    verifier = _FakeVerifier("SUPPORTED")
    session = create_session("s1", PROTOCOL, appointment_reason_text="a persistent cough")
    handlers = create_tool_handlers(session, verifier_llm=verifier)

    handlers["update_intake_record"]({"field": "chief_complaint", "polarity": "present", "value": "persistent cough", "confidence": 0.9})

    assert verifier.prompts == []


def test_a_volunteered_claim_tells_the_verifier_nothing_was_asked():
    verifier = _FakeVerifier("SUPPORTED")
    session, handlers = _with_patient_words("No, I don't smoke.", verifier)

    handlers["update_intake_record"](_denial("smoking_history", "No, I don't smoke"))

    assert "Nothing was asked about this" in verifier.prompts[0]


def test_all_the_facts_in_one_message_are_checked_with_a_single_verifier_call():
    verifier = _FakeVerifier("SUPPORTED")
    session, handlers = _with_patient_words("About two weeks ago. There's no fever, but I've been wheezing, and I have asthma.", verifier)
    calls = [
        {"name": "update_intake_record", "args": _present("onset", "two weeks", "About two weeks ago"), "id": "1"},
        {"name": "update_intake_record", "args": _denial("fever", "There's no fever"), "id": "2"},
        {"name": "update_intake_record", "args": _present("wheezing", "yes", "I've been wheezing"), "id": "3"},
        {"name": "update_intake_record", "args": _present("respiratory_history", "asthma", "I have asthma"), "id": "4"},
    ]

    handlers.prepare(calls)
    results = [handlers[c["name"]](c["args"]) for c in calls]

    assert all(r.ok for r in results)
    assert len(verifier.prompts) == 1  # one call for all four facts
    assert verifier.prompts[0].count("Claim ") == 4
    assert len(session.record.facts) == 4


def test_one_unsupported_fact_in_a_batch_is_rejected_and_the_others_still_save():
    class PerClaim:
        def invoke(self, messages):
            return AIMessage(content="1: SUPPORTED\n2: UNRELATED")

    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "About two weeks ago. I've had a cough."))
    handlers = create_tool_handlers(session, verifier_llm=PerClaim())
    calls = [
        {"name": "update_intake_record", "args": _present("onset", "two weeks", "About two weeks ago"), "id": "1"},
        {"name": "update_intake_record", "args": _present("medication_allergies", "penicillin", "I've had a cough"), "id": "2"},
    ]

    handlers.prepare(calls)
    results = [handlers[c["name"]](c["args"]) for c in calls]

    assert [r.ok for r in results] == [True, False]
    assert [f.field for f in session.record.facts] == ["onset"]


def test_a_failed_batch_check_refuses_every_fact_in_it(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda s: None)
    verifier = _FakeVerifier(error=RuntimeError("503 unavailable"))
    session, handlers = _with_patient_words("About two weeks ago, and no fever.", verifier)
    calls = [
        {"name": "update_intake_record", "args": _present("onset", "two weeks", "About two weeks ago"), "id": "1"},
        {"name": "update_intake_record", "args": _denial("fever", "no fever"), "id": "2"},
    ]

    handlers.prepare(calls)
    results = [handlers[c["name"]](c["args"]) for c in calls]

    assert [r.ok for r in results] == [False, False]
    assert all("Could not verify" in r.error for r in results)
    assert session.record.facts == []
    assert len(verifier.prompts) == 2  # tried, then retried once, then gave up: no per-fact retries after that


def test_a_verdict_is_not_reused_for_a_second_identical_request():
    verifier = _FakeVerifier("UNRELATED")  # refused each time, so the second request is a real second request
    session, handlers = _with_patient_words("About two weeks ago.", verifier)
    call = {"name": "update_intake_record", "args": _present("onset", "two weeks", "About two weeks ago"), "id": "1"}

    handlers.prepare([call])
    first = handlers["update_intake_record"](call["args"])
    second = handlers["update_intake_record"](call["args"])  # the same request again, outside any prepare

    assert first.ok is False and second.ok is False
    assert len(verifier.prompts) == 2  # one batched, one fresh


# ---------------------------------------------------------------------------
# Corrections follow the same rules
# (a correction that disagrees with the record is covered in test_conflicts.py)
# ---------------------------------------------------------------------------


def test_a_small_slip_in_copying_the_quote_is_accepted():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "I do take my albuterol inhaler sometimes, mostly at night."))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_present("medications_tried", "albuterol inhaler, sometimes", "I do took my albuterol inhaler sometimes"))

    assert result.ok is True


def test_a_quote_that_drops_a_negation_is_rejected_however_similar_it_looks():
    session = create_session("s1", PROTOCOL)
    session.transcript.append(_turn(0, "patient", "honestly I have no fever at all today"))
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_present("fever", "a fever today", "honestly I have a fever at all today"))

    assert result.ok is False
    assert "not found in anything the patient has said" in result.error
    assert session.record.facts == []
