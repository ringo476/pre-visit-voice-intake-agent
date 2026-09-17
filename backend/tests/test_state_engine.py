import pytest

from app.schemas.intake_record import QuestionEvent, Source
from app.schemas.protocol_config import ProtocolConfig, ProtocolField
from app.state_engine import (
    ProvenanceViolationError,
    apply_fact,
    create_empty_record,
    get_current_fact,
    get_missing_fields,
    is_record_complete,
    record_correction,
)

PROTOCOL = ProtocolConfig(
    protocol_id="test-protocol",
    name="Test protocol",
    fields=[
        ProtocolField(field="onset", label="Onset", category="timeline", required=True),
        ProtocolField(field="fever", label="Fever", category="associated_symptoms", required=True),
        ProtocolField(
            field="max_temperature",
            label="Max temperature",
            category="associated_symptoms",
            required=False,
            required_if="fever",
        ),
    ],
)


def test_apply_fact_appends_patient_reported():
    record = create_empty_record("s1", "test-protocol")
    record = apply_fact(record, "onset", "10 days ago", Source.PATIENT_REPORTED, "10 days ago", 0.9, [])
    assert get_current_fact(record, "onset").value == "10 days ago"


def test_apply_fact_rejects_denied_without_question_event():
    record = create_empty_record("s1", "test-protocol")
    with pytest.raises(ProvenanceViolationError):
        apply_fact(record, "fever", "false", Source.ASKED_AND_DENIED, None, 0.9, [])


def test_apply_fact_rejects_denied_with_mismatched_question_event():
    record = create_empty_record("s1", "test-protocol")
    events = [QuestionEvent(id="q1", field="onset", question_text="When did it start?", timestamp="t")]
    with pytest.raises(ProvenanceViolationError):
        apply_fact(record, "fever", "false", Source.ASKED_AND_DENIED, None, 0.9, events, question_event_id="q1")


def test_apply_fact_accepts_denied_with_matching_question_event():
    record = create_empty_record("s1", "test-protocol")
    events = [QuestionEvent(id="q1", field="fever", question_text="Any fever?", timestamp="t")]
    record = apply_fact(record, "fever", "false", Source.ASKED_AND_DENIED, "No", 0.95, events, question_event_id="q1")
    assert get_current_fact(record, "fever").source == Source.ASKED_AND_DENIED


def test_record_correction_preserves_original_and_links_via_supersedes():
    record = create_empty_record("s1", "test-protocol")
    record = apply_fact(record, "onset", "10 days ago", Source.PATIENT_REPORTED, "It started last Monday", 0.9, [])
    original_id = get_current_fact(record, "onset").id

    record = record_correction(record, "onset", "2 weeks ago", "Actually, two weeks ago", 0.92)

    current = get_current_fact(record, "onset")
    assert current.value == "2 weeks ago"
    assert current.supersedes == original_id
    assert current.status.value == "corrected"

    original = next(f for f in record.facts if f.id == original_id)
    assert original.value == "10 days ago"


def test_record_correction_rejects_field_with_no_current_fact():
    record = create_empty_record("s1", "test-protocol")
    with pytest.raises(ProvenanceViolationError):
        record_correction(record, "onset", "2 weeks ago", "...", 0.9)


def test_missing_fields_flags_required_fields_with_no_fact():
    record = create_empty_record("s1", "test-protocol")
    missing = [m.field for m in get_missing_fields(record, PROTOCOL)]
    assert "onset" in missing
    assert "fever" in missing
    assert "max_temperature" not in missing  # conditional, not yet triggered


def test_conditional_field_not_required_when_trigger_denied():
    record = create_empty_record("s1", "test-protocol")
    record = apply_fact(record, "onset", "10 days ago", Source.PATIENT_REPORTED, "x", 0.9, [])
    events = [QuestionEvent(id="q1", field="fever", question_text="Fever?", timestamp="t")]
    record = apply_fact(record, "fever", "false", Source.ASKED_AND_DENIED, "No", 0.9, events, question_event_id="q1")

    missing = [m.field for m in get_missing_fields(record, PROTOCOL)]
    assert "max_temperature" not in missing
    assert is_record_complete(record, PROTOCOL) is True


def test_conditional_field_required_once_trigger_affirmed():
    record = create_empty_record("s1", "test-protocol")
    record = apply_fact(record, "onset", "10 days ago", Source.PATIENT_REPORTED, "x", 0.9, [])
    record = apply_fact(record, "fever", "true", Source.PATIENT_REPORTED, "I've had a fever", 0.9, [])

    missing = [m.field for m in get_missing_fields(record, PROTOCOL)]
    assert "max_temperature" in missing
    assert is_record_complete(record, PROTOCOL) is False
