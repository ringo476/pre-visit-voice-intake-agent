import pytest

from app.schemas.intake_record import QuestionEvent, Source, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig, ProtocolField
from app.state_engine import (
    ProvenanceViolationError,
    apply_fact,
    create_empty_record,
    evidence_follows_question,
    find_asked_event,
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


def _ev(id_, field, asked_in_turn):
    return QuestionEvent(id=id_, field=field, question_text=field, timestamp="t", asked_in_turn=asked_in_turn)


def test_find_asked_event_ignores_events_that_were_never_spoken():
    events = [_ev("q1", "fever", None)]
    assert find_asked_event(events, "fever", before_turn=5) is None


def test_find_asked_event_ignores_other_fields():
    events = [_ev("q1", "onset", 1)]
    assert find_asked_event(events, "fever", before_turn=5) is None


def test_find_asked_event_requires_it_to_be_spoken_before_the_answer_turn():
    events = [_ev("q1", "fever", 4)]
    assert find_asked_event(events, "fever", before_turn=4) is None
    assert find_asked_event(events, "fever", before_turn=5).id == "q1"


def test_find_asked_event_returns_the_most_recent_spoken_question_for_the_field():
    events = [_ev("q1", "fever", 1), _ev("q2", "fever", 5), _ev("q3", "fever", None)]
    assert find_asked_event(events, "fever", before_turn=9).id == "q2"


def _turns(*pairs):
    return [TranscriptTurn(id=str(i), speaker=sp, text=tx, timestamp=str(i)) for i, (sp, tx) in enumerate(pairs)]


def test_evidence_must_come_from_a_patient_turn_after_the_question():
    transcript = _turns(("patient", "No fever."), ("agent", "Any fever?"), ("patient", "No, none at all."))
    event = _ev("q1", "fever", asked_in_turn=1)
    assert evidence_follows_question(transcript, event, "No, none at all") is True
    assert evidence_follows_question(transcript, event, "No fever") is False  # only said before the question


def test_evidence_matching_ignores_case_and_punctuation():
    transcript = _turns(("agent", "Any fever?"), ("patient", "No -- no fever, really!"))
    assert evidence_follows_question(transcript, _ev("q1", "fever", 0), "no no fever really") is True


def test_evidence_in_an_agent_turn_does_not_count():
    transcript = _turns(("agent", "Any fever?"), ("agent", "You said no fever."), ("patient", "Hmm."))
    assert evidence_follows_question(transcript, _ev("q1", "fever", 0), "no fever") is False


def test_empty_evidence_never_matches():
    transcript = _turns(("agent", "Any fever?"), ("patient", "No."))
    assert evidence_follows_question(transcript, _ev("q1", "fever", 0), "  ") is False


def test_evidence_must_match_whole_words_not_fragments_of_other_words():
    transcript = _turns(("agent", "Any fever?"), ("patient", "I know it started on a Monday."))
    assert evidence_follows_question(transcript, _ev("q1", "fever", 0), "no") is False
    assert evidence_follows_question(transcript, _ev("q1", "fever", 0), "I know it started") is True


def test_apply_fact_rejects_uncertain_without_question_event():
    record = create_empty_record("s1", "p")
    with pytest.raises(ProvenanceViolationError):
        apply_fact(record, "onset", "does not remember", Source.UNCERTAIN, "I don't remember", 0.8, [])


def test_apply_fact_accepts_uncertain_with_matching_question_event():
    record = create_empty_record("s1", "p")
    events = [QuestionEvent(id="q1", field="onset", question_text="Onset", timestamp="t")]
    record = apply_fact(record, "onset", "does not remember", Source.UNCERTAIN, "I don't remember", 0.8, events, question_event_id="q1")
    assert get_current_fact(record, "onset").source == Source.UNCERTAIN


def _protocol_with_dependent():
    return ProtocolConfig(
        protocol_id="p",
        name="P",
        keywords=[],
        fields=[
            ProtocolField(field="fever", label="Fever", category="c", required=True),
            ProtocolField(field="max_temperature", label="Max temp", category="c", required=False, required_if="fever"),
        ],
    )


def test_an_uncertain_answer_closes_the_field_so_it_is_not_asked_forever():
    protocol = _protocol_with_dependent()
    events = [QuestionEvent(id="q1", field="fever", question_text="Fever", timestamp="t")]
    record = apply_fact(create_empty_record("s1", "p"), "fever", "unsure", Source.UNCERTAIN, "not sure", 0.8, events, question_event_id="q1")
    assert get_missing_fields(record, protocol) == []


def test_an_uncertain_answer_does_not_activate_conditional_follow_ups():
    """'I'm not sure if I had a fever' must not trigger 'what was your max
    temperature' — unknown is not the same as present."""
    protocol = _protocol_with_dependent()
    events = [QuestionEvent(id="q1", field="fever", question_text="Fever", timestamp="t")]
    record = apply_fact(create_empty_record("s1", "p"), "fever", "unsure", Source.UNCERTAIN, "not sure", 0.8, events, question_event_id="q1")
    assert [m.field for m in get_missing_fields(record, protocol)] == []


def test_correcting_an_uncertain_answer_becomes_a_patient_reported_fact():
    events = [QuestionEvent(id="q1", field="onset", question_text="Onset", timestamp="t")]
    record = apply_fact(create_empty_record("s1", "p"), "onset", "does not remember", Source.UNCERTAIN, "don't remember", 0.8, events, question_event_id="q1")
    record = record_correction(record, "onset", "3 weeks ago", "Oh, it was three weeks ago", 0.9)
    assert get_current_fact(record, "onset").source == Source.PATIENT_REPORTED


def test_correcting_a_denial_to_a_yes_is_no_longer_labelled_a_denial():
    """Patient says no fever, then 'actually I did have one'. The corrected
    fact must not stay asked_and_denied, or the brief would still say the
    patient denies it."""
    events = [QuestionEvent(id="q1", field="fever", question_text="Fever", timestamp="t")]
    record = apply_fact(create_empty_record("s1", "p"), "fever", "no", Source.ASKED_AND_DENIED, "No fever", 0.9, events, question_event_id="q1")

    record = record_correction(record, "fever", "yes, 101 on Tuesday", "Actually I did have a fever, 101 on Tuesday", 0.9)

    fact = get_current_fact(record, "fever")
    assert fact.source == Source.PATIENT_REPORTED
    assert fact.question_event_id is None  # no longer backed by the old question
    assert fact.supersedes is not None  # the original denial is still preserved


def test_correcting_a_patient_reported_fact_keeps_its_source_and_proof():
    record = apply_fact(create_empty_record("s1", "p"), "onset", "10 days ago", Source.PATIENT_REPORTED, "last Monday", 0.9, [])
    record = record_correction(record, "onset", "2 weeks ago", "two weeks ago", 0.9)
    assert get_current_fact(record, "onset").source == Source.PATIENT_REPORTED
