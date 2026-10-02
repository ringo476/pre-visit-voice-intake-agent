import pytest

from app.schemas.intake_record import Polarity, QuestionEvent, Source, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig, ProtocolField
from app.state_engine import (
    ProvenanceViolationError,
    apply_fact,
    create_empty_record,
    derive_polarity,
    evidence_follows_question,
    evidence_in_patient_speech,
    evidence_in_text,
    find_asked_event,
    looks_like_denial,
    patient_turns_containing,
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


def test_evidence_in_text_matches_whole_words_ignoring_case_and_punctuation():
    assert evidence_in_text("I have a COUGH, and a fever!", "a cough and a fever") is True  # case and the comma are ignored
    assert evidence_in_text("I have a COUGH, and a fever!", "cough and fever") is False  # a missing word is not a match
    assert evidence_in_text("I know it started Monday", "no") is False
    assert evidence_in_text("anything", "   ") is False


def test_evidence_in_patient_speech_ignores_what_the_agent_said():
    transcript = _turns(("agent", "Any penicillin allergy?"), ("patient", "Not that I know of."))
    assert evidence_in_patient_speech(transcript, "Not that I know of") is True
    assert evidence_in_patient_speech(transcript, "penicillin allergy") is False


def test_patient_turns_containing_returns_the_sentence_the_quote_came_from():
    transcript = _turns(("patient", "Hello."), ("agent", "Hi."), ("patient", "No, I don't smoke. I have asthma."))
    assert patient_turns_containing(transcript, "I don't smoke") == "No, I don't smoke. I have asthma."


def test_patient_turns_containing_falls_back_to_the_latest_patient_message():
    transcript = _turns(("patient", "first"), ("patient", "second"))
    assert patient_turns_containing(transcript, "something not there") == "second"


def test_looks_like_denial_ignores_empty_values():
    assert looks_like_denial("no") is True
    assert looks_like_denial("None of them") is True
    assert looks_like_denial("   ") is False
    assert looks_like_denial("two weeks ago") is False


def test_any_correction_becomes_patient_reported_even_from_a_document_value():
    record = apply_fact(create_empty_record("s1", "p"), "medications_tried", "albuterol", Source.DOCUMENT_SOURCED, "Albuterol 90mcg", 0.9, [])
    record = record_correction(record, "medications_tried", "I stopped it last year", "I stopped it last year", 0.9)
    assert get_current_fact(record, "medications_tried").source == Source.PATIENT_REPORTED


def test_derive_polarity_from_the_source_or_the_wording_when_none_is_given():
    assert derive_polarity(Source.ASKED_AND_DENIED, "anything") == Polarity.ABSENT
    assert derive_polarity(Source.UNCERTAIN, "anything") == Polarity.UNKNOWN
    assert derive_polarity(Source.PATIENT_REPORTED, "no") == Polarity.ABSENT
    assert derive_polarity(Source.PATIENT_REPORTED, "two weeks ago") == Polarity.PRESENT


def test_a_stated_polarity_is_kept_whatever_the_value_looks_like():
    record = apply_fact(create_empty_record("s1", "p"), "fever", "not at all", Source.PATIENT_REPORTED, "not at all", 0.9, [], polarity=Polarity.ABSENT)
    assert get_current_fact(record, "fever").polarity == Polarity.ABSENT


def _protocol_dependent():
    return ProtocolConfig(
        protocol_id="p",
        name="P",
        keywords=[],
        fields=[
            ProtocolField(field="fever", label="Fever", category="c", required=True),
            ProtocolField(field="max_temperature", label="Max temp", category="c", required=False, required_if="fever"),
        ],
    )


def test_follow_ups_activate_from_the_stored_polarity_not_the_wording_of_the_value():
    protocol = _protocol_dependent()
    present = apply_fact(create_empty_record("s1", "p"), "fever", "I felt hot", Source.PATIENT_REPORTED, "I felt hot", 0.9, [], polarity=Polarity.PRESENT)
    absent_odd_wording = apply_fact(create_empty_record("s1", "p"), "fever", "not at all", Source.PATIENT_REPORTED, "not at all", 0.9, [], polarity=Polarity.ABSENT)
    unknown = apply_fact(create_empty_record("s1", "p"), "fever", "who knows", Source.PATIENT_REPORTED, "who knows", 0.9, [], polarity=Polarity.UNKNOWN)

    assert [m.field for m in get_missing_fields(present, protocol)] == ["max_temperature"]
    assert get_missing_fields(absent_odd_wording, protocol) == []   # "not at all" no longer slips through
    assert get_missing_fields(unknown, protocol) == []              # not knowing is not a yes
