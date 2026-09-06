import json
from pathlib import Path

from app.output.brief_generator import format_clinician_brief_as_text, generate_clinician_brief
from app.schemas.intake_record import QuestionEvent
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import apply_fact, create_empty_record

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def build_demo_record():
    record = create_empty_record("demo-session", PROTOCOL.protocol_id)
    events: list[QuestionEvent] = []

    reported = [
        ("chief_complaint", "persistent cough", "I've had this cough for around a week and a half"),
        ("onset", "approximately 10 days ago", "It started last Monday"),
        ("course", "gradually worsening", "It's probably a little worse now"),
        ("character", "primarily dry, occasional yellow sputum", "sometimes there's a little yellow mucus"),
        ("timing", "worse at night", "especially at night"),
        ("wheezing", "present", "I've had some wheezing"),
        ("respiratory_history", "previously prescribed an inhaler", "I have an inhaler, but I haven't used it for years"),
        ("inhaler_last_used", "not used in several years", "I haven't used it for years"),
        ("medications_tried", "cetirizine as needed", "I take cetirizine when I need it"),
        ("medication_allergies", "penicillin", "I'm allergic to penicillin"),
        ("patient_concerns", "worried the cough may become more serious", "I'm worried this might turn into something worse"),
    ]
    for field, value, evidence in reported:
        record = apply_fact(record, field, value, "patient_reported", evidence, 0.9, events)

    for field in ["fever", "chest_discomfort", "travel_exposure", "smoking_history"]:
        event = QuestionEvent(id=f"q_{field}", field=field, question_text=field, timestamp="t")
        events.append(event)
        record = apply_fact(record, field, "false", "asked_and_denied", "No", 0.9, events, question_event_id=event.id)

    # breathing_difficulty / inhaler_indication / allergy_reaction deliberately left unrecorded.
    return record


def test_places_denials_and_reports_in_the_right_sections():
    record = build_demo_record()
    brief = generate_clinician_brief(record, PROTOCOL)

    hpi = next(s for s in brief.sections if s.title == "History of present concern")
    joined = " ".join(hpi.lines).lower()
    assert "patient reports" in joined
    assert "denies" in joined

    allergies = next(s for s in brief.sections if s.title == "Allergies")
    assert "penicillin" in " ".join(allergies.lines).lower()


def test_never_renders_a_missing_field_as_a_denial():
    record = build_demo_record()  # breathing_difficulty was never asked
    brief = generate_clinician_brief(record, PROTOCOL)

    all_text = " ".join(line for s in brief.sections for line in s.lines).lower()
    assert "denies" not in all_text.split("breathing")[0][-40:] if "breathing" in all_text else True
    assert "Shortness of breath / difficulty breathing" in brief.clarifications


def test_lists_conditional_gaps_as_clarifications():
    record = build_demo_record()
    brief = generate_clinician_brief(record, PROTOCOL)
    assert "Inhaler: original indication" in brief.clarifications
    assert "Allergy: reaction description" in brief.clarifications


def test_includes_review_disclaimer():
    brief = generate_clinician_brief(build_demo_record(), PROTOCOL)
    assert "review and verify" in brief.disclaimer.lower()


def test_formats_to_readable_text():
    text = format_clinician_brief_as_text(generate_clinician_brief(build_demo_record(), PROTOCOL))
    assert "Reason for visit" in text
    assert "Information requiring clarification" in text
    assert "Generated from a patient conversation" in text
