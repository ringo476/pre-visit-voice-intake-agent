import uuid
from datetime import datetime, timezone

from app.policy_engine import build_evaluation_log_entry, evaluate_fact, evaluate_statement
from app.schemas.intake_record import Fact, FactStatus, Source


def make_fact(**overrides) -> Fact:
    defaults = dict(
        id=str(uuid.uuid4()),
        field="breathing_difficulty",
        value="mild shortness of breath on exertion",
        source=Source.PATIENT_REPORTED,
        evidence_span="a bit short of breath when I walk upstairs",
        confidence=0.8,
        status=FactStatus.UNCONFIRMED,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return Fact(**defaults)


def test_no_trigger_on_mild_finding():
    result = evaluate_fact(make_fact())
    assert result.triggered is False


def test_triggers_on_severe_breathing_difficulty():
    fact = make_fact(value="severe difficulty breathing", evidence_span="I can't breathe properly and it's getting worse")
    result = evaluate_fact(fact)
    assert result.triggered is True
    assert result.rule_id == "severe_breathing_difficulty"
    assert "911" in result.scripted_message or "emergency" in result.scripted_message.lower()


def test_triggers_on_severe_chest_pain_case_insensitive():
    fact = make_fact(field="chest_discomfort", value="Crushing chest pain", evidence_span="It feels like CRUSHING pressure on my chest")
    result = evaluate_fact(fact)
    assert result.triggered is True
    assert result.rule_id == "severe_chest_pain"


def test_field_rule_does_not_fire_on_unrelated_field():
    fact = make_fact(field="medications_tried", value="took severe cold medicine", evidence_span=None)
    result = evaluate_fact(fact)
    assert result.triggered is False


def test_triggers_on_limb_vascular_compromise():
    fact = make_fact(field="numbness_or_tingling", value="foot feels cold and numb", evidence_span="my foot is cold to touch and looks pale")
    result = evaluate_fact(fact)
    assert result.triggered is True
    assert result.rule_id == "limb_vascular_compromise"
    assert result.action == "emergency_escalation"


def test_triggers_urgent_escalation_for_inability_to_bear_weight():
    fact = make_fact(field="weight_bearing", value="cannot walk on it", evidence_span="I can't put any weight on it at all")
    result = evaluate_fact(fact)
    assert result.triggered is True
    assert result.rule_id == "unable_to_bear_weight"
    assert result.action == "urgent_escalation"  # time-sensitive, but not a 911-level emergency


def test_triggers_urgent_escalation_for_spreading_redness():
    fact = make_fact(field="redness", value="redness is spreading", evidence_span="the red area keeps getting bigger")
    result = evaluate_fact(fact)
    assert result.triggered is True
    assert result.rule_id == "spreading_redness"
    assert result.action == "urgent_escalation"


def test_evaluate_statement_loss_of_consciousness():
    result = evaluate_statement("I actually passed out yesterday for a few seconds")
    assert result.triggered is True
    assert result.rule_id == "loss_of_consciousness"


def test_evaluate_statement_confusion():
    result = evaluate_statement("My spouse says I've seemed really confused since this morning")
    assert result.triggered is True
    assert result.rule_id == "new_confusion"


def test_evaluate_statement_ordinary_no_trigger():
    result = evaluate_statement("I've just had a dry cough for about a week")
    assert result.triggered is False


def test_log_entry_records_both_triggered_and_not():
    triggered = evaluate_statement("I passed out")
    entry = build_evaluation_log_entry("fact_1", triggered)
    assert entry.triggered is True
    assert entry.rule_id == "loss_of_consciousness"

    not_triggered = evaluate_statement("just a mild cough")
    entry2 = build_evaluation_log_entry("fact_2", not_triggered)
    assert entry2.triggered is False
    assert entry2.rule_id is None
