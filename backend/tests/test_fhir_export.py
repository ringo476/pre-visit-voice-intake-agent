import json
from pathlib import Path

from app.output.fhir_export import generate_fhir_export
from app.schemas.intake_record import QuestionEvent
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import apply_fact, create_empty_record

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def find_resource(bundle, resource_type):
    return next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type)


def test_questionnaire_response_only_includes_reported_fields():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    record = apply_fact(record, "onset", "10 days ago", "patient_reported", "10 days ago", 0.9, [])

    bundle = generate_fhir_export(record, PROTOCOL)
    qr = find_resource(bundle, "QuestionnaireResponse")
    assert len(qr["item"]) == 1
    assert qr["item"][0]["linkId"] == "onset"
    assert qr["item"][0]["answer"][0]["valueString"] == "10 days ago"


def test_medication_statement_for_reported_medication():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    record = apply_fact(record, "medications_tried", "cetirizine as needed", "patient_reported", "I take cetirizine", 0.9, [])

    bundle = generate_fhir_export(record, PROTOCOL)
    med = find_resource(bundle, "MedicationStatement")
    assert med["medicationCodeableConcept"]["text"] == "cetirizine as needed"
    assert med["status"] == "active"


def test_allergy_intolerance_with_reaction():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    record = apply_fact(record, "medication_allergies", "penicillin", "patient_reported", "allergic to penicillin", 0.9, [])
    record = apply_fact(record, "allergy_reaction", "hives", "patient_reported", "I get hives", 0.9, [])

    bundle = generate_fhir_export(record, PROTOCOL)
    allergy = find_resource(bundle, "AllergyIntolerance")
    assert allergy["code"]["text"] == "penicillin"
    assert allergy["reaction"][0]["description"] == "hives"


def test_omits_allergy_intolerance_when_never_asked():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    bundle = generate_fhir_export(record, PROTOCOL)
    assert not any(e["resource"]["resourceType"] == "AllergyIntolerance" for e in bundle["entry"])


def test_an_uncertain_medication_is_listed_with_provenance_but_never_asserted():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    events = [QuestionEvent(id="q1", field="medications_tried", question_text="Meds", timestamp="t")]
    record = apply_fact(record, "medications_tried", "does not remember", "uncertain", "I don't remember", 0.8, events, question_event_id="q1")

    bundle = generate_fhir_export(record, PROTOCOL)

    qr = find_resource(bundle, "QuestionnaireResponse")
    item = next(i for i in qr["item"] if i["linkId"] == "medications_tried")
    assert {"url": "https://example.org/fhir/StructureDefinition/provenance-source", "valueCode": "uncertain"} in item["extension"]
    assert not [e for e in bundle["entry"] if e["resource"]["resourceType"] == "MedicationStatement"]


def test_an_uncertain_allergy_answer_does_not_create_an_allergy_record():
    record = create_empty_record("s1", PROTOCOL.protocol_id)
    events = [QuestionEvent(id="q1", field="medication_allergies", question_text="Allergies", timestamp="t")]
    record = apply_fact(record, "medication_allergies", "does not know", "uncertain", "I'm not sure", 0.8, events, question_event_id="q1")

    bundle = generate_fhir_export(record, PROTOCOL)

    assert not [e for e in bundle["entry"] if e["resource"]["resourceType"] == "AllergyIntolerance"]
