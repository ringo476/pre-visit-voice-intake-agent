"""Maps the intake record into a small, structurally-valid FHIR bundle
(QuestionnaireResponse + MedicationStatement + AllergyIntolerance) —
synthetic/demo data only, not wired to a real FHIR server."""

from app.schemas.intake_record import Fact, IntakeRecord, Polarity, Source
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import get_current_facts


def _is_reported(fact: Fact | None) -> bool:
    return fact is not None and fact.source != Source.NOT_ASKED


def _is_asserted(fact: Fact | None) -> bool:
    """Reported AND actually a claim about the patient. An 'unsure' answer is
    still listed in the QuestionnaireResponse with its provenance, but must
    never become a MedicationStatement or AllergyIntolerance — those assert
    something about the patient, and 'patient doesn't recall' asserts nothing."""
    return _is_reported(fact) and fact.source != Source.UNCERTAIN and fact.polarity != Polarity.UNKNOWN


def generate_fhir_export(record: IntakeRecord, protocol: ProtocolConfig) -> dict:
    current = get_current_facts(record)
    patient_ref = {"reference": f"Patient/{record.session_id}"}

    items = []
    for pf in protocol.fields:
        fact = current.get(pf.field)
        if not _is_reported(fact):
            continue
        extension = [{"url": "https://example.org/fhir/StructureDefinition/provenance-source", "valueCode": fact.source.value}]
        if fact.evidence_span:
            extension.append({"url": "https://example.org/fhir/StructureDefinition/evidence-quote", "valueString": fact.evidence_span})
        items.append(
            {
                "linkId": pf.field,
                "text": pf.label,
                "answer": [{"valueString": fact.value}],
                "extension": extension,
            }
        )

    questionnaire_response = {
        "resourceType": "QuestionnaireResponse",
        "status": "completed",
        "questionnaire": f"Questionnaire/{protocol.protocol_id}",
        "subject": patient_ref,
        "authored": record.updated_at,
        "item": items,
    }

    medication_statements = []
    for pf in protocol.fields:
        if pf.category != "medications":
            continue
        fact = current.get(pf.field)
        if not _is_asserted(fact):
            continue
        medication_statements.append(
            {
                "resourceType": "MedicationStatement",
                "status": "not-taken" if fact.polarity == Polarity.ABSENT else "active",
                "subject": patient_ref,
                "medicationCodeableConcept": {"text": fact.value},
                "informationSource": {"display": fact.source.value},
            }
        )

    allergy_fact = current.get("medication_allergies")
    allergy_reaction_fact = current.get("allergy_reaction")
    allergy_intolerances = []
    if _is_asserted(allergy_fact):
        allergy: dict = {
            "resourceType": "AllergyIntolerance",
            "clinicalStatus": {"text": "denied" if allergy_fact.polarity == Polarity.ABSENT else "active"},
            "patient": patient_ref,
            "code": {"text": allergy_fact.value},
        }
        if _is_reported(allergy_reaction_fact):
            allergy["reaction"] = [{"description": allergy_reaction_fact.value}]
        allergy_intolerances.append(allergy)

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": questionnaire_response},
            *({"resource": r} for r in medication_statements),
            *({"resource": r} for r in allergy_intolerances),
        ],
    }
