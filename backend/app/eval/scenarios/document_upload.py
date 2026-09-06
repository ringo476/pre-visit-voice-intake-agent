from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn, SeedDocument

document_upload_scenario = EvalScenario(
    id="document-upload",
    description=(
        "Patient uploads a prescription document; the agent must look it up via retrieve_uploaded_document "
        "and cite the extracted text verbatim as document_sourced evidence, never its own paraphrase."
    ),
    seed_documents=[
        SeedDocument(filename="prescription.pdf", mime_type="application/pdf", text="Albuterol inhaler 90mcg, 2 puffs as needed for wheeze"),
    ],
    turns=[
        ScenarioTurn(
            patient_utterance='Patient uploaded a document: "prescription.pdf".',
            tool_calls=[
                {"name": "retrieve_uploaded_document", "args": {"query": "medication name and dose"}},
                {
                    "name": "update_intake_record",
                    "args": {
                        "field": "medications_tried",
                        "value": "Albuterol inhaler 90mcg, 2 puffs as needed for wheeze",
                        "source": "document_sourced",
                        "evidence": "Albuterol inhaler 90mcg, 2 puffs as needed for wheeze",
                        "confidence": 0.95,
                    },
                },
            ],
            final_text="Thanks, I can see from that document you have an albuterol inhaler on file — do you still use it?",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["medications_tried"],
        expect_document_sourced_field="medications_tried",
    ),
)
