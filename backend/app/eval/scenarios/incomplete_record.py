from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

incomplete_record_scenario = EvalScenario(
    id="incomplete-at-end",
    description="Patient has to leave before the protocol is fully covered — the record must honestly reflect what's still open.",
    turns=[
        ScenarioTurn(
            patient_utterance="I've had a cough for a few days, but I actually have to run to a meeting, sorry.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "chief_complaint", "value": "cough", "source": "patient_reported", "evidence": "I've had a cough for a few days", "confidence": 0.85}},
                {"name": "generate_clinician_brief", "args": {"early_termination_reason": "Patient had to leave before intake was complete"}},
            ],
            final_text="No problem — I've saved what we covered so far, and your clinician will see what's still open.",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["chief_complaint"],
        fields_should_remain_missing=["onset", "course", "character", "fever", "breathing_difficulty"],
    ),
)
