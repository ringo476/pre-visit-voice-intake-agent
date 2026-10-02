from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

uncertain_answer_scenario = EvalScenario(
    id="uncertain-answer",
    description="Patient is asked a question and says they don't remember — this must be recorded as uncertain, never as asked_and_denied.",
    turns=[
        ScenarioTurn(
            patient_utterance="I've had a cough for a while now.",
            tool_calls=[
                {
                    "name": "update_intake_record",
                    "args": {
                        "field": "chief_complaint",
                        "value": "persistent cough",
                        "polarity": "present",
                        "evidence": "I've had a cough for a while now",
                        "confidence": 0.9,
                    },
                },
                {"name": "get_next_intake_question", "args": {}},
            ],
            final_text="I'm sorry to hear that. When did the cough start?",
        ),
        ScenarioTurn(
            patient_utterance="Honestly, I don't remember exactly when it started.",
            tool_calls=[
                {
                    "name": "update_intake_record",
                    "args": {
                        "field": "onset",
                        "value": "patient does not remember when it started",
                        "polarity": "unknown",
                        "evidence": "I don't remember exactly when it started",
                        "confidence": 0.85,
                    },
                },
            ],
            final_text="No problem, that happens. Has it been getting better, worse, or staying about the same?",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["onset"],
        fields_should_not_be_denied=["onset"],
        fields_should_be_uncertain=["onset"],
    ),
)
