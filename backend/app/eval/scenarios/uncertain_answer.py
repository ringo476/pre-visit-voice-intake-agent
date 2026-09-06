from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

uncertain_answer_scenario = EvalScenario(
    id="uncertain-answer",
    description="Patient doesn't know/remember an answer — this must never be recorded as asked_and_denied.",
    turns=[
        ScenarioTurn(
            patient_utterance="Honestly, I don't remember exactly how high my fever got, I didn't take my temperature.",
            tool_calls=[
                {
                    "name": "update_intake_record",
                    "args": {
                        "field": "max_temperature",
                        "value": "patient does not recall a specific reading",
                        "source": "patient_reported",
                        "evidence": "I don't remember exactly how high my fever got, I didn't take my temperature",
                        "confidence": 0.8,
                    },
                },
            ],
            final_text="No problem, that happens. Anything else about the fever you do remember, like when it started?",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["max_temperature"],
        fields_should_not_be_denied=["max_temperature"],
    ),
)
