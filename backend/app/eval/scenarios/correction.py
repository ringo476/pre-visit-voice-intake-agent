from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

correction_scenario = EvalScenario(
    id="mid-conversation-correction",
    description="Patient corrects the onset they gave earlier; the original statement must be preserved alongside the correction.",
    turns=[
        ScenarioTurn(
            patient_utterance="This cough started last Monday, so about 10 days ago.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "onset", "value": "approximately 10 days ago", "source": "patient_reported", "evidence": "started last Monday, so about 10 days ago", "confidence": 0.9}},
            ],
            final_text="Got it. Has it been getting better or worse?",
        ),
        ScenarioTurn(
            patient_utterance="Actually, wait — it started two weeks ago, not last Monday. I mixed up the days.",
            tool_calls=[
                {
                    "name": "record_patient_correction",
                    "args": {
                        "fact_id": "$LAST_FACT_ID:onset",
                        "field": "onset",
                        "new_value": "approximately 2 weeks ago",
                        "evidence": "it started two weeks ago, not last Monday",
                        "confidence": 0.93,
                    },
                },
            ],
            final_text="Thanks for the correction — I've updated that to about two weeks ago.",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["onset"],
        expect_correction=True,
    ),
)
