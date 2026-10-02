from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

correction_scenario = EvalScenario(
    id="mid-conversation-correction",
    description=(
        "Patient gives a different onset from the one on record. Ava does not overwrite it on that one "
        "statement: she asks which is right, and only the patient's answer changes it. The original is preserved."
    ),
    turns=[
        ScenarioTurn(
            patient_utterance="This cough started last Monday, so about 10 days ago.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "onset", "value": "approximately 10 days ago", "polarity": "present", "evidence": "started last Monday, so about 10 days ago", "confidence": 0.9}},
            ],
            final_text="Got it. Has it been getting better or worse?",
        ),
        ScenarioTurn(
            patient_utterance="Actually, wait — it started two weeks ago, not last Monday. I mixed up the days.",
            # Refused: this disagrees with the onset on record, so nothing changes yet.
            tool_calls=[
                {
                    "name": "record_patient_correction",
                    "args": {
                        "field": "onset",
                        "new_value": "approximately 2 weeks ago",
                        "evidence": "it started two weeks ago, not last Monday",
                        "confidence": 0.93,
                    },
                },
            ],
            final_text="Just to be sure about the onset: earlier I noted about 10 days ago, and just now it sounded like two weeks ago. Which is right?",
        ),
        ScenarioTurn(
            patient_utterance="Two weeks ago. I mixed up the days.",
            tool_calls=[
                {
                    "name": "record_patient_correction",
                    "args": {
                        "field": "onset",
                        "new_value": "approximately 2 weeks ago",
                        "evidence": "Two weeks ago. I mixed up the days",
                        "confidence": 0.93,
                    },
                },
            ],
            final_text="Thanks, I've updated that to about two weeks ago.",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["onset"],
        expect_correction=True,
        expect_confirmation_question=True,
    ),
)
