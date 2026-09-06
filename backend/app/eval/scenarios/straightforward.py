from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

straightforward_scenario = EvalScenario(
    id="straightforward",
    description="A typical cough intake with no corrections and nothing safety-relevant.",
    turns=[
        ScenarioTurn(
            patient_utterance="I've had this cough for around a week and a half. It doesn't seem to be going away.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "chief_complaint", "value": "persistent cough", "source": "patient_reported", "evidence": "I've had this cough for around a week and a half", "confidence": 0.95}},
                {"name": "update_intake_record", "args": {"field": "onset", "value": "approximately 10 days ago", "source": "patient_reported", "evidence": "around a week and a half", "confidence": 0.85}},
            ],
            final_text="I'm sorry it's been lingering. Has it been getting better, worse, or staying about the same?",
        ),
        ScenarioTurn(
            patient_utterance="It's probably a little worse now, especially at night.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "course", "value": "gradually worsening", "source": "patient_reported", "evidence": "a little worse now", "confidence": 0.85}},
                {"name": "update_intake_record", "args": {"field": "timing", "value": "worse at night", "source": "patient_reported", "evidence": "especially at night", "confidence": 0.9}},
            ],
            final_text="Is the cough dry, or are you bringing up mucus?",
        ),
        ScenarioTurn(
            patient_utterance="Mostly dry, but sometimes there's a little yellow mucus.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "character", "value": "primarily dry, occasional yellow sputum", "source": "patient_reported", "evidence": "sometimes there's a little yellow mucus", "confidence": 0.9}},
                {"name": "get_next_intake_question", "args": {}},
            ],
            final_text="Have you had any fever or chills?",
        ),
        ScenarioTurn(
            patient_utterance="No, no fever.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "fever", "value": "false", "source": "asked_and_denied", "evidence": "No, no fever", "confidence": 0.92, "question_event_id": "$LAST_QUESTION_EVENT_ID"}},
            ],
            final_text="Understood, no fever noted.",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["chief_complaint", "onset", "course", "character", "timing", "fever"],
        fields_should_not_be_denied=["chief_complaint", "onset", "course", "character", "timing"],
        expect_safety_trigger=False,
    ),
)
