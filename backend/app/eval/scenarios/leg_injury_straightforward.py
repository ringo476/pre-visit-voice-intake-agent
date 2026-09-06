from app.eval.types import EvalScenario, ScenarioExpectations, ScenarioTurn

# Proves the routing fix: this scenario runs against a DIFFERENT protocol
# (musculoskeletal-leg-injury, not respiratory-intake) and exercises fields
# that don't exist in the respiratory checklist at all — if protocol
# selection were still hardcoded, this scenario couldn't pass.
leg_injury_straightforward_scenario = EvalScenario(
    id="leg-injury-straightforward",
    description="A leg-injury intake, proving protocol routing works end-to-end for a non-respiratory complaint.",
    protocol_id="musculoskeletal-leg-injury",
    turns=[
        ScenarioTurn(
            patient_utterance="I twisted my ankle yesterday playing basketball and it's pretty swollen.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "chief_complaint", "value": "twisted ankle", "source": "patient_reported", "evidence": "I twisted my ankle yesterday playing basketball", "confidence": 0.95}},
                {"name": "update_intake_record", "args": {"field": "onset", "value": "yesterday", "source": "patient_reported", "evidence": "yesterday", "confidence": 0.9}},
                {"name": "update_intake_record", "args": {"field": "mechanism_of_injury", "value": "twisted while playing basketball", "source": "patient_reported", "evidence": "twisted my ankle yesterday playing basketball", "confidence": 0.9}},
                {"name": "update_intake_record", "args": {"field": "swelling", "value": "present", "source": "patient_reported", "evidence": "it's pretty swollen", "confidence": 0.9}},
            ],
            final_text="Sorry to hear that. Can you still put weight on it and walk?",
        ),
        ScenarioTurn(
            patient_utterance="Yeah, it hurts but I can walk on it okay.",
            tool_calls=[
                {"name": "update_intake_record", "args": {"field": "weight_bearing", "value": "able to bear weight, painful", "source": "patient_reported", "evidence": "I can walk on it okay", "confidence": 0.9}},
                {"name": "get_next_intake_question", "args": {}},
            ],
            final_text="Good, that's reassuring. Any redness or warmth around the ankle?",
        ),
    ],
    expectations=ScenarioExpectations(
        fields_should_be_covered=["chief_complaint", "onset", "mechanism_of_injury", "swelling", "weight_bearing"],
        expect_safety_trigger=False,
    ),
)
