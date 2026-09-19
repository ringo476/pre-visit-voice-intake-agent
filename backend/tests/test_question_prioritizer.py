"""Verifies the adaptive next-question ranking is genuinely guarded: a
valid pick is honored, but a hallucinated field, a malformed response, or a
model that raises must all fall back to None (the caller's signal to keep
the original fixed order) rather than ever propagating a bad value or
crashing the turn."""

from langchain_core.messages import AIMessage

from app.agent.question_prioritizer import rank_next_field
from app.state_engine import MissingField

MISSING = [
    MissingField(field="onset", label="Onset", category="timeline"),
    MissingField(field="fever", label="Fever or chills", category="associated_symptoms"),
    MissingField(field="breathing_difficulty", label="Shortness of breath", category="associated_symptoms"),
]


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)


class RaisingLLM:
    def invoke(self, messages):
        raise RuntimeError("simulated network failure")


def test_honors_a_valid_pick():
    llm = FakeLLM("breathing_difficulty")
    assert rank_next_field(MISSING, {}, llm) == "breathing_difficulty"


def test_is_case_and_punctuation_tolerant():
    llm = FakeLLM("Fever.")
    assert rank_next_field(MISSING, {}, llm) == "fever"


def test_falls_back_to_none_on_hallucinated_field():
    llm = FakeLLM("chest_pain_that_does_not_exist")
    assert rank_next_field(MISSING, {}, llm) is None


def test_falls_back_to_none_on_rambling_non_field_response():
    llm = FakeLLM("I think you should ask about the fever next since it's most urgent.")
    assert rank_next_field(MISSING, {}, llm) is None


def test_falls_back_to_none_when_model_raises():
    assert rank_next_field(MISSING, {}, RaisingLLM()) is None


def test_never_calls_the_model_with_fewer_than_two_options():
    llm = FakeLLM("onset")
    result = rank_next_field(MISSING[:1], {}, llm)
    assert result is None
    assert llm.calls == 0
