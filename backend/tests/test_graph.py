import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from app.agent.graph import run_agent_turn, run_opening_turn
from app.agent.session import create_session
from app.schemas.protocol_config import ProtocolConfig

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


class FakeLLM:
    """Stands in for the real Gemini-backed model: returns a scripted
    sequence of AIMessages on successive .invoke() calls, exactly like
    reasoner.ts's injectable ModelStep did in the TypeScript version."""

    def __init__(self, script: list[AIMessage]):
        self.script = script
        self.call_count = 0

    def invoke(self, messages):
        response = self.script[self.call_count]
        self.call_count = min(self.call_count + 1, len(self.script) - 1)
        return response


def tool_call_message(calls: list[dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": c["name"], "args": c["args"], "id": c.get("id", c["name"])} for c in calls])


def final_message(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


def test_no_tool_calls_produces_immediate_reply_and_logs_transcript():
    session = create_session("s1", PROTOCOL)
    llm = FakeLLM([final_message("Thanks, tell me more.")])

    result = run_agent_turn(session, "I've had a cough for 10 days.", llm=llm)

    assert result["reply_text"] == "Thanks, tell me more."
    assert len(session.transcript) == 2
    assert session.transcript[0].speaker == "patient"
    assert session.transcript[1].speaker == "agent"


def test_single_tool_call_executes_and_updates_state_before_final_reply():
    session = create_session("s1", PROTOCOL)
    llm = FakeLLM(
        [
            tool_call_message(
                [
                    {
                        "name": "update_intake_record",
                        "args": {"field": "onset", "value": "10 days ago", "source": "patient_reported", "evidence": "10 days ago", "confidence": 0.9},
                    }
                ]
            ),
            final_message("Got it, noted the onset."),
        ]
    )

    result = run_agent_turn(session, "It started 10 days ago.", llm=llm)

    assert result["reply_text"] == "Got it, noted the onset."
    assert len(session.record.facts) == 1


def test_multiple_tool_calls_in_one_round_all_execute():
    session = create_session("s1", PROTOCOL)
    llm = FakeLLM(
        [
            tool_call_message(
                [
                    {
                        "name": "update_intake_record",
                        "args": {"field": "onset", "value": "10 days ago", "source": "patient_reported", "evidence": "10 days ago", "confidence": 0.9},
                        "id": "call1",
                    },
                    {"name": "get_next_intake_question", "args": {}, "id": "call2"},
                ]
            ),
            final_message("Okay."),
        ]
    )

    run_agent_turn(session, "It started 10 days ago.", llm=llm)

    assert len(session.record.facts) == 1
    assert len(session.question_events) == 1


def test_loop_guard_raises_when_model_never_stops_calling_tools():
    session = create_session("s1", PROTOCOL)

    class AlwaysCallsToolsLLM:
        """Builds a fresh AIMessage each call — reusing the identical object
        across calls would give every round the same message id, which
        LangGraph's message-list reducer treats as an in-place update
        rather than a new turn, masking the infinite loop this test means
        to exercise."""

        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            return tool_call_message([{"name": "get_next_intake_question", "args": {}, "id": f"call{self.n}"}])

    with pytest.raises(GraphRecursionError):
        run_agent_turn(session, "hello", llm=AlwaysCallsToolsLLM())


class _TwoTurnDenialLLM:
    """Drives a realistic two-turn denial. Turn 1: ask what's missing, then
    ask it aloud. Turn 2 (patient answers): record the denial WITHOUT any
    question_event_id — the model has none in a later turn, because tool
    results from earlier turns are never replayed to it."""

    def __init__(self, session):
        self.session = session
        self.phase = 0

    def invoke(self, messages):
        self.phase += 1
        if self.phase == 1:
            return tool_call_message([{"name": "get_next_intake_question", "args": {}}])
        if self.phase == 2:
            return final_message("Have you had any fever?")
        if self.phase == 3:
            field = self.session.question_events[0].field
            return tool_call_message(
                [{"name": "update_intake_record", "args": {"field": field, "value": "false", "source": "asked_and_denied", "evidence": "No, not at all", "confidence": 0.9}}]
            )
        return final_message("Understood.")


def test_denial_is_recorded_on_the_turn_after_the_question_was_spoken():
    session = create_session("s1", PROTOCOL)
    llm = _TwoTurnDenialLLM(session)

    run_agent_turn(session, "hello", llm=llm)

    event = session.question_events[0]
    assert event.asked_in_turn == 1  # transcript: [patient hello, agent question]
    assert event.spoken_text == "Have you had any fever?"

    run_agent_turn(session, "No, not at all.", llm=llm)

    assert len(session.record.facts) == 1
    fact = session.record.facts[0]
    assert fact.source.value == "asked_and_denied"
    assert fact.question_event_id == event.id


def test_denial_in_the_same_turn_as_the_question_is_rejected():
    """The old behaviour: look up a question and immediately file a denial
    for it before the patient has said anything. The question hasn't been
    spoken yet, so there is nothing for the denial to answer."""
    session = create_session("s1", PROTOCOL)

    class SameTurnLLM:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return tool_call_message([{"name": "get_next_intake_question", "args": {}}])
            if self.n == 2:
                field = session.question_events[0].field
                return tool_call_message(
                    [{"name": "update_intake_record", "args": {"field": field, "value": "false", "source": "asked_and_denied", "evidence": "No, not at all", "confidence": 0.9}}]
                )
            return final_message("Understood.")

    run_agent_turn(session, "No, not at all.", llm=SameTurnLLM())

    assert session.record.facts == []


def test_only_the_question_actually_asked_this_turn_is_marked_as_spoken():
    session = create_session("s1", PROTOCOL)
    llm = FakeLLM(
        [
            tool_call_message([{"name": "get_next_intake_question", "args": {}, "id": "a"}]),
            tool_call_message([{"name": "get_next_intake_question", "args": {}, "id": "b"}]),
            final_message("Have you had any wheezing?"),
        ]
    )

    run_agent_turn(session, "hello", llm=llm)

    assert len(session.question_events) == 2
    assert session.question_events[0].asked_in_turn is None
    assert session.question_events[1].asked_in_turn == 1


def test_a_superseded_turn_never_marks_its_question_as_spoken():
    session = create_session("s1", PROTOCOL)

    class SupersededMidTurnLLM:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return tool_call_message([{"name": "get_next_intake_question", "args": {}}])
            session.turn_generation += 1  # the patient interrupted while this turn was in flight
            return final_message("Have you had any fever?")

    result = run_agent_turn(session, "hello", llm=SupersededMidTurnLLM(), turn_generation=session.turn_generation)

    assert result["superseded"] is True
    assert len(session.question_events) == 1
    assert session.question_events[0].asked_in_turn is None


def test_opening_turn_marks_its_question_as_spoken():
    session = create_session("s1", PROTOCOL)
    llm = FakeLLM(
        [
            tool_call_message([{"name": "get_next_intake_question", "args": {}}]),
            final_message("Hi, I see you're coming in about a cough. When did it start?"),
        ]
    )

    run_opening_turn(session, "persistent cough", llm=llm)

    assert session.question_events[0].asked_in_turn == 0


class _FakeVerifier:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.verdict)


def test_a_denial_the_verifier_rejects_is_not_recorded_and_the_field_stays_open():
    session = create_session("s1", PROTOCOL)
    llm = _TwoTurnDenialLLM(session)
    verifier = _FakeVerifier("OTHER")

    run_agent_turn(session, "hello", llm=llm, verifier_llm=verifier)
    run_agent_turn(session, "No, not at all.", llm=llm, verifier_llm=verifier)

    assert verifier.calls == 1
    assert session.record.facts == []
    asked_field = session.question_events[0].field
    from app.state_engine import get_missing_fields

    assert asked_field in [m.field for m in get_missing_fields(session.record, PROTOCOL)]


def test_a_denial_the_verifier_confirms_is_recorded():
    session = create_session("s1", PROTOCOL)
    llm = _TwoTurnDenialLLM(session)
    verifier = _FakeVerifier("NEGATIVE")

    run_agent_turn(session, "hello", llm=llm, verifier_llm=verifier)
    run_agent_turn(session, "No, not at all.", llm=llm, verifier_llm=verifier)

    assert [f.source.value for f in session.record.facts] == ["asked_and_denied"]
