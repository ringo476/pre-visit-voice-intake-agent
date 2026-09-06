import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from app.agent.graph import run_agent_turn
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


def test_asked_and_denied_flow_across_rounds_using_question_event_from_tool():
    session = create_session("s1", PROTOCOL)

    # Round 1: ask what's missing. Round 2: deny it using the just-issued
    # question_event_id. Round 3: final reply.
    class SequencedLLM:
        def __init__(self):
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            if self.call_count == 1:
                return tool_call_message([{"name": "get_next_intake_question", "args": {}}])
            if self.call_count == 2:
                field = session.question_events[0].field
                qid = session.question_events[0].id
                return tool_call_message(
                    [
                        {
                            "name": "update_intake_record",
                            "args": {"field": field, "value": "false", "source": "asked_and_denied", "evidence": "No, not at all", "confidence": 0.9, "question_event_id": qid},
                        }
                    ]
                )
            return final_message("Understood.")

    result = run_agent_turn(session, "hello", llm=SequencedLLM())
    assert result["reply_text"] == "Understood."
    assert len(session.record.facts) == 1
    assert session.record.facts[0].source.value == "asked_and_denied"
