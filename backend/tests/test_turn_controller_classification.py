"""Proves the routing fix end-to-end through the real turn_controller flow:
no protocol is chosen up front, a greeting doesn't force a premature
classification, a real complaint locks in the right protocol, and a later
unrelated complaint gets deferred rather than silently absorbed or
dropped."""

import os

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

from langchain_core.messages import AIMessage

from app.agent.session import create_session
from app.protocol.registry import get_protocol
from app.turn_controller import _classify_and_note


def make_llm(replies: list[AIMessage]):
    class FakeLLM:
        def __init__(self):
            self.n = 0

        def invoke(self, messages):
            msg = replies[min(self.n, len(replies) - 1)]
            self.n += 1
            return msg

    return FakeLLM()


def final(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


def test_greeting_does_not_classify_a_protocol():
    session = create_session("s1")
    note = _classify_and_note(session, "hi there")
    assert note is None
    assert session.protocol is None


def test_real_complaint_locks_in_the_matching_protocol():
    session = create_session("s1")
    note = _classify_and_note(session, "I've had a cough for about a week now")
    assert note is None
    assert session.protocol is not None
    assert session.protocol.protocol_id == "respiratory-intake"


def test_protocol_stays_locked_across_turns():
    session = create_session("s1")
    _classify_and_note(session, "I've had a cough for a week")
    assert session.protocol.protocol_id == "respiratory-intake"

    # A follow-up mentioning another respiratory keyword should not reclassify or produce a note.
    note = _classify_and_note(session, "I also have a sore throat")
    assert note is None
    assert session.protocol.protocol_id == "respiratory-intake"


def test_unrelated_complaint_after_lock_produces_a_deferral_note_not_a_switch():
    session = create_session("s1", get_protocol("respiratory-intake"))
    note = _classify_and_note(session, "by the way my ankle has also been really swollen")
    assert note is not None
    assert "different concern" in note.lower() or "separate" in note.lower() or "leg" in note.lower() or "musculoskeletal" in note.lower()
    # Protocol is NOT reassigned — no branching, no silent switch.
    assert session.protocol.protocol_id == "respiratory-intake"


def test_update_intake_record_rejects_a_field_outside_the_locked_protocol():
    """Regression test for the silent-drop bug: a fact for a field outside
    the current protocol must be rejected, not written and then vanish
    from the clinician brief."""
    from app.agent.tools import create_tool_handlers

    session = create_session("s1", get_protocol("respiratory-intake"))
    handlers = create_tool_handlers(session)
    result = handlers["update_intake_record"](
        {"field": "swelling", "value": "present", "source": "patient_reported", "evidence": "it's swollen", "confidence": 0.9}
    )
    assert result.ok is False
    assert len(session.record.facts) == 0
