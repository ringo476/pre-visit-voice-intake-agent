"""The LangGraph features the agent uses on purpose: one graph compiled once and
shared by every turn, a checkpointer that saves each session's state under its own
thread id, and an interrupt that pauses the run at the one irreversible step
(finalizing the intake) until the patient has heard the read-back and said yes."""

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

import app.agent.graph as graph_module
from app.agent.finalization import CLOSING_TEXT, is_clear_yes
from app.agent.graph import configure_checkpointer, get_graph, release_conversation, run_agent_turn
from app.agent.session import create_session, question_was_heard
from app.agent.tools import create_tool_handlers
from app.output.readback import build_readback
from app.schemas.intake_record import Polarity, Source
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import apply_fact

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


class Script:
    """A model that plays one scripted reply per call, and counts its calls."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def thread(session) -> dict:
    return {"configurable": {"thread_id": session.session_id}}


def complete_session(session_id: str = "s1"):
    """A session where every field on the checklist already has an answer."""
    session = create_session(session_id, PROTOCOL)
    for pf in PROTOCOL.fields:
        session.record = apply_fact(
            session.record, pf.field, "seeded", Source.PATIENT_REPORTED, "seed", 1.0, [], polarity=Polarity.PRESENT
        )
    return session


# --- compiled once -----------------------------------------------------------------------------


def test_the_graph_is_compiled_once_and_shared_by_every_turn_and_session(monkeypatch):
    compiles = []
    real = graph_module._compile
    monkeypatch.setattr(graph_module, "_compile", lambda saver: compiles.append(1) or real(saver))
    configure_checkpointer()  # drop the graph left over from earlier, so the count starts at zero

    for session_id in ("a", "b"):
        session = create_session(session_id, PROTOCOL)
        for text in ("hello", "I have a cough", "since Monday"):
            run_agent_turn(session, text, llm=Script([say("Okay.")]))

    assert len(compiles) == 1
    assert get_graph() is get_graph()


# --- checkpointer ------------------------------------------------------------------------------


def test_each_session_is_saved_under_its_own_thread():
    first, second = create_session("first", PROTOCOL), create_session("second", PROTOCOL)

    run_agent_turn(first, "hello", llm=Script([say("Reply for the first patient.")]))
    run_agent_turn(second, "hi", llm=Script([say("Reply for the second patient.")]))

    saved_first = get_graph().get_state(thread(first)).values["messages"]
    saved_second = get_graph().get_state(thread(second)).values["messages"]
    assert saved_first[-1].content == "Reply for the first patient."
    assert saved_second[-1].content == "Reply for the second patient."


def test_the_checkpoint_shows_what_the_model_saw_and_did_on_the_latest_turn():
    session = create_session("s1", PROTOCOL)
    script = Script([call("get_next_intake_question", {}), say("When did it start?")])

    run_agent_turn(session, "I have a cough", llm=script)

    saved = get_graph().get_state(thread(session)).values["messages"]
    assert [type(m).__name__ for m in saved] == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert saved[1].content == "I have a cough"
    assert saved[2].tool_calls[0]["name"] == "get_next_intake_question"
    assert "suggested" in json.loads(saved[3].content)["data"]
    assert saved[4].content == "When did it start?"


def test_each_turn_replaces_the_saved_history_instead_of_adding_to_it():
    session = create_session("s1", PROTOCOL)
    script = Script([say("First reply."), say("Second reply.")])

    run_agent_turn(session, "one", llm=script)
    run_agent_turn(session, "two", llm=script)

    saved = get_graph().get_state(thread(session)).values["messages"]
    # persona + transcript as it stood at the second turn (patient, agent, patient) + the new reply
    assert len(saved) == 1 + 3 + 1
    assert [m.content for m in saved[1:]] == ["one", "First reply.", "two", "Second reply."]


def test_releasing_a_conversation_forgets_its_saved_state():
    session = create_session("s1", PROTOCOL)
    run_agent_turn(session, "hello", llm=Script([say("Hi.")]))
    assert get_graph().get_state(thread(session)).values

    release_conversation("s1")

    assert not get_graph().get_state(thread(session)).values


# --- the pause at finalization -------------------------------------------------------------------


def finish_turn(session, script, **kwargs):
    return run_agent_turn(session, "That's everything.", llm=script, confirm_finalization=True, **kwargs)


def test_asking_to_finish_pauses_the_run_and_reads_the_answers_back():
    session = complete_session()
    script = Script([call("generate_clinician_brief", {})])

    result = finish_turn(session, script)

    assert result["reply_text"].startswith("Before I finish, let me read back what I have.")
    assert result["reply_text"].endswith("Is all of that correct?")
    assert session.brief_finalized is False  # the model asked, but nothing is finalized
    assert session.pending_readback_event_id is not None
    assert session.transcript[-1].speaker == "agent" and session.transcript[-1].text == result["reply_text"]
    assert get_graph().get_state(thread(session)).next == ("confirm_finalization",)
    assert script.calls == 1


def test_a_clear_yes_resumes_the_paused_run_and_finalizes_without_asking_the_model_again():
    session = complete_session()
    script = Script([call("generate_clinician_brief", {})])
    finish_turn(session, script)

    result = run_agent_turn(session, "Yes, that's all correct.", llm=script, confirm_finalization=True)

    assert session.brief_finalized is True
    assert result["reply_text"] == CLOSING_TEXT
    assert session.transcript[-1].text == CLOSING_TEXT
    assert script.calls == 1  # the resume did not call the model
    assert session.pending_readback_event_id is None
    assert get_graph().get_state(thread(session)).next == ()  # the run completed


def test_a_correction_does_not_finalize_and_is_handled_as_an_ordinary_turn():
    session = complete_session()
    script = Script([call("generate_clinician_brief", {}), say("Thanks, I will note that allergy.")])
    finish_turn(session, script)

    result = run_agent_turn(session, "No, I'm also allergic to penicillin.", llm=script, confirm_finalization=True)

    assert session.brief_finalized is False
    assert result["reply_text"] == "Thanks, I will note that allergy."  # the model handled the utterance
    assert script.calls == 2
    assert [t.speaker for t in session.transcript[-3:]] == ["agent", "patient", "agent"]
    assert get_graph().get_state(thread(session)).next == ()


@pytest.mark.parametrize(
    "reply",
    ["Yes but my fever started on Thursday", "Hmm, let me think", "Not really", "Yes, that is mostly right, I guess, apart from one thing"],
)
def test_anything_that_is_not_a_clear_yes_does_not_finalize(reply):
    session = complete_session()
    script = Script([call("generate_clinician_brief", {}), say("Okay.")])
    finish_turn(session, script)

    run_agent_turn(session, reply, llm=script, confirm_finalization=True)

    assert session.brief_finalized is False


def test_a_yes_to_a_read_back_the_patient_did_not_hear_to_the_end_does_not_finalize():
    session = complete_session()
    script = Script([call("generate_clinician_brief", {}), say("Let me go through it again.")])
    finish_turn(session, script, await_playback=True)  # the audio is still playing: not counted as heard yet

    run_agent_turn(session, "Yes", llm=script, confirm_finalization=True, await_playback=True)

    assert session.brief_finalized is False
    assert session.cut_off_question is None  # a read-back is not a checklist question to be asked again


def test_a_yes_after_the_read_back_played_to_the_end_finalizes():
    session = complete_session()
    script = Script([call("generate_clinician_brief", {})])
    finish_turn(session, script, await_playback=True)
    assert question_was_heard(session) is True  # the browser reported the audio finished

    run_agent_turn(session, "Yes", llm=script, confirm_finalization=True, await_playback=True)

    assert session.brief_finalized is True


def test_the_model_cannot_finish_by_passing_an_invented_reason():
    """The early-termination reason is whatever the model writes. It used to finalize at once,
    with required answers missing. Now it only earns a read-back, and without the patient's
    yes nothing is finalized."""
    session = create_session("s1", PROTOCOL)  # nothing recorded at all
    script = Script([call("generate_clinician_brief", {"early_termination_reason": "x"}), say("Okay.")])

    result = finish_turn(session, script)

    assert "I don't have any answers recorded yet." in result["reply_text"]
    assert "I still don't have answers for" in result["reply_text"]
    assert session.brief_finalized is False

    run_agent_turn(session, "I'm not sure what you mean", llm=script, confirm_finalization=True)
    assert session.brief_finalized is False


def test_a_run_superseded_while_it_pauses_does_not_speak_or_wait(monkeypatch):
    session = complete_session()

    def interrupted_while_building_the_readback(record, protocol):
        session.turn_generation += 1  # the patient talked over the turn
        return "Before I finish..."

    monkeypatch.setattr(graph_module, "build_readback", interrupted_while_building_the_readback)

    result = run_agent_turn(
        session, "That's everything.", llm=Script([call("generate_clinician_brief", {})]),
        confirm_finalization=True, turn_generation=session.turn_generation,
    )

    assert result["superseded"] is True
    assert session.pending_readback_event_id is None
    assert session.transcript[-1].speaker == "patient"


def test_losing_the_pending_marker_abandons_the_pause_instead_of_wedging_the_conversation():
    """For example after a restart: the saved run is still paused but the live session no longer
    knows it. The next utterance is simply an ordinary turn, and the stale pause is dropped."""
    session = complete_session()
    script = Script([call("generate_clinician_brief", {}), say("Okay, tell me more.")])
    finish_turn(session, script)
    session.pending_readback_event_id = None

    result = run_agent_turn(session, "Yes", llm=script, confirm_finalization=True)

    assert result["reply_text"] == "Okay, tell me more."
    assert session.brief_finalized is False
    assert get_graph().get_state(thread(session)).next == ()


def test_without_the_gate_the_tool_still_finalizes_at_once_as_before():
    session = complete_session()
    handlers = create_tool_handlers(session)

    result = handlers["generate_clinician_brief"]({})

    assert result.data["finalized"] is True and session.brief_finalized is True


def test_with_the_gate_the_tool_only_asks_for_the_pause():
    session = complete_session()
    handlers = create_tool_handlers(session, confirm_finalization=True)

    result = handlers["generate_clinician_brief"]({})

    assert result.data["awaiting_patient_confirmation"] is True
    assert session.brief_finalized is False


def test_the_gate_still_refuses_an_unfinished_intake_without_a_reason():
    handlers = create_tool_handlers(create_session("s1", PROTOCOL), confirm_finalization=True)

    result = handlers["generate_clinician_brief"]({})

    assert result.ok is False and "required fields still open" in result.error


# --- the plain-code pieces -----------------------------------------------------------------------


@pytest.mark.parametrize("reply", ["Yes", "yes.", "Yeah, that's correct", "That's right", "Sounds good", "Yes please", "Okay"])
def test_short_agreements_are_a_clear_yes(reply):
    assert is_clear_yes(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        "", "No", "No, that's wrong", "Yes but also penicillin", "Yes, except the fever", "Actually wait",
        "I don't think so", "Maybe", "Hmm", "Yes, that is right, and by the way I have another question for you",
    ],
)
def test_corrections_doubts_and_long_replies_are_not_a_clear_yes(reply):
    assert is_clear_yes(reply) is False


def test_the_read_back_is_built_from_the_record_in_checklist_order():
    session = create_session("s1", PROTOCOL)
    for field, polarity, value in [("onset", Polarity.PRESENT, "two weeks ago"), ("fever", Polarity.ABSENT, "no"), ("wheezing", Polarity.UNKNOWN, "unsure")]:
        session.record = apply_fact(session.record, field, value, Source.PATIENT_REPORTED, "q", 1.0, [], polarity=polarity)

    text = build_readback(session.record, PROTOCOL)

    assert "Onset: two weeks ago." in text
    assert "Fever or chills: no." in text
    assert "Wheezing: not sure." in text
    assert text.index("Onset:") < text.index("Fever or chills:") < text.index("Wheezing:")
    assert "I still don't have answers for" in text and text.endswith("Is all of that correct?")
