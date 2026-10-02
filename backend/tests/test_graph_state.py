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
from app.agent.graph import configure_checkpointer, get_graph, release_conversation, run_agent_turn, run_opening_turn
from app.agent.session import create_session, question_was_heard
from app.agent.tools import create_tool_handlers
from app.output.readback import build_readback
from app.schemas.intake_record import Polarity, Source, TranscriptTurn
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
        self.seen = []  # every message list the model was sent

    def invoke(self, messages):
        self.seen.append(list(messages))
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
    assert [type(m).__name__ for m in saved] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert saved[0].content == "I have a cough"
    assert saved[1].tool_calls[0]["name"] == "get_next_intake_question"
    assert "suggested" in json.loads(saved[2].content)["data"]
    assert saved[3].content == "When did it start?"


def conversation(model_input):
    """The part of a model input after the persona prompt, as (kind, text) pairs."""
    return [(type(m).__name__, m.content) for m in model_input[1:]]


def count_rebuilds(monkeypatch):
    rebuilds = []
    real = graph_module._transcript_messages
    monkeypatch.setattr(graph_module, "_transcript_messages", lambda turns: rebuilds.append(len(turns)) or real(turns))
    return rebuilds


def test_the_checkpointer_carries_the_conversation_so_a_normal_turn_sends_only_the_new_words(monkeypatch):
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    script = Script([say("First reply."), say("Second reply.")])

    run_agent_turn(session, "one", llm=script)
    run_agent_turn(session, "two", llm=script)

    assert isinstance(script.seen[1][0], SystemMessage)  # the persona prompt is added on every call
    assert conversation(script.seen[1]) == [("HumanMessage", "one"), ("AIMessage", "First reply."), ("HumanMessage", "two")]
    assert rebuilds == []  # the history came from the checkpoint, not from the transcript


def test_last_turns_tool_exchange_is_not_shown_to_the_model_on_the_next_turn():
    session = create_session("s1", PROTOCOL)
    script = Script([call("get_next_intake_question", {}), say("When did it start?"), say("Thanks.")])

    run_agent_turn(session, "I have a cough", llm=script)
    run_agent_turn(session, "Two weeks ago", llm=script)

    second_turn = script.seen[2]
    assert not any(isinstance(m, ToolMessage) for m in second_turn)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in second_turn)
    assert conversation(second_turn) == [
        ("HumanMessage", "I have a cough"), ("AIMessage", "When did it start?"), ("HumanMessage", "Two weeks ago"),
    ]


def test_when_the_saved_conversation_is_lost_it_is_rebuilt_from_the_transcript(monkeypatch):
    """For example after a restart or a dropped connection: the saved thread is empty but the
    transcript, reloaded from the database, is not."""
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    script = Script([say("A."), say("B."), say("C.")])
    run_agent_turn(session, "one", llm=script)
    run_agent_turn(session, "two", llm=script)
    release_conversation("s1")

    run_agent_turn(session, "three", llm=script)

    assert conversation(script.seen[2]) == [
        ("HumanMessage", "one"), ("AIMessage", "A."), ("HumanMessage", "two"), ("AIMessage", "B."), ("HumanMessage", "three"),
    ]
    assert rebuilds == [4]  # rebuilt once, from the four earlier turns


def test_a_line_the_app_spoke_outside_the_graph_is_picked_up_by_a_rebuild(monkeypatch):
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    script = Script([say("First.")])
    run_agent_turn(session, "one", llm=script)
    # e.g. a message replayed after a patient talked over it: the graph never saw this line
    session.transcript.append(TranscriptTurn(id="x", speaker="agent", text="Replayed line.", timestamp="t"))

    run_agent_turn(session, "two", llm=script)

    assert ("AIMessage", "Replayed line.") in conversation(script.seen[-1])
    assert rebuilds == [3]


def test_a_late_reply_from_a_turn_the_patient_talked_over_is_never_shown_to_the_model(monkeypatch):
    """The model call was already in flight when the patient interrupted, so its reply arrives
    after the turn was thrown away. It is saved in the thread, which therefore no longer matches
    the transcript, so the next turn heals the thread by rebuilding it from the transcript."""
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)

    class InterruptedInFlight:
        def invoke(self, messages):
            session.turn_generation += 1  # the patient interrupted while the model was answering
            return say("late reply that is thrown away")

    first = run_agent_turn(session, "one", llm=InterruptedInFlight(), turn_generation=session.turn_generation)
    assert first["superseded"] is True
    assert [t.speaker for t in session.transcript] == ["patient"]  # no reply was added to the transcript

    script = Script([say("Okay.")])
    run_agent_turn(session, "two", llm=script)

    assert conversation(script.seen[0]) == [("HumanMessage", "one"), ("HumanMessage", "two")]
    assert rebuilds == [1]


def test_a_turn_stopped_before_its_model_call_leaves_nothing_to_rebuild(monkeypatch):
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    session.turn_generation = 5

    first = run_agent_turn(session, "one", llm=Script([say("never asked")]), turn_generation=4)  # already superseded
    assert first["superseded"] is True

    script = Script([say("Okay.")])
    run_agent_turn(session, "two", llm=script)

    assert conversation(script.seen[0]) == [("HumanMessage", "one"), ("HumanMessage", "two")]
    assert rebuilds == []  # only an empty placeholder was saved, and it is simply dropped


def test_an_earlier_reply_is_shown_to_the_model_as_plain_text_even_if_it_arrived_as_content_blocks(monkeypatch):
    """A real model reply can come back as content blocks with metadata attached. What the model is
    shown for an earlier turn must be exactly its text, the same as the transcript."""
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    fancy = AIMessage(
        content=[{"type": "text", "text": "Hello there."}],
        additional_kwargs={"signature": "abc"},
        response_metadata={"finish_reason": "STOP"},
    )
    script = Script([fancy, say("Okay.")])

    run_agent_turn(session, "one", llm=script)
    run_agent_turn(session, "two", llm=script)

    earlier_reply = script.seen[1][2]  # persona, patient "one", then Ava's earlier reply
    assert earlier_reply.content == "Hello there."
    assert not earlier_reply.additional_kwargs and not earlier_reply.response_metadata
    assert rebuilds == []


def test_an_empty_reply_in_the_transcript_does_not_break_the_match(monkeypatch):
    rebuilds = count_rebuilds(monkeypatch)
    session = create_session("s1", PROTOCOL)
    script = Script([say(""), say("Okay.")])

    run_agent_turn(session, "one", llm=script)
    run_agent_turn(session, "two", llm=script)

    assert rebuilds == []
    assert conversation(script.seen[1]) == [("HumanMessage", "one"), ("HumanMessage", "two")]


def test_a_one_turn_note_is_sent_for_that_turn_only_and_never_saved():
    session = create_session("s1", PROTOCOL)
    script = Script([say("A."), say("B.")])

    run_agent_turn(session, "one", llm=script, system_note="NOTE-FOR-THIS-TURN")
    run_agent_turn(session, "two", llm=script)

    assert "NOTE-FOR-THIS-TURN" in [m.content for m in script.seen[0] if isinstance(m, SystemMessage)]
    assert "NOTE-FOR-THIS-TURN" not in [m.content for m in script.seen[1] if isinstance(m, SystemMessage)]
    saved = get_graph().get_state(thread(session)).values["messages"]
    assert not any(isinstance(m, SystemMessage) for m in saved)  # neither the persona nor any note is stored


def test_the_synthetic_opening_message_is_never_saved_into_the_conversation():
    session = create_session("s1", PROTOCOL)
    script = Script([say("Hi, I see you are coming in about a cough. When did it start?"), say("Thanks.")])

    run_opening_turn(session, "persistent cough", llm=script)
    saved = get_graph().get_state(thread(session)).values["messages"]
    assert [type(m).__name__ for m in saved] == ["AIMessage"]  # just Ava's line, not the trigger

    run_agent_turn(session, "Two weeks ago", llm=script)

    assert "[This is the start of the call" in script.seen[0][1].content  # the trigger was sent on the opening call
    assert conversation(script.seen[1]) == [
        ("AIMessage", "Hi, I see you are coming in about a cough. When did it start?"), ("HumanMessage", "Two weeks ago"),
    ]


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
    shown = conversation(script.seen[-1])  # the model was shown the read-back Ava spoke, then the patient's reply
    assert shown[-2][0] == "AIMessage" and shown[-2][1].startswith("Before I finish, let me read back what I have.")
    assert shown[-1] == ("HumanMessage", "No, I'm also allergic to penicillin.")
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
