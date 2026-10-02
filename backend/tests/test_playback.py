"""A question counts as asked only when the patient has heard all of it: the
browser reports that the reply's audio played to the end. If the patient talks
over Ava first, the question is never counted, and Ava asks it again."""

import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage

import app.turn_controller as turn_controller
from app.agent.graph import run_agent_turn, run_opening_turn
from app.agent.session import (
    CutOffQuestion,
    create_session,
    question_was_cut_off,
    question_was_heard,
)
from app.agent.tools import create_tool_handlers
from app.schemas.protocol_config import ProtocolConfig
from app.turn_controller import TranscriptionResult, handle_utterance

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def _tool_call(name: str, args: dict, call_id: str = "c") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _AsksThenRecordsLLM:
    """Turn 1: look up the next question and say it aloud. Any later turn: record
    a bare "No" against that question's field, then say "Understood." It also
    keeps every message list it was given, so tests can see what it was told."""

    def __init__(self, session, patient_quote: str):
        self.session = session
        self.patient_quote = patient_quote
        self.calls = 0
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(messages)
        self.calls += 1
        if self.calls == 1:
            return _tool_call("get_next_intake_question", {})
        if self.calls == 2:
            return AIMessage(content="Have you had any fever?")
        if self.calls == 3:
            field = self.session.question_events[0].field
            args = {"field": field, "polarity": "absent", "evidence": self.patient_quote, "confidence": 0.9}
            return _tool_call("update_intake_record", args)
        return AIMessage(content="Understood.")


def _ask_one_question(session, llm):
    run_agent_turn(session, "hello", llm=llm, await_playback=True)
    return session.question_events[0]


# --- when the question is registered ---------------------------------------------------------


def test_a_reply_that_is_final_but_has_not_played_yet_does_not_count_as_asked():
    session = create_session("s1", PROTOCOL)
    event = _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))

    assert event.asked_in_turn is None
    assert event.spoken_text is None
    assert session.playing_question is not None


def test_the_question_counts_the_moment_the_audio_finishes_and_not_before_or_after():
    session = create_session("s1", PROTOCOL)
    event = _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))

    assert question_was_heard(session) is True
    assert event.asked_in_turn == 1  # transcript: [patient hello, agent question]
    assert event.spoken_text == "Have you had any fever?"
    assert session.playing_question is None

    # a repeated or stray report changes nothing
    assert question_was_heard(session) is False
    assert event.asked_in_turn == 1


def test_a_client_that_never_reports_playback_never_gets_a_question_counted():
    """The failure that must be safe: no report means not counted, so nothing the
    patient says afterwards can be recorded as an answer to that question."""
    session = create_session("s1", PROTOCOL)
    llm = _AsksThenRecordsLLM(session, "No, not at all")
    _ask_one_question(session, llm)

    run_agent_turn(session, "No, not at all.", llm=llm, await_playback=True)

    assert [f.source.value for f in session.record.facts] == ["patient_reported"]


def test_the_opening_line_s_question_waits_for_playback_too():
    session = create_session("s1", PROTOCOL)
    llm = _ScriptedLLM([_tool_call("get_next_intake_question", {}), AIMessage(content="Hi! When did it start?")])

    run_opening_turn(session, "persistent cough", llm=llm, await_playback=True)
    assert session.question_events[0].asked_in_turn is None

    assert question_was_heard(session) is True
    assert session.question_events[0].asked_in_turn == 0


class _ScriptedLLM:
    def __init__(self, script):
        self.script = script
        self.n = 0
        self.seen: list[list] = []

    def invoke(self, messages):
        self.seen.append(messages)
        reply = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return reply


# --- a barge-in ------------------------------------------------------------------------------


def test_a_barge_in_before_the_audio_finishes_means_the_question_is_never_counted():
    session = create_session("s1", PROTOCOL)
    event = _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))

    question_was_cut_off(session)

    assert event.asked_in_turn is None
    assert session.playing_question is None
    assert session.cut_off_question is not None
    assert session.cut_off_question.field == event.field
    assert question_was_heard(session) is False  # a late report cannot revive it


def test_a_bare_no_in_the_interruption_is_not_taken_as_a_denial():
    """The interrupting patient says "No, wait, one more thing". They never heard the
    question, so that "No" is not an answer to it."""
    session = create_session("s1", PROTOCOL)
    llm = _AsksThenRecordsLLM(session, "No")
    _ask_one_question(session, llm)
    question_was_cut_off(session)

    run_agent_turn(session, "No, wait, one more thing.", llm=llm, await_playback=True)

    assert [f.source.value for f in session.record.facts] == ["patient_reported"]  # not asked_and_denied


def test_the_same_no_after_a_question_heard_in_full_is_a_denial():
    session = create_session("s1", PROTOCOL)
    llm = _AsksThenRecordsLLM(session, "No, not at all")
    _ask_one_question(session, llm)
    question_was_heard(session)

    run_agent_turn(session, "No, not at all.", llm=llm, await_playback=True)

    assert [f.source.value for f in session.record.facts] == ["asked_and_denied"]


def test_a_new_reply_that_replaces_one_still_playing_cuts_the_old_one_off():
    """A document upload starts a turn while the previous reply may still be playing;
    the browser stops that audio when the new reply arrives."""
    session = create_session("s1", PROTOCOL)
    event = _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))

    run_agent_turn(session, "I uploaded a file.", llm=_ScriptedLLM([AIMessage(content="Thanks.")]), await_playback=True)

    assert event.asked_in_turn is None
    assert session.playing_question is None  # the new reply asked nothing, and the old one is no longer "playing"


# --- asking it again -------------------------------------------------------------------------


def test_the_next_turn_is_told_which_question_was_cut_off_and_to_ask_it_again():
    session = create_session("s1", PROTOCOL)
    llm = _AsksThenRecordsLLM(session, "No")
    event = _ask_one_question(session, llm)
    question_was_cut_off(session)
    llm.seen.clear()

    run_agent_turn(session, "Actually, one more thing.", llm=llm, await_playback=True)

    notes = [m.content for m in llm.seen[0] if isinstance(m, SystemMessage)]
    cut_off_notes = [n for n in notes if "cut off" in n]
    assert len(cut_off_notes) == 1
    assert event.question_text in cut_off_notes[0]
    assert "Have you had any fever?" in cut_off_notes[0]
    assert "ask that question again" in cut_off_notes[0]


def test_the_note_lasts_for_one_turn_only():
    session = create_session("s1", PROTOCOL)
    llm = _AsksThenRecordsLLM(session, "No")
    _ask_one_question(session, llm)
    question_was_cut_off(session)

    run_agent_turn(session, "Actually, one more thing.", llm=llm, await_playback=True)
    assert session.cut_off_question is None

    later = _ScriptedLLM([AIMessage(content="Right.")])
    run_agent_turn(session, "Okay.", llm=later, await_playback=True)
    assert not [m for m in later.seen[0] if isinstance(m, SystemMessage) and "cut off" in m.content]


def test_get_next_intake_question_suggests_the_cut_off_field_even_when_the_ranker_prefers_another():
    session = create_session("s1", PROTOCOL)
    session.cut_off_question = CutOffQuestion(
        event_id="q1", field="medication_allergies", label="Medication allergies", reply_text="Any allergies?"
    )

    class Ranker:
        def invoke(self, messages):
            return AIMessage(content="fever")  # would normally win

    result = create_tool_handlers(session, llm=Ranker())["get_next_intake_question"]({})

    assert result.data["suggested"]["field"] == "medication_allergies"


def test_a_cut_off_question_that_has_since_been_answered_is_not_asked_again():
    session = create_session("s1", PROTOCOL)
    session.cut_off_question = CutOffQuestion(event_id="q1", field="onset", label="Onset", reply_text="When did it start?")
    handlers = create_tool_handlers(session)
    session.transcript.append(_patient_turn("It started 10 days ago."))
    handlers["update_intake_record"](
        {"field": "onset", "polarity": "present", "value": "10 days ago", "evidence": "10 days ago", "confidence": 0.9}
    )

    result = handlers["get_next_intake_question"]({})

    assert result.data["suggested"]["field"] != "onset"


def _patient_turn(text: str):
    from app.schemas.intake_record import TranscriptTurn

    return TranscriptTurn(id="p1", speaker="patient", text=text, timestamp="t")


# --- an interruption with no words -----------------------------------------------------------


def _stub_audio(monkeypatch, transcript_text: str):
    monkeypatch.setattr(turn_controller, "transcribe_utterance", lambda audio: TranscriptionResult(text=transcript_text, confidence=0.9))
    monkeypatch.setattr(turn_controller, "synthesize_speech", lambda text: b"audio")


def test_a_barge_in_that_says_nothing_makes_ava_repeat_what_she_was_saying(monkeypatch):
    _stub_audio(monkeypatch, "")  # a cough: nothing to transcribe
    session = create_session("s1", PROTOCOL)
    event = _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))
    question_was_cut_off(session)

    outcome = handle_utterance(session, b"cough")

    assert outcome.superseded is False
    assert outcome.reply_text == "Have you had any fever?"
    assert outcome.audio == b"audio"
    assert [t.text for t in session.transcript if t.speaker == "agent"] == ["Have you had any fever?"] * 2
    assert event.asked_in_turn is None  # the first telling was never heard

    assert question_was_heard(session) is True  # the repeat played to the end
    assert event.asked_in_turn == len(session.transcript) - 1


def test_silence_with_nothing_cut_off_stays_silent(monkeypatch):
    _stub_audio(monkeypatch, "")
    session = create_session("s1", PROTOCOL)

    outcome = handle_utterance(session, b"silence")

    assert outcome.superseded is True
    assert outcome.reply_text == ""


def test_a_repeat_is_dropped_if_a_newer_utterance_has_already_arrived(monkeypatch):
    _stub_audio(monkeypatch, "")
    session = create_session("s1", PROTOCOL)
    _ask_one_question(session, _AsksThenRecordsLLM(session, "No"))
    question_was_cut_off(session)
    session.turn_generation = 5

    outcome = handle_utterance(session, b"cough", generation=4)  # stale

    assert outcome.superseded is True
    assert session.cut_off_question is not None  # still owed, for the newer turn
