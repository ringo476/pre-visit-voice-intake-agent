"""A patient's new statement that disagrees with an answer already on record is
never written on that one statement. Ava asks which is right; only the answer to
that question, once it was heard, changes the record, and the old answer stays in
history."""

import json
import os
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.agent.graph import run_agent_turn
from app.agent.session import create_session, question_was_cut_off, question_was_heard
from app.agent.tools import create_tool_handlers
from app.schemas.document import UploadedDocument
from app.schemas.intake_record import Fact, FactStatus, Polarity, Source, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import classify_change, get_current_facts

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_PROTOCOL_PATH = Path(__file__).parent.parent / "app" / "protocol" / "respiratory_intake.json"
with open(_PROTOCOL_PATH, "r", encoding="utf-8") as f:
    PROTOCOL = ProtocolConfig(**json.load(f))


def _say(session, speaker: str, text: str) -> int:
    session.transcript.append(TranscriptTurn(id=f"t{len(session.transcript)}", speaker=speaker, text=text, timestamp="t"))
    return len(session.transcript) - 1


def _yes(field: str, value: str, evidence: str) -> dict:
    return {"field": field, "polarity": "present", "value": value, "evidence": evidence, "confidence": 0.9}


def _no(field: str, evidence: str) -> dict:
    return {"field": field, "polarity": "absent", "evidence": evidence, "confidence": 0.9}


def _dont_know(field: str, evidence: str) -> dict:
    return {"field": field, "polarity": "unknown", "evidence": evidence, "confidence": 0.9}


def _correction(field: str, polarity: str, evidence: str, value: str = "") -> dict:
    return {"field": field, "polarity": polarity, "new_value": value, "evidence": evidence, "confidence": 0.9}


class _Verifier:
    """Stands in for the separate verifier model: answers every claim with one verdict
    and keeps the prompts it was shown."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.prompts: list[str] = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        n = len(re.findall(r"(?m)^Claim \d+$", prompt))
        return AIMessage(content="\n".join(f"{i}: {self.verdict}" for i in range(1, n + 1)))


def _fever_on_record(spoken_fever_question: bool = True):
    """A session where the patient said they have had a fever, and it is on record."""
    session = create_session("s1", PROTOCOL)
    _say(session, "agent", "Have you had any fever?")
    _say(session, "patient", "Yes, I've had a fever since Tuesday.")
    result = create_tool_handlers(session)["update_intake_record"](_yes("fever", "fever since Tuesday", "I've had a fever since Tuesday"))
    assert result.ok and result.data["recorded"]
    return session, result.data["fact_id"]


def _patient_now_says_no(session):
    _say(session, "agent", "Okay, thanks. Anything else with the cough?")
    _say(session, "patient", "No, I don't have a fever.")


def _ava_asks_which_is_right_and_it_is_heard(session, event):
    """What really happens around the question: Ava says it, the audio plays to the end."""
    index = _say(session, "agent", "Just to be sure about fever: earlier I noted yes, fever since Tuesday, and just now it sounded like no. Which is right?")
    event.asked_in_turn, event.spoken_text = index, session.transcript[index].text
    return index


# --- the disagreement is not written ---------------------------------------------------------


def test_a_no_against_a_recorded_yes_is_not_written_and_a_question_is_logged():
    session, fact_id = _fever_on_record()
    _patient_now_says_no(session)
    handlers = create_tool_handlers(session)

    result = handlers["update_intake_record"](_no("fever", "I don't have a fever"))

    assert result.ok is True
    assert result.data["recorded"] is False
    assert result.data["needs_confirmation"] is True
    assert len(session.record.facts) == 1  # nothing changed
    assert get_current_facts(session.record)["fever"].polarity == Polarity.PRESENT

    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    assert event.confirms_fact_id == fact_id and event.field == "fever"
    assert event.asked_in_turn is None  # not heard yet

    message = result.data["message"]
    assert "Which is right" in message
    assert "yes, fever since Tuesday" in message and "no" in message
    assert "record_patient_correction" in message


def test_the_correction_tool_is_held_to_the_same_rule():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)

    result = create_tool_handlers(session)["record_patient_correction"](_correction("fever", "absent", "I don't have a fever"))

    assert result.ok is True and result.data["needs_confirmation"] is True
    assert len(session.record.facts) == 1


def test_asking_twice_in_one_turn_logs_one_question():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)
    handlers = create_tool_handlers(session, verifier_llm=_Verifier("SUPPORTED"))
    call = {"name": "update_intake_record", "args": _no("fever", "I don't have a fever"), "id": "1"}

    handlers.prepare([call])
    handlers["update_intake_record"](call["args"])
    handlers["record_patient_correction"](_correction("fever", "absent", "I don't have a fever"))

    assert len([e for e in session.question_events if e.confirms_fact_id]) == 1


def test_an_invented_quote_cannot_make_ava_ask_anything():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)

    result = create_tool_handlers(session)["update_intake_record"](_no("fever", "I never had any fever whatsoever"))

    assert result.ok is False
    assert [e for e in session.question_events if e.confirms_fact_id] == []


# --- the patient's answer changes it ---------------------------------------------------------


def test_after_the_question_is_heard_the_patients_answer_replaces_the_old_one_and_keeps_it_in_history():
    session, old_id = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", "No fever. I mixed it up with my sister.")

    result = create_tool_handlers(session)["record_patient_correction"](_correction("fever", "absent", "No fever"))

    assert result.ok is True and result.data["corrected"] is True
    assert [f.polarity for f in session.record.facts] == [Polarity.PRESENT, Polarity.ABSENT]  # history kept
    new = session.record.facts[1]
    assert new.supersedes == old_id and new.status == FactStatus.CORRECTED
    assert new.source == Source.PATIENT_REPORTED  # not asked_and_denied: it is a change, not a first answer
    assert get_current_facts(session.record)["fever"] is new


def test_the_update_tool_works_for_the_confirmed_change_too():
    session, old_id = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", "The second one, no fever.")

    result = create_tool_handlers(session)["update_intake_record"](_no("fever", "no fever"))

    assert result.ok is True and result.data["recorded"] is True
    assert session.record.facts[1].supersedes == old_id


def test_a_question_that_was_not_heard_to_the_end_does_not_unlock_the_change():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    # Ava's question was cut off or never played: its event was never stamped.
    _say(session, "agent", "Just to be sure about fever: which is right?")
    _say(session, "patient", "No fever.")

    result = create_tool_handlers(session)["record_patient_correction"](_correction("fever", "absent", "No fever"))

    assert result.data["needs_confirmation"] is True
    assert len(session.record.facts) == 1
    assert len([e for e in session.question_events if e.confirms_fact_id]) == 2  # asked again, in a new turn


def test_words_said_before_the_question_do_not_answer_it():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)  # "No, I don't have a fever." comes first
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", "Hmm, let me think.")

    result = create_tool_handlers(session)["record_patient_correction"](_correction("fever", "absent", "I don't have a fever"))

    assert result.data["needs_confirmation"] is True  # the quote is from before the question
    assert len(session.record.facts) == 1


def test_a_confirmation_belongs_to_the_answer_it_was_about():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", "No fever.")
    create_tool_handlers(session)["record_patient_correction"](_correction("fever", "absent", "No fever"))

    # later the patient flips again: the old confirmation was about the OLD answer
    _say(session, "agent", "Okay.")
    _say(session, "patient", "Actually I did have a fever last night.")
    result = create_tool_handlers(session)["update_intake_record"](_yes("fever", "fever last night", "I did have a fever last night"))

    assert result.data["needs_confirmation"] is True
    assert len(session.record.facts) == 2


def test_if_the_patient_keeps_the_earlier_answer_nothing_changes():
    session, _ = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", "Fever, since Tuesday. Sorry, I misspoke.")

    result = create_tool_handlers(session)["update_intake_record"](_yes("fever", "fever since Tuesday", "Fever, since Tuesday"))

    assert result.ok is True and result.data["recorded"] is False and result.data["already_recorded"] is True
    assert len(session.record.facts) == 1


# --- what is not a disagreement --------------------------------------------------------------


def test_saying_the_same_thing_again_adds_nothing():
    session, _ = _fever_on_record()
    _say(session, "agent", "Any chills with it?")
    _say(session, "patient", "Yeah I've had a fever since Tuesday, like I said.")

    result = create_tool_handlers(session)["update_intake_record"](_yes("fever", "fever since Tuesday", "I've had a fever since Tuesday"))

    assert result.data["already_recorded"] is True
    assert len(session.record.facts) == 1
    assert [e for e in session.question_events if e.confirms_fact_id] == []


def test_adding_a_detail_replaces_the_answer_without_a_question():
    session, old_id = _fever_on_record()
    _say(session, "agent", "How high did it get?")
    _say(session, "patient", "It's been a fever since Tuesday, around 101 at night.")

    result = create_tool_handlers(session)["update_intake_record"](
        _yes("fever", "fever since Tuesday, around 101 at night", "fever since Tuesday, around 101 at night")
    )

    assert result.ok is True and result.data["recorded"] is True
    assert session.record.facts[1].supersedes == old_id
    assert [e for e in session.question_events if e.confirms_fact_id] == []


def test_settling_an_i_dont_know_replaces_it_without_a_question():
    session = create_session("s1", PROTOCOL)
    _say(session, "patient", "I'm not sure about a fever. Actually no, I haven't had one.")
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"](_dont_know("fever", "I'm not sure about a fever"))

    result = handlers["update_intake_record"](_no("fever", "no, I haven't had one"))

    assert result.data["recorded"] is True
    assert get_current_facts(session.record)["fever"].polarity == Polarity.ABSENT
    assert [e for e in session.question_events if e.confirms_fact_id] == []


def test_a_yes_that_turns_into_not_sure_is_a_disagreement():
    session, _ = _fever_on_record()
    _say(session, "agent", "Is it still going on?")
    _say(session, "patient", "Honestly I'm not sure about the fever now.")

    result = create_tool_handlers(session)["update_intake_record"](_dont_know("fever", "I'm not sure about the fever now"))

    assert result.data["needs_confirmation"] is True


def test_a_different_value_is_a_disagreement():
    session = create_session("s1", PROTOCOL)
    _say(session, "patient", "It started 3 weeks ago. No wait, it was 5 days ago.")
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"](_yes("onset", "3 weeks ago", "It started 3 weeks ago"))

    result = handlers["update_intake_record"](_yes("onset", "5 days ago", "it was 5 days ago"))

    assert result.data["needs_confirmation"] is True
    assert len(session.record.facts) == 1


def test_the_patients_own_words_replace_the_booking_reason_without_a_question():
    session = create_session("s1", PROTOCOL, appointment_reason_text="persistent cough")
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"]({"field": "chief_complaint", "polarity": "present", "value": "persistent cough", "confidence": 0.9})
    assert session.record.facts[0].source == Source.INFERRED
    _say(session, "patient", "I've got a cough that will not go away, and my chest hurts.")

    result = handlers["update_intake_record"](_yes("chief_complaint", "cough that will not go away and chest pain", "a cough that will not go away"))

    assert result.data["recorded"] is True
    assert session.record.facts[1].supersedes == session.record.facts[0].id
    assert [e for e in session.question_events if e.confirms_fact_id] == []


def test_a_claim_quoted_from_a_document_is_not_the_patient_contradicting_themselves():
    session = create_session("s1", PROTOCOL)
    session.documents.append(UploadedDocument(id="d1", filename="rx.pdf", mime_type="application/pdf", text="Salbutamol inhaler 100mcg as needed", uploaded_at="t"))
    _say(session, "patient", "I use an inhaler sometimes.")
    handlers = create_tool_handlers(session)
    handlers["update_intake_record"](_yes("medications_tried", "inhaler", "I use an inhaler sometimes"))

    result = handlers["update_intake_record"](
        {"field": "medications_tried", "polarity": "present", "value": "salbutamol 100mcg", "evidence": "Salbutamol inhaler 100mcg", "confidence": 0.9}
    )

    assert result.ok is True and result.data["recorded"] is True
    assert [e for e in session.question_events if e.confirms_fact_id] == []


# --- the independent reader ------------------------------------------------------------------


def _confirmed_setup(reply: str):
    session, _ = _fever_on_record()
    _patient_now_says_no(session)
    create_tool_handlers(session)["update_intake_record"](_no("fever", "I don't have a fever"))
    (event,) = [e for e in session.question_events if e.confirms_fact_id]
    _ava_asks_which_is_right_and_it_is_heard(session, event)
    _say(session, "patient", reply)
    return session


def test_the_verifier_is_shown_the_question_the_patient_was_answering():
    session = _confirmed_setup("The second one.")
    verifier = _Verifier("SUPPORTED")

    result = create_tool_handlers(session, verifier_llm=verifier)["record_patient_correction"](
        _correction("fever", "absent", "The second one")
    )

    assert result.ok is True
    prompt = verifier.prompts[0]
    assert "Which is right?" in prompt  # without it, "the second one" means nothing
    assert "The second one." in prompt
    assert "Fever or chills" in prompt


def test_a_confirmed_change_the_verifier_rejects_is_not_made():
    session = _confirmed_setup("Hmm, I really am not sure.")

    result = create_tool_handlers(session, verifier_llm=_Verifier("CONTRADICTED"))["record_patient_correction"](
        _correction("fever", "absent", "I really am not sure")
    )

    assert result.ok is False
    assert len(session.record.facts) == 1
    assert get_current_facts(session.record)["fever"].polarity == Polarity.PRESENT


# --- how the answer is compared --------------------------------------------------------------


def _fact(polarity: Polarity, value: str) -> Fact:
    return Fact(
        id="f1", field="x", value=value, source=Source.PATIENT_REPORTED, evidence_span="e", confidence=0.9,
        status=FactStatus.UNCONFIRMED, timestamp="t", polarity=polarity,
    )


@pytest.mark.parametrize(
    "current, polarity, value, expected",
    [
        (_fact(Polarity.PRESENT, "fever since Tuesday"), Polarity.PRESENT, "Fever, since Tuesday!", "same"),
        (_fact(Polarity.PRESENT, "fever since Tuesday"), Polarity.PRESENT, "yes", "same"),
        (_fact(Polarity.PRESENT, "fever since Tuesday"), Polarity.PRESENT, "fever since Tuesday at night", "update"),
        (_fact(Polarity.PRESENT, "yes"), Polarity.PRESENT, "since Tuesday", "update"),
        (_fact(Polarity.PRESENT, "3 weeks"), Polarity.PRESENT, "5 days", "conflict"),
        (_fact(Polarity.PRESENT, "dry cough"), Polarity.PRESENT, "wet cough", "conflict"),
        (_fact(Polarity.PRESENT, "fever"), Polarity.ABSENT, "no", "conflict"),
        (_fact(Polarity.ABSENT, "no"), Polarity.PRESENT, "fever", "conflict"),
        (_fact(Polarity.ABSENT, "no"), Polarity.ABSENT, "none", "same"),
        (_fact(Polarity.PRESENT, "fever"), Polarity.UNKNOWN, "not sure", "conflict"),
        (_fact(Polarity.ABSENT, "no"), Polarity.UNKNOWN, "not sure", "conflict"),
        (_fact(Polarity.UNKNOWN, "patient does not know"), Polarity.ABSENT, "no", "update"),
        (_fact(Polarity.UNKNOWN, "patient does not know"), Polarity.PRESENT, "fever", "update"),
        (_fact(Polarity.UNKNOWN, "patient does not know"), Polarity.UNKNOWN, "no idea", "same"),
    ],
)
def test_how_a_new_answer_relates_to_the_one_on_record(current, polarity, value, expected):
    assert classify_change(current, polarity, value) == expected


def test_when_two_answers_land_in_the_same_clock_tick_the_later_one_is_current():
    session, _ = _fever_on_record()
    first = session.record.facts[0]
    later = first.model_copy(update={"id": "f2", "polarity": Polarity.ABSENT, "value": "no"})
    session.record = session.record.model_copy(update={"facts": [first, later]})

    assert first.timestamp == later.timestamp
    assert get_current_facts(session.record)["fever"].id == "f2"


# --- the whole flow, turn by turn ------------------------------------------------------------


class _Script:
    """A model that plays one scripted step per call: a list of AIMessages per turn."""

    def __init__(self, turns: list[list[AIMessage]]):
        self.turns = turns
        self.turn = -1
        self.step = 0

    def start_turn(self):
        self.turn += 1
        self.step = 0

    def invoke(self, messages):
        message = self.turns[self.turn][self.step]
        self.step += 1
        return message


def _tool(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{len(args)}"}])


def _run(session, script: _Script, patient_text: str):
    script.start_turn()
    return run_agent_turn(session, patient_text, llm=script, await_playback=True)


def test_a_full_conversation_from_yes_to_no_goes_through_the_question():
    session = create_session("s1", PROTOCOL)
    script = _Script(
        [
            [_tool("update_intake_record", _yes("fever", "fever since Tuesday", "I've had a fever since Tuesday")), AIMessage(content="Okay, thanks.")],
            [_tool("update_intake_record", _no("fever", "I don't have a fever")), AIMessage(content="Just to be sure about fever: earlier I noted yes, and just now it sounded like no. Which is right?")],
            [_tool("record_patient_correction", _correction("fever", "absent", "No fever")), AIMessage(content="Thanks, I've changed that to no fever.")],
        ]
    )

    _run(session, script, "Yes, I've had a fever since Tuesday.")
    _run(session, script, "No, I don't have a fever.")
    assert len(session.record.facts) == 1  # one statement did not change it

    assert question_was_heard(session) is True  # the "which is right?" audio played to the end
    _run(session, script, "No fever. I mixed it up.")

    assert [f.polarity for f in session.record.facts] == [Polarity.PRESENT, Polarity.ABSENT]
    assert get_current_facts(session.record)["fever"].polarity == Polarity.ABSENT


def test_if_the_patient_talks_over_the_question_the_change_waits():
    session = create_session("s1", PROTOCOL)
    script = _Script(
        [
            [_tool("update_intake_record", _yes("fever", "fever since Tuesday", "I've had a fever since Tuesday")), AIMessage(content="Okay, thanks.")],
            [_tool("update_intake_record", _no("fever", "I don't have a fever")), AIMessage(content="Just to be sure about fever: which is right?")],
            [_tool("record_patient_correction", _correction("fever", "absent", "No fever")), AIMessage(content="Which is right, a fever or no fever?")],
        ]
    )

    _run(session, script, "Yes, I've had a fever since Tuesday.")
    _run(session, script, "No, I don't have a fever.")
    question_was_cut_off(session)  # the patient started talking before it finished
    _run(session, script, "No fever.")

    assert len(session.record.facts) == 1  # a "no fever" in an interruption does not settle it
    assert len([e for e in session.question_events if e.confirms_fact_id]) == 2  # so Ava asked again
