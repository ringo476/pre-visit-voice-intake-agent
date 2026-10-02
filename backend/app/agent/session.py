"""All server-side state for one patient conversation. Tool handlers read
and mutate this; the model never sees or touches it directly, only through
the 8 tool calls."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.schemas.document import UploadedDocument
from app.schemas.intake_record import IntakeRecord, QuestionEvent, SafetyEvaluation, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import create_empty_record

UNCLASSIFIED_PROTOCOL_ID = "unclassified"


@dataclass
class AssistanceRequest:
    id: str
    reason: str
    timestamp: str


@dataclass
class PlayingQuestion:
    """A question that is inside the reply now being played to the patient.
    It is not yet counted as asked: that happens only when the browser reports
    that the audio played to the end (see question_was_heard)."""

    event_id: str
    reply_index: int  # position of Ava's reply in session.transcript


@dataclass
class CutOffQuestion:
    """A question whose reply was interrupted before it finished playing, so
    the patient did not hear all of it. It was never counted as asked, and the
    next turn must ask it again."""

    event_id: str
    field: str
    label: str
    reply_text: str


@dataclass
class SessionState:
    session_id: str
    record: IntakeRecord
    # None until the patient actually describes a complaint — see
    # app/protocol/classifier.py. Before that, the tool layer allows
    # conversation and safety checks but not protocol-dependent tools
    # (get_next_intake_question, retrieve_existing_patient_context,
    # generate_clinician_brief), which fail gracefully rather than crash.
    protocol: Optional[ProtocolConfig] = None
    question_events: list[QuestionEvent] = field(default_factory=list)
    transcript: list[TranscriptTurn] = field(default_factory=list)
    safety_log: list[SafetyEvaluation] = field(default_factory=list)
    assistance_requests: list[AssistanceRequest] = field(default_factory=list)
    documents: list[UploadedDocument] = field(default_factory=list)
    brief_finalized: bool = False
    # Bumped by main.py on every new recorded utterance (and on a client
    # barge-in signal). A turn captures this value when it starts; if it no
    # longer matches by the time a Gemini call would run, that turn has been
    # superseded by a newer one and stops itself rather than making further
    # model calls or writing facts from stale context.
    turn_generation: int = 0
    # Live-call state about the audio currently reaching the patient. Neither
    # is persisted: a dropped connection loses it, and the safe outcome of
    # losing it is that the question simply never counts as asked.
    playing_question: Optional[PlayingQuestion] = None
    cut_off_question: Optional[CutOffQuestion] = None
    # None until the patient explicitly consents to this conversation being
    # handled by an AI assistant — required before the voice call proceeds
    # at all (see main.py's /api/appointments/mock and the WebSocket's
    # rejection of a session with no consent on file).
    consent_given_at: Optional[str] = None
    # Set once, at booking time (main.py's /api/appointments/mock), and
    # persisted rather than kept only in server memory — so the exact
    # appointment reason/time survive a restart between booking and the
    # patient actually joining the call, the same durability guarantee
    # everything else in the session now has.
    appointment_reason_text: Optional[str] = None
    appointment_when_text: Optional[str] = None


def create_session(
    session_id: str,
    protocol: Optional[ProtocolConfig] = None,
    consent_given_at: Optional[str] = None,
    appointment_reason_text: Optional[str] = None,
    appointment_when_text: Optional[str] = None,
) -> SessionState:
    protocol_id = protocol.protocol_id if protocol else UNCLASSIFIED_PROTOCOL_ID
    return SessionState(
        session_id=session_id,
        protocol=protocol,
        record=create_empty_record(session_id, protocol_id),
        consent_given_at=consent_given_at,
        appointment_reason_text=appointment_reason_text,
        appointment_when_text=appointment_when_text,
    )


def assign_protocol(session: SessionState, protocol: ProtocolConfig) -> None:
    """Locks in the protocol once the patient's chief complaint has been
    classified. Only ever called once per session — a mid-conversation
    switch to a different complaint type is deliberately NOT supported;
    the agent is instructed to acknowledge it and defer it instead (see
    instructions.py), not silently reassign the checklist mid-flow."""
    session.protocol = protocol
    session.record = session.record.model_copy(update={"protocol_id": protocol.protocol_id})


def mark_question_asked(session: SessionState, event: QuestionEvent, reply_index: int) -> None:
    """Counts `event` as asked: Ava's reply at `reply_index` in the transcript
    is the turn that carried it, and its text is what was actually said."""
    event.asked_in_turn = reply_index
    event.spoken_text = session.transcript[reply_index].text


def _event_by_id(session: SessionState, event_id: str) -> Optional[QuestionEvent]:
    return next((e for e in session.question_events if e.id == event_id), None)


def question_is_playing(session: SessionState, event_id: str, reply_index: int) -> None:
    """The reply carrying this question has just been sent to the patient as
    audio. It does not count as asked until question_was_heard."""
    session.playing_question = PlayingQuestion(event_id=event_id, reply_index=reply_index)


def question_was_heard(session: SessionState) -> bool:
    """The browser reports the reply played to the end, so the question in it
    has now been heard in full and counts as asked. Returns True when a
    question was counted, so the caller knows to save the session."""
    playing, session.playing_question = session.playing_question, None
    if playing is None or playing.reply_index >= len(session.transcript):
        return False
    event = _event_by_id(session, playing.event_id)
    if event is None:
        return False
    mark_question_asked(session, event, playing.reply_index)
    return True


def question_was_cut_off(session: SessionState) -> None:
    """The patient started talking, or a newer reply replaced this one, before
    the audio finished. The question stays uncounted and is remembered so the
    next turn asks it again."""
    playing, session.playing_question = session.playing_question, None
    if playing is None or playing.reply_index >= len(session.transcript):
        return
    event = _event_by_id(session, playing.event_id)
    if event is None:
        return
    session.cut_off_question = CutOffQuestion(
        event_id=event.id,
        field=event.field,
        label=event.question_text,
        reply_text=session.transcript[playing.reply_index].text,
    )


def repeat_cut_off_question(session: SessionState) -> Optional[str]:
    """For a barge-in that produced no words (a cough, a noise): there is
    nothing to reply to, so Ava says the interrupted message again. Returns the
    text to speak, or None if nothing was cut off. The repeat is its own turn in
    the transcript, and its question counts only once it plays to the end."""
    cut, session.cut_off_question = session.cut_off_question, None
    if cut is None:
        return None
    session.transcript.append(
        TranscriptTurn(id=str(uuid.uuid4()), speaker="agent", text=cut.reply_text, timestamp=datetime.now(timezone.utc).isoformat())
    )
    question_is_playing(session, cut.event_id, len(session.transcript) - 1)
    return cut.reply_text
