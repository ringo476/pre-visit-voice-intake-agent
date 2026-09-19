"""All server-side state for one patient conversation. Tool handlers read
and mutate this; the model never sees or touches it directly, only through
the 8 tool calls."""

from dataclasses import dataclass, field
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
