"""Builds the handler map for one session. Every handler validates its args
against the Pydantic schema, calls exactly one backend module (state
engine, safety engine, or RAG retriever), and mutates `session` in place.
The model itself has no path to state other than through these handlers."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from app.agent.answer_verifier import AnswerVerificationError, verify_answer
from app.agent.question_prioritizer import rank_next_field
from app.agent.session import AssistanceRequest, SessionState
from app.documents.document_store import retrieve_from_documents
from app.rag.retriever import retrieve_follow_up_guidance, retrieve_prior_chart
from app.policy_engine import SafetyResult, build_evaluation_log_entry, evaluate_fact, evaluate_statement
from app.schemas.intake_record import QuestionEvent, Source
from app.schemas.tool_schemas import (
    CheckSafetyProtocolArgs,
    GenerateClinicianBriefArgs,
    GetNextIntakeQuestionArgs,
    RecordPatientCorrectionArgs,
    RequestHumanAssistanceArgs,
    RetrieveExistingPatientContextArgs,
    RetrieveUploadedDocumentArgs,
    UpdateIntakeRecordArgs,
)
from app.state_engine import (
    SPOKEN_QUESTION_SOURCES,
    ProvenanceViolationError,
    apply_fact,
    evidence_follows_question,
    evidence_in_patient_speech,
    evidence_in_text,
    find_asked_event,
    get_current_facts,
    get_missing_fields,
    looks_like_denial,
    patient_reply_after,
    patient_turns_containing,
    record_correction,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

EVIDENCE_REQUIRED_SOURCES = {Source.PATIENT_REPORTED, Source.ASKED_AND_DENIED, Source.UNCERTAIN, Source.DOCUMENT_SOURCED}


@dataclass
class ToolResult:
    ok: bool
    data: Optional[dict] = None
    error: Optional[str] = None


def _ok(data: dict) -> ToolResult:
    return ToolResult(ok=True, data=data)


def _fail(error: str) -> ToolResult:
    return ToolResult(ok=False, error=error)


def _safety_payload(result: SafetyResult) -> Optional[dict]:
    if not result.triggered:
        return None
    return {"triggered": True, "action": result.action, "scripted_message": result.scripted_message}


ToolHandler = Callable[[dict], ToolResult]


# What an independent reading of the patient's reply must say for each label
# the main model can choose. See answer_verifier.py.
_EXPECTED_VERDICT = {Source.ASKED_AND_DENIED: "negative", Source.UNCERTAIN: "unsure"}
_VERDICT_MEANING = {
    "negative": 'a clear "no" (record it as asked_and_denied)',
    "unsure": 'an "I don\'t know" (record it as uncertain)',
    "other": 'neither a clear "no" nor an "I don\'t know" — for example a "yes", details, or an unclear answer (record what they actually said as patient_reported, or ask again)',
}


def create_tool_handlers(
    session: SessionState,
    llm: Optional[BaseChatModel] = None,
    verifier_llm: Optional[BaseChatModel] = None,
) -> dict[str, ToolHandler]:
    def _label_for(field: str) -> str:
        if session.protocol is not None:
            return next((f.label for f in session.protocol.fields if f.field == field), field)
        return field

    def _meaning_mismatch(
        field: str, question: Optional[str], patient_reply: str, expected: str, claimed: str
    ) -> Optional[ToolResult]:
        """Asks the separate verifier model what the patient's reply actually
        was and compares it with what the main model claimed. Returns a failing
        ToolResult on a mismatch or if the verifier cannot answer (fail
        closed), or None when the claim holds up. Skipped only when no
        verifier is configured (offline tests / no API key)."""
        if verifier_llm is None:
            return None
        try:
            verdict = verify_answer(
                topic=_label_for(field), question=question, patient_reply=patient_reply, llm=verifier_llm
            )
        except AnswerVerificationError as e:
            logger.warning(
                "answer verification unavailable, refusing the save",
                extra={"session_id": session.session_id, "field": field, "error": str(e)},
            )
            return _fail(
                f'Could not verify the patient\'s answer for "{field}", so it was not recorded. '
                f"Ask the patient again and confirm what they meant."
            )
        if verdict != expected:
            logger.info(
                "answer verification mismatch",
                extra={"session_id": session.session_id, "field": field, "claimed": claimed, "verdict": verdict},
            )
            return _fail(
                f'Cannot record "{field}" as {claimed}: an independent reading of the '
                f"patient's reply says it is {_VERDICT_MEANING[verdict]}."
            )
        return None

    def _check_patient_statement(field: str, value: str, evidence: Optional[str], claimed: str) -> Optional[ToolResult]:
        """Checks for a fact recorded as the patient's own words (a new
        patient_reported fact or a correction): the quote must really be
        something the patient said, and a recorded "no" must really be a "no"
        in the sentence it came from. The meaning check applies only to
        denial-like values, because a false absence is the clinically
        dangerous direction and checking every fact would add a model call per
        fact."""
        if not evidence_in_patient_speech(session.transcript, evidence or ""):
            logger.info("quote not found in patient speech", extra={"session_id": session.session_id, "field": field})
            return _fail(
                f'Cannot record "{field}" as {claimed}: the evidence quote was not found in anything the '
                f"patient has said. Quote their actual words verbatim."
            )
        if looks_like_denial(value):
            spoken = find_asked_event(session.question_events, field, before_turn=len(session.transcript))
            return _meaning_mismatch(
                field,
                spoken.spoken_text if spoken else None,
                patient_turns_containing(session.transcript, evidence or ""),
                expected="negative",
                claimed=claimed,
            )
        return None

    def update_intake_record(raw_args: dict) -> ToolResult:
        try:
            args = UpdateIntakeRecordArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if args.source in EVIDENCE_REQUIRED_SOURCES and not args.evidence:
            return _fail(f'Source "{args.source.value}" requires a verbatim evidence quote')

        # Once a protocol is locked in, reject facts for fields outside its
        # checklist — otherwise they'd be written successfully but then
        # silently vanish from the clinician brief, which only ever renders
        # fields it recognizes from the active protocol.
        if session.protocol is not None:
            known_fields = {f.field for f in session.protocol.fields}
            if args.field not in known_fields:
                return _fail(
                    f'"{args.field}" is not part of the current intake ({session.protocol.name}). '
                    f"If this is a separate concern, acknowledge it to the patient and let them know "
                    f"to raise it after this conversation or with their clinician — do not record it here."
                )

        # Labels the model could otherwise assert on its own word. Each one
        # claims a specific origin, so each is checked against that origin.
        if args.source == Source.PATIENT_REPORTED:
            rejected = _check_patient_statement(args.field, args.value, args.evidence, args.source.value)
            if rejected:
                return rejected
        elif args.source == Source.DOCUMENT_SOURCED:
            if not session.documents:
                return _fail(
                    f'Cannot record "{args.field}" as document_sourced: no document has been uploaded in this '
                    f"conversation. Record only what the patient said, or ask them."
                )
            if not any(evidence_in_text(d.text, args.evidence or "") for d in session.documents):
                return _fail(
                    f'Cannot record "{args.field}" as document_sourced: the evidence quote was not found in the '
                    f"text of any uploaded document. Quote the document text exactly (retrieve_uploaded_document)."
                )
        elif args.source == Source.INFERRED:
            if not (args.field == "chief_complaint" and session.appointment_reason_text):
                return _fail(
                    f'Cannot record "{args.field}" as inferred. "inferred" is only for the visit reason taken '
                    f"from the booking (chief_complaint). Record only what the patient said or a document says; "
                    f"if it has not come up yet, ask."
                )

        # A denial ("no") or an "I don't know" is only valid in answer to a
        # question the agent really spoke, so the server resolves that
        # question itself instead of trusting an id from the model (see
        # find_asked_event). Anything the patient volunteers unprompted is
        # recorded as patient_reported and never goes through this path.
        question_event_id = None
        if args.source in SPOKEN_QUESTION_SOURCES:
            last_patient_turn = max(
                (i for i, t in enumerate(session.transcript) if t.speaker == "patient"), default=-1
            )
            event = find_asked_event(session.question_events, args.field, before_turn=last_patient_turn)
            if event is None:
                return _fail(
                    f'Cannot record "{args.field}" as {args.source.value}: no question about it has been '
                    f"asked to the patient yet. Ask it first (get_next_intake_question), or if the patient "
                    f"volunteered this themselves, record it as patient_reported."
                )
            if not evidence_follows_question(session.transcript, event, args.evidence or ""):
                return _fail(
                    f'Cannot record "{args.field}" as {args.source.value}: the evidence quote was not found '
                    f"in what the patient said after being asked. Quote their actual words verbatim."
                )
            question_event_id = event.id

            # Meaning check, by a separate model: the quote is real, but does the
            # reply actually say what the main model labelled it, about this topic?
            rejected = _meaning_mismatch(
                args.field,
                event.spoken_text or event.question_text,
                patient_reply_after(session.transcript, event),
                expected=_EXPECTED_VERDICT[args.source],
                claimed=args.source.value,
            )
            if rejected:
                return rejected

        try:
            session.record = apply_fact(
                session.record,
                args.field,
                args.value,
                args.source,
                args.evidence,
                args.confidence,
                session.question_events,
                question_event_id=question_event_id,
            )
        except ProvenanceViolationError as e:
            return _fail(str(e))

        new_fact = session.record.facts[-1]
        safety = evaluate_fact(new_fact)
        session.safety_log.append(build_evaluation_log_entry(new_fact.id, safety))
        return _ok({"recorded": True, "fact_id": new_fact.id, "safety": _safety_payload(safety)})

    def record_patient_correction(raw_args: dict) -> ToolResult:
        try:
            args = RecordPatientCorrectionArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        rejected = _check_patient_statement(args.field, args.new_value, args.evidence, "patient_reported")
        if rejected:
            return rejected

        try:
            session.record = record_correction(
                session.record, args.field, args.new_value, args.evidence, args.confidence
            )
        except ProvenanceViolationError as e:
            return _fail(str(e))

        new_fact = session.record.facts[-1]
        safety = evaluate_fact(new_fact)
        session.safety_log.append(build_evaluation_log_entry(new_fact.id, safety))
        return _ok({"corrected": True, "fact_id": new_fact.id, "safety": _safety_payload(safety)})

    def check_safety_protocol(raw_args: dict) -> ToolResult:
        try:
            args = CheckSafetyProtocolArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if args.statement:
            result = evaluate_statement(args.statement)
            session.safety_log.append(build_evaluation_log_entry("statement-check", result))
            return _ok({"triggered": result.triggered, "action": result.action, "scripted_message": result.scripted_message})

        # No statement given: re-scan every current fact in case something
        # concerning was recorded without an explicit safety check.
        for fact in get_current_facts(session.record).values():
            result = evaluate_fact(fact)
            if result.triggered:
                session.safety_log.append(build_evaluation_log_entry(fact.id, result))
                return _ok({"triggered": True, "action": result.action, "scripted_message": result.scripted_message})
        return _ok({"triggered": False})

    def get_next_intake_question(raw_args: dict) -> ToolResult:
        try:
            GetNextIntakeQuestionArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if session.protocol is None:
            return _ok(
                {
                    "missing_fields": [],
                    "done": False,
                    "message": "No specific complaint has been identified yet. Ask the patient what brings them in today before using this tool.",
                }
            )

        missing = get_missing_fields(session.record, session.protocol)
        if not missing:
            return _ok(
                {
                    "missing_fields": [],
                    "done": True,
                    "message": (
                        "Everything required has been covered. Before finishing, read back the key facts "
                        "you've gathered to the patient in your own words and ask them to confirm or correct "
                        "anything, then call generate_clinician_brief."
                    ),
                }
            )

        next_field = missing[0]
        if llm is not None:
            # Advisory reordering only — see question_prioritizer.py for the
            # guardrail: any failure here silently keeps the fixed-order
            # default above, and nothing is ever dropped from `missing`,
            # only reordered for which gets asked this turn.
            ranked = rank_next_field(missing, get_current_facts(session.record), llm)
            if ranked:
                next_field = next(m for m in missing if m.field == ranked)

        guidance_results = retrieve_follow_up_guidance(
            session.protocol.protocol_id, f"{next_field.label} {next_field.field}", k=1
        )
        guidance_text = guidance_results[0][0].page_content if guidance_results else None

        question_event = QuestionEvent(
            id=str(uuid.uuid4()),
            field=next_field.field,
            question_text=next_field.label,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        session.question_events.append(question_event)

        return _ok(
            {
                "missing_fields": [m.model_dump() for m in missing],
                "suggested": {
                    "field": next_field.field,
                    "label": next_field.label,
                    "guidance": guidance_text,
                },
                "done": False,
            }
        )

    def retrieve_existing_patient_context(raw_args: dict) -> ToolResult:
        try:
            args = RetrieveExistingPatientContextArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if session.protocol is None:
            return _ok({"results": [], "message": "No specific complaint has been identified yet."})

        query = args.query or session.protocol.name
        results = retrieve_prior_chart(session.protocol.protocol_id, query, k=2)
        return _ok({"results": [{"text": doc.page_content, "score": score} for doc, score in results]})

    def retrieve_uploaded_document(raw_args: dict) -> ToolResult:
        try:
            args = RetrieveUploadedDocumentArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if not session.documents:
            return _ok({"results": [], "message": "No documents have been uploaded yet."})

        results = retrieve_from_documents(session.session_id, args.query, k=2)
        return _ok(
            {
                "results": [
                    {"filename": doc.metadata.get("filename", "document"), "text": doc.page_content, "score": score}
                    for doc, score in results
                ]
            }
        )

    def generate_clinician_brief(raw_args: dict) -> ToolResult:
        try:
            args = GenerateClinicianBriefArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        if session.protocol is None:
            return _fail("Cannot generate a brief before a chief complaint has been identified.")

        missing = get_missing_fields(session.record, session.protocol)
        if missing and not args.early_termination_reason:
            fields = ", ".join(m.field for m in missing)
            return _fail(
                f"Cannot finalize: required fields still open: {fields}. "
                f"Pass early_termination_reason to finalize anyway."
            )

        session.brief_finalized = True
        return _ok({"finalized": True, "missing_fields": [m.model_dump() for m in missing]})

    def request_human_assistance(raw_args: dict) -> ToolResult:
        try:
            args = RequestHumanAssistanceArgs(**raw_args)
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        session.assistance_requests.append(
            AssistanceRequest(
                id=str(uuid.uuid4()), reason=args.reason, timestamp=datetime.now(timezone.utc).isoformat()
            )
        )
        return _ok({"flagged": True})

    return {
        "update_intake_record": update_intake_record,
        "record_patient_correction": record_patient_correction,
        "check_safety_protocol": check_safety_protocol,
        "get_next_intake_question": get_next_intake_question,
        "retrieve_existing_patient_context": retrieve_existing_patient_context,
        "retrieve_uploaded_document": retrieve_uploaded_document,
        "generate_clinician_brief": generate_clinician_brief,
        "request_human_assistance": request_human_assistance,
    }
