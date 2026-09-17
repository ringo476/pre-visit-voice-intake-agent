"""Builds the handler map for one session. Every handler validates its args
against the Pydantic schema, calls exactly one backend module (state
engine, safety engine, or RAG retriever), and mutates `session` in place.
The model itself has no path to state other than through these handlers."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from pydantic import ValidationError

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
    ProvenanceViolationError,
    apply_fact,
    get_current_facts,
    get_missing_fields,
    record_correction,
)

EVIDENCE_REQUIRED_SOURCES = {Source.PATIENT_REPORTED, Source.ASKED_AND_DENIED, Source.DOCUMENT_SOURCED}


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


def create_tool_handlers(session: SessionState) -> dict[str, ToolHandler]:
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

        try:
            session.record = apply_fact(
                session.record,
                args.field,
                args.value,
                args.source,
                args.evidence,
                args.confidence,
                session.question_events,
                question_event_id=args.question_event_id,
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
                    "question_event_id": question_event.id,
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
