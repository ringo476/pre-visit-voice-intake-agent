"""Builds the handler map for one session. Every handler validates its args
against the Pydantic schema, calls exactly one backend module (state
engine, safety engine, or RAG retriever), and mutates `session` in place.
The model itself has no path to state other than through these handlers."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from app.agent.answer_verifier import AnswerVerificationError, Claim, verify_claims
from app.agent.question_prioritizer import rank_next_field
from app.agent.session import AssistanceRequest, SessionState
from app.documents.document_store import retrieve_from_documents
from app.rag.retriever import retrieve_follow_up_guidance, retrieve_prior_chart
from app.policy_engine import SafetyResult, build_evaluation_log_entry, evaluate_fact, evaluate_statement
from app.schemas.intake_record import Polarity, QuestionEvent, Source
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
    classify_change,
    derive_polarity,
    evidence_follows_question,
    evidence_in_patient_speech,
    QUOTE_MATCH_THRESHOLD,
    evidence_in_text,
    evidence_match_score,
    find_asked_event,
    find_confirmation_event,
    get_current_fact,
    get_current_facts,
    get_missing_fields,
    looks_like_denial,
    patient_reply_after,
    patient_turns_containing,
    record_correction,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


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


_VERDICT_MEANING = {
    "contradicted": "the words it was taken from say the opposite, or something different, about this topic",
    "unrelated": "the words it was taken from are not about this topic at all",
}


class _Handlers(dict):
    """The handler map plus `prepare`, which the graph calls once for each
    assistant message that asks for tools. prepare checks every fact proposed
    in that message with ONE call to the verifier model, ahead of the
    handlers running, so a turn costs one extra short call however many facts
    it records. Handlers also work without prepare (they then verify their own
    claim), which is how they are used directly in tests."""

    prepare: Callable[[list], None]


@dataclass
class _Resolved:
    """A proposed fact after every check that needs no model."""

    field: str
    polarity: Polarity
    value: str
    evidence: Optional[str]
    confidence: float
    source: Source                    # worked out by the server, never chosen by the model
    question_event_id: Optional[str]
    claim: Optional[Claim]            # None when there is nothing to verify (the booking reason)
    supersedes_current: bool = False  # replaces the field's current answer (kept in history) instead of sitting beside it


def create_tool_handlers(
    session: SessionState,
    llm: Optional[BaseChatModel] = None,
    verifier_llm: Optional[BaseChatModel] = None,
) -> _Handlers:
    verdict_cache: dict[str, object] = {}
    # Handlers are built once per turn. A "which is right?" question asked in this
    # turn is remembered by the fact it concerns, so the prepare pass and the real
    # call (or a retry) do not log it twice.
    confirm_events: dict[str, QuestionEvent] = {}

    def _label_for(field: str) -> str:
        if session.protocol is not None:
            return next((f.label for f in session.protocol.fields if f.field == field), field)
        return field

    def _resolve(raw_args: dict, correction: bool) -> Union[_Resolved, ToolResult]:
        """Everything about a proposed fact that can be decided in plain code:
        is the request well formed, is the field real, is the quote really
        something the patient said (or a document says), and WHICH LABEL the
        fact therefore earns. The model proposes a field, a polarity, a value
        and a quote; it does not get to choose the label."""
        try:
            if correction:
                a = RecordPatientCorrectionArgs(**raw_args)
                value_in, legacy_source = a.new_value, None
            else:
                a = UpdateIntakeRecordArgs(**raw_args)
                value_in, legacy_source = a.value, a.source
        except ValidationError as e:
            return _fail(f"Invalid arguments: {e}")

        field = a.field
        if session.protocol is not None:
            known_fields = {f.field for f in session.protocol.fields}
            if field not in known_fields:
                return _fail(
                    f'"{field}" is not part of the current intake ({session.protocol.name}). '
                    f"If this is a separate concern, acknowledge it to the patient and let them know "
                    f"to raise it after this conversation or with their clinician — do not record it here."
                )

        polarity = a.polarity
        if polarity is None:
            polarity = derive_polarity(legacy_source or Source.PATIENT_REPORTED, value_in)
            logger.warning(
                "tool call left out polarity; derived it",
                extra={"session_id": session.session_id, "field": field, "derived": polarity.value},
            )

        value = value_in.strip()
        if polarity == Polarity.PRESENT and value and looks_like_denial(value):
            return _fail(
                f'Cannot record "{field}": the value "{value}" reads as a no, but the polarity is "present". '
                f'Use polarity "absent" for a no, or "unknown" if the patient does not know.'
            )
        value = value or {Polarity.PRESENT: "yes", Polarity.ABSENT: "no", Polarity.UNKNOWN: "patient does not know"}[polarity]

        evidence = (a.evidence or "").strip()
        if not evidence:
            # The one fact that has no quote: the visit reason already known from the booking.
            if (
                not correction
                and field == "chief_complaint"
                and session.appointment_reason_text
                and polarity == Polarity.PRESENT
            ):
                return _Resolved(field, polarity, value, None, a.confidence, Source.INFERRED, None, None)
            return _fail(
                f'Cannot record "{field}": it requires a verbatim evidence quote of the patient\'s own words. '
                f"(The only fact recorded without one is the visit reason taken from the booking.)"
            )

        # Where did the quote really come from?
        in_speech = evidence_in_patient_speech(session.transcript, evidence)
        if in_speech:
            score = evidence_match_score(" ".join(t.text for t in session.transcript if t.speaker == "patient"), evidence)
            if score < 1.0:  # accepted as a near copy, not an exact one: keep the rate visible
                logger.info(
                    "quote matched approximately",
                    extra={"session_id": session.session_id, "field": field, "score": round(score, 3), "threshold": QUOTE_MATCH_THRESHOLD},
                )
        document = None
        if not in_speech and not correction:
            document = next((d for d in session.documents if evidence_in_text(d.text, evidence)), None)
        if not in_speech and document is None:
            where = (
                "anything the patient has said"
                if correction or not session.documents
                else "anything the patient has said or any uploaded document"
            )
            logger.info("quote not found", extra={"session_id": session.session_id, "field": field})
            return _fail(f'Cannot record "{field}": the evidence quote was not found in {where}. Quote the exact words.')

        last_patient_turn = max((i for i, t in enumerate(session.transcript) if t.speaker == "patient"), default=-1)

        # Does the patient's statement meet an answer already on record? A repeat
        # adds nothing, a detail or a settled doubt replaces the old answer, and a
        # disagreement is not settled on one statement: the patient is asked which
        # is right, and the change is accepted only after they have answered that
        # question. (A claim quoted from a document is not the patient contradicting
        # themselves, so it is left alone.)
        supersedes_current = False
        confirmation = None
        current = get_current_fact(session.record, field) if document is None else None
        if current is not None:
            if current.source == Source.INFERRED:
                supersedes_current = True  # the booking's guess, replaced by the patient's own words
            else:
                change = classify_change(current, polarity, value)
                if change == "same":
                    return _ok(
                        {
                            "recorded": False,
                            "already_recorded": True,
                            "message": f'"{field}" already has this answer on record; nothing to change.',
                        }
                    )
                if change == "conflict":
                    confirmation = find_confirmation_event(
                        session.question_events, current.id, before_turn=last_patient_turn
                    )
                    if confirmation is None or not evidence_follows_question(session.transcript, confirmation, evidence):
                        return _ask_which_is_right(field, current, polarity, value)
                supersedes_current = True

        # Which label does that earn? Worked out here, from what actually happened.
        source = Source.PATIENT_REPORTED
        event = None
        asked_text: Optional[str] = None
        if document is not None:
            source = Source.DOCUMENT_SOURCED
            said = document.text[:2000]
        else:
            if polarity != Polarity.PRESENT and not correction:
                candidate = find_asked_event(session.question_events, field, before_turn=last_patient_turn)
                if candidate is not None and evidence_follows_question(session.transcript, candidate, evidence):
                    event = candidate
                    source = Source.ASKED_AND_DENIED if polarity == Polarity.ABSENT else Source.UNCERTAIN
            if event is not None:
                said = patient_reply_after(session.transcript, event)
                asked_text = event.spoken_text or event.question_text
            else:
                said = patient_turns_containing(session.transcript, evidence)
                spoken = find_asked_event(session.question_events, field, before_turn=len(session.transcript))
                asked_text = (spoken.spoken_text or spoken.question_text) if spoken else None
            if confirmation is not None:
                # The patient is answering "which is right?", so that is what the
                # independent reader is shown: a bare "the second one" or "yes"
                # means nothing without it.
                source, event = Source.PATIENT_REPORTED, None
                said = patient_reply_after(session.transcript, confirmation)
                asked_text = confirmation.spoken_text or confirmation.question_text

        claim = Claim(
            topic=_label_for(field),
            polarity=polarity.value,
            value=value if polarity == Polarity.PRESENT else "",
            said=said,
            asked=asked_text,
            from_document=document is not None,
        )
        return _Resolved(
            field, polarity, value, evidence, a.confidence, source, event.id if event else None, claim, supersedes_current
        )

    def _answer_text(polarity: Polarity, value: str) -> str:
        if polarity == Polarity.ABSENT:
            return "no"
        if polarity == Polarity.UNKNOWN:
            return "not sure"
        return "yes" if value.strip().lower() in ("yes", "") else f"yes, {value}"

    def _ask_which_is_right(field: str, current, polarity: Polarity, value: str) -> ToolResult:
        """The patient's statement contradicts what is on record. Nothing is
        written. A question is logged for the field, tied to the exact fact in
        dispute, so that Ava asking it (and the patient answering it) is what
        later allows the change; and the model is told what to ask."""
        label = _label_for(field)
        event = confirm_events.get(current.id)
        if event is None:
            event = QuestionEvent(
                id=str(uuid.uuid4()),
                field=field,
                question_text=f"Confirm: {label}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                confirms_fact_id=current.id,
            )
            session.question_events.append(event)
            confirm_events[current.id] = event
        on_record, just_said = _answer_text(current.polarity, current.value), _answer_text(polarity, value)
        logger.info(
            "new statement contradicts the record, asking the patient which is right",
            extra={"session_id": session.session_id, "field": field, "on_record": current.polarity.value, "new": polarity.value},
        )
        return _ok(
            {
                "recorded": False,
                "needs_confirmation": True,
                "field": field,
                "on_record": on_record,
                "just_said": just_said,
                "message": (
                    f'Nothing was changed. On "{label}" the record says: {on_record}. What the patient just said '
                    f"reads as: {just_said}. One statement is not enough to overwrite an answer they gave. Ask the "
                    f"patient which is right, in one short question that names both, for example: "
                    f'"Just to be sure about {label.lower()}: earlier I noted {on_record}, and just now it sounded like '
                    f'{just_said}. Which is right?" When they answer, record their answer with '
                    f"record_patient_correction. If they keep the earlier answer, record nothing."
                ),
            }
        )

    def _cache_key(name: str, raw_args: dict) -> str:
        return name + json.dumps(raw_args, sort_keys=True, default=str)

    def _verify(resolved: _Resolved, name: str, raw_args: dict) -> Optional[ToolResult]:
        """Independent check that the words really say what is being recorded.
        Returns a failing ToolResult, or None when the claim holds up. Skipped
        only when no verifier is configured (offline tests / no API key). A
        verifier that cannot answer refuses the save (fails closed)."""
        if verifier_llm is None or resolved.claim is None:
            return None
        outcome = verdict_cache.pop(_cache_key(name, raw_args), None)
        if outcome is None:
            try:
                outcome = verify_claims([resolved.claim], verifier_llm)[0]
            except AnswerVerificationError as e:
                outcome = e
        if isinstance(outcome, AnswerVerificationError):
            logger.warning(
                "answer verification unavailable, refusing the save",
                extra={"session_id": session.session_id, "field": resolved.field, "error": str(outcome)},
            )
            return _fail(
                f'Could not verify the answer for "{resolved.field}", so it was not recorded. '
                f"Ask the patient again and confirm what they meant."
            )
        if outcome != "supported":
            logger.info(
                "answer verification mismatch",
                extra={
                    "session_id": session.session_id,
                    "field": resolved.field,
                    "polarity": resolved.polarity.value,
                    "verdict": outcome,
                },
            )
            return _fail(
                f'Cannot record "{resolved.field}" as {resolved.polarity.value}: an independent reading says '
                f"{_VERDICT_MEANING[outcome]}. Re-read what was said and record it under the right field with "
                f"the right polarity, or ask the patient again."
            )
        return None

    def prepare(calls: list) -> None:
        if verifier_llm is None:
            return
        pending: list[tuple[str, Claim]] = []
        for call in calls:
            name = call.get("name")
            if name not in ("update_intake_record", "record_patient_correction"):
                continue
            raw = call.get("args") or {}
            resolved = _resolve(raw, correction=(name == "record_patient_correction"))
            if isinstance(resolved, ToolResult) or resolved.claim is None:
                continue
            pending.append((_cache_key(name, raw), resolved.claim))
        if not pending:
            return
        try:
            verdicts = verify_claims([claim for _, claim in pending], verifier_llm)
            for (key, _), verdict in zip(pending, verdicts):
                verdict_cache[key] = verdict
        except AnswerVerificationError as e:
            for key, _ in pending:
                verdict_cache[key] = e

    def update_intake_record(raw_args: dict) -> ToolResult:
        resolved = _resolve(raw_args, correction=False)
        if isinstance(resolved, ToolResult):
            return resolved
        rejected = _verify(resolved, "update_intake_record", raw_args)
        if rejected:
            return rejected
        try:
            if resolved.supersedes_current:
                # The field already had an answer. The new one replaces it as the
                # current answer and links back to it; the old one stays in history.
                session.record = record_correction(
                    session.record,
                    resolved.field,
                    resolved.value,
                    resolved.evidence or "",
                    resolved.confidence,
                    polarity=resolved.polarity,
                )
            else:
                session.record = apply_fact(
                    session.record,
                    resolved.field,
                    resolved.value,
                    resolved.source,
                    resolved.evidence,
                    resolved.confidence,
                    session.question_events,
                    question_event_id=resolved.question_event_id,
                    polarity=resolved.polarity,
                )
        except ProvenanceViolationError as e:
            return _fail(str(e))

        new_fact = session.record.facts[-1]
        safety = evaluate_fact(new_fact)
        session.safety_log.append(build_evaluation_log_entry(new_fact.id, safety))
        return _ok(
            {
                "recorded": True,
                "fact_id": new_fact.id,
                "source": new_fact.source.value,
                "safety": _safety_payload(safety),
            }
        )

    def record_patient_correction(raw_args: dict) -> ToolResult:
        resolved = _resolve(raw_args, correction=True)
        if isinstance(resolved, ToolResult):
            return resolved
        rejected = _verify(resolved, "record_patient_correction", raw_args)
        if rejected:
            return rejected
        try:
            session.record = record_correction(
                session.record,
                resolved.field,
                resolved.value,
                resolved.evidence or "",
                resolved.confidence,
                polarity=resolved.polarity,
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
        # A question the patient talked over has to be asked again, so it
        # takes priority over the usual ordering and over the ranker below.
        cut_off = session.cut_off_question
        repeat = next((m for m in missing if cut_off and m.field == cut_off.field), None)
        if repeat is not None:
            next_field = repeat
        elif llm is not None:
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

    handlers = _Handlers({
        "update_intake_record": update_intake_record,
        "record_patient_correction": record_patient_correction,
        "check_safety_protocol": check_safety_protocol,
        "get_next_intake_question": get_next_intake_question,
        "retrieve_existing_patient_context": retrieve_existing_patient_context,
        "retrieve_uploaded_document": retrieve_uploaded_document,
        "generate_clinician_brief": generate_clinician_brief,
        "request_human_assistance": request_human_assistance,
    })
    handlers.prepare = prepare
    return handlers
