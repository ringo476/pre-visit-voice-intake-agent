"""Pydantic models validating the arguments for each of the 8 tools the
reasoning model may call. Each tool is a narrow RPC into exactly one
backend module — these schemas are the only surface the model can act
through."""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.intake_record import Polarity, Source


class UpdateIntakeRecordArgs(BaseModel):
    field: str
    # The direction of the patient's answer. The tool the model sees requires
    # it. Optional here only so a call that leaves it out is still handled (the
    # server then derives it, logs it, and verifies the claim like any other).
    polarity: Optional[Polarity] = None
    value: str = ""
    # The patient's (or a document's) exact words. Required except for the one
    # fact taken from the booking rather than from speech.
    evidence: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Not part of the tool the model sees any more: the server works out the
    # source label itself from where the quote really came from. Still accepted
    # and ignored so a model that sends one does not fail validation.
    source: Optional[Source] = None
    question_event_id: Optional[str] = None


class RecordPatientCorrectionArgs(BaseModel):
    field: str
    polarity: Optional[Polarity] = None
    new_value: str = ""
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class CheckSafetyProtocolArgs(BaseModel):
    statement: Optional[str] = None


class GetNextIntakeQuestionArgs(BaseModel):
    pass


class RetrieveExistingPatientContextArgs(BaseModel):
    query: Optional[str] = None


class RetrieveUploadedDocumentArgs(BaseModel):
    query: str


class GenerateClinicianBriefArgs(BaseModel):
    early_termination_reason: Optional[str] = None


class RequestHumanAssistanceArgs(BaseModel):
    reason: str


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "update_intake_record": UpdateIntakeRecordArgs,
    "record_patient_correction": RecordPatientCorrectionArgs,
    "check_safety_protocol": CheckSafetyProtocolArgs,
    "get_next_intake_question": GetNextIntakeQuestionArgs,
    "retrieve_existing_patient_context": RetrieveExistingPatientContextArgs,
    "retrieve_uploaded_document": RetrieveUploadedDocumentArgs,
    "generate_clinician_brief": GenerateClinicianBriefArgs,
    "request_human_assistance": RequestHumanAssistanceArgs,
}

TOOL_NAMES = list(TOOL_ARG_MODELS.keys())
