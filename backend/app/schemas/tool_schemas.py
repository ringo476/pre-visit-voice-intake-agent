"""Pydantic models validating the arguments for each of the 8 tools the
reasoning model may call. Each tool is a narrow RPC into exactly one
backend module — these schemas are the only surface the model can act
through."""

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.intake_record import Source


class UpdateIntakeRecordArgs(BaseModel):
    field: str
    value: str
    source: Source
    # Required for patient_reported / asked_and_denied / document_sourced;
    # the backend rejects a write claiming those sources without one.
    evidence: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Required by the backend when source == asked_and_denied.
    question_event_id: Optional[str] = None


class RecordPatientCorrectionArgs(BaseModel):
    fact_id: str
    field: str
    new_value: str
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
