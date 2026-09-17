"""LLM-facing tool declarations (OpenAI function-calling / JSON-schema
format — langchain-google-genai converts these to Gemini's native format at
call time; we verified this conversion happens without needing a real API
key). This is the model's entire surface for affecting anything; everything
else in the backend is unreachable to it. Kept as hand-written, explicit
schemas — rather than auto-derived from the Pydantic arg models in
schemas/tool_schemas.py — so the exact contract the model sees is easy to
audit at a glance."""

import copy
from typing import Optional

SOURCE_ENUM = ["patient_reported", "asked_and_denied", "document_sourced", "inferred", "not_asked"]

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "update_intake_record",
        "description": (
            "Record one structured fact extracted from what the patient just said. Always include the "
            "patient's own words as the evidence quote."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "The protocol field this fact is about."},
                "value": {"type": "string", "description": "The extracted value, in plain text."},
                "source": {
                    "type": "string",
                    "enum": SOURCE_ENUM,
                    "description": "Where this fact came from. Use asked_and_denied only together with the question_event_id from get_next_intake_question for that same field.",
                },
                "evidence": {
                    "type": "string",
                    "description": "The patient's own words backing this fact, verbatim. Required unless source is not_asked or inferred.",
                },
                "confidence": {"type": "number", "description": "0 to 1 confidence in this extraction."},
                "question_event_id": {
                    "type": "string",
                    "description": "Required when source is asked_and_denied. Comes from a prior get_next_intake_question call.",
                },
            },
            "required": ["field", "value", "source", "confidence"],
        },
    },
    {
        "name": "record_patient_correction",
        "description": (
            "Use when the patient wants to redo or correct something already recorded for a field — including "
            "when they say it was captured wrong and ask you to ask them again. Preserves the original statement; "
            "just name the field, the current value for it is looked up automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "The field being corrected."},
                "new_value": {"type": "string", "description": "The corrected value."},
                "evidence": {"type": "string", "description": "The patient's own words making the correction, verbatim."},
                "confidence": {"type": "number", "description": "0 to 1 confidence in this correction."},
            },
            "required": ["field", "new_value", "evidence", "confidence"],
        },
    },
    {
        "name": "check_safety_protocol",
        "description": (
            "Call this whenever the patient says anything that might be a safety concern (e.g. severe "
            "breathing difficulty, chest pain, confusion, loss of consciousness). A deterministic system, "
            "not you, decides whether to escalate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "The concerning statement the patient made, verbatim."},
            },
        },
    },
    {
        "name": "get_next_intake_question",
        "description": "Ask what the protocol still needs covered. Returns the next open field and a question_event_id to use if the patient denies it.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "retrieve_existing_patient_context",
        "description": "Look up anything already known about this patient from prior visits, to avoid re-asking what's already documented.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for, e.g. 'prior inhaler use' or 'known allergies'."},
            },
        },
    },
    {
        "name": "retrieve_uploaded_document",
        "description": "Search the text extracted from any document the patient has uploaded this session (e.g. a photo of a prescription, a lab result PDF). Use this before asking about something the document might already answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'medication name and dose' or 'test result value'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "generate_clinician_brief",
        "description": "Finalize the intake once the conversation is complete. Refuses if required fields are still open, unless early_termination_reason is given.",
        "parameters": {
            "type": "object",
            "properties": {
                "early_termination_reason": {
                    "type": "string",
                    "description": "Only set if the patient needs to end the conversation before all required fields are covered.",
                },
            },
        },
    },
    {
        "name": "request_human_assistance",
        "description": "Call when the patient asks for a human, or you are unsure how to proceed safely. Does not modify the clinical record.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why human assistance is being requested."},
            },
            "required": ["reason"],
        },
    },
]


def build_tool_definitions(field_names: Optional[list[str]] = None) -> list[dict]:
    """Returns TOOL_DEFINITIONS, constraining the `field` argument of
    update_intake_record and record_patient_correction to an exact enum of
    `field_names` when given (i.e. once a protocol is locked and its real
    field list is known). Without this, the model only sees `field` as a
    free-text string with a couple of illustrative examples, and will
    occasionally invent a plausible-sounding name (e.g. "timing_pattern")
    instead of the protocol's actual field ("timing") — caught by
    tools.py's validation, but only after a wasted extra round trip to
    the model. An enum makes that guess structurally impossible instead of
    catching it after the fact."""
    definitions = copy.deepcopy(TOOL_DEFINITIONS)
    if not field_names:
        return definitions
    for tool in definitions:
        if tool["name"] in ("update_intake_record", "record_patient_correction"):
            tool["parameters"]["properties"]["field"]["enum"] = field_names
    return definitions
