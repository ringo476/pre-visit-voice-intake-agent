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

POLARITY_ENUM = ["present", "absent", "unknown"]

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "update_intake_record",
        "description": (
            "Record one structured fact taken from what the patient just said. Say which topic it is about, "
            "which direction the patient's answer goes (present, absent or unknown), and quote their exact "
            "words. The system works out where the fact came from; you do not label it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "description": "The checklist topic this fact is about."},
                "polarity": {
                    "type": "string",
                    "enum": POLARITY_ENUM,
                    "description": (
                        "The direction of the patient's answer about this topic. present: they say it is so "
                        "(give the detail in value). absent: they say no, it is not. unknown: they say they do "
                        "not know or do not remember."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The detail, in plain words, when polarity is present (for example '3 weeks ago'). Leave empty for absent or unknown.",
                },
                "evidence": {
                    "type": "string",
                    "description": "The patient's own words backing this, copied exactly from what they said. If it came from an uploaded document, copy the document's text exactly. Only the visit reason taken from the booking (chief_complaint) may be recorded without a quote.",
                },
                "confidence": {"type": "number", "description": "0 to 1 confidence in this extraction."},
            },
            "required": ["field", "polarity", "confidence"],
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
                "polarity": {
                    "type": "string",
                    "enum": POLARITY_ENUM,
                    "description": "The direction of the corrected answer: present, absent (no) or unknown (does not know).",
                },
                "new_value": {"type": "string", "description": "The corrected detail when polarity is present. Leave empty for absent or unknown."},
                "evidence": {"type": "string", "description": "The patient's own words making the correction, copied exactly."},
                "confidence": {"type": "number", "description": "0 to 1 confidence in this correction."},
            },
            "required": ["field", "polarity", "evidence", "confidence"],
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
        "description": "Ask what the protocol still needs covered. Returns the next open field to ask the patient about. Asking it is what later lets a denial of that field be recorded.",
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
