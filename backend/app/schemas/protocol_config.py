"""Describes an intake protocol's field checklist — the deterministic source
of truth for what's required. Retrieval/RAG may influence phrasing or
ordering, but never what counts as complete."""

from typing import Optional

from pydantic import BaseModel


class ProtocolField(BaseModel):
    field: str
    label: str
    category: str
    required: bool
    # If set, this field is only required once the named field has an
    # affirmed, non-NOT_ASKED fact recorded (e.g. inhaler_last_used is only
    # required once respiratory_history mentions an inhaler).
    required_if: Optional[str] = None


class ProtocolConfig(BaseModel):
    protocol_id: str
    name: str
    fields: list[ProtocolField]
    # Deterministic keyword set used by protocol/classifier.py to figure out
    # which checklist applies from what the patient actually says — not
    # chosen via a UI picker, and not left to the LLM to decide freely.
    keywords: list[str] = []
