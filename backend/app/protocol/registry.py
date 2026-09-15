"""The set of available intake protocols. Adding a new complaint type means:
add a protocol JSON here, add its RAG reference data under
app/rag/data/<protocol_id>/, and register it below — nothing else in the
state engine, safety engine, agent, or LangGraph needs to change, since none
of them know what any specific field means."""

import json
from pathlib import Path

from app.schemas.protocol_config import ProtocolConfig

_PROTOCOL_DIR = Path(__file__).parent

DEFAULT_PROTOCOL_ID = "respiratory-intake"

_PROTOCOL_FILES = {
    "respiratory-intake": "respiratory_intake.json",
    "musculoskeletal-leg-injury": "musculoskeletal_leg_injury.json",
    "allergy-reaction": "allergy_reaction.json",
}


def _load(filename: str) -> ProtocolConfig:
    with open(_PROTOCOL_DIR / filename, "r", encoding="utf-8") as f:
        return ProtocolConfig(**json.load(f))


PROTOCOLS: dict[str, ProtocolConfig] = {protocol_id: _load(filename) for protocol_id, filename in _PROTOCOL_FILES.items()}


def get_protocol(protocol_id: str | None) -> ProtocolConfig:
    """Falls back to the default protocol for an unknown or missing id,
    rather than failing a session over a bad query param."""
    if protocol_id and protocol_id in PROTOCOLS:
        return PROTOCOLS[protocol_id]
    return PROTOCOLS[DEFAULT_PROTOCOL_ID]
