from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioTurn:
    patient_utterance: str
    # Tool calls a competent agent should make in response: [{"name": str, "args": dict}, ...]
    tool_calls: list[dict]
    # What the agent says after those tool calls resolve.
    final_text: str


@dataclass
class SeedDocument:
    filename: str
    mime_type: str
    text: str


@dataclass
class ScenarioExpectations:
    # Fields that must never end up asked_and_denied by the end.
    fields_should_not_be_denied: list[str] = field(default_factory=list)
    # Fields that must be covered (any source other than not_asked) by the end.
    fields_should_be_covered: list[str] = field(default_factory=list)
    # Fields expected to still be missing/open at the end.
    fields_should_remain_missing: list[str] = field(default_factory=list)
    expect_safety_trigger: Optional[bool] = None
    expect_correction: Optional[bool] = None
    # A field that must end up recorded with source document_sourced, proving document provenance was preserved.
    expect_document_sourced_field: Optional[str] = None


@dataclass
class EvalScenario:
    id: str
    description: str
    turns: list[ScenarioTurn]
    expectations: ScenarioExpectations
    # Pre-populates session.documents before any turns run — stands in for a real upload+extraction.
    seed_documents: list[SeedDocument] = field(default_factory=list)
    # Which protocol this scenario runs against. Defaults to the respiratory
    # showcase protocol; the leg-injury scenario overrides this to prove
    # routing actually works for a non-respiratory complaint.
    protocol_id: str = "respiratory-intake"
