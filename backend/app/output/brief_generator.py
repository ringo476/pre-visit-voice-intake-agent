"""Pure, deterministic template render — no LLM involved in producing the
brief, so there is no "smoothed prose might invent a claim" risk to guard
against: every line traces directly to one current Fact."""

from dataclasses import dataclass, field

from app.schemas.intake_record import Fact, IntakeRecord, Source
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import get_current_facts, get_missing_fields

CATEGORY_TO_SECTION = {
    "reason": "Reason for visit",
    "timeline": "History of present concern",
    "character": "History of present concern",
    "associated_symptoms": "History of present concern",
    "history": "Relevant history",
    "medications": "Medications",
    "allergies": "Allergies",
    "concerns": "Patient's primary concern",
}

SECTION_ORDER = [
    "Reason for visit",
    "History of present concern",
    "Relevant history",
    "Medications",
    "Allergies",
    "Patient's primary concern",
]


def _describe_fact(label: str, fact: Fact) -> str:
    """Renders one fact into a provenance-labeled sentence. Critically,
    NOT_ASKED never reaches here (callers skip it), so absence of
    information can never render as a denial."""
    if fact.source == Source.PATIENT_REPORTED:
        return f"Patient reports {label.lower()}: {fact.value}."
    if fact.source == Source.ASKED_AND_DENIED:
        return f"Patient denies {label.lower()}."
    if fact.source == Source.DOCUMENT_SOURCED:
        return f"Chart notes: {fact.value}."
    if fact.source == Source.INFERRED:
        return f"{label} (inferred): {fact.value}."
    return ""  # NOT_ASKED


@dataclass
class ClinicianBriefSection:
    title: str
    lines: list[str] = field(default_factory=list)


@dataclass
class ClinicianBrief:
    sections: list[ClinicianBriefSection]
    clarifications: list[str]
    disclaimer: str


def generate_clinician_brief(record: IntakeRecord, protocol: ProtocolConfig) -> ClinicianBrief:
    current = get_current_facts(record)
    section_lines: dict[str, list[str]] = {title: [] for title in SECTION_ORDER}

    for pf in protocol.fields:
        fact = current.get(pf.field)
        if fact is None or fact.source == Source.NOT_ASKED:
            continue  # absence -> clarifications, never a claim here
        line = _describe_fact(pf.label, fact)
        if not line:
            continue
        section = CATEGORY_TO_SECTION.get(pf.category, "Relevant history")
        section_lines[section].append(line)

    sections = [
        ClinicianBriefSection(title=title, lines=section_lines[title])
        for title in SECTION_ORDER
        if section_lines[title]
    ]

    clarifications = [m.label for m in get_missing_fields(record, protocol)]

    return ClinicianBrief(
        sections=sections,
        clarifications=clarifications,
        disclaimer="Generated from a patient conversation. Review and verify before clinical use.",
    )


def format_clinician_brief_as_text(brief: ClinicianBrief) -> str:
    parts: list[str] = []
    for section in brief.sections:
        parts.append(section.title)
        parts.append(" ".join(section.lines))
        parts.append("")
    if brief.clarifications:
        parts.append("Information requiring clarification")
        parts.extend(f"- {c}" for c in brief.clarifications)
        parts.append("")
    parts.append(brief.disclaimer)
    return "\n".join(parts).strip()
