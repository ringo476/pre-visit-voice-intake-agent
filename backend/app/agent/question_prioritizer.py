"""Adaptive next-question selection: asks the reasoning model which of the
still-missing fields is most valuable to ask about next, rather than always
walking a fixed protocol order — the same principle behind real adaptive
triage tools (ask whatever's most clinically informative given what's
already known, not the next line in a script).

This is deliberately advisory only, never authoritative — the model's
answer is free-text, not a tool call, and gets checked against the real set
of missing fields before it's trusted at all. A hallucinated field name, a
malformed response, a network failure, or no API key at all must never
block the conversation: the caller (tools.py's get_next_intake_question)
always has the original fixed order as an unconditional fallback. Nothing
is ever skipped by this — every field still missing stays missing and will
be asked eventually regardless of what gets picked first; the only thing
this changes is which one gets asked *this* turn.

`llm` is typed as the generic LangChain BaseChatModel, not anything
Gemini-specific, on purpose: this is exactly the kind of low-stakes,
easily-fallback-guarded decision suited to a small, cheap, even local model
(e.g. via Ollama) instead of spending a full call on the main reasoning
model — swapping one in later needs no change here."""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.schemas.intake_record import Fact
from app.state_engine import MissingField


def _extract_text(content: object) -> str:
    """Duplicated from graph.py rather than imported, to avoid a circular
    import (graph.py imports tools.py, which imports this module)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def rank_next_field(missing: list[MissingField], current_facts: dict[str, Fact], llm: BaseChatModel) -> Optional[str]:
    """Returns the field name the model considers most valuable to ask
    about next, or None if there's nothing to prioritize between, the call
    fails for any reason, or the response doesn't match a real missing
    field. Never raises."""
    if len(missing) < 2:
        return None  # nothing to choose between

    known = "; ".join(f"{f.field}: {f.value}" for f in current_facts.values()) or "nothing yet"
    options = "\n".join(f"- {m.field}: {m.label}" for m in missing)
    prompt = (
        "You are helping prioritize a clinical intake checklist for a phone conversation.\n"
        f"Already known: {known}\n\n"
        "Fields still needed (field_name: label):\n"
        f"{options}\n\n"
        "Considering patient safety and which answer would be most clinically informative given what's "
        "already known, which ONE field should be asked about next? "
        "Respond with ONLY the exact field_name and nothing else — no punctuation, no explanation."
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        candidate = _extract_text(response.content).strip().strip(".:\"'").lower()
    except Exception:
        return None

    valid = {m.field.lower(): m.field for m in missing}
    return valid.get(candidate)
