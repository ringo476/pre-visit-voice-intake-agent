"""The words Ava speaks to read the recorded answers back to the patient before
the intake can be finalized. Built by plain code from the current facts, never by
the model, so what the patient hears is exactly what is on record."""

from app.schemas.intake_record import Fact, IntakeRecord, Polarity, Source
from app.schemas.protocol_config import ProtocolConfig
from app.state_engine import get_current_facts, get_missing_fields


def _spoken_answer(fact: Fact) -> str:
    if fact.polarity == Polarity.ABSENT:
        return "no"
    if fact.polarity == Polarity.UNKNOWN:
        return "not sure"
    value = fact.value.strip()
    return value if value and value.lower() != "yes" else "yes"


def build_readback(record: IntakeRecord, protocol: ProtocolConfig) -> str:
    current = get_current_facts(record)
    answers = []
    for pf in protocol.fields:
        fact = current.get(pf.field)
        if fact is None or fact.source == Source.NOT_ASKED:
            continue
        answers.append(f"{pf.label}: {_spoken_answer(fact)}.")

    text = "Before I finish, let me read back what I have. "
    text += " ".join(answers) if answers else "I don't have any answers recorded yet."
    missing = get_missing_fields(record, protocol)
    if missing:
        text += " I still don't have answers for " + ", ".join(m.label.lower() for m in missing) + "."
    return text + " Is all of that correct?"
