"""The clinical state engine — pure functions, no I/O. Enforces the core
provenance invariant: a fact can only claim ASKED_AND_DENIED if a matching
question was actually logged first. This is what the tool layer calls;
the reasoning model never touches this directly."""

import re
import uuid
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from app.schemas.intake_record import Fact, FactStatus, IntakeRecord, Polarity, QuestionEvent, Source, TranscriptTurn
from app.schemas.protocol_config import ProtocolConfig

AFFIRMATIVE_SOURCES = {Source.PATIENT_REPORTED, Source.DOCUMENT_SOURCED, Source.INFERRED}
# Both of these are answers to a question the agent must really have asked:
# "no" and "I don't know" each close a field without the patient volunteering
# anything, so each needs proof the question was spoken first.
SPOKEN_QUESTION_SOURCES = {Source.ASKED_AND_DENIED, Source.UNCERTAIN}


class ProvenanceViolationError(Exception):
    """Raised whenever a write would violate a provenance invariant. Callers
    (the tool layer) are expected to catch this and report a tool error
    back to the model rather than let state silently corrupt."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_empty_record(session_id: str, protocol_id: str) -> IntakeRecord:
    now = _now()
    return IntakeRecord(session_id=session_id, protocol_id=protocol_id, facts=[], created_at=now, updated_at=now)


def derive_polarity(source: Source, value: str) -> Polarity:
    """Fallback for code that records a fact without stating its polarity
    (older call sites, saved data). The live tool states it explicitly.
    Only this fallback ever reads the wording of the value."""
    source = Source(source)
    if source == Source.ASKED_AND_DENIED:
        return Polarity.ABSENT
    if source == Source.UNCERTAIN:
        return Polarity.UNKNOWN
    return Polarity.ABSENT if _looks_like_denial(value) else Polarity.PRESENT


def apply_fact(
    record: IntakeRecord,
    field: str,
    value: str,
    source: Source,
    evidence_span: Optional[str],
    confidence: float,
    question_events: list[QuestionEvent],
    question_event_id: Optional[str] = None,
    polarity: Optional[Polarity] = None,
) -> IntakeRecord:
    """Appends a new fact to the record. Never mutates or removes existing facts."""
    source = Source(source)
    if source in SPOKEN_QUESTION_SOURCES:
        if not question_event_id:
            raise ProvenanceViolationError(
                f'Cannot record "{field}" as {source.value} without a question_event_id'
            )
        event = next((e for e in question_events if e.id == question_event_id), None)
        if event is None:
            raise ProvenanceViolationError(
                f'question_event_id "{question_event_id}" does not match any logged question event'
            )
        if event.field != field:
            raise ProvenanceViolationError(
                f'question_event_id "{question_event_id}" logged a question about '
                f'"{event.field}", not "{field}"'
            )

    timestamp = _now()
    fact = Fact(
        id=str(uuid.uuid4()),
        field=field,
        value=value,
        source=source,
        evidence_span=evidence_span,
        confidence=confidence,
        status=FactStatus.UNCONFIRMED,
        timestamp=timestamp,
        question_event_id=question_event_id,
        polarity=polarity or derive_polarity(source, value),
    )
    return record.model_copy(update={"facts": [*record.facts, fact], "updated_at": timestamp})


def find_asked_event(
    question_events: list[QuestionEvent], field: str, before_turn: int
) -> Optional[QuestionEvent]:
    """The most recent question about `field` that was actually spoken in an
    agent turn earlier than `before_turn` (an index into the transcript).
    The server resolves this itself rather than trusting an id from the
    model: the model only ever sees an event id inside the tool result of
    the turn that created it, so it has nothing valid to hand back on the
    turn where the patient actually answers."""
    candidates = [
        e
        for e in question_events
        if e.field == field and e.asked_in_turn is not None and e.asked_in_turn < before_turn
    ]
    return max(candidates, key=lambda e: e.asked_in_turn, default=None)


def _normalize_for_match(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def patient_reply_after(transcript: list[TranscriptTurn], event: QuestionEvent) -> str:
    """Everything the patient said after the agent spoke `event`'s question."""
    if event.asked_in_turn is None:
        return ""
    return " ".join(t.text for t in transcript[event.asked_in_turn + 1 :] if t.speaker == "patient")


# How closely a quote must resemble the words it claims to come from. A model
# asked to copy a sentence often changes a word or drops one ("took" for
# "take"), and speech-to-text can word things slightly differently, so an
# exact-only match rejected genuine statements. The threshold is calibrated in
# tests/test_quote_matching.py: it sits in the gap between the worst slip that
# should be accepted and the best invented or paraphrased quote that must not be.
QUOTE_MATCH_THRESHOLD = 0.88
# A quote this short must match exactly. One changed letter in a three-word
# quote can be a different word with a different meaning.
MIN_FUZZY_WORDS = 4
# Words that flip or change the meaning of a sentence. A near-match is never
# allowed to add, drop or change one of these, or any number, because a
# similarity score treats "I have no fever" and "I have a fever" as nearly
# identical. ("t" is what remains of "can't" / "don't" once punctuation goes.)
_MEANING_WORDS = {
    "no", "not", "never", "none", "nothing", "nobody", "without", "neither", "nor", "cannot", "t",
}


def _critical_words(words: list[str]) -> Counter:
    return Counter(w for w in words if w in _MEANING_WORDS or any(ch.isdigit() for ch in w))


def evidence_match_score(text: str, evidence: str) -> float:
    """How well `evidence` matches some stretch of `text`, from 0 to 1. A copy
    of the words (ignoring case and punctuation) scores 1. Otherwise the quote
    is compared with every stretch of the text of about its length, but only
    stretches that have exactly the same negations and numbers are considered,
    and quotes under MIN_FUZZY_WORDS words get no near-match at all."""
    needle = _normalize_for_match(evidence)
    if not needle:
        return 0.0
    haystack = _normalize_for_match(text)
    if f" {needle} " in f" {haystack} ":
        return 1.0

    quote = needle.split()
    words = haystack.split()
    if len(quote) < MIN_FUZZY_WORDS or not words:
        return 0.0

    wanted = _critical_words(quote)
    matcher = SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(needle)
    best = 0.0
    for size in range(max(1, len(quote) - 2), len(quote) + 3):
        for start in range(0, len(words) - size + 1):
            window = words[start : start + size]
            if _critical_words(window) != wanted:
                continue
            matcher.set_seq1(" ".join(window))
            if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
                continue
            best = max(best, matcher.ratio())
    return best


def evidence_in_text(text: str, evidence: str) -> bool:
    """True if `evidence` is a copy of words in `text` (ignoring case and
    punctuation, whole words only), or close enough to count as one (see
    evidence_match_score)."""
    return evidence_match_score(text, evidence) >= QUOTE_MATCH_THRESHOLD


def evidence_in_patient_speech(transcript: list[TranscriptTurn], evidence: str) -> bool:
    """True if the quote is something the PATIENT said at some point in the
    conversation. Used for facts the patient volunteered, which have no
    particular question to anchor to."""
    return evidence_in_text(" ".join(t.text for t in transcript if t.speaker == "patient"), evidence)


def patient_turns_containing(transcript: list[TranscriptTurn], evidence: str) -> str:
    """The patient message(s) the quote came from, so a meaning check can read
    the whole sentence the quote was taken from rather than the quote alone.
    Falls back to the patient's most recent message if no single message holds
    the whole quote."""
    matching = [t.text for t in transcript if t.speaker == "patient" and evidence_in_text(t.text, evidence)]
    if matching:
        return " ".join(matching)
    patient_turns = [t.text for t in transcript if t.speaker == "patient"]
    return patient_turns[-1] if patient_turns else ""


def looks_like_denial(value: str) -> bool:
    """Public wrapper: does this recorded value read as a "no"? Empty values
    do not count."""
    return bool(value.strip()) and _looks_like_denial(value)


def evidence_follows_question(transcript: list[TranscriptTurn], event: QuestionEvent, evidence: str) -> bool:
    """True if `evidence` is a quote from something the patient said AFTER
    the agent spoke `event`'s question. Punctuation and case are ignored,
    since the model's quote and the STT transcript routinely differ in
    those alone."""
    if event.asked_in_turn is None:
        return False
    return evidence_in_text(patient_reply_after(transcript, event), evidence)


def record_correction(
    record: IntakeRecord,
    field: str,
    new_value: str,
    evidence_span: str,
    confidence: float,
    polarity: Optional[Polarity] = None,
) -> IntakeRecord:
    """Appends a corrected version of the current fact for `field`. The
    prior fact is left untouched; the new fact's `supersedes` links back to
    it. Looks up the current fact by field name rather than taking a prior
    fact id from the caller — an id from an earlier update_intake_record
    result is only ever visible within that same turn's own tool-call
    history, never in later turns (each turn rebuilds its message list from
    session.transcript alone), so a caller several turns later has no
    reliable id to supply. The field name, by contrast, is always known."""
    prior = get_current_fact(record, field)
    if prior is None:
        raise ProvenanceViolationError(
            f'Cannot correct "{field}" — nothing has been recorded for it yet; use update_intake_record instead'
        )

    # A correction is always the patient's own new statement, so it is always
    # patient_reported, whatever the prior fact was. Letting the prior source
    # leak through filed "I did have a fever after all" as asked_and_denied
    # (rendering "Patient denies fever") and kept a document or inferred label
    # on a value the patient had just contradicted in their own words.
    new_source = Source.PATIENT_REPORTED

    timestamp = _now()
    corrected = Fact(
        id=str(uuid.uuid4()),
        field=field,
        value=new_value,
        source=new_source,
        evidence_span=evidence_span,
        confidence=confidence,
        status=FactStatus.CORRECTED,
        timestamp=timestamp,
        supersedes=prior.id,
        polarity=polarity or derive_polarity(new_source, new_value),
        # A changed source means the new value was not given in answer to the
        # logged question, so it must not inherit that question's proof.
        question_event_id=prior.question_event_id if new_source == prior.source else None,
    )
    return record.model_copy(update={"facts": [*record.facts, corrected], "updated_at": timestamp})


def get_current_facts(record: IntakeRecord) -> dict[str, Fact]:
    """The current (non-superseded) fact for every field that has one, keyed by field name."""
    superseded = {f.supersedes for f in record.facts if f.supersedes}
    current: dict[str, Fact] = {}
    for fact in record.facts:
        if fact.id in superseded:
            continue
        existing = current.get(fact.field)
        # Facts are appended in the order they were recorded, so on a tie (two
        # writes landing in the same clock tick) the later one is the current one.
        if existing is None or fact.timestamp >= existing.timestamp:
            current[fact.field] = fact
    return current


def get_current_fact(record: IntakeRecord, field: str) -> Optional[Fact]:
    return get_current_facts(record).get(field)


_GENERIC_YES = {"yes", "yeah", "yep", "yup"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Grammar words and hedges that may be added to an answer without being in the
# patient's quote. Negations are deliberately not here.
_FILLER_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "is", "are", "was", "were",
    "be", "been", "it", "its", "i", "my", "me", "with", "for", "from", "that", "this", "also", "actually",
    "just", "very", "really", "about", "around", "roughly", "approximately", "some", "yes",
}


def _added_words_are_in_quote(old: str, new: str, quote: str) -> bool:
    """`old`, `new` and `quote` are already lowercased with punctuation removed.
    True if every word the new answer adds to the old one is either a filler word
    or appears in the patient's quote. This is what stops a model from keeping
    the old text and tacking on a detail the patient never said."""
    added = set(new.split()) - set(old.split()) - _FILLER_WORDS
    return added <= set(quote.split())


def classify_change(current: Fact, polarity: Polarity, value: str, quote: Optional[str] = None) -> str:
    """How a new answer about a field that already has one relates to it.

    "same"     - nothing new; recording it again would only add a duplicate.
    "update"   - the new answer adds to or settles the old one, so it can replace
                 it: a detail added to a yes ("fever" -> "fever since Tuesday"), or
                 an answer where the patient had said they did not know.
    "conflict" - it disagrees with an answer the patient gave: yes against no, a
                 definite answer now being doubted, or a different value. A
                 disagreement is never settled on one noisy statement; the
                 patient is asked which is right."""
    if current.polarity == Polarity.UNKNOWN:
        return "same" if polarity == Polarity.UNKNOWN else "update"
    if polarity != current.polarity:
        return "conflict"
    if polarity == Polarity.ABSENT:
        return "same"

    old, new = _normalize_for_match(current.value), _normalize_for_match(value)
    if old == new or new in _GENERIC_YES:
        return "same"
    if not old or not new:
        return "update"
    # A number the patient already gave (a temperature, a dose, a duration) may not
    # change or grow on the strength of one statement: 104 becoming 102, 104.5 or
    # "104 and 102" is a disagreement even though the old text is still inside the
    # new. Adding a number to an answer that had none is still just added detail.
    old_numbers = sorted(_NUMBER.findall(current.value))
    if old_numbers and old_numbers != sorted(_NUMBER.findall(value)):
        return "conflict"
    if old in _GENERIC_YES or f" {old} " in f" {new} " or f" {new} " in f" {old} ":
        # Only added detail so far, but the detail has to be the patient's: a word
        # that is in neither the old answer nor the quote was made up.
        if quote is not None and not _added_words_are_in_quote(old, new, _normalize_for_match(quote)):
            return "conflict"
        return "update"
    return "conflict"


def find_confirmation_event(
    question_events: list[QuestionEvent], fact_id: str, before_turn: int
) -> Optional[QuestionEvent]:
    """The most recent question that asked the patient to settle a disagreement
    with the fact `fact_id`, and that was actually heard (stamped) in an agent
    turn earlier than `before_turn`. Tied to the fact's id, so a confirmation
    about an older version of the answer can never approve a change to this one."""
    candidates = [
        e
        for e in question_events
        if e.confirms_fact_id == fact_id and e.asked_in_turn is not None and e.asked_in_turn < before_turn
    ]
    return max(candidates, key=lambda e: e.asked_in_turn, default=None)


class MissingField(BaseModel):
    field: str
    label: str
    category: str


_DENIAL_WORDS = {"no", "none", "false", "negative", "denied", "n/a", "na"}


def _looks_like_denial(value: str) -> bool:
    """A fact's source (e.g. patient_reported) only says how the value was
    obtained, not what it says — a fact can be patient_reported AND a
    denial ("no medication allergies") at once. This is a best-effort check
    of the value text itself, since the model phrases denials consistently
    as leading "no"/"none" in this codebase's own tool-call data."""
    first_word = value.strip().lower().split(" ")[0].strip(".,!?") if value.strip() else ""
    return value.strip() == "" or first_word in _DENIAL_WORDS


def get_missing_fields(record: IntakeRecord, protocol: ProtocolConfig) -> list[MissingField]:
    """Fields the protocol still needs. A conditional field (`required_if`)
    only becomes required once the referenced field has an affirmative
    current fact — a denial or an unasked field never triggers its
    dependents."""
    current = get_current_facts(record)

    def is_covered(field: str) -> bool:
        fact = current.get(field)
        return fact is not None and fact.source != Source.NOT_ASKED

    def is_affirmed(field: str) -> bool:
        fact = current.get(field)
        return fact is not None and fact.source in AFFIRMATIVE_SOURCES and fact.polarity == Polarity.PRESENT

    missing: list[MissingField] = []
    for pf in protocol.fields:
        required_now = pf.required or (pf.required_if is not None and is_affirmed(pf.required_if))
        if required_now and not is_covered(pf.field):
            missing.append(MissingField(field=pf.field, label=pf.label, category=pf.category))
    return missing


def is_record_complete(record: IntakeRecord, protocol: ProtocolConfig) -> bool:
    return len(get_missing_fields(record, protocol)) == 0
