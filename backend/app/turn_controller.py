"""One full voice turn: batch-transcribe the patient's recorded utterance,
run it through the agent's LangGraph (which is completely unaware this call
originated from voice at all), then synthesize the reply.

The Google Cloud Speech-to-Text and Text-to-Speech calls live directly in
this file rather than in their own modules — each is a single small
function with exactly one caller (this file), so a separate module bought
no real isolation. If a second caller or a provider swap ever shows up,
split them back out then.

Barge-in is detected client-side (see the frontend's voice hook), which
stops local audio playback the instant it detects the patient speaking
again — but that alone doesn't stop the turn that's still being processed
server-side. main.py bumps session.turn_generation on every new recorded
utterance and on a client barge-in signal; the generation a turn started
with is threaded through to the agent graph (see agent/graph.py), which
checks it before each further model/tool call and stops itself once a
newer utterance has superseded it, rather than finishing in the background
and landing a stale reply in the transcript.
"""

from dataclasses import dataclass
from typing import Optional

from app.agent.graph import run_agent_turn, run_opening_turn
from app.agent.session import SessionState, assign_protocol
from app.protocol.classifier import classify_complaint
from app.protocol.registry import get_protocol
from app.retry import call_with_retry

# Below this, a transcription is trusted structurally but not blindly: it's
# passed to the reasoning model in the transcript as usual (rejecting it
# outright would mean the patient's actual words could never even reach a
# clarifying follow-up), but any fact extracted from that specific turn
# never gets to claim it was clearly heard — see LOW_CONFIDENCE_NOTE below.
STT_CONFIDENCE_THRESHOLD = 0.6

_cached_stt_client = None
_cached_tts_client = None


def _get_stt_client():
    global _cached_stt_client
    if _cached_stt_client is None:
        from google.cloud import speech

        _cached_stt_client = speech.SpeechClient()
    return _cached_stt_client


def _get_tts_client():
    global _cached_tts_client
    if _cached_tts_client is None:
        from google.cloud import texttospeech

        _cached_tts_client = texttospeech.TextToSpeechClient()
    return _cached_tts_client


@dataclass
class TranscriptionResult:
    text: str
    confidence: Optional[float]


def transcribe_utterance(audio: bytes, encoding: str = "WEBM_OPUS", sample_rate_hertz: int = 48000) -> TranscriptionResult:
    """This is the canonical, auditable transcript: the reasoning model
    (agent/graph.py) only ever sees text that came out of this function,
    never raw audio — so every fact's evidence quote can be checked against
    real transcript text instead of trusted on the model's word.

    Batch mode (record-a-full-utterance, then transcribe) is used instead
    of bidirectional streaming recognition: simpler and just as correct at
    this project's scale, and responsive barge-in is delivered by the
    frontend's own voice-activity detection, not server-side streaming
    endpointing."""
    client = _get_stt_client()
    response = call_with_retry(
        lambda: client.recognize(
            config={"encoding": encoding, "sample_rate_hertz": sample_rate_hertz, "language_code": "en-US"},
            audio={"content": audio},
        ),
        what="Speech-to-Text",
    )
    if not response.results:
        return TranscriptionResult(text="", confidence=None)
    alternative = response.results[0].alternatives[0]
    return TranscriptionResult(text=alternative.transcript, confidence=alternative.confidence)


def synthesize_speech(text: str) -> bytes:
    """Converts the agent's reply text into playable audio. Unlike
    transcribe_utterance, this side has no provenance requirement — what
    the agent chooses to say is never itself a clinical fact, so any TTS
    voice/engine is fine here."""
    client = _get_tts_client()
    response = call_with_retry(
        lambda: client.synthesize_speech(
            input={"text": text},
            voice={"language_code": "en-US", "ssml_gender": "FEMALE"},
            audio_config={"audio_encoding": "MP3"},
        ),
        what="Text-to-Speech",
    )
    if not response.audio_content:
        raise RuntimeError("Text-to-speech returned no audio content")
    return response.audio_content


@dataclass
class TurnOutcome:
    transcript: str
    reply_text: str
    audio: bytes
    superseded: bool = False


def handle_utterance(session: SessionState, audio: bytes, generation: Optional[int] = None) -> TurnOutcome:
    transcription = transcribe_utterance(audio)
    if not transcription.text.strip():
        # A silence-only or too-short clip transcribes to nothing — sending
        # an empty message to Gemini is rejected outright ("contents are
        # required"). Nothing meaningful was said, so there's nothing to
        # run through the agent or send back; `superseded` already means
        # "the client gets nothing for this attempt" to main.py, which
        # fits here too.
        return TurnOutcome(transcript="", reply_text="", audio=b"", superseded=True)

    # A low-confidence transcription still gets sent through — rejecting it
    # outright would mean whatever the patient actually said could never
    # even reach a clarifying follow-up. But it shouldn't be trusted the
    # same as a clean one: flag it so the model knows to record anything it
    # extracts this turn as less certain, or check back with the patient,
    # rather than silently treating "salbutamol" misheard as "salicylate"
    # with full confidence.
    low_confidence_note = None
    if transcription.confidence is not None and transcription.confidence < STT_CONFIDENCE_THRESHOLD:
        low_confidence_note = (
            f"The speech-to-text transcription of the patient's last message has low confidence "
            f"({transcription.confidence:.2f}); some words may be mistranscribed. Use a lower confidence "
            f"value when recording any fact extracted from this specific message, and if anything critical "
            f"was unclear, briefly confirm it with the patient rather than assuming it was heard correctly."
        )

    return run_text_turn(session, transcription.text, generation=generation, extra_system_note=low_confidence_note)


def run_text_turn(
    session: SessionState,
    patient_text: str,
    generation: Optional[int] = None,
    extra_system_note: Optional[str] = None,
) -> TurnOutcome:
    """Runs one turn from already-known text rather than recorded audio —
    used when a document upload triggers a synthetic 'patient' turn (see
    main.py's upload endpoint). Shares the agent graph and TTS step with
    the voice path; only the STT step is skipped since there's no audio.

    `generation` is None for the document-upload path (nothing there races
    against a barge-in) and the live turn_generation snapshot for voice
    turns. When the turn comes back superseded, no audio is synthesized for
    a reply nobody is waiting to hear.

    `extra_system_note` (currently only the low-STT-confidence warning from
    handle_utterance) is combined with the deterministic classifier note,
    when both are present, rather than one silently overwriting the other."""
    notes = [n for n in (_classify_and_note(session, patient_text), extra_system_note) if n]
    system_note = "\n".join(notes) if notes else None
    result = run_agent_turn(session, patient_text, system_note=system_note, turn_generation=generation)
    if result.get("superseded"):
        return TurnOutcome(transcript=patient_text, reply_text="", audio=b"", superseded=True)
    audio_out = synthesize_speech(result["reply_text"])
    return TurnOutcome(transcript=patient_text, reply_text=result["reply_text"], audio=audio_out)


def run_opening_line(session: SessionState, reason_text: str, when_text: Optional[str] = None) -> TurnOutcome:
    """Generates and synthesizes Ava's opening line for a session whose
    chief complaint (and, when given, appointment date/time) is already
    known upfront (session.protocol is already locked at creation, not
    discovered live) — e.g. simulating a real pre-visit call where the
    clinic's booking already named a reason and a time for the visit.
    `transcript` comes back empty: no patient utterance kicked this off,
    Ava is speaking first."""
    result = run_opening_turn(session, reason_text, when_text=when_text)
    audio_out = synthesize_speech(result["reply_text"])
    return TurnOutcome(transcript="", reply_text=result["reply_text"], audio=audio_out)


def _classify_and_note(session: SessionState, patient_text: str) -> Optional[str]:
    """Deterministic pre-step run before every turn — which protocol
    applies is decided by classify_complaint(), never left to the model.

    - No protocol locked yet + a match -> lock it in, silently (the model
      just carries on naturally; it doesn't need to be told classification
      happened).
    - Protocol already locked + this turn matches a DIFFERENT protocol ->
      the checklist is NOT reassigned mid-conversation. Instead a
      deterministic note tells the model to acknowledge the new topic and
      defer it, rather than either silently absorbing it into the wrong
      checklist or dropping it.
    - No match at all (greeting, small talk, or already-consistent with
      the locked protocol) -> nothing to do.
    """
    matched_id = classify_complaint(patient_text)
    if matched_id is None:
        return None

    if session.protocol is None:
        assign_protocol(session, get_protocol(matched_id))
        return None

    if matched_id != session.protocol.protocol_id:
        other = get_protocol(matched_id)
        return (
            f"The patient's last message sounds like it may describe a different concern "
            f"(possibly: {other.name}) than today's intake ({session.protocol.name}). Briefly "
            f"acknowledge it and let the patient know they can raise it after this conversation or "
            f"with their clinician directly — do not try to record or fully discuss it now."
        )

    return None
