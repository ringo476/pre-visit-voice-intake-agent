"""Verifies the STT-confidence routing: a low-confidence transcription still
reaches the reasoning model (never silently dropped), but carries an
explicit warning note so the model knows to treat anything it extracts from
that turn as less certain — the fix for a real gap found by reading the
Deepgram healthcare-voice-agent article: STT confidence was captured from
Google's API but never actually used anywhere downstream."""

import os

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import app.turn_controller as turn_controller
from app.agent.session import create_session
from app.turn_controller import STT_CONFIDENCE_THRESHOLD, TranscriptionResult, handle_utterance


def _capture_system_note(monkeypatch):
    """Stubs out the real Gemini call and real TTS synthesis so this test
    exercises only the routing logic, and records what system_note
    run_agent_turn actually received."""
    captured = {}

    def fake_run_agent_turn(session, patient_utterance, llm=None, system_note=None, turn_generation=None, ranking_llm=None):
        captured["system_note"] = system_note
        return {"reply_text": "ok", "superseded": False}

    monkeypatch.setattr(turn_controller, "run_agent_turn", fake_run_agent_turn)
    monkeypatch.setattr(turn_controller, "synthesize_speech", lambda text: b"fake-audio")
    return captured


def test_low_confidence_transcription_adds_a_warning_note(monkeypatch):
    monkeypatch.setattr(turn_controller, "transcribe_utterance", lambda audio: TranscriptionResult(text="my fever is high", confidence=0.3))
    captured = _capture_system_note(monkeypatch)

    session = create_session("s1")
    outcome = handle_utterance(session, audio=b"fake-audio-bytes")

    assert outcome.superseded is False
    assert captured["system_note"] is not None
    assert "low confidence" in captured["system_note"].lower()
    assert "0.3" in captured["system_note"]


def test_high_confidence_transcription_adds_no_warning_note(monkeypatch):
    monkeypatch.setattr(turn_controller, "transcribe_utterance", lambda audio: TranscriptionResult(text="my fever is high", confidence=0.95))
    captured = _capture_system_note(monkeypatch)

    session = create_session("s1")
    handle_utterance(session, audio=b"fake-audio-bytes")

    assert captured["system_note"] is None


def test_missing_confidence_score_does_not_crash_or_warn(monkeypatch):
    """Some STT responses may not return a confidence score at all
    (confidence=None) — this must not be treated as low confidence, and
    must not raise a comparison error (None < threshold)."""
    monkeypatch.setattr(turn_controller, "transcribe_utterance", lambda audio: TranscriptionResult(text="my fever is high", confidence=None))
    captured = _capture_system_note(monkeypatch)

    session = create_session("s1")
    handle_utterance(session, audio=b"fake-audio-bytes")

    assert captured["system_note"] is None


def test_threshold_is_a_real_module_constant_not_a_magic_number():
    assert 0.0 < STT_CONFIDENCE_THRESHOLD < 1.0
