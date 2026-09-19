"""FastAPI server: REST endpoints for health/brief/FHIR/document-upload, and
a WebSocket endpoint that carries the voice turns (binary audio in,
JSON control/state messages + binary audio out)."""

import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.agent.session import SessionState, create_session
from app.db import init_db
from app.documents.document_ingest import SUPPORTED_IMAGE_MIME_TYPES, UnsupportedDocumentTypeError, extract_text
from app.documents.document_store import add_document, clear_session, create_uploaded_document
from app.output.brief_generator import format_clinician_brief_as_text, generate_clinician_brief
from app.output.fhir_export import generate_fhir_export
from app.persistence import list_sessions, load_session, save_session
from app.protocol.registry import get_protocol
from app.state_engine import get_missing_fields
from app.turn_controller import TurnOutcome, handle_utterance, run_opening_line, run_text_turn

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

# Stands in for a real appointment/booking system: in an actual pre-visit
# voice agent, the clinic's calendar already carries a date/time and a
# reason for the visit by the time the call happens, so both are known
# upfront rather than discovered live from what the patient says. No real
# scheduling/EHR integration exists here — one of these three is picked at
# random per session so the "protocol already known" flow can be
# demonstrated against any of the three checklists.
MOCK_APPOINTMENTS = [
    {"protocol_id": "respiratory-intake", "reason_text": "chest discomfort and fever"},
    {"protocol_id": "musculoskeletal-leg-injury", "reason_text": "pain and swelling in the left ankle after a fall"},
    {"protocol_id": "allergy-reaction", "reason_text": "a skin rash and reaction after starting a new medication"},
]


def _mock_appointment_details() -> dict:
    appointment = random.choice(MOCK_APPOINTMENTS)
    visit_time = datetime.now() + timedelta(days=random.randint(1, 3), hours=random.randint(0, 8))
    return {
        "protocol_id": appointment["protocol_id"],
        "reason_text": appointment["reason_text"],
        "when_text": visit_time.strftime("%A, %B %d at %I:%M %p"),
    }

app = FastAPI(title="Pre-Visit Voice Intake Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    await init_db()


# In-memory mirror of whatever's currently connected, for fast access
# within a live process (upload_document, get_brief, get_fhir all read this
# directly). The durable copy lives in the database via app/persistence.py —
# saved after every turn, reloadable by session_id even after a restart, so
# this dict being wiped on process exit no longer means the conversation is
# gone.
sessions: dict[str, SessionState] = {}
# Lets the HTTP document-upload endpoint push results to the same live
# WebSocket a voice turn would use.
session_sockets: dict[str, WebSocket] = {}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/sessions")
async def get_sessions(limit: int = 50):
    """Lists past intakes (finished or not) straight from durable storage —
    the feature that only became possible once sessions survive past one
    process's lifetime."""
    return {"sessions": await list_sessions(limit)}


async def _get_session_from_memory_or_db(session_id: str) -> Optional[SessionState]:
    """Checks the live in-memory dict first (no DB round trip for an active
    conversation), falling back to durable storage — the case that matters
    for a clinician looking up a finished intake after the process that
    handled it has since restarted."""
    session = sessions.get(session_id)
    if session is not None:
        return session
    return await load_session(session_id)


@app.get("/api/sessions/{session_id}/brief")
async def get_brief(session_id: str):
    session = await _get_session_from_memory_or_db(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if session.protocol is None:
        raise HTTPException(status_code=409, detail="No chief complaint has been identified in this conversation yet")
    brief = generate_clinician_brief(session.record, session.protocol)
    return {
        "brief": {
            "sections": [{"title": s.title, "lines": s.lines} for s in brief.sections],
            "clarifications": brief.clarifications,
            "disclaimer": brief.disclaimer,
        },
        "text": format_clinician_brief_as_text(brief),
    }


@app.get("/api/sessions/{session_id}/fhir")
async def get_fhir(session_id: str):
    session = await _get_session_from_memory_or_db(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if session.protocol is None:
        raise HTTPException(status_code=409, detail="No chief complaint has been identified in this conversation yet")
    return generate_fhir_export(session.record, session.protocol)


async def _push_state_delta(ws: WebSocket, session: SessionState) -> None:
    missing_fields = get_missing_fields(session.record, session.protocol) if session.protocol else []
    await ws.send_json(
        {
            "type": "state_delta",
            "record": session.record.model_dump(),
            "protocol_id": session.protocol.protocol_id if session.protocol else None,
            "protocol_name": session.protocol.name if session.protocol else None,
            "missing_fields": [m.model_dump() for m in missing_fields],
            "safety_log": [e.model_dump() for e in session.safety_log],
            "assistance_requests": [
                {"id": a.id, "reason": a.reason, "timestamp": a.timestamp} for a in session.assistance_requests
            ],
            "documents": [
                {"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at} for d in session.documents
            ],
            "brief_finalized": session.brief_finalized,
        }
    )


async def _send_turn_outcome(ws: WebSocket, session: SessionState, outcome: TurnOutcome) -> None:
    """Shared by the WS voice-turn handler and the HTTP document-upload
    handler so both push results identically. `outcome.transcript` is empty
    for the opening line (Ava speaks first, before the patient has said
    anything) — no patient transcript line to send in that case."""
    if outcome.transcript:
        await ws.send_json({"type": "transcript", "speaker": "patient", "text": outcome.transcript})
    await ws.send_json({"type": "transcript", "speaker": "agent", "text": outcome.reply_text})
    await _push_state_delta(ws, session)
    await ws.send_json({"type": "voice_state", "state": "speaking"})
    await ws.send_bytes(outcome.audio)


@app.post("/api/sessions/{session_id}/documents")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")

    allowed = file.content_type == "application/pdf" or file.content_type in SUPPORTED_IMAGE_MIME_TYPES
    if not allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported document type: {file.content_type}")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (10MB limit)")

    try:
        extracted = extract_text(data, file.content_type)
    except UnsupportedDocumentTypeError as e:
        raise HTTPException(status_code=415, detail=str(e))

    doc = create_uploaded_document(file.filename, file.content_type, extracted.text)
    session.documents.append(doc)
    add_document(session.session_id, doc)

    response = {"document_id": doc.id, "filename": doc.filename, "extraction_method": extracted.method}

    ws = session_sockets.get(session_id)
    if ws is not None:
        note = (
            f'Patient uploaded a document: "{file.filename}". No readable text could be extracted automatically '
            f"(it may be a scanned image with no text layer) — ask the patient to describe what it says."
            if extracted.method == "pdf_text_empty"
            else f'Patient uploaded a document: "{file.filename}".'
        )
        try:
            outcome = await asyncio.to_thread(run_text_turn, session, note)
            await _send_turn_outcome(ws, session, outcome)
            await save_session(session)
        except Exception as e:  # noqa: BLE001 - report but don't fail the upload response
            print(f"[{session_id}] document turn failed: {e}")

    return response


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()

    # A client that already has a session_id from a prior connection (saved
    # client-side, e.g. after a dropped connection or server restart) can
    # ask to pick that exact conversation back up instead of starting a
    # fresh mock appointment — the whole point of persisting to a real
    # database rather than keeping SessionState only in server memory.
    resume_id = websocket.query_params.get("resume")
    session = await load_session(resume_id) if resume_id else None
    is_resumed = session is not None

    if session is None:
        # Simulating a real pre-visit call: the appointment's date/time and
        # reason are already known (standing in for a booking/EHR record),
        # so the protocol is locked immediately rather than discovered live
        # from what the patient says (contrast with turn_controller's
        # deterministic classifier, still used if that reason ever changes
        # mid-conversation).
        appointment = _mock_appointment_details()
        session_id = str(uuid.uuid4())
        session = create_session(session_id, protocol=get_protocol(appointment["protocol_id"]))
    else:
        session_id = session.session_id

    sessions[session_id] = session
    session_sockets[session_id] = websocket

    await websocket.send_json({"type": "session_started", "session_id": session_id})
    await _push_state_delta(websocket, session)

    if is_resumed:
        # Replay what was already said as plain transcript entries — no
        # audio, since past TTS output was never stored, only the text.
        # The patient can just carry on talking from here.
        for turn in session.transcript:
            await websocket.send_json({"type": "transcript", "speaker": turn.speaker, "text": turn.text})
        await websocket.send_json({"type": "voice_state", "state": "listening"})
    else:
        try:
            await websocket.send_json({"type": "voice_state", "state": "thinking"})
            opening_outcome = await asyncio.to_thread(
                run_opening_line, session, appointment["reason_text"], appointment["when_text"]
            )
            await _send_turn_outcome(websocket, session, opening_outcome)
            await save_session(session)
        except Exception as e:  # noqa: BLE001
            print(f"[{session_id}] opening line failed: {e}")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue

            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "barge_in":
                    # Invalidates whatever turn is currently in flight (see
                    # agent/graph.py's staleness check) so it stops making
                    # further model/tool calls instead of finishing in the
                    # background and landing a stale reply in the transcript.
                    session.turn_generation += 1
                    print(f"[{session_id}] barge-in signaled by client")
                continue

            if "bytes" in message and message["bytes"] is not None:
                session.turn_generation += 1
                my_generation = session.turn_generation
                await websocket.send_json({"type": "voice_state", "state": "thinking"})
                try:
                    outcome = await asyncio.to_thread(handle_utterance, session, message["bytes"], my_generation)
                    if not outcome.superseded:
                        await _send_turn_outcome(websocket, session, outcome)
                        await save_session(session)
                    else:
                        # Empty transcription or a stale/interrupted turn: no
                        # reply to speak, but the client still needs telling
                        # to stop waiting and start listening again.
                        await websocket.send_json({"type": "voice_state", "state": "listening"})
                except (WebSocketDisconnect, RuntimeError):
                    # Client already gone (e.g. it disconnected while this
                    # turn was still running) — nothing to send back to.
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[{session_id}] turn failed: {e}")
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        await websocket.send_json({"type": "voice_state", "state": "listening"})
                    except (WebSocketDisconnect, RuntimeError):
                        break
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)
        session_sockets.pop(session_id, None)
        clear_session(session_id)
