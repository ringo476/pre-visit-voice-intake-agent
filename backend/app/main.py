"""FastAPI server: REST endpoints for health/brief/FHIR/document-upload, and
a WebSocket endpoint that carries the voice turns (binary audio in,
JSON control/state messages + binary audio out)."""

import json
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.agent.session import SessionState, create_session
from app.documents.document_ingest import SUPPORTED_IMAGE_MIME_TYPES, UnsupportedDocumentTypeError, extract_text
from app.documents.document_store import add_document, clear_session, create_uploaded_document
from app.output.brief_generator import format_clinician_brief_as_text, generate_clinician_brief
from app.output.fhir_export import generate_fhir_export
from app.state_engine import get_missing_fields
from app.turn_controller import TurnOutcome, handle_utterance, run_text_turn

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

app = FastAPI(title="Pre-Visit Voice Intake Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One in-memory session per active patient conversation. Swappable for a
# real DB later without touching state-engine logic.
sessions: dict[str, SessionState] = {}
# Lets the HTTP document-upload endpoint push results to the same live
# WebSocket a voice turn would use.
session_sockets: dict[str, WebSocket] = {}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/sessions/{session_id}/brief")
def get_brief(session_id: str):
    session = sessions.get(session_id)
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
def get_fhir(session_id: str):
    session = sessions.get(session_id)
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
    handler so both push results identically."""
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
            outcome = run_text_turn(session, note)
            await _send_turn_outcome(ws, session, outcome)
        except Exception as e:  # noqa: BLE001 - report but don't fail the upload response
            print(f"[{session_id}] document turn failed: {e}")

    return response


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()

    # No protocol is chosen up front — the patient just starts talking, and
    # turn_controller's deterministic classifier picks the checklist from
    # what they actually say (see app/protocol/classifier.py).
    session_id = str(uuid.uuid4())
    session = create_session(session_id)
    sessions[session_id] = session
    session_sockets[session_id] = websocket

    await websocket.send_json({"type": "session_started", "session_id": session_id})
    await _push_state_delta(websocket, session)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] != "websocket.receive":
                continue

            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "barge_in":
                    # Batch STT/TTS means there's nothing server-side in
                    # flight to cancel — the frontend stopping local
                    # playback the instant it detects speech IS the
                    # interruption. Logged for visibility.
                    print(f"[{session_id}] barge-in signaled by client")
                continue

            if "bytes" in message and message["bytes"] is not None:
                await websocket.send_json({"type": "voice_state", "state": "thinking"})
                try:
                    outcome = handle_utterance(session, message["bytes"])
                    await _send_turn_outcome(websocket, session, outcome)
                except Exception as e:  # noqa: BLE001
                    print(f"[{session_id}] turn failed: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})
                    await websocket.send_json({"type": "voice_state", "state": "listening"})
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)
        session_sockets.pop(session_id, None)
        clear_session(session_id)
