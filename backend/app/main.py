"""FastAPI server: REST endpoints for health/brief/FHIR/document-upload, and
a WebSocket endpoint that carries the voice turns (binary audio in,
JSON control/state messages + binary audio out)."""

import asyncio
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from app.agent.session import SessionState, create_session
from app.db import DATABASE_URL, init_db
from app.documents.document_ingest import SUPPORTED_IMAGE_MIME_TYPES, UnsupportedDocumentTypeError, extract_text
from app.documents.document_store import add_document, clear_session, create_uploaded_document
from app.identity import issue_access_token, verify_access_token
from app.logging_config import configure_logging, get_logger
from app.output.brief_generator import format_clinician_brief_as_text, generate_clinician_brief
from app.output.fhir_export import generate_fhir_export
from app.persistence import list_sessions, load_session, record_access, save_session
from app.protocol.registry import get_protocol
from app.state_engine import get_missing_fields
from app.turn_controller import TurnOutcome, handle_utterance, run_opening_line, run_text_turn

configure_logging()
logger = get_logger(__name__)

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
    logger.info("server startup complete", extra={"database_url_scheme": DATABASE_URL.split("://")[0]})


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
async def health():
    """Actually checks the database is reachable, not just that the process
    is alive — a process that's up but can't reach its database is not
    healthy, and a load balancer or orchestrator relying on this endpoint
    needs to know that."""
    try:
        from sqlalchemy import text

        from app.db import get_session_factory

        async with get_session_factory()() as db:
            await db.execute(text("SELECT 1"))
        return {"ok": True, "database": "reachable"}
    except Exception as e:  # noqa: BLE001
        logger.exception("health check: database unreachable")
        return JSONResponse(status_code=503, content={"ok": False, "database": "unreachable", "error": str(e)})


@app.get("/api/sessions")
async def get_sessions(limit: int = 50):
    """Lists past intakes (finished or not) straight from durable storage —
    the feature that only became possible once sessions survive past one
    process's lifetime."""
    return {"sessions": await list_sessions(limit)}


@app.post("/api/appointments/mock")
async def book_mock_appointment(payload: dict = Body(...)):
    """Stands in for a real clinic's booking/EHR system generating a
    pre-visit call link and sending it to the patient (SMS/email) ahead of
    time — there's no real scheduling integration or messaging provider
    here, so this endpoint plays that role directly instead. What it
    returns (session_id + a signed, time-limited access_token) is exactly
    what a real link would embed; the frontend calling this on page load is
    the simulated stand-in for "the patient already received and opened
    that link," not a shortcut in the verification itself — the token
    issued here is genuinely checked, not trusted blindly, by the
    WebSocket endpoint below.

    Requires explicit consent up front: {"consent": true}. Without it, no
    session is created at all — consent isn't a checkbox that happens to
    exist somewhere, it's a precondition for a session coming into being."""
    if payload.get("consent") is not True:
        raise HTTPException(status_code=400, detail="Patient consent is required before a session can be created.")

    appointment = _mock_appointment_details()
    session_id = str(uuid.uuid4())
    session = create_session(
        session_id,
        protocol=get_protocol(appointment["protocol_id"]),
        consent_given_at=datetime.now(timezone.utc).isoformat(),
        appointment_reason_text=appointment["reason_text"],
        appointment_when_text=appointment["when_text"],
    )
    await save_session(session)
    await record_access(session_id, "session_booked")

    token = issue_access_token(session_id)
    logger.info("mock appointment booked", extra={"session_id": session_id, "protocol_id": appointment["protocol_id"]})
    return {"session_id": session_id, "access_token": token}


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
    await record_access(session_id, "brief_viewed")
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
    await record_access(session_id, "fhir_exported")
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
        except Exception:  # noqa: BLE001 - report but don't fail the upload response
            logger.exception("document turn failed", extra={"session_id": session_id})

    return response


@app.websocket("/ws")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()

    # Every connection — the very first one right after booking, or a later
    # reconnect after a dropped connection or server restart — must present
    # the signed access token issued by /api/appointments/mock. This is the
    # actual identity check: without it, anyone who guessed or intercepted
    # a session_id could connect as that patient. The token is verified,
    # not just present-checked — tampered, expired, or made-up tokens are
    # rejected the same as a missing one.
    token = websocket.query_params.get("token")
    session_id = verify_access_token(token)
    if session_id is None:
        await websocket.send_json({"type": "error", "message": "Missing or invalid access token."})
        await websocket.close(code=4401)
        return

    session = await load_session(session_id)
    if session is None:
        # A structurally valid, correctly-signed token for a session that
        # doesn't actually exist in the database shouldn't happen in normal
        # operation (booking always persists before issuing a token), but
        # never trust a client-supplied value into "this must be fine."
        await websocket.send_json({"type": "error", "message": "This appointment could not be found."})
        await websocket.close(code=4404)
        return
    if session.consent_given_at is None:
        await websocket.send_json({"type": "error", "message": "Patient consent was not recorded for this appointment."})
        await websocket.close(code=4403)
        return

    # A session with no transcript yet has never actually spoken to the
    # patient — this is the first real connection, and the opening line
    # still needs to run. Any transcript already present means a previous
    # connection got at least that far, so this is a reconnect: replay what
    # was said instead of speaking the opening line a second time.
    is_resumed = len(session.transcript) > 0

    sessions[session_id] = session
    session_sockets[session_id] = websocket
    await record_access(session_id, "voice_reconnected" if is_resumed else "voice_connected")

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
                run_opening_line, session, session.appointment_reason_text, session.appointment_when_text
            )
            await _send_turn_outcome(websocket, session, opening_outcome)
            await save_session(session)
        except Exception:  # noqa: BLE001
            logger.exception("opening line failed", extra={"session_id": session_id})

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
                    logger.info("barge-in signaled by client", extra={"session_id": session_id})
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
                    logger.exception("turn failed", extra={"session_id": session_id})
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
