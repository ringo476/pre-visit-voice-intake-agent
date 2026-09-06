# Pre-Visit Voice Intake Agent

A voice-based pre-visit intake assistant. A patient talks naturally about what's going on before their
appointment; the system asks adaptive follow-up questions and turns the conversation into a structured,
provenance-labeled visit brief for the clinician. It is **not** a diagnostic tool — it collects, clarifies,
and organizes information, and never diagnoses, rules out a condition, or recommends treatment.

Showcase protocol: a respiratory intake (persistent cough), covering onset, course, character, associated
symptoms, relevant history, medications, allergies, and patient concerns.

**Stack**: Python backend (FastAPI + LangGraph + Pydantic + Chroma), simple React (Vite) frontend.

## Why it's built this way

The core design principle is a strict separation between **conversation** and **clinical state**:

- A conversational model (Gemini Flash) owns natural dialogue and can only affect anything through 8
  narrowly-scoped tool calls, orchestrated by a **LangGraph** graph.
- Everything clinically important — fact provenance, the "not asked ≠ denied" rule, safety escalation, and
  final brief generation — lives in deterministic Python code that the model can *trigger* but never
  *override*. A clinician-authored keyword-rule safety engine runs on every fact written, independent of
  whether the model thought to check.
- Speech-to-text and document extraction are separate, dedicated steps (not the same model doing the
  reasoning), so every fact's evidence quote can be checked against real transcript/document text instead
  of trusted on the model's word.
- RAG uses the standard toolchain — real embeddings (Gemini's embedding model, with an offline
  feature-hashing fallback when no API key is set) indexed in **Chroma** — not a hand-rolled trick.

## Architecture

```
Patient audio
   -> Speech-to-Text (Google Cloud, batch per utterance)   [canonical transcript / evidence source]
   -> LangGraph:  reason (Gemini, tool-calling) <-> tools (deterministic Python handlers)
        reason --(no tool_calls)--> END
        reason --(tool_calls)--> tools --> reason  (loop)
   -> reply text -> Text-to-Speech -> audio back to patient

Frontend runs its own voice-activity detection so the patient can interrupt the agent mid-reply
(barge-in) — the frontend stops playback locally the instant it detects speech; this is the entire
interruption mechanism, since STT/TTS are per-utterance batch calls rather than a live stream.

Document upload:
   file -> extract_text() router [deterministic, not the LLM]:
             application/pdf -> pypdf (local, no network)
             image/*         -> Google Cloud Vision OCR
   -> indexed per-session in Chroma -> agent notified via a synthetic "patient" turn
   -> agent can call retrieve_uploaded_document, same pattern as retrieve_existing_patient_context
```

`reason` is the only node that talks to Gemini, and its only power is to speak or request a tool call.
`tools` is plain Python — the only node with authority to change anything (state engine, safety engine,
RAG). The LLM is injectable (see `app/agent/graph.py`), so the whole graph — and the eval suite built on
top of it — is fully testable with a scripted fake model and no network access.

The 8 tools the model can call — each a narrow RPC into exactly one backend module:

| Tool | What it does |
|---|---|
| `update_intake_record` | Record one fact, with a verbatim evidence quote and a `source` (patient_reported / asked_and_denied / document_sourced / inferred / not_asked) |
| `record_patient_correction` | Correct an earlier fact without erasing it — appends a new version |
| `check_safety_protocol` | Ask the deterministic safety engine to evaluate a concerning statement |
| `get_next_intake_question` | Ask what's still missing per the protocol checklist, with RAG-suggested phrasing |
| `retrieve_existing_patient_context` | Look up prior-chart context (RAG, synthetic demo data) |
| `retrieve_uploaded_document` | Look up text extracted from a document the patient uploaded this session |
| `generate_clinician_brief` | Finalize — refuses if required fields are still open, unless given an early-termination reason |
| `request_human_assistance` | Flag for a human; no effect on the clinical record |

## Project structure

```
/backend
  app/
    schemas/        Pydantic models (IntakeRecord, Fact, tool args, protocol config, document)
    protocol/       classifier.py, registry.py + one JSON checklist per complaint type (respiratory, leg injury)
    state_engine.py provenance rules, corrections, missing-field tracker (pure functions)
    policy_engine.py + rules.json — deterministic keyword-rule safety engine
    rag/            embeddings.py, retriever.py — Chroma-backed retrieval, real or offline-hashing embeddings
    documents/      document_ingest.py (extraction router), document_store.py (per-session Chroma store)
    agent/          session.py, tools.py, tool_definitions.py, instructions.py, graph.py (the LangGraph)
    turn_controller.py  STT -> classify -> agent graph -> TTS, one full voice turn
    output/         brief_generator.py, fhir_export.py
    eval/           types.py, scenarios/, runner.py — 7 synthetic scenarios through the real graph
    main.py         FastAPI app: REST + WebSocket
  tests/            pytest — 81 tests, no credentials required
/frontend           React + Vite: welcome screen, live 3-pane conversation view (+ document upload), completion/clinician view
```

## Running it

**Prerequisites**: Python 3.12+, Node.js 20+.

```bash
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Tests and eval (no credentials required — these exercise the deterministic backend and the LangGraph
orchestration directly, with a scripted stand-in for Gemini):**

```bash
cd backend
.venv\Scripts\python.exe -m pytest -q          # 67 tests
.venv\Scripts\python.exe -m app.eval.runner    # 6 synthetic scenarios through the real graph
```

**Running the app live** (needs credentials — see below):

```bash
cd backend
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8080
```
```bash
cd frontend
npm install
npm run dev
```
Then open `http://localhost:5173` in a real browser (mic access won't work in a sandboxed one).

### Credentials for the live voice pipeline

Copy `backend/.env.example` to `backend/.env` and fill in:

- `GEMINI_API_KEY` — reasoning/tool-calling (Gemini Flash) and RAG embeddings
- `GOOGLE_APPLICATION_CREDENTIALS` — path to a GCP service account JSON with Cloud Speech-to-Text, Cloud
  Text-to-Speech, and Cloud Vision enabled

Without these: all 67 tests and the eval suite still run (RAG falls back to an offline hashing embedding,
and the LangGraph tests/eval use a scripted fake model), and the frontend UI works and shows a clear
connection/microphone error rather than crashing.

## What's verified vs. what isn't

**Fully tested (67 automated tests, no external dependency):** state engine provenance rules, safety
engine, RAG retrieval (real Chroma vector store), document extraction router (real PDF text extraction via
a generated test PDF; OCR path exercised with an injected fake), the full LangGraph orchestration loop
(including a genuine loop-guard/recursion test), output generation (brief + FHIR), and the 6-scenario eval
suite run through the real graph.

**Verified structurally but not with real audio:** the backend boots and the WebSocket handshake/session
state push work end-to-end (confirmed live); the frontend renders correctly, including the document-upload
UI, and handles a denied-microphone case gracefully. A real end-to-end voice conversation (mic → STT →
Gemini → TTS) has **not** been tested, since this environment has neither API credentials nor a real
microphone — that's the one thing to smoke-test after adding your own keys.

## Safety design

Three layers, deliberately not resting on the model's judgment alone:

1. **Conversational instructions** ([`app/agent/instructions.py`](backend/app/agent/instructions.py)) — never diagnose, one question at a time, acknowledge uncertainty, escalate rather than improvise.
2. **Deterministic safety policy** ([`app/policy_engine.py`](backend/app/policy_engine.py), rules in [`rules.json`](backend/app/rules.json)) — plain keyword-rule evaluation that runs on *every* fact write, regardless of whether the model called `check_safety_protocol`.
3. **Clinician review** — every generated brief carries "Generated from a patient conversation. Review and verify before clinical use."
