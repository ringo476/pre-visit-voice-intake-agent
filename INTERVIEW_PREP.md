# Interview Prep — Pre-Visit Voice Intake Agent

A grilling-style Q&A set for defending this project in an Applied AI interview. Questions are grouped by theme and get progressively harder within each group. Answers are written the way you should actually say them — grounded in what's really in the code, not generic AI-buzzword answers.

---

## 0. The project flow — say this out loud to open the conversation

**One-breath version:** "It's a voice agent that calls a patient before their clinic appointment, already knowing why they're coming in, has a structured conversation to fill in a clinical checklist, keeps a strict paper trail of what was actually said versus assumed, runs every fact through a deterministic safety scanner regardless of what the AI 'thinks,' and hands the clinician a structured, provenance-labeled brief at the end — with the underlying LLM's power deliberately limited to 'talk, or request one of 8 pre-approved actions,' nothing more."

**Step by step, in order:**

1. **The premise.** A patient has a real, already-booked appointment. This is simulated with `MOCK_APPOINTMENTS` in [main.py](backend/app/main.py) — a random pick of one of 3 complaint types (respiratory, leg injury, allergic reaction) plus a random future date/time, standing in for a real clinic calendar/EHR feed. The complaint isn't discovered by asking "what's wrong" — it's already known, the way a real outbound pre-visit call would already know from the booking record.

2. **Ava speaks first.** The instant the WebSocket connects, before the patient says a word, the backend locks in the protocol (`create_session(..., protocol=...)`) and generates an opening line referencing the real appointment reason and time directly — never "what brings you in today." This isn't one quick canned line: the model runs its normal tool-calling loop first (checks what's missing, records the already-known complaint as an `inferred` fact) and only then produces the spoken greeting — genuinely 2-3 real network round-trips before a single word is said, which is why a visible "thinking" state matters here.

3. **The patient talks, the browser listens.** A volume-threshold voice-activity detector in the browser starts recording on detected speech and stops after enough silence. That clip goes over the WebSocket as raw audio bytes — never text at this stage.

4. **Speech becomes a transcript, once, cleanly.** The backend sends that audio to Google Cloud Speech-to-Text and gets back plain text. This is the *only* place audio ever becomes words — the reasoning model never touches raw audio, only this transcript. That matters later: every recorded fact has to quote this transcript verbatim as its evidence.

5. **The transcript enters the LangGraph loop.** Two nodes: `reason` (Gemini, bound to exactly 8 named tools) and `tools` (plain deterministic Python). `reason` either produces a plain reply (loop ends — that's what gets spoken) or requests tool calls. `tools` executes those requests through real Python code, never trusting the model's account of what happened, and feeds results back to `reason`, which runs again. One "turn" can bounce back and forth a few times before the model has enough information to actually reply.

6. **Every attempted state change gets independently re-validated, never just trusted.** The model can *request* `update_intake_record`, but the tool layer: (a) validates argument shape with Pydantic, (b) rejects any field name outside the locked protocol's closed field list, (c) for an `asked_and_denied` claim, requires a real, previously-logged `question_event_id` proving that field was actually asked about — no fabricating a denial that never happened, (d) runs every new fact through a deterministic keyword safety scanner *unconditionally*, whether or not the model thought to check, and (e) never overwrites a fact — a correction appends a new one linked back to the original, so nothing silently disappears.

7. **The reply gets spoken.** Whatever text `reason` settles on goes to Google Cloud Text-to-Speech, comes back as audio, and streams to the browser over the same WebSocket. The frontend plays it while watching for the patient to talk over it (barge-in) — which immediately stops playback and starts a fresh recording instead of letting the old reply keep finishing pointlessly in the background.

8. **This repeats until the checklist is satisfied.** `get_missing_fields` recomputes, every turn, which required fields still have no real answer — including conditional fields that only activate once something else is genuinely confirmed (e.g. "describe the allergic reaction" only becomes required if a real allergy was affirmed, never if it was denied). Once nothing required is missing, the model can call `generate_clinician_brief`, which refuses to run early if anything's still open.

9. **The output is structured, not a transcript dump.** The final brief groups facts by category and flags provenance concerns; it can also export as a FHIR bundle — the standard format real clinical systems exchange structured data in — so the record is shaped to interoperate, not just to be read once.

10. **None of this is tested by hoping it works.** The LLM is injectable everywhere (`llm=` parameter throughout), so the entire deterministic core — state engine, safety engine, classifier, the full graph loop — has 80+ fast, hermetic unit tests plus a 7-scenario behavioral eval suite, all running with a scripted fake model and zero network access.

---

## 1. System design & architecture

**Q: Walk me through what happens between the patient speaking and the agent replying, in as much technical detail as you can.**
A: Browser detects speech via a volume threshold, records until 2.5s of silence, sends the clip over a WebSocket. The backend runs it through Google Cloud Speech-to-Text to get canonical text — the reasoning model never touches raw audio. A deterministic keyword classifier (or, now, a pre-known booking reason) decides which checklist applies. The text goes into a LangGraph two-node loop: a `reason` node (Gemini, bound to 8 tools) and a `tools` node (plain Python). The loop continues until the model returns plain text with no tool calls — that's the reply, which goes to Google Cloud Text-to-Speech and back to the browser as audio.

**Q: Why separate speech-to-text from the reasoning model instead of using a realtime multimodal voice API that does both at once?**
A: Auditability. If STT is a separate, dedicated step, every fact's "evidence quote" can be checked against a real, saved transcript. If a bundled realtime API is doing speech understanding and reasoning in one black box, you're trusting the model's own account of what it heard — you can't independently verify it. That mattered enough to accept a small latency cost for a real correctness guarantee.

**Q: Why LangGraph and not just a single well-crafted prompt with a while-loop yourself?**
A: I could have written the loop by hand — it's not complex (`reason → tools → reason`). LangGraph mainly buys structured state management (`MessagesState`) and a clean seam for injecting a fake LLM in tests. The real reason to use *a* graph library rather than raw prompt engineering is that this is fundamentally a tool-calling agent, not a single completion — the model needs to see the results of its own actions before deciding what to say, which requires actual control flow, not just a bigger prompt.

**Q: What's the actual blast radius of the LLM in this system? What can it never do, no matter what it's prompted to do?**
A: It can only call one of 8 named functions. It cannot: write directly to the record (only `update_intake_record` can, and that goes through Pydantic validation plus a field-enum check first), decide whether something's a safety emergency (a separate deterministic keyword engine does that, unconditionally, on every fact write), pick which patient's data it's touching (no `session_id` argument exists — it's bound server-side), or decide the conversation's structural shape (protocol classification is keyword-based, not the model's call).

**Q: This whole thing runs synchronously per turn. How would you scale it to handle many concurrent patients?**
A: The blocking I/O calls (STT, Gemini, TTS) are already wrapped in `asyncio.to_thread`, so one slow call doesn't freeze the whole server's event loop — that was a real bug I found and fixed after a turn taking too long caused the WebSocket to look dead. For true multi-patient scale, the next bottleneck would be the default thread pool size for `asyncio.to_thread`, and Chroma's lack of a real async client — RAG lookups would still need thread-offloading regardless of how async the rest got.

---

## 2. The provenance model (expect this to get poked hard)

**Q: What stops the model from just lying about what the patient said?**
A: It can't fabricate a denial. Every fact has a `source` tag, and the one enforced-in-code rule: `asked_and_denied` requires a `question_event_id` matching a real, logged `QuestionEvent` for that exact field. If the model tries to mark something denied without a matching logged question, `apply_fact()` raises `ProvenanceViolationError` and the write is rejected outright — not flagged for review later, rejected at write time.

**Q: Couldn't the model just call `get_next_intake_question` to manufacture a fake question_event_id, then immediately claim a denial?**
A: No — `get_next_intake_question` returns a *real* `QuestionEvent`, logged into `session.question_events` by the deterministic handler, not something the model invents. The model can only ever reference an ID that our own code actually created and can look up. It has no path to generate an ID that would pass validation without a genuine matching event existing.

**Q: What happens to a fact once it's written — can it be edited?**
A: Never mutated. `record_patient_correction` appends a new fact linked via `supersedes` to the original; the original stays in the record. A clinician can see both what was first said and what was later corrected — nothing silently disappears.

**Q: Isn't "not asked" vs "asked and denied" a distinction most systems wouldn't bother making? Why does it matter here?**
A: Because collapsing them is a real patient-safety failure mode: if a field that was never actually covered gets rendered the same as "patient confirmed no," a clinician might skip following up on something nobody ever actually checked. The provenance tag is the whole point of the project's honesty guarantee — it's not decoration.

---

## 3. Safety engine & guardrails

**Q: Why is safety checking a separate system instead of a tool the model calls when it thinks something's concerning?**
A: Because "when it thinks to" is exactly the failure mode to design around. `check_safety_protocol` exists as a tool the model *can* call, but the real guarantee is that `evaluate_fact()` runs automatically inside `update_intake_record`, on *every* fact write, regardless of whether the model asked for a check. Even a model that completely fails to notice a red flag still gets caught by this second, independent layer.

**Q: This is just keyword matching. Isn't that trivially easy to fool, and wouldn't a small classifier model do a much better job?**
A: Yes, keyword matching has real limits — it can miss paraphrased danger signs and can false-positive on incidental word use. But the design goal wasn't "best possible detector," it was "a detector whose behavior is 100% predictable and auditable, with zero chance of being talked out of firing." A classifier model reintroduces exactly the kind of judgment-call uncertainty this layer exists to remove. If I were hardening this for production, I'd want both — a learned classifier for coverage, plus this deterministic layer as the non-negotiable floor.

**Q: What's the actual difference between your two escalation tiers?**
A: `emergency_escalation` (911-level — severe breathing difficulty, chest pain radiating, anaphylaxis airway swelling, loss of consciousness) tells the patient to stop and call emergency services immediately. `urgent_escalation` (be-seen-today-level — can't bear weight, spreading redness, dizziness during an allergic reaction) tells them to go to urgent care rather than wait for the scheduled appointment, but isn't a 911 situation.

**Q: You added a third protocol for allergic reactions later. Did you have to duplicate any safety rules for it?**
A: No — I deliberately reused the existing field name `breathing_difficulty` for the allergy protocol's dyspnea field, so the pre-existing `severe_breathing_difficulty` rule (keyed only by field name, not by protocol) automatically covers it too, with zero duplication. I did add two new rules specific to allergy: airway swelling and reaction-related dizziness, since those didn't have an equivalent in the other two protocols.

**Q: What is negative prompting, and where does it stop being enough?**
A: Instructions phrased as "never do X" in the system prompt — this project has several ("never diagnose," "never recommend changing medication"). It stops being enough anywhere the consequence of the model ignoring the instruction is serious, which is why nothing safety-critical here relies on it alone — the safety engine, the provenance gate, and the field-enum are all guardrails: code-level checks that hold regardless of whether the model behaves.

---

## 4. Tool calling & structured output

**Q: Does the model send you raw JSON text that you parse?**
A: No. Gemini's function-calling has a real structured response type for "I want to call this tool" — LangChain's wrapper converts that into a Python `AIMessage.tool_calls` list of already-parsed dictionaries before my code ever sees it. I never manually parse model-generated JSON text.

**Q: Where does Pydantic actually come in, and what does it catch that the JSON schema sent to the model doesn't?**
A: The schema sent to Gemini (`tool_definitions.py`) shapes what the model is *likely* to send, but nothing stops a model from sending something malformed anyway — wrong type, missing field, invalid enum value. `UpdateIntakeRecordArgs(**raw_args)` (a Pydantic model) is the actual gate: it validates the raw dict before any business logic runs, and a `ValidationError` gets turned into a plain-English rejection sent back to the model as a tool result.

**Q: You mentioned the model once invented a field name that didn't exist. Walk me through exactly what happened and how you fixed it.**
A: The `field` argument was originally just `{"type": "string"}` with two example names in the description. For "timing pattern," the model reasonably guessed `timing_pattern` instead of the protocol's real field, `timing`. It got rejected, but only after a full network round-trip. The fix: `build_tool_definitions(field_names)` dynamically injects a JSON-schema `enum` of the *actual* locked protocol's real field names into that argument. Once it's a closed list, guessing wrong is structurally impossible, not just penalized after the fact.

**Q: Why does that fix require caching the LLM client per protocol instead of one global model instance?**
A: Because the tool schema itself now depends on which protocol is locked — `.bind_tools()` bakes the schema into the model object at bind time. One global model built at first use couldn't reflect a per-protocol enum. I switched to a small dict cache keyed by `protocol_id` (or `None` before a protocol is known), so each distinct protocol gets its own correctly-scoped bound client, built once and reused.

**Q: Why can a single model turn sometimes take 2-3 real round trips instead of one?**
A: Not because of how many facts it writes — it can bundle several tool calls into one message. It's because the model can't know whether those calls succeeded until it's told, and it can't decide what to actually say until it knows that. So the minimum is always two trips: one to act, one to see the results and respond. A third happens only if something got rejected and needs correcting.

---

## 5. RAG

**Q: What's actually being retrieved, and why do you need retrieval at all for a project this scale?**
A: Two small per-protocol stores: a synthetic "prior chart" note (so the agent can reference things like a documented past allergy without re-asking) and follow-up guidance (phrasing/rationale hints for what to ask next). Honestly, at this scale, retrieval isn't solving a "can't fit it in the prompt" problem — it's demonstrating the RAG pattern cleanly: real embeddings, real vector search, a fallback path when no API key is present.

**Q: You mentioned Chroma uses HNSW under the hood. Was that a deliberate scaling decision?**
A: No, and I'd say so directly if asked — I checked the installed package and confirmed Chroma's persistent vector index is HNSW-based, but that's just what Chroma always uses, not something I chose for this project's scale. With a handful of documents per store, brute-force exact search would already be instant; HNSW isn't buying anything real here, it's just incidental to the library.

**Q: You found a real chromadb bug. What was it and how did you actually diagnose it, rather than just retry your way past it?**
A: Chroma's default *ephemeral* in-process client silently shares one underlying engine across every collection created in that process. Under enough concurrent collections, that shared engine would return `.count()` correctly while `.similarity_search()` came back empty, and `.get()` sometimes raised an internal error — for data that demonstrably still existed. I confirmed it wasn't a timing issue by writing isolated repro scripts and testing longer retries and eager loading, both of which failed to fix it reliably. The actual fix was giving every store its own persistent, on-disk client via a unique `persist_directory`, which sidesteps the shared engine entirely.

**Q: Why embeddings and not just keyword search for retrieval, given how deterministic everything else in this system is?**
A: Retrieval quality (finding the *most relevant* reference sentence, not just one with matching words) is one of the few places where "approximately right by meaning" is actually the goal, unlike protocol classification or safety checking, where "exactly, auditably correct" is non-negotiable. It's also a deliberate real embedding model integration (`gemini-embedding-001`) with a deterministic offline fallback when no key is present, so the app still runs — just with lower ranking quality — without credentials.

---

## 6. Multi-protocol design & its real limitations

**Q: How does the system decide which checklist applies, and why not let the model decide?**
A: A plain keyword-scoring function, `classify_complaint()` — highest-scoring protocol wins, no match returns `None`. This is deliberately kept out of the model's hands because it's a structural decision: it determines which safety rules are even reachable for the rest of the session. Too foundational to leave to "whatever the model felt like this time."

**Q: What happens if a patient mentions two unrelated complaints in the very first message — say, a cold and thigh pain?**
A: Honestly — right now, one gets silently dropped. I traced this exactly: "cold" and "thigh" score 1-1 across the two protocols, and on a tie the classifier keeps whichever was checked first (declaration order), with zero acknowledgment that a second complaint was even mentioned. That's worse than the mid-conversation case, where a *later* topic switch at least generates a deflection note telling the patient to raise it separately.

**Q: How would you fix that?**
A: Detect the near-tie and, instead of silently picking one, have the deterministic layer instruct the model to ask the patient which one they'd like to focus on today — a subjective, patient-reported preference, not the AI making a clinical urgency judgment. The one that's deprioritized would get saved as a new always-available free-text field (something like `secondary_concern_note`), with the same evidence/provenance tagging as everything else, rendered in the final brief as "Also mentioned, not the focus of this visit" instead of vanishing.

**Q: Would that fix risk missing a genuine emergency hiding in the "less important" topic?**
A: No, and that's worth checking explicitly rather than assuming: the safety keyword scanner runs on raw fact text regardless of which protocol won the classification — it isn't gated by the classifier's choice. As long as whatever gets said about the deprioritized topic still gets written down somewhere (even as the free-text side note) and passes through that same scan, an emergency wouldn't get missed just because it lost a topic tie-break.

---

## 7. Concurrency & the barge-in bug (good "tell me about a bug you fixed" material)

**Q: Tell me about a real bug you found in this system, not something hypothetical.**
A: Interrupting the agent mid-reply ("barge-in") stopped local audio playback client-side, but the *server-side* turn that was interrupted just kept running to completion in the background regardless — writing facts and appending its stale reply to the transcript after the fact. That stale reply would then confuse the *next* real turn, which replays the whole transcript from scratch: I watched a real log where this caused the model to re-extract and re-record seven facts it had already captured moments earlier, because a half-finished, ignored reply had polluted the conversation history it was replaying.

**Q: How did you actually fix that, and why not just use `asyncio.Task.cancel()`?**
A: I added a generation counter on the session, bumped on every new recorded utterance and on a barge-in signal. A turn captures that number when it starts; the graph's `reason` and `tools` nodes check it before doing any further work, and stop immediately — no further model calls, nothing written — the instant a newer utterance has superseded them. True task cancellation would be cleaner in principle, but the blocking work runs via `asyncio.to_thread`, and a Python thread can't be forcibly killed once started — you can only ignore its eventual result, which is effectively what the generation check does, just made explicit instead of implicit.

**Q: How did you verify that fix actually worked, rather than just assuming the logic was right?**
A: I wrote a standalone script with a fake LLM that bumps the generation counter *inside* its own `.invoke()` call — simulating a barge-in arriving mid-flight — and asserted four things directly: the model was only called once (the would-be second call was skipped), zero facts were written from the interrupted attempt, its reply never landed in the transcript, and the turn correctly reported itself as superseded. All four passed before I touched the live server.

---

## 8. Testing & evaluation

**Q: How do you test a system whose core behavior depends on a non-deterministic LLM?**
A: The LLM is injectable everywhere it's used (`run_agent_turn`, `build_graph` all take an optional `llm` parameter). Tests and the eval suite pass in a scripted fake model, so the entire deterministic core — state engine, safety engine, classifier, the full LangGraph loop, output generation — is tested with zero network access and zero flakiness. 81 tests, all of them fast and hermetic.

**Q: What does the eval suite measure that unit tests don't?**
A: End-to-end behavioral metrics across 7 synthetic scenarios run through the *real* graph (fake LLM, everything else real): red-flag recall (does the safety engine actually catch every scripted emergency), false-escalation rate (does it over-trigger on benign statements), and percentage of recorded facts linked to a real evidence quote. Unit tests check that individual functions are correct; this checks that the assembled pipeline behaves correctly end to end.

**Q: What's NOT tested, and why does that gap exist?**
A: The actual live voice path — real microphone, real Gemini, real Google Cloud STT/TTS — has only been exercised manually, not in an automated test, since that requires live credentials and a live network by definition. The deterministic core doesn't need any of that to be fully verified; the live integration is the one place automated coverage genuinely can't reach without becoming a flaky, credential-dependent test.

---

## 9. Curveballs / "what would you do differently"

**Q: If you were told this had to go to production tomorrow, what's the first thing you'd flag as not ready?**
A: The safety engine only scans facts and explicit safety-tool calls — nothing currently scans the model's *final spoken reply* for a policy violation before it's spoken. If the model slipped and said something diagnosis-shaped despite the instructions, nothing would catch it. Everything else that matters has a guardrail behind it; the reply text itself doesn't yet.

**Q: What's the difference between this and a "real" pre-visit voice agent used by an actual clinic?**
A: This is inbound and browser-based — the patient opens a tab and talks. A real one is outbound: it dials the patient's phone 1-3 days before the appointment, already knows the chief complaint from the clinic's calendar/EHR, and opens by referencing it directly rather than discovering it live. I actually built a simulation of that last part — protocol locked at session start from a mock "booking record," Ava speaks first — but the telephony (placing an actual phone call), the scheduler (deciding *when* to call), and a real appointment data model are all genuinely missing infrastructure, not just simplified versions of something that exists.

**Q: Why does the final output include a FHIR export? What's that actually for?**
A: FHIR is the standard data format clinical systems (EHRs) exchange structured health data in. Producing a (mocked) FHIR bundle — QuestionnaireResponse, MedicationStatement, AllergyIntolerance — demonstrates that the structured record isn't just for display, it's shaped to actually interoperate with real clinical infrastructure, and it inherits the same honesty rule as everything else: it only ever includes fields that were actually reported, never fabricates an entry for something never asked.

**Q: What would you say is the single most defensible design decision in this whole project?**
A: Keeping the LLM's power strictly to "speak, or request one of 8 named actions" and putting every actual state change behind deterministic Python that re-validates everything independently. It's the one decision that every other guardrail in the system — provenance, safety, field validation — is really just a specific instance of.

---

## 10. The earlier bug history — problems faced building this, before today's testing round

**Q: A model you depended on got deprecated mid-project. What happened?**
A: `gemini-2.5-flash` started returning a 404, "no longer available to new users" — a live external dependency changing under me with zero warning, mid-build. The fix itself was trivial (switch the default to `gemini-3.6-flash` via the `GEMINI_MODEL` env var the code already read), but the real lesson is architectural: an agent is only as stable as a fast-moving vendor's model lineup. In production I'd want a model allowlist with a fallback, not a single hardcoded default.

**Q: Gemini's responses started breaking your parsing code partway through. Why?**
A: Newer Gemini models sometimes return `AIMessage.content` as a list of content blocks (`[{"type": "text", "text": "..."}]`) instead of a plain string — a silent shape change, not something clearly documented. Fixed with a small `_extract_text()` normalizer in `graph.py` that handles both shapes, used everywhere reply text gets read.

**Q: You mentioned a "model prefilling" crash. What was the actual rule you violated?**
A: Gemini requires the last message in a request to be a user turn or a function response — it rejects a request that ends on a system message. I was appending a steering "system_note" *after* replaying the full transcript, which made a system message the last thing in the list whenever that note was present. Fixed by moving the note to right after the persona system message, before the transcript replay, so the list always ends on the patient's own turn. I applied the same lesson proactively when building the opening-line generator later — using a `HumanMessage` for the synthetic "start the call" trigger instead of a second `SystemMessage`, avoiding the identical failure before it ever happened live.

**Q: What happens if a patient's recording is silence, or too short to transcribe?**
A: Originally, it crashed the turn — an empty string sent to Gemini fails with "contents are required." Fixed by having `handle_utterance` check for a blank Speech-to-Text result up front and return a no-op, superseded outcome before ever reaching the model. This is also *why* saying "hello" too early on a call can seem to vanish without a trace: a short or quiet utterance can come back blank from STT and get silently dropped by this exact guard, invisible in the conversation log.

**Q: You mentioned a crash-on-crash around disconnects. What was actually happening?**
A: A client disconnecting mid-turn spammed `RuntimeError('Cannot call "receive" once a disconnect message has been received.')` on every dev reload. Fixed with an explicit check for the disconnect message type ahead of the generic fallback. That surfaced a second problem: the error-reporting path itself then crashed trying to send an error to an already-closed socket (`Cannot call "send" once a close message has been sent`) — so both the main turn-processing block and the error-reporting fallback got wrapped to just break the loop cleanly on either exception, instead of one failure cascading into another.

**Q: When you added the third (allergy) protocol, did the classifier work correctly first try?**
A: No — and I caught it with a script, not by inspection. `classify_complaint("my throat feels like its closing and I feel dizzy")` returned `None`. The multi-word keyword `"throat closing"` didn't literally substring-match the natural phrasing "throat feels like its closing," and `"dizzy"` wasn't in the keyword list at all — only the field name `dizziness_or_fainting` existed, which nobody actually says out loud. Fixed by adding the missing keywords and phrasing variants, then re-ran the same script to confirm the fix, rather than assuming it worked.

**Q: Is there an operational hazard with Chroma, separate from the shared-engine bug you already mentioned?**
A: Yes, a distinct issue. If the live backend and the pytest suite run concurrently against the same on-disk `.chroma_data` directory on Windows, file locking causes stale state — tests fail with a dimension mismatch even though nothing about the embeddings changed. It's not a code bug, it's an operational rule: never run the test suite and the live server against the same Chroma directory at once. My working pattern: find the exact process holding the port via `netstat`, kill only that specific PID, delete `.chroma_data`, then test cleanly. I did once use a broader `taskkill` that matched more processes than intended, and flagged that directly rather than treating it as a non-issue.

---

## 11. Live debugging session — today's round of real bugs found by actually using the product

**Q: Tell me about a bug you found today, live, that had nothing to do with the LLM at all.**
A: Audio overlap. `playAudio()` on the frontend created a brand-new `Audio` element and played it on every incoming message, with no check for whether a previous reply was still playing. If a second reply's audio arrived before the first had finished, the old `Audio` object just kept playing in the background — nothing ever called `.pause()` on it — while the new one started on top. Two genuinely separate `<audio>` elements playing at once. The fix was one line: call `stopPlayback()` at the top of `playAudio()` before starting anything new.

**Q: What actually caused replies to pile up and overlap in the first place — wasn't that supposed to be impossible?**
A: The voice-activity detector had no concept of "a turn is already in flight." It only checked two things: is audio currently playing (barge-in path), or am I already mid-recording. There was no third check for "I already sent a recording and I'm still waiting on the backend's reply." So saying something again while the agent was silently "thinking" (a real multi-second Gemini + TTS round trip) queued up a second, fully independent turn — and since the backend processes turns strictly in order but with almost no gap between two already-buffered messages, the replies landed back-to-back with no natural pause, sounding like overlapping speech. The fix added a `turnInFlightRef` that locks the mic out from starting a new recording until the current turn's reply actually starts arriving — including a special case for the very first turn (the opening line), since that one isn't triggered by the patient's own recording at all, so the usual lock-on-send never applied to it.

**Q: You added a "thinking" indicator for the opening line. What was actually wrong before that, and why did it take three network round-trips just to say hello?**
A: The opening line isn't a canned string — it runs through the exact same `reason ⇄ tools` loop as any other turn. From the real backend log: the model calls `get_next_intake_question`, then `update_intake_record` (to log the already-known complaint as an `inferred` fact), then `get_next_intake_question` again, and only after those three real Gemini calls does it produce the actual greeting text. Meanwhile `main.py` never sent a `"thinking"` state message during that specific window (it only did that for regular turns), so the frontend sat on a blank "Listening" screen the whole time — genuinely working, but looking frozen. The fix was one added line sending `{"type": "voice_state", "state": "thinking"}` right before generating the opening line.

**Q: What happened when the backend got restarted mid-session, and why does that matter?**
A: The UI reset to the welcome screen as expected (the WebSocket died), but audio kept playing in the background — a real bug, not an artifact of the restart. `ws.onclose` only reset the React state; it never called `stopPlayback()`, never cleared the voice-activity-detector's interval, never released the microphone. `endSession()` (the "I'm done" button path) did all of that correctly — `onclose` just didn't share that cleanup. I refactored the shared logic into one `cleanupLocalMedia()` function called from both places, so any disconnect — intentional or not — actually releases everything instead of leaving an orphaned audio element and a still-running VAD loop behind a UI that claims nothing is happening.

**Q: Walk me through the false-positive anaphylaxis escalation you found, and what it revealed about the safety engine's real weakness.**
A: A patient statement got the exact scripted anaphylaxis message read back verbatim, despite the conversation being about unrelated medications. Reading `policy_engine.py` end to end, the root issue is that `_matches_keyword()` was pure substring matching with zero negation awareness — a sentence *denying* a symptom ("no swelling on my face") contains the identical keyword text as a sentence *reporting* it. The fix scans backward from each keyword match for a nearby negation word (no, not, denies, without, never...) within an 8-word window, and only counts it as a real hit if there isn't one. I verified it against both the exact denial case and a harder compound sentence ("denies any face swelling or throat closing") where the negation word is several words before the second symptom — both correctly suppressed — while confirming real positive reports ("face swelling getting worse," "passed out") still fire normally.

**Q: Separately, a patient explicitly denied a medication allergy, and the system still asked them to describe the allergic reaction. What was that bug?**
A: `is_affirmed()`, which decides whether a conditional field like "describe the reaction" becomes required, only checked the fact's `source` (was this `patient_reported`, as opposed to `not_asked`) — it never looked at the fact's actual `value`. A fact saying "no medication allergies" with `source: patient_reported` was treated identically to a fact saying "yes, allergic to penicillin," because both have an "affirmative" source category, which just means "the patient told us this directly" — it says nothing about whether the answer was yes or no. The fix added a value-level check for denial-shaped text ("no", "none", etc. as the leading word) before treating a conditional field as triggered, verified against both the real denial case (correctly stops requiring the follow-up now) and a genuine allergy case (still correctly requires it).

**Q: Did you find anything you noticed but decided NOT to fix, and why?**
A: Yes — the model sometimes ignores `get_next_intake_question`'s own suggested next field and asks about something else instead, based on its own conversational judgment (e.g. the tool said "suspected_trigger is still missing" but the model asked about affected body areas instead, since the patient had just mentioned "hives"). I traced it directly in the logs rather than guessing. I didn't patch it, because the tool's suggestion is deliberately advisory, not a hard constraint — forcing an exact question order would make the conversation feel robotic, and in the one case I observed, the patient volunteered the missing field anyway on the very next turn, so no data was actually lost. It's a named, real limitation, not a data-integrity bug: worth mentioning proactively if asked "what would you improve further," not worth a defensive patch that trades naturalness for a marginal, unconfirmed benefit.

---

## 12. Model authoring: when to trust a model vs deterministic code

**Q: Why does this project use hand-written JSON checklists instead of a model deciding what to ask, live?**
A: Because a missed required safety question in a clinical context is a patient-safety and liability issue, not just a UX rough edge. If the question list is something the model improvises turn to turn, "why didn't it ask about anaphylaxis red flags this time" has no better answer than "the model didn't feel like it" — that's not acceptable in a regulated clinical setting. A clinician-authored, versioned checklist gives every session for a given complaint type a guaranteed minimum bar, and an auditable reason for every question asked.

**Q: Three hand-typed JSON files obviously don't scale to real production. Would you use a model to solve that, and how?**
A: At build time, not runtime. I'd give a model a complaint category plus real source material — clinical guidelines, review-of-systems templates — and have it *draft* a candidate checklist: fields, required-ness, safety keywords. A clinician reviews and signs off on that draft before it ever reaches a real patient. The deployed, patient-facing conversation still runs against the reviewed, versioned, deterministic artifact — never a live model opinion about what's clinically necessary. That gets the model's speed for authoring at scale without exposing any patient to the model's unreviewed judgment at the moment it actually matters.

**Q: Is "build time vs runtime" just a one-off idea for the checklist, or does it show up elsewhere in this project?**
A: It's the same pattern behind every guardrail in this system, not a one-off. `rules.json` (the safety engine) is a clinician-authored file, reviewed and versioned — not something an LLM decides live. The tool schema's field `enum` is built once from the locked protocol's real field list, not guessed by the model each turn. The protocol JSON files themselves are the same idea. In every case, a human authors and reviews something *before* any patient interaction happens, and the live system can only trigger that reviewed artifact — never redefine it on the fly. If an interviewer asks "what's the unifying design principle here," this is it.

**Q: Would you ever trust a model's live judgment for anything in a system like this?**
A: Yes — for exactly the things that are recoverable and low-stakes if wrong: phrasing, tone, deciding which of several already-approved facts to mention first, small talk. The dividing line isn't "LLMs are untrustworthy," it's "is a mistake here reviewable after the fact, or does it directly become a patient-safety incident in real time." Conversational judgment calls are the former; anything touching the clinical record or an escalation decision is the latter, and that's exactly the line this project draws between the `reason` node's freedom and the `tools` node's determinism.

---

## 13. Known gaps — the honest, consolidated "not solved yet" list

This is the one section to have fully loaded if asked "what would you do next" or "what's the weakest part of this." Two of these are genuinely deep and haven't come up elsewhere in this document; the rest are pointers back to where they're already covered in detail.

**Q: You verify a denial can't be fabricated. Does anything verify a recorded value is actually *true* to what the patient said?**
A: No, and this is the deepest real gap in the whole system. The provenance system enforces the *claim* — a real, logged question exists for this field before you're allowed to say "denied." It does not, and structurally cannot, enforce the *content* — whether the value or evidence quote actually matches what's in `session.transcript`. `apply_fact()` doesn't even receive the transcript as an argument; it physically has no way to cross-check against it. The only thing standing between the model and a fabricated-but-plausible evidence quote is a prompt instruction ("use the patient's own words, never paraphrase") — which is a request, not an enforced rule, exactly like every other "never do X" instruction in this project. Closing this would need either a second verification pass (a cheap similarity check between the claimed evidence and the actual turn's transcript text before allowing the write) or human review before facts are finalized — neither exists today.

**Q: You fixed the safety engine's negation blindness. Is the keyword matching fully sound now?**
A: No — I fixed negation, but a separate, narrower gap remains. `evaluate_fact()` only checks a rule if `rule["field"] == fact.field`, so a fact about medication history can never trip an anaphylaxis rule. But `evaluate_statement()` — the path used when the model calls `check_safety_protocol` with a free-text statement, rather than a structured fact — checks the statement against *every* rule's keywords with no field filter at all, by design, since the model is describing something not yet written as a fact and doesn't know which field it belongs to. That's a reasonable reason for it to exist, but it does mean a statement about one topic could still theoretically trip a rule "about" something unrelated if the wording happens to overlap — the exact class of bug I fixed for negation still has a narrower, unresolved sibling here.

**Q: What other gaps have already come up, that are worth having ready as a quick list?**
A: In order of how often I'd expect this to get asked: (1) the model's final spoken reply is never itself scanned for a policy violation before being said out loud — only facts and explicit tool calls are checked (Section 9). (2) Two unrelated complaints mentioned in the same first message silently drop one on a scoring tie, with no acknowledgment (Section 6). (3) The model can ignore `get_next_intake_question`'s own suggested field and ask about something else instead — advisory, not enforced (Section 11). (4) This is inbound and browser-based; a real outbound agent needs telephony, a call scheduler, and a real EHR/booking integration, none of which exist here (Section 9). (5) The live audio path (real mic, real STT/TTS) is verified manually only — it can't be covered by an automated test without becoming credential-dependent and flaky by definition (Section 8).
