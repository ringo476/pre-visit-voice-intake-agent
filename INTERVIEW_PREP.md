# Interview Prep — Pre-Visit Voice Intake Agent

A grilling-style Q&A set for defending this project in an Applied AI interview. Questions are grouped by theme and get progressively harder within each group. Answers are written the way you should actually say them — grounded in what's really in the code, not generic AI-buzzword answers.

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
