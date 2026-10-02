"""The agent's LangGraph: a loop between reasoning (Gemini bound to the 8
tools) and tool execution (our own deterministic handlers), with one pause.

    reason --(no tool_calls)--> END
    reason --(has tool_calls)--> tools --> reason (loop)
    tools --(the model asked to finish)--> confirm_finalization --> END

`reason` is the only node that talks to Gemini, and its only power is to
speak or request a tool call. `tools` is plain Python — it's the only node
with authority to change anything (state engine, safety engine, RAG).

Three LangGraph features are used on purpose:

  * The graph is compiled ONCE (get_graph) and serves every session. What differs
    per run (the live session, this turn's tool handlers, the model) is passed in
    at run time as a RunContext, never captured in the graph.
  * A checkpointer saves the graph's state after every step, under a thread id
    equal to the session id. Each turn replaces the saved messages with the
    history rebuilt from the verbatim transcript, so a checkpoint shows exactly
    what the model saw and did on the latest turn, and a run can be paused.
  * confirm_finalization uses interrupt(). Finishing the intake is the one
    irreversible step, so the run pauses there with the read-back for the
    patient, and resumes (Command(resume=...)) with their next utterance.
    Only an explicit yes from a patient who heard the read-back finalizes.

The LLM is injectable so this whole graph is testable with a scripted fake
model and no network access; production wiring uses the real Gemini-backed
model from get_llm().
"""

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from app.agent.finalization import CLOSING_TEXT, is_clear_yes
from app.agent.instructions import AGENT_PERSONA_INSTRUCTIONS
from app.agent.session import (
    CutOffQuestion,
    SessionState,
    mark_question_asked,
    question_is_playing,
    question_was_cut_off,
    readback_was_heard,
    speak_readback,
)
from app.output.readback import build_readback
from app.logging_config import get_logger
from app.retry import call_with_retry
from app.agent.tool_definitions import build_tool_definitions
from app.agent.tools import ToolResult, create_tool_handlers
from app.schemas.protocol_config import ProtocolConfig
from app.schemas.intake_record import TranscriptTurn

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6

_cached_llms: dict[Optional[str], BaseChatModel] = {}
# Guards _cached_llms specifically. This dict is written from inside
# asyncio.to_thread calls (main.py's handle_utterance/run_opening_line path),
# which means genuinely different OS threads — not just different asyncio
# tasks on one thread — can race on "check if cached, then build and store."
# A plain threading.Lock is required here, not an asyncio.Lock: the race is
# between real threads, and asyncio.Lock only protects coroutines sharing a
# single thread's event loop, so it would not actually close this race.
_cached_llms_lock = threading.Lock()
_cached_verifier_llm: Optional[BaseChatModel] = None


def get_llm(protocol: Optional[ProtocolConfig] = None) -> BaseChatModel:
    """Lazily constructs the real Gemini-backed model on first use, not at
    import time — so importing this module never fails just because
    GEMINI_API_KEY isn't set. Tests never call this; they pass a scripted
    fake model into run_agent_turn instead.

    Cached per protocol (keyed by protocol_id, or None before a protocol is
    locked) rather than as one single global model: once the protocol is
    known, its tool schema is built with an exact enum of that protocol's
    real field names (see tool_definitions.build_tool_definitions), so the
    model can no longer guess a plausible-but-wrong field name. There are
    only a handful of protocols, so caching one bound client per protocol_id
    costs nothing beyond the first turn of each.

    Double-checked locking: the fast, common path (already cached) never
    touches the lock at all; the lock is only ever acquired on a cache miss,
    and the cache is checked again once inside it — closing a real race
    where two patients' first turns for the same protocol land on different
    threads at nearly the same moment, previously letting both build a
    separate client and one silently overwrite the other's wasted work."""
    global _cached_llms
    cache_key = protocol.protocol_id if protocol else None
    if cache_key in _cached_llms:
        return _cached_llms[cache_key]

    with _cached_llms_lock:
        if cache_key in _cached_llms:  # someone else built it while we waited for the lock
            return _cached_llms[cache_key]

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Required to run the live voice pipeline; not required for "
                "the state-engine/tool-layer/eval test suite."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        field_names = [f.field for f in protocol.fields] if protocol else None
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key).bind_tools(build_tool_definitions(field_names))
        _cached_llms[cache_key] = llm
        return llm


def get_verifier_llm() -> BaseChatModel:
    """The separate model that double-checks a recorded "no" / "I don't know"
    against the patient's actual reply (see answer_verifier.py). Deliberately
    NOT bound to any tools: it can only answer with a word, never act.
    Uses GEMINI_VERIFIER_MODEL if set — point it at a smaller, cheaper model —
    otherwise the same model as the main agent. Same double-checked locking
    as get_llm, for the same reason (first calls from different threads)."""
    global _cached_verifier_llm
    if _cached_verifier_llm is not None:
        return _cached_verifier_llm

    with _cached_llms_lock:
        if _cached_verifier_llm is not None:
            return _cached_verifier_llm

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Required to run the live voice pipeline.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = os.environ.get("GEMINI_VERIFIER_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        _cached_verifier_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        return _cached_verifier_llm


def _tool_result_to_payload(result: ToolResult) -> dict:
    payload: dict = {"ok": result.ok}
    if result.data is not None:
        payload["data"] = result.data
    if result.error is not None:
        payload["error"] = result.error
    return payload


class AgentState(MessagesState):
    """What the graph carries from step to step, and what the checkpointer saves:
    the conversation messages and two small flags. The clinical record is never in
    here. It lives in SessionState, which only the tool handlers change."""

    finalize_requested: bool
    finalize_outcome: Optional[str]


@dataclass
class RunContext:
    """What one run needs that must NOT be saved in a checkpoint: the live session,
    this turn's tool handlers and the model. It is handed to the compiled graph at
    run time, so a single compiled graph serves every session and every turn."""

    session: SessionState
    handlers: dict
    model: BaseChatModel
    turn_generation: Optional[int] = None

    def is_stale(self) -> bool:
        """main.py bumps session.turn_generation on every new utterance and on a
        barge-in, so a turn whose generation no longer matches has been superseded."""
        return self.turn_generation is not None and self.session.turn_generation != self.turn_generation


def _reason(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    if ctx.is_stale():
        return {"messages": [AIMessage(content="")]}
    logger.debug(
        "calling model",
        extra={
            "session_id": ctx.session.session_id,
            "message_count": len(state["messages"]),
            "messages": [
                {"role": type(m).__name__, "content_preview": str(m.content)[:80], "tool_calls": getattr(m, "tool_calls", None)}
                for m in state["messages"]
            ],
        },
    )
    response = call_with_retry(lambda: ctx.model.invoke(state["messages"]), what="Gemini reasoning call")
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _run_tools(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    last = state["messages"][-1]
    if ctx.is_stale():
        return {
            "messages": [
                ToolMessage(content=json.dumps({"ok": False, "error": "superseded"}), tool_call_id=call["id"])
                for call in last.tool_calls
            ]
        }
    # Check every fact proposed in this message with ONE verifier call, ahead of
    # the handlers running. If that fails for any reason, each handler falls
    # back to verifying its own claim, so nothing is ever saved unverified.
    prepare = getattr(ctx.handlers, "prepare", None)
    if prepare is not None:
        try:
            prepare(last.tool_calls)
        except Exception:  # noqa: BLE001
            logger.exception("batched verification failed; handlers will verify individually")
    tool_messages: list[BaseMessage] = []
    finalize_requested = False
    for call in last.tool_calls:
        handler = ctx.handlers.get(call["name"])
        result = handler(call["args"]) if handler else ToolResult(ok=False, error=f"Unknown tool: {call['name']}")
        if result.ok and result.data and result.data.get("awaiting_patient_confirmation"):
            finalize_requested = True
        tool_messages.append(ToolMessage(content=json.dumps(_tool_result_to_payload(result)), tool_call_id=call["id"]))
    return {"messages": tool_messages, "finalize_requested": finalize_requested}


def _after_tools(state: AgentState) -> str:
    return "confirm_finalization" if state.get("finalize_requested") else "reason"


def _confirm_finalization(state: AgentState, runtime: Runtime[RunContext]) -> dict:
    """Pauses the run until the patient has heard the read-back and answered.

    interrupt() stops the graph here and hands the read-back text to whoever ran
    it. When the patient next speaks, the run is resumed with their words and this
    node starts again from its first line, which is why everything above the
    interrupt must be harmless to repeat (building the text has no side effects).
    Whether they agreed is decided by plain code, not by a model, and only an
    explicit yes from a patient who heard the whole read-back finalizes."""
    ctx = runtime.context
    answer = interrupt({"kind": "confirm_finalization", "readback": build_readback(ctx.session.record, ctx.session.protocol)})
    confirmed = bool(answer.get("heard")) and is_clear_yes(answer.get("patient_said", ""))
    if not confirmed:
        return {"finalize_outcome": "not_confirmed"}
    ctx.session.brief_finalized = True
    return {"finalize_outcome": "confirmed", "messages": [AIMessage(content=CLOSING_TEXT)]}


def _compile(checkpointer) -> object:
    graph = StateGraph(AgentState, context_schema=RunContext)
    graph.add_node("reason", _reason)
    graph.add_node("tools", _run_tools)
    graph.add_node("confirm_finalization", _confirm_finalization)
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", _should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", _after_tools, {"confirm_finalization": "confirm_finalization", "reason": "reason"})
    graph.add_edge("confirm_finalization", END)
    return graph.compile(checkpointer=checkpointer)


# The one compiled graph and the checkpointer behind it. InMemorySaver keeps the saved
# state in this process, which is right for development and for the tests. For a
# deployment that must survive a restart, pass a SqliteSaver or PostgresSaver to
# configure_checkpointer(): nothing else in this file changes.
_checkpointer = InMemorySaver()
_graph = None
_graph_lock = threading.Lock()


def get_graph():
    """The compiled graph, built on first use and then shared by every turn of every session."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = _compile(_checkpointer)
    return _graph


def configure_checkpointer(saver=None) -> None:
    """Replaces the checkpointer (a fresh in-memory one by default) and recompiles on next use."""
    global _checkpointer, _graph
    with _graph_lock:
        _checkpointer = saver if saver is not None else InMemorySaver()
        _graph = None


def release_conversation(session_id: str) -> None:
    """Forgets the saved graph state for a call that has ended, so checkpoints don't pile up."""
    try:
        _checkpointer.delete_thread(session_id)
    except Exception:  # noqa: BLE001 - cleanup must never break a disconnect
        logger.warning("could not delete saved graph state", extra={"session_id": session_id})


def _make_context(
    session: SessionState,
    llm: Optional[BaseChatModel],
    turn_generation: Optional[int],
    ranking_llm: Optional[BaseChatModel],
    verifier_llm: Optional[BaseChatModel],
    confirm_finalization: Optional[bool],
) -> RunContext:
    """Builds this run's context. `llm` is injectable for tests.

    `ranking_llm` backs the adaptive next-question ordering in question_prioritizer.py,
    and `verifier_llm` backs the independent check on every recorded answer
    (answer_verifier.py). Both default to the real models, but only when `llm` was not
    overridden: tests inject rigid, positional scripted fakes as `llm`, and an unplanned
    extra call would silently consume and shift every later scripted response. A test
    that wants one of them must opt in explicitly.

    `confirm_finalization` follows the same rule: on in the real path, where finishing
    the intake pauses for the patient's yes, and off for scripted fakes unless a test
    opts in."""
    model = llm or get_llm(session.protocol)
    if ranking_llm is None and llm is None:
        ranking_llm = model
    if verifier_llm is None and llm is None:
        verifier_llm = get_verifier_llm()
    if confirm_finalization is None:
        confirm_finalization = llm is None
    handlers = create_tool_handlers(
        session, ranking_llm, verifier_llm=verifier_llm, confirm_finalization=confirm_finalization
    )
    return RunContext(session=session, handlers=handlers, model=model, turn_generation=turn_generation)


def _run_config(session: SessionState) -> dict:
    # The thread id names this call's saved state in the checkpointer, so two patients never share one.
    return {"configurable": {"thread_id": session.session_id}, "recursion_limit": MAX_TOOL_ROUNDS * 2 + 2}


def _fresh_input(messages: list[BaseMessage]) -> dict:
    # Every turn rebuilds the model's input from the verbatim transcript, so the saved history is
    # replaced, not added to: RemoveMessage(REMOVE_ALL_MESSAGES) clears it before the new list goes in.
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages],
        "finalize_requested": False,
        "finalize_outcome": None,
    }


def _waiting_for_confirmation(graph, config) -> bool:
    return any(task.interrupts for task in graph.get_state(config).tasks)


def _extract_text(content: object) -> str:
    """Newer Gemini models return AIMessage.content as a list of content
    blocks (e.g. [{"type": "text", "text": "..."}]) rather than a plain
    string — normalize either shape to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(parts)
    return str(content)


def _mark_spoken_question(session: SessionState, events_before: int, await_playback: bool = False) -> None:
    """Picks out the question the agent just asked, once its reply is final.
    get_next_intake_question logs a QuestionEvent the moment the tool runs,
    which is before the agent has said anything — so logging alone proves
    nothing was asked. Only the last event created this turn counts (the
    persona asks one question per turn, and an earlier lookup in the same
    turn may have been for a field the model then didn't ask about), and
    only after the reply has been appended to the transcript, so
    asked_in_turn is that agent turn's index. A turn that was superseded
    never reaches here, so its events stay unstamped and can never back a
    denial.

    With `await_playback` (every live call) the question is not stamped yet:
    a reply that is final is not a reply the patient has heard. It is stamped
    by question_was_heard when the browser reports the audio played to the
    end, and if the patient talks over it first, it is never stamped."""
    new_events = session.question_events[events_before:]
    if not new_events or not session.transcript[-1].text.strip():
        return  # nothing was logged, or the reply was empty so nothing was actually said
    event = new_events[-1]
    reply_index = len(session.transcript) - 1
    if await_playback:
        question_is_playing(session, event.id, reply_index)
    else:
        mark_question_asked(session, event, reply_index)


def _cut_off_note(cut: CutOffQuestion) -> str:
    return (
        f'Your previous message was cut off when the patient started speaking, so they did not hear all of it, '
        f'including your question about "{cut.label}". You had said: "{cut.reply_text}" '
        f"Because they never heard that question, a bare yes or no in their new message is not an answer to it: "
        f"record something about that topic only if they clearly say it themselves. Respond to what they just "
        f"said, then ask that question again (get_next_intake_question will suggest it)."
    )


def _reply_text(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return _extract_text(msg.content)
    return ""


def _append_agent_turn(session: SessionState, text: str) -> None:
    session.transcript.append(
        TranscriptTurn(id=str(uuid.uuid4()), speaker="agent", text=text, timestamp=datetime.now(timezone.utc).isoformat())
    )


def run_agent_turn(
    session: SessionState,
    patient_utterance: str,
    llm: Optional[BaseChatModel] = None,
    system_note: Optional[str] = None,
    turn_generation: Optional[int] = None,
    ranking_llm: Optional[BaseChatModel] = None,
    verifier_llm: Optional[BaseChatModel] = None,
    await_playback: bool = False,
    confirm_finalization: Optional[bool] = None,
) -> dict:
    """Runs one full patient turn: logs the utterance, replays the
    session's transcript-so-far as the conversation history (tool-call
    exchanges within a turn are not persisted across turns — the durable
    state is session.record/documents, queryable again via tools, not raw
    chat history), and returns the agent's reply text.

    `system_note`, when given, is a deterministic aside appended for this
    turn only (e.g. "the patient's last message sounds like a different
    complaint type") — it steers the model's phrasing without ever being
    stored in session.transcript, which must stay a verbatim record of what
    was actually said for evidence-quote integrity.

    `turn_generation`, when given, is checked against session.turn_generation
    after the graph runs: if a newer utterance (or a barge-in) superseded
    this one while it was in flight, its reply is dropped instead of being
    appended to the transcript — an interrupted turn's late answer should
    never land in the conversation after the fact.

    `await_playback` is True on a live call, where the reply reaches the patient
    as audio: the question in it counts as asked only once the browser reports
    the audio finished (see session.question_was_heard). A reply still marked as
    playing when a new turn starts never finished, so it counts as cut off, and
    this turn is told to ask that question again.

    When the model asks to finish the intake, the run pauses at the read-back
    (see _confirm_finalization) and this function returns the read-back as the
    reply. The patient's next utterance resumes that paused run instead of
    starting a new one: a clear yes finalizes, anything else is handled as an
    ordinary turn.
    """
    if await_playback:
        question_was_cut_off(session)
    session.transcript.append(
        TranscriptTurn(
            id=str(uuid.uuid4()), speaker="patient", text=patient_utterance, timestamp=datetime.now(timezone.utc).isoformat()
        )
    )

    graph = get_graph()
    config = _run_config(session)
    ctx = _make_context(session, llm, turn_generation, ranking_llm, verifier_llm, confirm_finalization)

    if session.pending_readback_event_id is not None:
        # The graph is paused at the read-back, and this utterance is the patient's answer to it.
        heard = readback_was_heard(session)
        waiting = _waiting_for_confirmation(graph, config)
        session.pending_readback_event_id = None  # consumed, whatever happens next
        if waiting:
            resumed = graph.invoke(Command(resume={"patient_said": patient_utterance, "heard": heard}), config, context=ctx)
            if turn_generation is not None and session.turn_generation != turn_generation:
                return {"reply_text": "", "superseded": True}
            if resumed.get("finalize_outcome") == "confirmed":
                reply_text = _reply_text(resumed)
                _append_agent_turn(session, reply_text)
                return {"reply_text": reply_text, "superseded": False}
            # Not a clear yes (a correction, a doubt, or a read-back they did not hear in full):
            # nothing is finalized, and this utterance is handled as an ordinary turn below.

    events_before = len(session.question_events)

    messages: list[BaseMessage] = [SystemMessage(content=AGENT_PERSONA_INSTRUCTIONS)]
    if system_note:
        messages.append(SystemMessage(content=system_note))
    if session.cut_off_question is not None:
        messages.append(SystemMessage(content=_cut_off_note(session.cut_off_question)))
    for turn in session.transcript:
        message_cls = HumanMessage if turn.speaker == "patient" else AIMessage
        messages.append(message_cls(content=turn.text))

    result = graph.invoke(_fresh_input(messages), config, context=ctx)

    if turn_generation is not None and session.turn_generation != turn_generation:
        return {"reply_text": "", "superseded": True}

    interrupts = result.get("__interrupt__")
    if interrupts:
        # The model asked to finish and the run is now paused at the read-back.
        readback = interrupts[0].value["readback"]
        speak_readback(session, readback, await_playback)
        session.cut_off_question = None
        return {"reply_text": readback, "superseded": False}

    reply_text = _reply_text(result)
    _append_agent_turn(session, reply_text)
    _mark_spoken_question(session, events_before, await_playback)
    session.cut_off_question = None  # this turn has had its chance to ask it again

    return {"reply_text": reply_text, "superseded": False}


def run_opening_turn(
    session: SessionState,
    reason_text: str,
    when_text: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    turn_generation: Optional[int] = None,
    ranking_llm: Optional[BaseChatModel] = None,
    await_playback: bool = False,
) -> dict:
    """Generates Ava's very first line, spoken before the patient has said
    anything — used when the chief complaint (and, when given, the
    appointment's date/time) is already known upfront, e.g. from a booking
    record, instead of discovering the complaint live from what the patient
    says. No patient transcript turn gets logged here (nothing was said
    yet); only the resulting opening line is appended, as the first entry
    in session.transcript.

    The trigger is sent as a HumanMessage rather than a trailing
    SystemMessage: Gemini requires the final turn in a request to be a user
    message or a function response (see run_agent_turn's system_note
    ordering, fixed for the same reason) — a synthetic instruction still
    needs to occupy the "user" slot to satisfy that.
    """
    graph = get_graph()
    config = _run_config(session)
    ctx = _make_context(session, llm, turn_generation, ranking_llm, None, None)
    events_before = len(session.question_events)

    when_clause = f' scheduled for {when_text},' if when_text else ""
    opening_trigger = (
        f"[This is the start of the call. The patient has an upcoming appointment{when_clause} "
        f'regarding: "{reason_text}". Open the conversation by warmly greeting them, mentioning when '
        f"the appointment is, and referencing this reason directly, then ask your first relevant "
        f'question. Do not ask an open-ended "what brings you in today" question — you already know '
        f"why they're here.]"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=AGENT_PERSONA_INSTRUCTIONS),
        HumanMessage(content=opening_trigger),
    ]

    result = graph.invoke(_fresh_input(messages), config, context=ctx)

    if turn_generation is not None and session.turn_generation != turn_generation:
        return {"reply_text": "", "superseded": True}

    reply_text = _reply_text(result)
    _append_agent_turn(session, reply_text)
    _mark_spoken_question(session, events_before, await_playback)

    return {"reply_text": reply_text, "superseded": False}
