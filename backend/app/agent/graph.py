"""The agent's LangGraph: a two-node loop between reasoning (Gemini bound
to the 8 tools) and tool execution (our own deterministic handlers).

    reason --(no tool_calls)--> END
    reason --(has tool_calls)--> tools --> reason (loop)

`reason` is the only node that talks to Gemini, and its only power is to
speak or request a tool call. `tools` is plain Python — it's the only node
with authority to change anything (state engine, safety engine, RAG). The
LLM is injectable so this whole graph is testable with a scripted fake
model and no network access; production wiring (server.py) uses the real
Gemini-backed model from get_llm().
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph

from app.agent.instructions import AGENT_PERSONA_INSTRUCTIONS
from app.agent.session import SessionState
from app.agent.tool_definitions import build_tool_definitions
from app.agent.tools import ToolResult, create_tool_handlers
from app.schemas.protocol_config import ProtocolConfig
from app.schemas.intake_record import TranscriptTurn

MAX_TOOL_ROUNDS = 6

_cached_llms: dict[Optional[str], BaseChatModel] = {}


def get_llm(protocol: Optional[ProtocolConfig] = None) -> BaseChatModel:
    """Lazily constructs the real Gemini-backed model on first use, not at
    import time — so importing this module never fails just because
    GEMINI_API_KEY isn't set. Tests never call this; they pass a scripted
    fake model into run_agent_turn/build_graph instead.

    Cached per protocol (keyed by protocol_id, or None before a protocol is
    locked) rather than as one single global model: once the protocol is
    known, its tool schema is built with an exact enum of that protocol's
    real field names (see tool_definitions.build_tool_definitions), so the
    model can no longer guess a plausible-but-wrong field name. There are
    only a handful of protocols, so caching one bound client per protocol_id
    costs nothing beyond the first turn of each."""
    global _cached_llms
    cache_key = protocol.protocol_id if protocol else None
    if cache_key in _cached_llms:
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


def _tool_result_to_payload(result: ToolResult) -> dict:
    payload: dict = {"ok": result.ok}
    if result.data is not None:
        payload["data"] = result.data
    if result.error is not None:
        payload["error"] = result.error
    return payload


def build_graph(session: SessionState, llm: Optional[BaseChatModel] = None, turn_generation: Optional[int] = None):
    """Compiles the graph for one session. `llm` is injectable for tests.

    `turn_generation` is the value session.turn_generation held when this
    turn started. main.py bumps session.turn_generation on every new
    recorded utterance and on a client barge-in signal — so if it no longer
    matches by the time reason()/run_tools() would run, a newer utterance
    has superseded this one. Rather than let an interrupted turn keep making
    Gemini calls and writing facts nobody's waiting on (which is what was
    happening before: an interrupted turn ran to completion in the
    background regardless, its stale reply landing in the transcript and
    confusing the next turn into re-extracting facts already recorded), each
    node checks first and stops immediately once stale."""
    handlers = create_tool_handlers(session)
    model = llm or get_llm(session.protocol)

    def _is_stale() -> bool:
        return turn_generation is not None and session.turn_generation != turn_generation

    def reason(state: MessagesState) -> MessagesState:
        if _is_stale():
            return {"messages": [AIMessage(content="")]}
        print(f"[{session.session_id}] calling model with {len(state['messages'])} messages:")
        for m in state["messages"]:
            role = type(m).__name__
            tool_calls = getattr(m, "tool_calls", None)
            print(f"  {role}: content={str(m.content)[:80]!r} tool_calls={tool_calls}")
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def run_tools(state: MessagesState) -> MessagesState:
        last = state["messages"][-1]
        if _is_stale():
            return {
                "messages": [
                    ToolMessage(content=json.dumps({"ok": False, "error": "superseded"}), tool_call_id=call["id"])
                    for call in last.tool_calls
                ]
            }
        tool_messages: list[BaseMessage] = []
        for call in last.tool_calls:
            handler = handlers.get(call["name"])
            result = handler(call["args"]) if handler else ToolResult(ok=False, error=f"Unknown tool: {call['name']}")
            tool_messages.append(
                ToolMessage(content=json.dumps(_tool_result_to_payload(result)), tool_call_id=call["id"])
            )
        return {"messages": tool_messages}

    graph = StateGraph(MessagesState)
    graph.add_node("reason", reason)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "reason")

    return graph.compile()


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


def run_agent_turn(
    session: SessionState,
    patient_utterance: str,
    llm: Optional[BaseChatModel] = None,
    system_note: Optional[str] = None,
    turn_generation: Optional[int] = None,
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
    """
    session.transcript.append(
        TranscriptTurn(
            id=str(uuid.uuid4()), speaker="patient", text=patient_utterance, timestamp=datetime.now(timezone.utc).isoformat()
        )
    )

    compiled = build_graph(session, llm=llm, turn_generation=turn_generation)

    messages: list[BaseMessage] = [SystemMessage(content=AGENT_PERSONA_INSTRUCTIONS)]
    if system_note:
        messages.append(SystemMessage(content=system_note))
    for turn in session.transcript:
        message_cls = HumanMessage if turn.speaker == "patient" else AIMessage
        messages.append(message_cls(content=turn.text))

    result = compiled.invoke({"messages": messages}, config={"recursion_limit": MAX_TOOL_ROUNDS * 2 + 2})

    if turn_generation is not None and session.turn_generation != turn_generation:
        return {"reply_text": "", "superseded": True}

    reply_text = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply_text = _extract_text(msg.content)
            break

    session.transcript.append(
        TranscriptTurn(id=str(uuid.uuid4()), speaker="agent", text=reply_text, timestamp=datetime.now(timezone.utc).isoformat())
    )

    return {"reply_text": reply_text, "superseded": False}
