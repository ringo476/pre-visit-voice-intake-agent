"""RAG primitives: build a Chroma collection, query it. The static
reference stores below are one per protocol (app/rag/data/<protocol_id>/),
and the per-session uploaded-document store (app/documents/document_store.py)
is built on these same two functions.

Each logical store gets its OWN persistent Chroma client (a unique
`persist_directory`), rather than sharing chromadb's default in-process
client across all of them. This was not a stylistic choice — chromadb's
default ephemeral client silently shares one underlying engine instance
across every collection created in the process (confirmed: separate
`chromadb.Client()`/`EphemeralClient()` objects still read each other's
collections), and under this app's collection count that shared engine
was observed to intermittently return empty/erroring results for a
collection that demonstrably still had its data (`.count()` correct,
`.similarity_search()` empty, `.get()` sometimes raising an internal
error) — a real bug in that shared in-memory engine, not a timing issue a
retry can paper over. Giving each store its own on-disk SQLite-backed
client (via a unique `persist_directory`) sidesteps the shared-engine bug
entirely; disk cost is negligible at this project's scale (a handful of
short reference documents per store).
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.protocol.registry import PROTOCOLS
from app.rag.embeddings import get_embeddings

_DIR = Path(__file__).parent
_CHROMA_ROOT = Path(__file__).parent.parent.parent / ".chroma_data"


def _persist_dir(*parts: str) -> str:
    return str(_CHROMA_ROOT.joinpath(*parts))


def build_index(
    texts: list[str],
    metadatas: list[dict],
    collection_name: str,
    persist_directory: str,
    embeddings: Optional[Embeddings] = None,
    fresh: bool = True,
) -> Chroma:
    """`fresh=True` (the default) deletes any leftover data at
    `persist_directory` first, so static reference stores always rebuild
    cleanly from their JSON source rather than silently accumulating
    duplicates across repeated process starts."""
    if fresh:
        shutil.rmtree(persist_directory, ignore_errors=True)
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings or get_embeddings(),
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def retrieve(store: Chroma, query: str, k: int = 2) -> list[tuple[Document, float]]:
    return store.similarity_search_with_score(query, k=k)


def _load_json(protocol_id: str, filename: str) -> list[dict]:
    path = _DIR / "data" / protocol_id / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_prior_chart_store(protocol_id: str) -> Chroma:
    docs = _load_json(protocol_id, "prior_chart.json")
    return build_index(
        texts=[d["text"] for d in docs],
        metadatas=[{"id": d["id"], "field": d["field"], "date": d["date"]} for d in docs],
        collection_name="prior-chart",
        persist_directory=_persist_dir("protocols", protocol_id, "prior-chart"),
    )


def _build_follow_up_store(protocol_id: str) -> Chroma:
    docs = _load_json(protocol_id, "follow_up_guidance.json")
    return build_index(
        texts=[d["text"] for d in docs],
        metadatas=[{"id": d["id"], "related_fields": ",".join(d["related_fields"])} for d in docs],
        collection_name="follow-up-guidance",
        persist_directory=_persist_dir("protocols", protocol_id, "follow-up-guidance"),
    )


# Built eagerly at import time so a failure surfaces at startup, not on the
# first patient turn. Cheap: only a couple of protocols, a handful of short
# documents each.
_prior_chart_stores: dict[str, Chroma] = {pid: _build_prior_chart_store(pid) for pid in PROTOCOLS}
_follow_up_stores: dict[str, Chroma] = {pid: _build_follow_up_store(pid) for pid in PROTOCOLS}


def retrieve_prior_chart(protocol_id: str, query: str, k: int = 2) -> list[tuple[Document, float]]:
    """Backs the retrieve_existing_patient_context tool — synthetic, single-patient chart context, scoped to the active protocol."""
    return retrieve(_prior_chart_stores[protocol_id], query, k)


def retrieve_follow_up_guidance(protocol_id: str, query: str, k: int = 2) -> list[tuple[Document, float]]:
    """Backs the get_next_intake_question tool's phrasing/rationale suggestions, scoped to the active protocol."""
    return retrieve(_follow_up_stores[protocol_id], query, k)
