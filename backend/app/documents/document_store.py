"""Per-session uploaded-document storage and retrieval. Each session gets
its own Chroma collection — its own persistent client, in fact, same
reasoning as the static reference stores in rag/retriever.py — built
lazily and added to incrementally as documents arrive."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.embeddings import get_embeddings
from app.rag.retriever import retrieve
from app.schemas.document import UploadedDocument

_SESSIONS_ROOT = Path(__file__).parent.parent.parent / ".chroma_data" / "sessions"

_session_stores: dict[str, Chroma] = {}


def create_uploaded_document(filename: str, mime_type: str, text: str) -> UploadedDocument:
    return UploadedDocument(
        id=str(uuid.uuid4()),
        filename=filename,
        mime_type=mime_type,
        text=text,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )


def _get_or_create_store(session_id: str) -> Chroma:
    store = _session_stores.get(session_id)
    if store is None:
        persist_directory = str(_SESSIONS_ROOT / session_id)
        # Session ids are UUIDs, so a collision with leftover data is
        # essentially impossible, but clear it defensively in case a prior
        # process crashed mid-session without cleaning up.
        shutil.rmtree(persist_directory, ignore_errors=True)
        store = Chroma(collection_name="documents", embedding_function=get_embeddings(), persist_directory=persist_directory)
        _session_stores[session_id] = store
    return store


def add_document(session_id: str, doc: UploadedDocument) -> None:
    store = _get_or_create_store(session_id)
    store.add_texts(texts=[doc.text], metadatas=[{"id": doc.id, "filename": doc.filename}], ids=[doc.id])


def retrieve_from_documents(session_id: str, query: str, k: int = 2) -> list[tuple[Document, float]]:
    store = _session_stores.get(session_id)
    if store is None:
        return []
    return retrieve(store, query, k)


def clear_session(session_id: str) -> None:
    """Called when a session ends, so its Chroma collection doesn't linger in memory or on disk."""
    store = _session_stores.pop(session_id, None)
    if store is not None:
        try:
            store.delete_collection()
        except Exception:
            pass
    shutil.rmtree(str(_SESSIONS_ROOT / session_id), ignore_errors=True)
