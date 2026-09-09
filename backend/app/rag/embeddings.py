"""Embeddings provider for the RAG stores. Production uses Gemini's real
embedding model through LangChain; tests inject a small offline
feature-hashing embedding (no network, no API key) that implements the same
`langchain_core.embeddings.Embeddings` interface Chroma expects — so the
retrieval logic itself is exercised against a real vector store either way,
only the vectors' source changes."""

import hashlib
import os
import re
from typing import Optional

from langchain_core.embeddings import Embeddings

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "has", "have", "had", "in", "on", "at", "to",
    "of", "and", "or", "for", "that", "this", "from", "not", "no", "note", "noted", "seen", "ago",
    "any", "does", "do", "did", "with", "as", "it", "its", "be", "been", "there", "known", "about",
}


def _normalize(token: str) -> str:
    """Very small suffix stripper so "allergies"/"allergy", "medications"/"medication" hash to the same bucket."""
    if len(token) > 4:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("es"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [_normalize(t) for t in tokens if t not in _STOPWORDS]


class HashingEmbeddings(Embeddings):
    """Deterministic, fully offline embedding used for tests and any
    environment without a Gemini API key. Not used in production."""

    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = _tokenize(text)
        for tok in tokens:
            vec[int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self.dims] += 1.0
        for a, b in zip(tokens, tokens[1:]):
            bigram = f"{a}_{b}"
            vec[int(hashlib.sha256(bigram.encode()).hexdigest(), 16) % self.dims] += 0.5
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_cached_embeddings: Optional[Embeddings] = None


def get_embeddings() -> Embeddings:
    """Real Gemini embeddings if GEMINI_API_KEY/GOOGLE_API_KEY is set,
    otherwise the offline hashing fallback — so the app still runs (with
    lower-quality retrieval ranking) without credentials."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _cached_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    else:
        _cached_embeddings = HashingEmbeddings()

    return _cached_embeddings
