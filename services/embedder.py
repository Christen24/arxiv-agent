"""
Local embedding wrapper using sentence-transformers.

Uses all-MiniLM-L6-v2 — runs on CPU, no GPU required.
Avoids burning through any hosted API free tier on the
highest-volume call type (one call per chunk).
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings. Returns list of float vectors."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
