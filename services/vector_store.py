"""
Vector store wrapper — Chroma persistent client, keyed by arxiv_id.

Supports two retrieval modes:
  - Dense only (Chroma cosine similarity)
  - Hybrid: Dense + BM25, fused with Reciprocal Rank Fusion (RRF)

BM25 index is rebuilt from Chroma metadata at query time (a single paper
is only tens to low-hundreds of chunks, so this is near-instant).

IMPORTANT: The confidence gate in qa_loop uses the top hit's raw *dense*
cosine similarity, NOT the fused RRF score. RRF scores (Σ 1/(k + rank),
k=60) top out around ~0.03 for a chunk ranked #1 by both retrievers —
thresholding on them would silently break the QA loop.
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from services.embedder import embed_texts, embed_query

_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".chroma_store")
_chroma_client: chromadb.PersistentClient | None = None

_bm25_cache: dict[str, list[dict]] = {}

_RRF_K = 60


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(_CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
    return _chroma_client


def collection_exists(arxiv_id: str) -> bool:
    """Check whether we already have embeddings stored for this paper."""
    client = _get_client()
    col_name = _safe_collection_name(arxiv_id)
    try:
        col = client.get_collection(col_name)
        return col.count() > 0
    except Exception:
        return False


def index_chunks(chunks: list[dict], arxiv_id: str) -> str:
    """
    Embed and store chunks in a Chroma collection named after the paper.

    Each chunk dict must have keys: text, chunk_id, section, arxiv_id.
    Returns the collection name (used as vector_store_ref in AgentState).
    """
    client = _get_client()
    col_name = _safe_collection_name(arxiv_id)

    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {"section": c["section"], "arxiv_id": c["arxiv_id"], "text": c["text"]}
        for c in chunks
    ]

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        embeddings = embed_texts(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_meta,
            documents=batch_texts,
        )

    return col_name


def _get_all_chunks(arxiv_id: str) -> list[dict]:
    """Fetch all chunks from a Chroma collection (cached for BM25 index rebuild)."""
    global _bm25_cache
    if arxiv_id in _bm25_cache:
        return _bm25_cache[arxiv_id]

    client = _get_client()
    col_name = _safe_collection_name(arxiv_id)
    try:
        collection = client.get_collection(col_name)
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []

    results = collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    chunks = []
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results.get("metadatas") else {}
        text = results["documents"][i] if results.get("documents") else ""
        chunks.append({
            "chunk_id": doc_id,
            "text": text,
            "section": meta.get("section", ""),
            "arxiv_id": meta.get("arxiv_id", arxiv_id),
        })
    _bm25_cache[arxiv_id] = chunks
    return chunks


def _bm25_rank(query: str, chunks: list[dict]) -> list[tuple[int, float]]:
    """
    Run BM25 over chunk texts.
    Returns list of (chunk_index, bm25_score), sorted desc by score.
    """
    if not chunks:
        return []

    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked = [(i, float(scores[i])) for i in range(len(scores))]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _reciprocal_rank_fusion(
    dense_ranking: list[int],
    bm25_ranking: list[int],
    k: int = _RRF_K,
) -> list[int]:
    """
    Fuse two rankings using Reciprocal Rank Fusion.
    Returns chunk indices sorted by fused score (desc).
    """
    scores: dict[int, float] = {}

    for rank, idx in enumerate(dense_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return fused


def query_similar(
    query: str,
    arxiv_id: str,
    top_k: int = 5,
    use_hybrid: bool = True,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks to the query.

    If use_hybrid=True, runs dense (Chroma) + BM25, fused with RRF.
    The returned 'similarity' field is always the raw dense cosine
    similarity (used for confidence gating), regardless of retrieval mode.

    Returns list of dicts with keys:
        text, section, chunk_id, arxiv_id, similarity
    """
    client = _get_client()
    col_name = _safe_collection_name(arxiv_id)

    try:
        collection = client.get_collection(col_name)
    except Exception:
        return []

    col_count = collection.count()
    if col_count == 0:
        return []

    # --- Dense retrieval ---
    q_embedding = embed_query(query)
    dense_n = min(col_count, max(top_k * 3, 20))
    dense_results = collection.query(
        query_embeddings=[q_embedding],
        n_results=dense_n,
        include=["documents", "metadatas", "distances"],
    )

    if not dense_results or not dense_results.get("ids"):
        return []

    dense_hits: dict[str, dict] = {}
    dense_order: list[str] = []
    for i, doc_id in enumerate(dense_results["ids"][0]):
        distance = dense_results["distances"][0][i] if dense_results.get("distances") else 1.0
        similarity = 1.0 - distance
        meta = dense_results["metadatas"][0][i] if dense_results.get("metadatas") else {}
        text = dense_results["documents"][0][i] if dense_results.get("documents") else ""

        dense_hits[doc_id] = {
            "text": text,
            "section": meta.get("section", ""),
            "chunk_id": doc_id,
            "arxiv_id": meta.get("arxiv_id", arxiv_id),
            "similarity": similarity,
        }
        dense_order.append(doc_id)

    if use_hybrid:
        all_chunks = _get_all_chunks(arxiv_id)
        if all_chunks:
            id_to_idx = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}

            bm25_ranked = _bm25_rank(query, all_chunks)
            bm25_order = [idx for idx, _ in bm25_ranked]

            dense_idx_order = [
                id_to_idx[cid] for cid in dense_order if cid in id_to_idx
            ]

            fused_indices = _reciprocal_rank_fusion(dense_idx_order, bm25_order)

            hits: list[dict] = []
            for chunk_idx in fused_indices[:top_k]:
                chunk = all_chunks[chunk_idx]
                cid = chunk["chunk_id"]
                if cid in dense_hits:
                    hits.append(dense_hits[cid])
                else:
                    chunk_emb = embed_texts([chunk["text"]])[0]
                    import numpy as np
                    q_vec = np.array(q_embedding)
                    c_vec = np.array(chunk_emb)
                    sim = float(np.dot(q_vec, c_vec) / (
                        np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-9
                    ))
                    hits.append({
                        "text": chunk["text"],
                        "section": chunk["section"],
                        "chunk_id": cid,
                        "arxiv_id": chunk["arxiv_id"],
                        "similarity": sim,
                    })
            return hits

    return [dense_hits[cid] for cid in dense_order[:top_k]]


def _safe_collection_name(arxiv_id: str) -> str:
    """Chroma collection names must be 3-63 chars, alphanumeric + underscores."""
    name = "arxiv_" + arxiv_id.replace(".", "_").replace("/", "_")
    return name[:63]
