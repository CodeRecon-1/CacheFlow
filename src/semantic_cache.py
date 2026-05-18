"""
Semantic cache: stores prompt embeddings in ChromaDB for similarity search.
Falls back gracefully if ChromaDB is unavailable.
"""
import hashlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None

COLLECTION_NAME = "prompt_cache"


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    try:
        from database import get_chroma
        _chroma_client = get_chroma()
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        return _collection
    except Exception as e:
        logger.warning(f"ChromaDB unavailable: {e}")
        return None


def semantic_add(entry_id: str, prompt: str, model: str):
    """Index a prompt in ChromaDB for future semantic lookup."""
    col = _get_collection()
    if col is None:
        return
    try:
        col.upsert(
            ids=[entry_id],
            documents=[prompt],
            metadatas=[{"model": model, "ts": time.time()}]
        )
    except Exception as e:
        logger.warning(f"ChromaDB upsert failed: {e}")


def semantic_search(prompt: str, model: str, threshold: float = 0.85, n: int = 1) -> Optional[dict]:
    """
    Search for semantically similar cached prompts.
    Returns {"entry_id": ..., "similarity": ...} or None.
    """
    col = _get_collection()
    if col is None:
        return None
    try:
        count = col.count()
        if count == 0:
            return None

        results = col.query(
            query_texts=[prompt],
            n_results=min(n, count),
            where={"model": model} if model else None
        )
        if not results["ids"] or not results["ids"][0]:
            return None

        # chromadb returns distances (lower = more similar for cosine)
        distances = results["distances"][0]
        ids = results["ids"][0]

        for dist, eid in zip(distances, ids):
            similarity = 1.0 - dist  # cosine: distance → similarity
            if similarity >= threshold:
                return {"entry_id": eid, "similarity": round(similarity, 4)}
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}")
    return None


def semantic_delete(entry_id: str):
    col = _get_collection()
    if col is None:
        return
    try:
        col.delete(ids=[entry_id])
    except Exception:
        pass


def semantic_clear():
    global _collection
    col = _get_collection()
    if col is None:
        return
    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
        _collection = None
    except Exception:
        pass
