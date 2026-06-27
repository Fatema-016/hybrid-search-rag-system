"""
Cross-encoder reranker.

Hybrid retrieval (Phase 3) ranks via RRF over two independent rankers,
which never directly scores query-passage relevance together. A
cross-encoder does exactly that -- scoring the (query, passage) pair
jointly -- at the cost of being too slow to run over a whole corpus.
Pattern: retrieve a WIDE candidate set cheaply via hybrid search, then
rerank only those candidates down to a clean, precise top-k.
"""

import torch
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    return _cross_encoder


def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    """Score each candidate directly against the query; return the
    top_k most relevant, sorted descending by relevance score."""
    if not documents:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]