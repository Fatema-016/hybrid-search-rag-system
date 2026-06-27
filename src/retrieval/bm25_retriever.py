"""
BM25 (sparse keyword) retrieval.

Design notes:
- BM25Retriever has no native save/load. Rather than pickle the retriever
  object itself (risky across rank_bm25 versions), we pickle the
  underlying Document corpus and rebuild BM25Retriever from it at load
  time -- rebuilding from already-chunked text is fast 
  so this avoids re-running ingestion while
  staying pickle-safe.
- chunk_id is used to dedupe on append, mirroring vector_store.py's
  idempotency: re-ingesting the same paper won't create duplicates.
"""

import os
import pickle
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

CORPUS_PATH = "chroma_db/bm25_corpus.pkl"


def _chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    documents = []
    for chunk in chunks:
        authors = chunk["authors"]
        metadata = {
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "title": chunk["title"],
            "authors": ", ".join(authors) if isinstance(authors, list) else authors,
            "chunk_id": chunk["chunk_id"],
        }
        documents.append(Document(page_content=chunk["text"], metadata=metadata))
    return documents


def load_corpus() -> List[Document]:
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save_corpus(documents: List[Document]) -> None:
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(documents, f)


def add_chunks_to_bm25(chunks: List[Dict]) -> None:
    """Append new chunks to the persisted BM25 corpus, deduping by chunk_id."""
    existing = load_corpus()
    existing_ids = {doc.metadata["chunk_id"] for doc in existing}

    new_documents = _chunks_to_documents(chunks)
    deduped_new = [d for d in new_documents if d.metadata["chunk_id"] not in existing_ids]

    save_corpus(existing + deduped_new)


def get_bm25_retriever(k: int = 5) -> BM25Retriever:
    """Load the persisted corpus and build a fresh BM25Retriever from it."""
    documents = load_corpus()
    if not documents:
        raise ValueError("BM25 corpus is empty — run add_chunks_to_bm25() during ingestion first.")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever