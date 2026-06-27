"""
ChromaDB vector store wired to a local embedding model (bge-base-en-v1.5).

Design notes:
- chunk_id (from chunker.py) is reused as the Chroma document ID, making
  ingestion idempotent: re-running ingestion on the same paper overwrites
  existing chunks instead of duplicating them.
- bge-base-en-v1.5 is an asymmetric retrieval model: passages are embedded
  as-is, but queries should get an instruction prefix prepended before
  encoding. wrap this seamlessly inside a custom retriever class.
"""

import os
from typing import List, Dict, Any
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embedding_model = None
_vector_store = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        # Determine device accelerator automatically
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=PERSIST_DIR,
        )
    return _vector_store


def add_chunks(chunks: List[Dict]) -> None:
    """Embed and store chunks. Metadata values must be simple types
    (Chroma rejects lists), so authors is joined into a string here."""
    store = get_vector_store()

    documents, ids = [], []
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
        ids.append(chunk["chunk_id"])

    store.add_documents(documents=documents, ids=ids)


class BGEAsymmetricRetriever(BaseRetriever):
    """Custom retriever wrapper that guarantees the BGE query prefix
    is prepended to every incoming string query during retrieval operations."""
    vectorstore: Chroma
    search_kwargs: Dict[str, Any]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        instructed_query = QUERY_INSTRUCTION + query
        # Using standard vectorstore retrieval under the hood with the formatted query
        return self.vectorstore.similarity_search(instructed_query, **self.search_kwargs)


def get_dense_retriever(k: int = 5) -> BaseRetriever:
    """Exposes a clean, native LangChain retriever instance ready for 
    EnsembleRetriever binding in Phase 3."""
    return BGEAsymmetricRetriever(
        vectorstore=get_vector_store(),
        search_kwargs={"k": k}
    )