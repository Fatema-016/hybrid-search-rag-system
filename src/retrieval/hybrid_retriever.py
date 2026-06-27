"""
Hybrid retrieval: combines dense (semantic) and BM25 (sparse/keyword)
retrieval via LangChain's EnsembleRetriever, which applies Reciprocal
Rank Fusion (RRF) automatically.

Starting weights: 0.6 semantic / 0.4 BM25 -- to be tuned in Phase 7
based on documented DeepEval context-precision A/B results.
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from src.retrieval.vector_store import get_dense_retriever
from src.retrieval.bm25_retriever import get_bm25_retriever
from src.retrieval.reranker import rerank_documents

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def get_hybrid_retriever(k: int = 5, weights: tuple = (SEMANTIC_WEIGHT, BM25_WEIGHT)) -> EnsembleRetriever:
    dense_retriever = get_dense_retriever(k=k)
    bm25_retriever = get_bm25_retriever(k=k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=list(weights),
    )


class RerankedRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    wide_k: int
    final_k: int

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        candidates = self.base_retriever.invoke(query)[: self.wide_k]
        return rerank_documents(query, candidates, top_k=self.final_k)


def get_reranked_retriever(wide_k: int = 20, final_k: int = 5) -> BaseRetriever:
    hybrid = get_hybrid_retriever(k=wide_k)
    return RerankedRetriever(base_retriever=hybrid, wide_k=wide_k, final_k=final_k)