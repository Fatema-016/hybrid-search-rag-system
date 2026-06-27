import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_retriever import get_hybrid_retriever, get_reranked_retriever

WIDE_K = 15
FINAL_K = 5

queries_and_keywords = [
    ("Virginia Tech", "Virginia Tech"),
    ("What models were compared in the résumé screening study, and what job role was used?", "DeepSeek"),
    ("What dataset was used in the AR-RAG ablation study?", "Midjourney"),
]

hybrid = get_hybrid_retriever(k=WIDE_K)
reranked = get_reranked_retriever(wide_k=WIDE_K, final_k=FINAL_K)


def find_position(docs, keyword):
    return next((i for i, d in enumerate(docs) if keyword in d.page_content), None)


for query, keyword in queries_and_keywords:
    print(f"\n=== Query: {query} ===")
    print(f"Looking for: '{keyword}'")

    wide_results = hybrid.invoke(query)
    print(f"Position in hybrid top-{len(wide_results)} (wide net):  "
          f"{find_position(wide_results, keyword)}")

    hybrid_top5 = wide_results[:FINAL_K]
    print(f"Position in hybrid top-{FINAL_K} (current production): "
          f"{find_position(hybrid_top5, keyword)}")

    reranked_results = reranked.invoke(query)
    print(f"Position in RERANKED top-{FINAL_K}:                     "
          f"{find_position(reranked_results, keyword)}")