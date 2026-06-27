import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.retrieval.reranker import get_cross_encoder

WIDE_K = 20

queries_and_keywords = [
    ("Which university is one of the AR-RAG authors affiliated with?", "Virginia Tech"),
    ("What models were compared in the résumé screening study, and what job role was used?", "DeepSeek"),
    ("What dataset was used in the AR-RAG ablation study?", "Midjourney"),
]

hybrid = get_hybrid_retriever(k=WIDE_K)
cross_encoder = get_cross_encoder()

for query, keyword in queries_and_keywords:
    print(f"\n{'='*70}\nQuery: {query}\n{'='*70}")
    candidates = hybrid.invoke(query)[:WIDE_K]

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    for rank, (doc, score) in enumerate(ranked[:7], start=1):  # top 7 only
        has_keyword = " <-- TARGET" if keyword in doc.page_content else ""
        print(f"\n[{rank}] score={score:.3f}  {doc.metadata['chunk_id']}{has_keyword}")
        print(f"    {doc.page_content[:250]}")