import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.retrieval.reranker import get_cross_encoder

WIDE_K = 20
checks = [
    "What models were compared in the résumé screening study, and what job role was used?",
    "What was GPT-4o-mini's success rate under descriptive injection in a homogeneous candidate pool?",
]

hybrid = get_hybrid_retriever(k=WIDE_K)
cross_encoder = get_cross_encoder()

for query in checks:
    print(f"\n=== {query} ===")
    candidates = hybrid.invoke(query)[:WIDE_K]
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    for rank, (doc, score) in enumerate(ranked[:3], start=1):
        print(f"[{rank}] {doc.metadata['chunk_id']}")
        print(f"    {doc.page_content[:300]}")