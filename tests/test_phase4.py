import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.generation.llm_service import generate_answer

retriever = get_hybrid_retriever(k=5)
question = "What does AR-RAG propose for image generation, and what dataset was used in the ablation study?"

chunks = retriever.invoke(question)[:5]  # cap the union output to a clean top-5
print(f"Retrieved {len(chunks)} chunks for context.\n")

answer = generate_answer(question, chunks)
print("=== Answer ===")
print(answer)

print("\n=== Source list (for citation mapping) ===")
for i, doc in enumerate(chunks, start=1):
    meta = doc.metadata
    print(f"[{i}] {meta['title']} -- page {meta['page_number']} (chunk_id: {meta['chunk_id']})")