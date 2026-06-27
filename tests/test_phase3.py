import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.arxiv_fetcher import fetch_paper_by_id
from src.ingestion.pdf_loader import extract_pages_from_bytes
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, get_dense_retriever
from src.retrieval.bm25_retriever import add_chunks_to_bm25
from src.retrieval.hybrid_retriever import get_hybrid_retriever

paper = fetch_paper_by_id("2506.06962")
pages = extract_pages_from_bytes(paper["pdf_bytes"], paper["arxiv_id"])
chunks = chunk_pages(pages, {"title": paper["title"], "authors": paper["authors"]})

print(f"Syncing {len(chunks)} chunks into Chroma + BM25...")
add_chunks(chunks)          # idempotent -- already there from Phase 2
add_chunks_to_bm25(chunks)  # BM25 corpus is currently empty, this populates it
print("Done.\n")


def print_results(label, results):
    print(f"=== {label} ===")
    for i, doc in enumerate(results):
        print(f"  [{i+1}] page {doc.metadata['page_number']}: {doc.page_content[:120]}")
    print()


exact_query = "Virginia Tech"  # an author affiliation -- exact term

print_results("Dense only", get_dense_retriever(k=3).invoke(exact_query))
print_results("Hybrid (semantic + BM25)", get_hybrid_retriever(k=3).invoke(exact_query))

concept_query = "what is retrieval augmented generation for image generation"
print_results("Hybrid -- conceptual query", get_hybrid_retriever(k=3).invoke(concept_query))