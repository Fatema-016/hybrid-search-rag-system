import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.arxiv_fetcher import fetch_paper_by_id
from src.ingestion.pdf_loader import extract_pages_from_bytes
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks, get_dense_retriever

paper = fetch_paper_by_id("2506.06962")
pages = extract_pages_from_bytes(paper["pdf_bytes"], paper["arxiv_id"])
chunks = chunk_pages(pages, {"title": paper["title"], "authors": paper["authors"]})

print(f"Embedding and storing {len(chunks)} chunks...")
add_chunks(chunks)
print("Done.")

retriever = get_dense_retriever(k=3)
results = retriever.invoke("what is retrieval augmented generation for image generation")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print("Title:", doc.metadata["title"])
    print("Page:", doc.metadata["page_number"])
    print("Chunk ID:", doc.metadata["chunk_id"])
    print("Text:", doc.page_content[:200])