"""
Chunking utilities.

Takes per-page text (from pdf_loader / arxiv_fetcher) plus paper-level
metadata (title, authors) and produces chunks tagged with everything
needed for citation rendering later: title, authors, page number, and
a unique chunk_id.

Design note: chunking is done PER PAGE, not across the whole document,
so page_number stays accurate per chunk. Tradeoff: a sentence straddling
a page break gets split. 
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_pages(pages: List[Dict], paper_metadata: Dict) -> List[Dict]:
    """
    Args:
        pages: List of {"source", "page_number", "text"} from pdf_loader.
        paper_metadata: {"title": str, "authors": list[str]}.

    Returns:
        List of chunk dicts:
        {"chunk_id", "text", "source", "page_number", "title", "authors"}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    title = paper_metadata.get("title", "Unknown title")
    authors = paper_metadata.get("authors", [])

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{page['source']}_p{page['page_number']}_c{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": page["source"],
                "page_number": page["page_number"],
                "title": title,
                "authors": authors,
            })

    return chunks