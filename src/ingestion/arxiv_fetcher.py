"""
arXiv fetching utilities.

Returns paper metadata (title, authors, arxiv_id) plus raw PDF bytes —
mirrors pdf_loader's bytes-first design so the chunker (Step 4) can treat
arXiv-sourced and manually-uploaded PDFs identically once both have been
reduced to "metadata dict + raw bytes".
"""

import arxiv
import requests
from typing import List, Dict


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search arXiv and return paper metadata only (no PDF download yet).

    Returns:
        List of dicts:
        {"arxiv_id", "title", "authors", "summary", "pdf_url"}
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for r in client.results(search):
        results.append({
            "arxiv_id": r.get_short_id(),
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "summary": r.summary.strip(),
            "pdf_url": r.pdf_url,
        })
    return results


def fetch_pdf_bytes(pdf_url: str) -> bytes:
    """Download a PDF's raw bytes from its arXiv URL."""
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    return response.content


def fetch_paper_by_id(arxiv_id: str) -> Dict:
    """
    Fetch one paper's full metadata + raw PDF bytes by arXiv ID
    (e.g. "2606.27287").

    Returns:
        {"arxiv_id", "title", "authors", "summary", "pdf_bytes"}
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    pdf_bytes = fetch_pdf_bytes(result.pdf_url)

    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip(),
        "authors": [a.name for a in result.authors],
        "summary": result.summary.strip(),
        "pdf_bytes": pdf_bytes,
    }