"""
PDF loading utilities.

Design note: every function here accepts raw bytes (a file stream) rather
than a filesystem path. This means this module works identically whether
the bytes come from a local file (now, Phase 1) or an S3 object (Phase 6) —
only the *caller* changes, never this code.
"""

from io import BytesIO
from typing import List, Dict
from pypdf import PdfReader


def extract_pages_from_bytes(pdf_bytes: bytes, source_name: str) -> List[Dict]:
    """
    Extract text page-by-page from a PDF given as raw bytes.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        source_name: Human-readable identifier for this PDF (filename or
                      arXiv ID) — carried through as metadata.

    Returns:
        List of dicts, one per page:
        {"source": source_name, "page_number": int (1-indexed), "text": str}
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip blank or scanned (non-text) pages for now

        pages.append({
            "source": source_name,
            "page_number": i + 1,
            "text": text,
        })

    return pages


def load_pdf_from_path(file_path: str) -> List[Dict]:
    """
    Convenience wrapper for local files during development.
    This is the ONLY function that changes in Phase 6 — a parallel
    function will read bytes from S3 instead and call
    extract_pages_from_bytes() the exact same way.
    """
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()

    source_name = file_path.replace("\\", "/").split("/")[-1]
    return extract_pages_from_bytes(pdf_bytes, source_name)