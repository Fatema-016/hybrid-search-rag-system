"""
Prompt templates for grounded RAG generation.

Design: retrieved chunks are numbered and formatted with their key
metadata inline, so the LLM can cite a specific source ("[2]") rather
than gesturing vaguely at "the context". These numbers map directly to
the source list shown in the UI's citation panel (Phase 5) -- a [2] in
the answer and source #2 in the sidebar refer to the same chunk.
"""

from typing import List
from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a research paper assistant. Answer the user's \
question using ONLY the numbered source excerpts provided below.

Rules:
- Cite the specific source number in square brackets, e.g. [1] or [2], \
immediately after any claim, number, or fact you state.
- If multiple sources support a claim, cite all of them, e.g. [1][3].
- If the answer is not contained in the provided sources, say so plainly \
-- do not use outside knowledge or guess.
- Be precise with numbers, model names, and technical terms -- copy them \
exactly as they appear in the sources rather than paraphrasing them.
"""


def format_sources_block(chunks: List[Document]) -> str:
    """Number each retrieved chunk and format it with its metadata."""
    blocks = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        blocks.append(
            f"[{i}] (Title: {meta.get('title')}, Page: {meta.get('page_number')})\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: List[Document]) -> str:
    sources_block = format_sources_block(chunks)
    return (
        f"Sources:\n{sources_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (remember to cite sources with [n]):"
    )