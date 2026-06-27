import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.ingestion.pdf_loader import extract_pages_from_bytes
from src.ingestion.arxiv_fetcher import search_arxiv, fetch_paper_by_id
from src.ingestion.chunker import chunk_pages
from src.retrieval.vector_store import add_chunks
from src.retrieval.bm25_retriever import add_chunks_to_bm25, load_corpus
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.generation.llm_service import generate_answer
from src.storage.s3_service import upload_pdf_bytes


st.set_page_config(page_title="Research Paper Assistant", layout="wide")


@st.cache_resource
def load_retriever():
    return get_hybrid_retriever(k=5)


def corpus_is_empty() -> bool:
    return len(load_corpus()) == 0


def ingest_and_refresh(chunks):
    """Shared by both ingestion paths: write to both stores, then
    invalidate the cached retriever so it rebuilds with new data."""
    add_chunks(chunks)
    add_chunks_to_bm25(chunks)
    load_retriever.clear()


# ---------- Sidebar: ingestion ----------
with st.sidebar:
    st.header("Ingest a paper")

    tab1, tab2 = st.tabs(["Upload PDF", "Search arXiv"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
        title_input = st.text_input("Paper title")
        authors_input = st.text_input("Authors (comma-separated)")

        if st.button("Ingest uploaded PDF", disabled=uploaded_file is None):
            pdf_bytes = uploaded_file.read()
            source_name = uploaded_file.name
            pages = extract_pages_from_bytes(pdf_bytes, source_name)
            upload_pdf_bytes(pdf_bytes, f"papers/{source_name}")

            metadata = {
                "title": title_input or source_name,
                "authors": [a.strip() for a in authors_input.split(",")] if authors_input else [],
            }
            chunks = chunk_pages(pages, metadata)

            with st.spinner(f"Embedding {len(chunks)} chunks..."):
                ingest_and_refresh(chunks)
            st.success(f"Ingested '{metadata['title']}' -- {len(chunks)} chunks")

    with tab2:
        query = st.text_input("Search query", placeholder="e.g. retrieval augmented generation")
        if st.button("Search") and query:
            st.session_state.arxiv_results = search_arxiv(query, max_results=5)

        for r in st.session_state.get("arxiv_results", []):
            st.write(f"**{r['title']}**")
            st.caption(", ".join(r["authors"][:3]) + ("..." if len(r["authors"]) > 3 else ""))
            if st.button("Ingest this paper", key=f"ingest_{r['arxiv_id']}"):
                with st.spinner(f"Fetching and embedding '{r['title']}'..."):
                    paper = fetch_paper_by_id(r["arxiv_id"])
                    upload_pdf_bytes(paper["pdf_bytes"], f"papers/{paper['arxiv_id']}.pdf")
                    pages = extract_pages_from_bytes(paper["pdf_bytes"], paper["arxiv_id"])
                    chunks = chunk_pages(pages, {"title": paper["title"], "authors": paper["authors"]})
                    ingest_and_refresh(chunks)
                st.success(f"Ingested '{r['title']}' -- {len(chunks)} chunks")

    st.divider()
    ingested_titles = sorted({doc.metadata["title"] for doc in load_corpus()})
    st.caption(f"{len(ingested_titles)} paper(s) in the knowledge base:")
    for t in ingested_titles:
        st.caption(f"- {t}")

    st.divider()
    st.caption("Semantic Search: bge-base-en-v1.5 (local) | Keyword Search: BM25 | LLM: Groq llama-3.3-70b-versatile")


# ---------- Main panel: chat ----------
st.title("AI/ML Research Assistant")
st.caption("Hybrid search (semantic + BM25) RAG over research papers, with grounded citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources"):
                for i, doc in enumerate(msg["sources"], start=1):
                    meta = doc.metadata
                    st.markdown(f"**[{i}]** {meta['title']} -- page {meta['page_number']} (`{meta['chunk_id']}`)")
                    st.text(doc.page_content[:300] + "...")

if corpus_is_empty():
    st.info("Ingest at least one paper from the sidebar before asking questions.")
else:
    user_question = st.chat_input("Ask a question about your ingested papers...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                retriever = load_retriever()
                chunks = retriever.invoke(user_question)[:5]  # cap union output to top-5
                try:
                    answer = generate_answer(user_question, chunks)
                except Exception as e:
                    answer = f"Generation failed: {e}"

            st.write(answer)
            with st.expander("View sources"):
                for i, doc in enumerate(chunks, start=1):
                    meta = doc.metadata
                    st.markdown(f"**[{i}]** {meta['title']} -- page {meta['page_number']} (`{meta['chunk_id']}`)")
                    st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": chunks})