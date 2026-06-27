"""
LLM generation service -- Groq (llama-3.3-70b-versatile) for the
main RAG answer generation.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,  # deterministic: factual grounding
        )
    return _llm


def generate_answer(question: str, retrieved_chunks) -> str:
    """
    Args:
        question: user's query.
        retrieved_chunks: list of Document objects, already sliced to a
                           clean top-k (the caller's job -- see Phase 3 note
                           on EnsembleRetriever returning a union, not top-k).
    Returns:
        Answer text with inline [n] citation markers.
    """
    llm = get_llm()
    user_prompt = build_user_prompt(question, retrieved_chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content