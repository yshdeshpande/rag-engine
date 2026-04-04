"""Prompt templates for RAG generation with citation support."""

from src.chunking.base import Chunk

SYSTEM_PROMPT = """\
You are a precise, knowledgeable assistant. Answer the user's question using \
ONLY the provided context passages. Follow these rules strictly:

1. Base your answer entirely on the context. Do not use prior knowledge.
2. Cite sources using [1], [2], etc. matching the passage numbers.
3. If the context doesn't contain enough information, say so clearly.
4. Be concise — don't repeat the question or pad your answer.\
"""

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided context to answer this question."
)


def format_context(chunks: list[tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into numbered passages for the prompt.

    Each passage includes its source doc_id and chunk index so the LLM
    can cite them, and the user can trace back to the original document.
    """
    if not chunks:
        return "No relevant passages found."

    passages = []
    for i, (chunk, score) in enumerate(chunks, start=1):
        header = f"[{i}] (source: {chunk.doc_id}, chunk {chunk.chunk_index}, score: {score:.3f})"
        passages.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(passages)


def build_user_message(query: str, context: str) -> str:
    """Combine the context and user query into the user message."""
    return f"""Context passages:
{context}

---

Question: {query}"""
