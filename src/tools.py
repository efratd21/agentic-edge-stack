"""Part 3 — the RAG retrieval flow from Part 2, exposed as a single agent tool.

`rag_search` is the one tool the agent can choose to call. Its description names
the corpus domain (AI agents) so the model uses it for in-domain questions and
skips it for general ones. A relevance threshold turns a weak top hit into the
`NO_RELEVANT_CONTEXT` sentinel, letting the agent fall back to a direct answer
instead of grounding on an irrelevant chunk.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .config import settings
from .rag import RagIndex, logged_search

# Build the FAISS index once, lazily, on first tool use (not at import time).
_index: RagIndex | None = None


def _get_index() -> RagIndex:
    global _index
    if _index is None:
        _index = RagIndex.from_corpus()
    return _index


@tool
def rag_search(query: str) -> str:
    """Search the local knowledge base of documents about AI AGENTS — agent loops,
    tool calling, ReAct and other reasoning patterns, retrieval-augmented
    generation (RAG), agent memory, and multi-agent systems. Use this for any
    question about how AI agents work. Do NOT use it for general-knowledge
    questions unrelated to AI agents. Returns the top-3 relevant passages, or the
    string NO_RELEVANT_CONTEXT if nothing relevant is found.
    """
    hits = logged_search(_get_index(), query)  # also appends to rag_retrieval.log
    if not hits or hits[0].score < settings.relevance_threshold:
        return "NO_RELEVANT_CONTEXT"
    return "\n\n".join(f"[score={h.score:.2f}] {h.chunk.text}" for h in hits)
