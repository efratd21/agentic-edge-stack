"""Part 3 — the rag_search tool and its relevance fallback.

Uses real retrieval (no stubbed model): an in-corpus query returns passages, an
off-corpus query falls through to the NO_RELEVANT_CONTEXT sentinel.
"""

from __future__ import annotations

from src.tools import rag_search


def test_in_corpus_query_returns_passages():
    out = rag_search.invoke({"query": "how does tool calling work in AI agents"})
    assert out != "NO_RELEVANT_CONTEXT"
    assert "score=" in out


def test_off_corpus_query_returns_sentinel():
    out = rag_search.invoke({"query": "who won the 1998 football world cup"})
    assert out == "NO_RELEVANT_CONTEXT"
