"""Part 2 — RAG pipeline tests.

Pure-logic tests (chunking) need nothing; the retrieval tests exercise the real
`nomic-embed-text` embedder via Ollama, so they double as an integration check
that cosine retrieval actually separates in-corpus from off-corpus queries.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import settings
from src.rag import RagIndex, _split_sections, _window, embed, load_corpus


# --- pure logic: chunking -------------------------------------------------- #
def test_window_keeps_short_text_whole():
    assert _window("short text", size=100, overlap=10) == ["short text"]


def test_window_splits_large_text_within_size():
    pieces = _window("x" * 5000, size=1800, overlap=200)
    assert len(pieces) > 1
    assert all(len(p) <= 1800 for p in pieces)


def test_split_sections_on_markdown_headings():
    secs = dict(_split_sections("# A\nalpha\n## B\nbeta"))
    assert secs == {"A": "alpha", "B": "beta"}


def test_corpus_chunks_do_not_exceed_window():
    chunks = load_corpus()
    assert len(chunks) > 0
    assert all(len(c.text) <= settings.chunk_size for c in chunks)


# --- integration: embeddings + retrieval (requires Ollama) ----------------- #
def test_embeddings_normalized_and_correct_dim():
    v = embed(["hello world"])
    assert v.shape == (1, settings.embed_dim)
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)


def test_cosine_self_similarity_is_one():
    v = embed(["a passage about AI agents"])
    assert float((v @ v.T)[0, 0]) > 0.999


@pytest.fixture(scope="module")
def index() -> RagIndex:
    return RagIndex.from_corpus()


def test_in_corpus_scores_higher_than_off_corpus(index):
    in_topic = index.search("what is the ReAct pattern")[0].score
    off_topic = index.search("best chocolate cake recipe")[0].score
    assert in_topic > off_topic
    assert in_topic > settings.relevance_threshold


def test_search_returns_top_k(index):
    assert len(index.search("tool calling")) == settings.top_k
