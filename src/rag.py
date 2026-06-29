"""In-memory RAG pipeline: load -> chunk -> embed -> FAISS (cosine) -> search.

Part 2 of the assignment. The index is built once from `data/corpus/` and held
in memory. Embeddings come from `nomic-embed-text` served by Ollama (no local
sentence-transformers download). Similarity is cosine, implemented as inner
product over L2-normalized vectors (`IndexFlatIP`).

CLI:
    python -m src.rag "how does an agent decide to use a tool?"
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from ollama import Client

from .config import settings

# One Ollama client for the whole module, pointed at the configured host.
_client = Client(host=settings.ollama_host)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A retrievable unit of text plus where it came from (for the log)."""

    text: str
    source: str   # corpus file name
    section: str  # nearest markdown heading


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk


# --------------------------------------------------------------------------- #
# 1. Load + chunk
# --------------------------------------------------------------------------- #
def load_corpus(corpus_dir: str | Path | None = None) -> list[Chunk]:
    """Read every .md/.txt file under the corpus dir into structure-aware chunks.

    We split on markdown headings first so each chunk stays topically coherent,
    then window any oversized section into ~chunk_size pieces with overlap.
    """
    corpus_dir = Path(corpus_dir or settings.corpus_dir)
    files = sorted([*corpus_dir.glob("*.md"), *corpus_dir.glob("*.txt")])
    if not files:
        raise FileNotFoundError(f"No .md/.txt files found in {corpus_dir!r}")

    chunks: list[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for section_title, section_body in _split_sections(text):
            for piece in _window(section_body, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(text=piece, source=path.name, section=section_title))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections on ATX headings (#, ##, ...)."""
    sections: list[tuple[str, str]] = []
    current_title = "(intro)"
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    # Drop empty bodies (e.g. a heading immediately followed by another heading).
    return [(t, b) for t, b in sections if b]


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character window with overlap. No-op if the text fits in one."""
    if len(text) <= size:
        return [text] if text.strip() else []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        out.append(text[i : i + size].strip())
        i += step
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# 2. Embed (nomic-embed-text via Ollama, with required task prefixes)
# --------------------------------------------------------------------------- #
def embed(texts: list[str], *, is_query: bool = False) -> np.ndarray:
    """Embed texts and L2-normalize so inner product == cosine.

    nomic-embed-text is instruction-tuned and requires task prefixes:
    'search_document:' for stored passages, 'search_query:' for queries.
    Omitting these measurably degrades retrieval.
    """
    prefix = "search_query: " if is_query else "search_document: "
    vecs = [
        _client.embeddings(model=settings.embed_model, prompt=prefix + t)["embedding"]
        for t in texts
    ]
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # critical: cosine via normalized inner product
    return arr


# --------------------------------------------------------------------------- #
# 3. Index + search
# --------------------------------------------------------------------------- #
class RagIndex:
    """A FAISS cosine index over the corpus, built once and queried many times."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build an index over an empty corpus.")
        self.chunks = chunks
        vecs = embed([c.text for c in chunks])  # documents
        if vecs.shape[1] != settings.embed_dim:
            raise ValueError(
                f"Embedding dim {vecs.shape[1]} != configured EMBED_DIM "
                f"{settings.embed_dim}. Index/model mismatch."
            )
        self.index = faiss.IndexFlatIP(settings.embed_dim)
        self.index.add(vecs)

    @classmethod
    def from_corpus(cls, corpus_dir: str | Path | None = None) -> "RagIndex":
        return cls(load_corpus(corpus_dir))

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        qv = embed([query], is_query=True)
        scores, idx = self.index.search(qv, k)
        hits: list[Hit] = []
        for rank, (j, s) in enumerate(zip(idx[0], scores[0]), start=1):
            if j == -1:  # FAISS pads with -1 when fewer than k vectors exist
                continue
            hits.append(Hit(rank=rank, score=float(s), chunk=self.chunks[j]))
        return hits


# --------------------------------------------------------------------------- #
# 4. Retrieval logging (the Part 2 deliverable)
# --------------------------------------------------------------------------- #
def _get_logger() -> logging.Logger:
    Path(settings.rag_log).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rag_retrieval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(settings.rag_log, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def logged_search(index: RagIndex, query: str, k: int | None = None) -> list[Hit]:
    """Search and append a human-readable record to logs/rag_retrieval.log."""
    hits = index.search(query, k)
    logger = _get_logger()
    logger.info("QUERY: %s", query)
    if not hits:
        logger.info("  (no hits)")
    for h in hits:
        preview = h.chunk.text[:200].replace("\n", " ")
        logger.info(
            "  [#%d score=%.3f src=%s §%s] %s",
            h.rank, h.score, h.chunk.source, h.chunk.section, preview,
        )
    return hits


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m src.rag "your query here"', file=sys.stderr)
        raise SystemExit(2)

    query = " ".join(argv)
    index = RagIndex.from_corpus()
    hits = logged_search(index, query)

    print(f'\nQUERY: "{query}"')
    print(f"Built index over {len(index.chunks)} chunks; top-{settings.top_k}:\n")
    if not hits:
        print("  (no relevant chunks found)")
    for h in hits:
        print(f"  #{h.rank}  score={h.score:.3f}  {h.chunk.source} § {h.chunk.section}")
        snippet = h.chunk.text[:300].replace("\n", " ")
        print(f"      {snippet}...\n")
    print(f"(logged to {settings.rag_log})")


if __name__ == "__main__":
    main()
