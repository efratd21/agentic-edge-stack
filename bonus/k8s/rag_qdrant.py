"""Bonus 3 — the RAG retrieval flow backed by Qdrant on Kubernetes.

This is a drop-in alternative to the in-memory FAISS index (`src/rag.py`): same
corpus, same chunking, same `nomic-embed-text` embeddings — only the vector
store changes from a process-local FAISS index to a standalone Qdrant service
running in the cluster. It deliberately *reuses* `load_corpus`, `embed`, `Chunk`
and `Hit` from `src.rag` so the only thing that differs is where vectors live.

Prerequisite: Qdrant is deployed (see bonus/k8s/README.md) and reachable at
`settings.qdrant_url` (e.g. via `kubectl port-forward svc/qdrant 6333:6333`).

CLI:
    python bonus/k8s/rag_qdrant.py --build                 # (re)build the collection
    python bonus/k8s/rag_qdrant.py "how does an agent use a tool?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable whether this is run as a script or a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402

from src.config import settings  # noqa: E402
from src.rag import Chunk, Hit, embed, load_corpus  # noqa: E402


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


class QdrantIndex:
    """A Qdrant collection over the corpus — the same contract as RagIndex."""

    def __init__(self, client: QdrantClient | None = None):
        self.client = client or _client()
        self.collection = settings.qdrant_collection

    # --- build ------------------------------------------------------------- #
    @classmethod
    def build(cls, corpus_dir: str | Path | None = None, *, recreate: bool = True) -> "QdrantIndex":
        """Load → chunk → embed the corpus and upsert it into Qdrant.

        `recreate=True` drops any existing collection first so a rebuild is
        deterministic (the assignment corpus is tiny, so this is cheap).
        """
        self = cls()
        chunks = load_corpus(corpus_dir)
        vecs = embed([c.text for c in chunks])  # normalized float32, dim = embed_dim
        if vecs.shape[1] != settings.embed_dim:
            raise ValueError(
                f"Embedding dim {vecs.shape[1]} != EMBED_DIM {settings.embed_dim}."
            )

        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=settings.embed_dim, distance=models.Distance.COSINE
                ),
            )

        points = [
            models.PointStruct(
                id=i,
                vector=vecs[i].tolist(),
                payload={"text": c.text, "source": c.source, "section": c.section},
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(self.collection, points=points, wait=True)
        return self

    # --- search ------------------------------------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or settings.top_k
        qv = embed([query], is_query=True)[0]
        res = self.client.query_points(
            collection_name=self.collection,
            query=qv.tolist(),
            limit=k,
            with_payload=True,
        )
        hits: list[Hit] = []
        for rank, p in enumerate(res.points, start=1):
            payload = p.payload or {}
            chunk = Chunk(
                text=payload.get("text", ""),
                source=payload.get("source", "?"),
                section=payload.get("section", "?"),
            )
            hits.append(Hit(rank=rank, score=float(p.score), chunk=chunk))
        return hits


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]

    if not argv or argv == ["--build"]:
        print(f"Building Qdrant collection '{settings.qdrant_collection}' "
              f"at {settings.qdrant_url} …")
        idx = QdrantIndex.build()
        info = idx.client.get_collection(idx.collection)
        print(f"OK — {info.points_count} points, dim={settings.embed_dim}, cosine.")
        if not argv:
            print('\nNow query it: python bonus/k8s/rag_qdrant.py "your question"')
        return

    query = " ".join(a for a in argv if a != "--build")
    idx = QdrantIndex.build() if "--build" in argv else QdrantIndex()
    hits = idx.search(query)

    print(f'\nQUERY: "{query}"  (top-{settings.top_k} from Qdrant)\n')
    if not hits:
        print("  (no hits — is the collection built?)")
    for h in hits:
        print(f"  #{h.rank}  score={h.score:.3f}  {h.chunk.source} § {h.chunk.section}")
        print(f"      {h.chunk.text[:300].strip()}...\n")


if __name__ == "__main__":
    main()
