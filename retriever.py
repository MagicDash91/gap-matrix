"""Hybrid retriever manual: dense (VectorStore) + sparse (BM25), digabung Reciprocal Rank Fusion.

RRF(d) = Σ_r  1 / (k_rrf + rank_r(d)),  dengan rank dimulai dari 1 dan k_rrf = 60.
RRF hanya memakai peringkat, sehingga skor cosine dan skor BM25 yang skalanya
berbeda tidak perlu dikalibrasi.
"""
from __future__ import annotations

from .bm25 import BM25
from .config import INDEX_DIR
from .ingest import Chunk
from .vector_store import VectorStore


class HybridIndex:
    def __init__(self, chunks: list[Chunk], store: VectorStore):
        self.chunks = chunks
        self.store = store
        self.bm25 = BM25([c.text for c in chunks])
        self.by_paper: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_paper.setdefault(c.paper_id, []).append(i)

    # -- build / load ---------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "HybridIndex":
        vecs = embedder.embed([c.text for c in chunks])
        store = VectorStore([c.chunk_id for c in chunks], vecs)
        store.save(INDEX_DIR)
        return cls(chunks, store)

    @classmethod
    def load(cls, chunks: list[Chunk]) -> "HybridIndex":
        store = VectorStore.load(INDEX_DIR)
        lookup = {c.chunk_id: c for c in chunks}
        return cls([lookup[i] for i in store.ids], store)

    # -- search ---------------------------------------------------------------
    def search(self, query: str, embedder, paper_id: str | None = None, k: int = 6,
               rrf_k: int = 60, pool: int = 50) -> list[Chunk]:
        """Cari chunk relevan. Bila paper_id diberikan, pencarian dibatasi pada paper itu."""
        cand = self.by_paper.get(paper_id, []) if paper_id else list(range(len(self.chunks)))
        if not cand:
            return []
        qv = embedder.embed([query])[0]
        dense = [i for i, _ in self.store.search(qv, candidates=cand, k=pool)]
        bm = self.bm25.scores(query, cand)
        sparse = [i for i in sorted(bm, key=lambda i: -bm[i])[:pool] if bm[i] > 0]
        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.chunks[i] for i in best]
