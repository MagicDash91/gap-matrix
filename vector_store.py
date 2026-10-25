"""Vector store sederhana berbasis numpy (pengganti FAISS / Chroma).

Vektor disimpan sebagai matriks float32 yang sudah dinormalisasi L2, sehingga
cosine similarity = dot product: sim = V · q. Pencarian dilakukan brute-force,
yang lebih dari cukup untuk ribuan chunk dari 18 paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def l2_normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


class VectorStore:
    def __init__(self, ids: list[str], vectors: np.ndarray):
        if len(ids) != len(vectors):
            raise ValueError("Jumlah id dan vektor tidak sama")
        self.ids = list(ids)
        self.vectors = l2_normalize(np.asarray(vectors, dtype="float32"))

    def search(self, query_vec: np.ndarray, candidates: list[int] | None = None, k: int = 10) -> list[tuple[int, float]]:
        """Kembalikan (indeks, skor cosine) terurut menurun, opsional dibatasi pada indeks kandidat."""
        q = l2_normalize(np.asarray(query_vec, dtype="float32"))
        rows = np.arange(len(self.ids)) if candidates is None else np.asarray(candidates)
        sims = self.vectors[rows] @ q
        order = np.argsort(-sims)[:k]
        return [(int(rows[i]), float(sims[i])) for i in order]

    # -- persistence -----------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunk_ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        return cls(json.loads((folder / "chunk_ids.json").read_text()), np.load(folder / "vectors.npy"))
