"""BM25 Okapi yang ditulis manual (tanpa rank_bm25, LangChain, atau LlamaIndex).

Skor dokumen d untuk kueri q:

    score(d, q) = Σ_t∈q  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(t)      = ln( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )

f(t,d) = frekuensi term t di dokumen d, |d| = panjang dokumen, avgdl = rata-rata
panjang dokumen, N = jumlah dokumen, n(t) = jumlah dokumen yang memuat t.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Stopword kecil Indonesia + Inggris; cukup untuk membuang kata fungsi paling sering.
STOPWORDS = set("""
yang dan di ke dari untuk dengan pada adalah ini itu atau dalam tidak akan juga oleh sebagai
the a an and or of to in on for with by is are was were be been this that these those it its as at from
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in corpus]
        self.lens = [sum(c.values()) for c in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(self.lens) / self.n) if self.n else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        self.idf = {t: math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0) for t, n_t in df.items()}

    def score(self, query: str, idx: int) -> float:
        tf, dl = self.docs[idx], self.lens[idx]
        s = 0.0
        for t in tokenize(query):
            f = tf.get(t, 0)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += self.idf[t] * f * (self.k1 + 1) / denom
        return s

    def scores(self, query: str, candidates: list[int] | None = None) -> dict[int, float]:
        ids = candidates if candidates is not None else range(self.n)
        return {i: self.score(query, i) for i in ids}
