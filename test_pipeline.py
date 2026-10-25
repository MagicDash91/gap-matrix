"""Uji unit offline (tanpa API key & tanpa unduh model): python -m pytest -q"""
import math
import os

import numpy as np

os.environ["MOCK_LLM"] = "1"

from src.bm25 import BM25, tokenize  # noqa: E402
from src.ingest import Chunk, chunk_pages, cut_references  # noqa: E402
from src.llm import HashEmbedder, parse_json  # noqa: E402
from src.retriever import HybridIndex  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

CHUNKS = [
    Chunk("A-c0", "A", 1, "BM25 hybrid retrieval with cross-encoder reranking on troubleshooting tickets"),
    Chunk("A-c1", "A", 2, "We evaluate faithfulness and hallucination rate of the assistant"),
    Chunk("B-c0", "B", 1, "Knowledge graph GraphRAG for customer service tickets"),
    Chunk("B-c1", "B", 3, "Telecom 3GPP standards question answering with neural routing"),
]


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"value": "x"}\n```') == {"value": "x"}
    assert parse_json('Jawaban: {"a": 1} selesai')["a"] == 1


def test_tokenize_drops_stopwords():
    assert tokenize("The SOP untuk instalasi dan troubleshooting") == ["sop", "instalasi", "troubleshooting"]


def test_bm25_matches_formula():
    bm = BM25(["kode error sap", "instalasi modem", "error modem"], k1=1.5, b=0.75)
    # hitung manual untuk term "sap" di dokumen 0: n=3, n(t)=1, f=1, |d|=3, avgdl=7/3
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 3 / (7 / 3)))
    assert math.isclose(bm.score("sap", 0), expected, rel_tol=1e-9)
    s = bm.scores("error modem")
    assert max(s, key=s.get) == 2


def test_vector_store_cosine_and_persistence(tmp_path):
    vs = VectorStore(["x", "y", "z"], np.array([[1, 0], [0, 1], [1, 1]], dtype="float32"))
    top = vs.search(np.array([1, 0.1]), k=2)
    assert top[0][0] == 0 and top[1][0] == 2
    assert vs.search(np.array([1, 0]), candidates=[1, 2], k=1)[0][0] == 2
    vs.save(tmp_path)
    again = VectorStore.load(tmp_path)
    assert again.ids == ["x", "y", "z"] and np.allclose(again.vectors, vs.vectors)


def test_chunking_keeps_pages_and_overlap():
    page = "\n\n".join(f"Paragraf nomor {i} " + "kata " * 60 for i in range(10))
    chunks = chunk_pages("PX", [page, page], size=500, overlap=80)
    assert len(chunks) > 4
    assert {c.page for c in chunks} == {1, 2}


def test_cut_references():
    out = cut_references(["Intro text", "Results\n\nReferences\n[1] Lewis et al.", "[2] other"])
    assert len(out) == 2 and "Lewis" not in out[-1]


def test_hybrid_search_filters_by_paper():
    emb = HashEmbedder()
    idx = HybridIndex(CHUNKS, VectorStore([c.chunk_id for c in CHUNKS], emb.embed([c.text for c in CHUNKS])))
    hits = idx.search("hallucination faithfulness", emb, paper_id="A", k=2)
    assert hits[0].chunk_id == "A-c1"
    assert all(h.paper_id == "A" for h in hits)
    assert idx.search("3GPP telecom routing", emb, k=1)[0].chunk_id == "B-c1"


class _FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return _FakeResp(self.payload)


def test_http_clients_build_correct_requests():
    from src.config import Settings
    from src.llm import GemmaLLM, GroqLLM

    s = Settings(groq_api_key="k1", google_api_key="k2", rpm_sleep=0)
    groq = GroqLLM(s)
    groq.session = _FakeSession({"choices": [{"message": {"content": '{"value": "ok"}'}}]})
    assert groq.generate_json("halo") == {"value": "ok"}
    url, kw = groq.session.calls[0]
    assert url.endswith("/chat/completions") and kw["headers"]["Authorization"] == "Bearer k1"
    assert kw["json"]["messages"][0]["content"] == "halo"

    gem = GemmaLLM(s)
    gem.session = _FakeSession({"candidates": [{"content": {"parts": [{"text": '{"value": "ya"}'}]}}]})
    assert gem.generate_json("halo") == {"value": "ya"}
    assert ":generateContent" in gem.session.calls[0][0]
