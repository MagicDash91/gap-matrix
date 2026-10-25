"""Klien LLM via HTTP langsung (requests), tanpa SDK dan tanpa framework RAG.

- Groq   : endpoint OpenAI-compatible  POST https://api.groq.com/openai/v1/chat/completions
- Gemini : REST                        POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Embedding lokal: sentence-transformers (model dijalankan di laptop, bukan API)
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import requests

from .config import Settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def parse_json(text: str) -> dict:
    """Parse output LLM secara toleran: buang code fence, ambil objek {...} pertama."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Tidak ada objek JSON dalam output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class _HTTPLLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.session = requests.Session()

    def _call(self, prompt: str) -> str:  # pragma: no cover - diimplementasi subclass
        raise NotImplementedError

    def generate(self, prompt: str, retries: int = 5) -> str:
        delay = 5.0
        for attempt in range(retries):
            try:
                out = self._call(prompt)
                time.sleep(self.s.rpm_sleep)          # hormati rate limit tier gratis
                return out or ""
            except (requests.RequestException, KeyError, IndexError) as exc:
                if attempt == retries - 1:
                    raise
                print(f"  [retry {attempt + 1}] {type(exc).__name__}: {exc}")
                time.sleep(delay)
                delay *= 2
        return ""

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt)
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            fixed = self.generate("Ubah teks berikut menjadi SATU objek JSON valid saja, tanpa penjelasan:\n\n" + raw)
            return parse_json(fixed)


class GroqLLM(_HTTPLLM):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.groq_api_key:
            raise RuntimeError("Set GROQ_API_KEY di environment / file .env")

    def _call(self, prompt: str) -> str:
        r = self.session.post(
            GROQ_URL, timeout=120,
            headers={"Authorization": f"Bearer {self.s.groq_api_key}", "Content-Type": "application/json"},
            json={"model": self.s.groq_model, "temperature": self.s.temperature,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        r.raise_for_status()                          # 429/5xx → RequestException → retry
        return r.json()["choices"][0]["message"]["content"]


class GemmaLLM(_HTTPLLM):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.google_api_key:
            raise RuntimeError("Set GOOGLE_API_KEY (atau GEMINI_API_KEY) di environment / file .env")

    def _call(self, prompt: str) -> str:
        r = self.session.post(
            GEMINI_URL.format(model=self.s.gemma_model), timeout=120,
            headers={"x-goog-api-key": self.s.google_api_key, "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": self.s.temperature}},
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class LocalEmbedder:
    """Embedding lokal sentence-transformers (paraphrase-multilingual-mpnet-base-v2, 768 dimensi)."""

    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embed_model)

    def embed(self, texts: list[str], batch: int = 32) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return np.asarray(vecs, dtype="float32")


# ---------------------------------------------------------------------------
# Mock untuk dry-run / unit test tanpa API key dan tanpa unduh model.
# ---------------------------------------------------------------------------
class HashEmbedder:
    """Embedding bag-of-words ter-hash (deterministik). Hanya untuk uji offline."""

    dim = 512

    def embed(self, texts: list[str], batch: int = 0) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype="float32")
        for r, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                arr[r, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


class MockLLM:
    """Mengembalikan JSON berformat benar berisi potongan konteks — untuk menguji alur, bukan isi."""

    def generate(self, prompt: str) -> str:
        ctx = prompt.split("KONTEKS:")[-1][:160].replace("\n", " ").strip()
        return json.dumps({"value": f"[MOCK] {ctx[:80]}", "evidence": ctx[:120], "chunk_ids": [],
                           "present": False, "analysis": "[MOCK] analisis gap", "answer": f"[MOCK] {ctx[:80]}"})

    def generate_json(self, prompt: str) -> dict:
        return parse_json(self.generate(prompt))


def build_llm(s: Settings):
    if s.mock:
        return MockLLM()
    return GroqLLM(s) if s.llm_provider == "groq" else GemmaLLM(s)


def build_embedder(s: Settings):
    return HashEmbedder() if s.mock else LocalEmbedder(s)
