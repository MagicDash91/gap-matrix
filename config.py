"""Konfigurasi pipeline: skema kolom gap matrix (sesuai template PPT), aspek gap, dan pengaturan model."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # .env opsional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "texts"
INDEX_DIR = DATA_DIR / "index"
OUTPUT_DIR = ROOT / "outputs"
PROMPT_DIR = ROOT / "prompts"

# Konteks domain yang disisipkan ke prompt (use-case kelompok)
DOMAIN = os.getenv(
    "DOMAIN_DESC",
    "RAG untuk asisten pengetahuan operasional di industri (telekomunikasi/ISP, IT support, SOP, troubleshooting)",
)


@dataclass
class Settings:
    # --- Generator (tahap 6 RAG). groq = sesuai infrastruktur mata kuliah; gemini = Gemma 4 via Gemini API
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemma_model: str = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    # --- Embedding lokal (sentence-transformers), sesuai PPT: paraphrase-multilingual-mpnet-base-v2 (768 dim)
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    # --- Retrieval
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))        # karakter
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))  # karakter
    top_k: int = int(os.getenv("TOP_K", "6"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    rpm_sleep: float = float(os.getenv("RPM_SLEEP", "2.0"))      # jeda antar-panggilan API (rate limit tier gratis)
    mock: bool = os.getenv("MOCK_LLM", "0") == "1"

    @property
    def llm_label(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.gemma_model


# ---------------------------------------------------------------------------
# Kolom gap matrix = template PPT: Paper | DOI | Problem | Metode | Dataset |
# Temuan | Keterbatasan | Future work. "Paper" dan "DOI" diambil dari
# data/papers.csv; enam kolom lainnya diekstraksi oleh LLM dari chunk paper.
# Setiap kolom punya query retrieval sendiri.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixField:
    key: str
    label: str
    query: str
    instruction: str


FIELDS: list[MatrixField] = [
    MatrixField("problem", "Problem",
                "problem challenge motivation difficulty existing approach limitation hallucination knowledge",
                "Ringkas masalah/research problem utama yang diangkat paper, maks. 30 kata."),
    MatrixField("metode", "Metode",
                "proposed method framework architecture pipeline retrieval chunking reranking embedding LLM agent graph",
                "Jelaskan metode/arsitektur RAG yang diusulkan beserta LLM dan model embedding yang dipakai, maks. 40 kata."),
    MatrixField("dataset", "Dataset",
                "dataset corpus documents tickets manuals questions evaluation set benchmark collected",
                "Sebutkan dataset/korpus beserta ukurannya bila disebut, maks. 25 kata. Tulis 'Tidak dilaporkan' bila tidak ada."),
    MatrixField("temuan", "Temuan",
                "results findings outperform improvement accuracy recall MRR achieved compared baseline",
                "Ringkas temuan utama beserta angka kuantitatif bila ada, maks. 40 kata."),
    MatrixField("keterbatasan", "Keterbatasan",
                "limitations threats to validity drawback only small sample single domain synthetic",
                "Ringkas keterbatasan yang dinyatakan penulis atau yang tampak jelas dari desain eksperimen, maks. 30 kata."),
    MatrixField("future_work", "Future work",
                "future work next steps plan extend further research",
                "Ringkas arah penelitian lanjutan yang disebut penulis, maks. 30 kata. Tulis 'Tidak dinyatakan' bila tidak ada."),
]

# ---------------------------------------------------------------------------
# Aspek biner untuk matriks cakupan. Dipilih dari kebutuhan use-case kelompok
# (Generative AI Operational Knowledge Assistant untuk ISP): aspek dengan
# cakupan rendah = kandidat kontribusi paper kelompok.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Aspect:
    key: str
    label: str
    question: str


ASPECTS: list[Aspect] = [
    Aspect("id_lowres", "Bahasa Indonesia",
           "Apakah studi mengevaluasi dokumen atau pertanyaan berbahasa Indonesia?"),
    Aspect("telco", "Domain telko / ISP",
           "Apakah domainnya telekomunikasi, jaringan, atau penyedia layanan internet (ISP)?"),
    Aspect("tiket", "Tiket / troubleshooting",
           "Apakah sistem memakai tiket insiden, log keluhan, atau prosedur troubleshooting untuk menyelesaikan masalah?"),
    Aspect("sop", "SOP / kebijakan internal",
           "Apakah korpus berisi SOP, manual prosedur, kebijakan, atau aturan internal organisasi?"),
    Aspect("hybrid", "Hybrid retrieval (sparse+dense)",
           "Apakah retriever menggabungkan pencarian leksikal/sparse (BM25, TF-IDF) dengan dense embedding?"),
    Aspect("rerank", "Reranking",
           "Apakah ada tahap reranking (cross-encoder, LLM reranker, dsb.)?"),
    Aspect("graph", "Knowledge graph",
           "Apakah studi memakai knowledge graph atau GraphRAG?"),
    Aspect("multiturn", "Multi-turn / agentic",
           "Apakah sistem mendukung percakapan multi-turn, retrieval iteratif, atau arsitektur agen?"),
    Aspect("sitasi", "Sitasi sumber di jawaban",
           "Apakah jawaban menyertakan sitasi/tautan ke dokumen sumber?"),
    Aspect("halusinasi", "Ukur halusinasi / faithfulness",
           "Apakah studi mengukur halusinasi, faithfulness, atau factual consistency secara eksplisit?"),
    Aspect("eskalasi", "Klarifikasi / eskalasi ke manusia",
           "Apakah sistem bisa bertanya balik, menolak menjawab, atau mengeskalasi ke petugas manusia?"),
    Aspect("open_local", "LLM open-weight / lokal",
           "Apakah studi memakai LLM open-weight yang dapat dijalankan lokal/on-premise?"),
    Aspect("produksi", "Diuji di produksi / pengguna nyata",
           "Apakah sistem dievaluasi dalam deployment produksi atau oleh pengguna nyata (bukan hanya offline)?"),
]
