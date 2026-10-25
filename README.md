# RAG Pipeline (Python Murni) untuk Literature Gap Matrix — Asisten Pengetahuan Operasional Industri

Pipeline Retrieval-Augmented Generation yang membaca **20 artikel hasil penelusuran literatur** (`data/papers.csv`) lalu mengisi **literature gap matrix** sesuai template mata kuliah:

| Paper | DOI | Problem | Metode | Dataset | Temuan | Keterbatasan | Future work |
|---|---|---|---|---|---|---|---|

Pipeline ini **tidak memakai LangChain, LlamaIndex, FAISS, maupun SDK LLM**. Setiap tahap RAG ditulis manual dengan Python, numpy, dan requests:

| Tahap RAG (materi kuliah) | File | Implementasi |
|---|---|---|
| 1. Document processing | `src/ingest.py` | Unduh PDF open-access (`requests`), ekstrak teks per halaman (`pypdf`), buang daftar pustaka; fallback abstrak OpenAlex |
| 2. Chunking | `src/ingest.py` → `chunk_pages()` | Chunk berbasis paragraf + overlap karakter, nomor halaman disimpan |
| 3. Embedding | `src/llm.py` → `LocalEmbedder` | `sentence-transformers` lokal: `paraphrase-multilingual-mpnet-base-v2` (768 dim) |
| 4. Penyimpanan vektor | `src/vector_store.py` | Matriks numpy ter-normalisasi L2, cosine = dot product, simpan ke `.npy` |
| 5. Retrieval | `src/bm25.py`, `src/retriever.py` | BM25 Okapi ditulis manual + dense search, digabung **Reciprocal Rank Fusion**; dapat dibatasi per paper |
| 6. Augmented generation | `src/llm.py`, `src/extract.py`, `prompts/` | HTTP langsung ke Groq (OpenAI-compatible) atau Gemini REST; LLM mengisi JSON `{value, evidence, chunk_ids}` per sel |
| Analisis + ekspor | `src/gap.py`, `src/export.py` | Cakupan 13 aspek gap, draf analisis paragraf, Excel/Markdown/CSV |

**Use-case kelompok:** *Generative AI Operational Knowledge Assistant* untuk perusahaan ISP. Asisten menjawab pertanyaan tentang SOP, troubleshooting, dan error SAP berdasarkan knowledge perusahaan, lengkap dengan sumbernya.

## Cara menjalankan

```bash
pip install -r requirements.txt
cp .env.example .env                      # isi GROQ_API_KEY
python run_pipeline.py --focus "Knowledge assistant operasional ISP: SOP, troubleshooting, error SAP"
```

Hasilnya ada di `outputs/gap_matrix.xlsx` (sheet: Gap Matrix, Cakupan Aspek, Ringkasan Gap, Bukti, Analisis), `.md`, dan `.csv`.

**Tanya-jawab atas 20 artikel** memakai index yang sama, dengan jawaban bersitasi `[P05, hlm. 3]`:

```bash
python ask.py "Metode retrieval apa yang dipakai untuk tiket troubleshooting?"
python ask.py --paper P11 "Berapa persen chunk yang benar masuk top-4?"
python ask.py                              # mode interaktif
```

Opsi lain:

```bash
python run_pipeline.py --only P05 P11 P16     # uji sebagian paper dulu
python run_pipeline.py --stage ingest         # hanya document processing + chunking
python run_pipeline.py --stage index          # + embedding + vector store
python run_pipeline.py --stage extract        # pakai index yang sudah ada
MOCK_LLM=1 python run_pipeline.py --offline   # dry-run tanpa API key, internet, dan unduh model
python -m pytest -q                           # 8 uji unit (BM25 vs rumus, cosine, RRF, chunking, klien HTTP)
python scripts/compare.py                     # akurasi ekstraksi aspek vs matriks referensi
python scripts/build_reference.py             # bangun matriks final dari data/reference_matrix.json
```

Untuk memakai Gemma 4 (Gemini API) sebagai pengganti Groq, set `LLM_PROVIDER=gemini` dan `GOOGLE_API_KEY` di `.env`.

**Paper berbayar** (Elsevier: P12, P13; P19 dan P20 sudah tersedia sebagai PDF lokal): unduh PDF-nya lewat akses kampus lalu simpan sebagai `data/pdfs/P12.pdf` dan `data/pdfs/P13.pdf`. Pipeline otomatis memakai file lokal sebelum mencoba mengunduh.

## Validasi

`data/reference_matrix.json` berisi matriks hasil kurasi manual untuk 20 paper. `scripts/compare.py` membandingkan aspek hasil ekstraksi pipeline dengan referensi tersebut. Keluaran LLM, terutama paragraf analisis, berstatus **draf** dan wajib diverifikasi terhadap paper aslinya.
