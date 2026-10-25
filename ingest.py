"""Tahap 1 — Ingest: unduh PDF, ekstrak teks per halaman, lalu chunking.

Urutan sumber teks per paper:
  1. data/pdfs/<ID>.pdf  (letakkan manual untuk paper berbayar dari Elsevier/IEEE/Springer)
  2. unduh otomatis dari kolom `pdf_url` (arXiv, ACL Anthology, MDPI, Frontiers — open access)
  3. fallback: abstrak dari OpenAlex berdasarkan DOI/judul (ditandai source=abstract)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import pandas as pd
import requests

from .config import DATA_DIR, PDF_DIR, TEXT_DIR, Settings

HEADERS = {"User-Agent": "rag-gap-matrix/1.0 (academic use)"}


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    page: int
    text: str


def load_papers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "papers.csv", dtype=str).fillna("")


def download_pdf(row: pd.Series) -> bool:
    path = PDF_DIR / f"{row['id']}.pdf"
    if path.exists():
        return True
    if not row["pdf_url"]:
        return False
    try:
        r = requests.get(row["pdf_url"], headers=HEADERS, timeout=60)
        if r.ok and r.content[:4] == b"%PDF":
            path.write_bytes(r.content)
            return True
    except requests.RequestException as exc:
        print(f"  [{row['id']}] gagal unduh: {exc}")
    return False


def openalex_abstract(row: pd.Series) -> str:
    """Rekonstruksi abstrak dari inverted index OpenAlex."""
    url = (f"https://api.openalex.org/works/doi:{row['doi']}" if row["doi"]
           else f"https://api.openalex.org/works?search={requests.utils.quote(row['title'])}&per-page=1")
    try:
        data = requests.get(url, headers=HEADERS, timeout=30).json()
        work = data["results"][0] if "results" in data else data
        inv = work.get("abstract_inverted_index") or {}
        words = sorted(((p, w) for w, ps in inv.items() for p in ps))
        return " ".join(w for _, w in words)
    except Exception:  # noqa: BLE001
        return ""


def pdf_pages(path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(p.extract_text() or "") for p in reader.pages]


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # sambung kata terpotong tanda hubung
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def cut_references(pages: list[str]) -> list[str]:
    """Buang bagian daftar pustaka supaya retrieval tidak mengambil judul paper lain."""
    out = []
    for p in pages:
        m = re.search(r"\n\s*(References|REFERENCES|Bibliography|DAFTAR PUSTAKA|REFERENCE)\s*\n", p)
        if m:
            out.append(p[: m.start()])
            break
        out.append(p)
    return out


def chunk_pages(paper_id: str, pages: list[str], size: int, overlap: int) -> list[Chunk]:
    """Chunking berbasis paragraf dengan batas ukuran + overlap, menyimpan nomor halaman."""
    chunks: list[Chunk] = []
    n = 0
    for page_no, page in enumerate(pages, start=1):
        paras = [p.strip() for p in re.split(r"\n\s*\n", clean(page)) if len(p.strip()) > 40]
        buf = ""
        for para in paras:
            if len(buf) + len(para) > size and buf:
                chunks.append(Chunk(f"{paper_id}-c{n:03d}", paper_id, page_no, buf))
                n += 1
                buf = buf[-overlap:] + " " + para
            else:
                buf = (buf + "\n" + para).strip()
            while len(buf) > size * 1.6:              # paragraf sangat panjang → potong keras
                chunks.append(Chunk(f"{paper_id}-c{n:03d}", paper_id, page_no, buf[:size]))
                n += 1
                buf = buf[size - overlap :]
        if buf:
            chunks.append(Chunk(f"{paper_id}-c{n:03d}", paper_id, page_no, buf))
            n += 1
    return chunks


def run(s: Settings, offline: bool = False) -> list[Chunk]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    papers = load_papers()
    all_chunks: list[Chunk] = []
    manifest = []
    for _, row in papers.iterrows():
        pid = row["id"]
        source = "pdf"
        has_pdf = (PDF_DIR / f"{pid}.pdf").exists() or (not offline and download_pdf(row))
        if has_pdf:
            pages = cut_references(pdf_pages(PDF_DIR / f"{pid}.pdf"))
        else:
            txt_cache = TEXT_DIR / f"{pid}.abstract.txt"
            if txt_cache.exists():
                abstract = txt_cache.read_text()
            else:
                abstract = "" if offline else openalex_abstract(row)
                if abstract:
                    txt_cache.write_text(abstract)
            pages, source = ([f"{row['title']}\n\n{abstract}"] if abstract else []), "abstract"
        chunks = chunk_pages(pid, pages, s.chunk_size, s.chunk_overlap)
        # Chunk metadata: judul + venue selalu tersedia sebagai konteks
        chunks.insert(0, Chunk(f"{pid}-meta", pid, 0,
                               f"Judul: {row['title']}\nPenulis: {row['authors']} ({row['year']})\nVenue: {row['venue']} — {row['publisher']}"))
        all_chunks.extend(chunks)
        manifest.append({"id": pid, "source": source if pages else "metadata-only", "n_chunks": len(chunks)})
        print(f"  {pid}: {manifest[-1]['source']:<13} {len(chunks):>4} chunk")
    (TEXT_DIR / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in all_chunks))
    pd.DataFrame(manifest).to_csv(TEXT_DIR / "manifest.csv", index=False)
    return all_chunks


def load_chunks() -> list[Chunk]:
    lines = (TEXT_DIR / "chunks.jsonl").read_text().splitlines()
    return [Chunk(**json.loads(l)) for l in lines if l.strip()]
