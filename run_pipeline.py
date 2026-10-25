"""RAG pipeline untuk menghasilkan Literature Gap Matrix (domain: RAG untuk asisten pengetahuan operasional industri).

Contoh:
    python run_pipeline.py                      # semua tahap
    python run_pipeline.py --stage ingest       # hanya unduh + chunking
    python run_pipeline.py --only P01 P07       # uji pada sebagian paper
    python run_pipeline.py --focus "Knowledge assistant operasional ISP"
    MOCK_LLM=1 python run_pipeline.py --offline # dry-run tanpa API & internet
"""
from __future__ import annotations

import argparse

from src import export, extract, gap, ingest
from src.config import OUTPUT_DIR, Settings
from src.retriever import HybridIndex
from src.llm import build_embedder, build_llm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "ingest", "index", "extract"], default="all")
    ap.add_argument("--only", nargs="*", help="ID paper tertentu, mis. P01 P02")
    ap.add_argument("--focus", default="", help="Fokus usulan paper kelompok untuk analisis gap")
    ap.add_argument("--offline", action="store_true", help="Jangan mengunduh (pakai PDF lokal saja)")
    ap.add_argument("--name", default="gap_matrix", help="Nama file keluaran")
    args = ap.parse_args()

    s = Settings()
    print(f"LLM: {s.llm_provider}/{s.llm_label} | Embedding lokal: {s.embed_model} | mock={s.mock}")

    if args.stage in ("all", "ingest", "index"):
        print("\n[1/5] Ingest & chunking")
        chunks = ingest.run(s, offline=args.offline)
        if args.stage == "ingest":
            return
    else:
        chunks = ingest.load_chunks()

    embedder = build_embedder(s)
    if args.stage in ("all", "index"):
        print(f"\n[2/5] Embedding {len(chunks)} chunk + membangun vector store numpy + BM25 manual")
        index = HybridIndex.build(chunks, embedder)
        if args.stage == "index":
            return
    else:
        index = HybridIndex.load(chunks)

    llm = build_llm(s)
    print("\n[3/5] Ekstraksi kolom gap matrix per paper")
    matrix, aspects = extract.run(s, index, llm, embedder, only=args.only)

    print("\n[4/5] Cakupan aspek & analisis gap")
    cov = gap.coverage(aspects)
    analysis = gap.analyse(matrix, cov, llm, focus=args.focus)

    print("\n[5/5] Ekspor")
    xlsx = export.to_excel(matrix, aspects, cov, analysis, name=args.name)
    md = export.to_markdown(matrix, cov, analysis, name=args.name)
    matrix.to_csv(OUTPUT_DIR / f"{args.name}.csv", index=False)
    print(f"Selesai:\n  {xlsx}\n  {md}")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
