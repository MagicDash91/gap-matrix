"""Tanya-jawab RAG atas 20 artikel yang sudah diindeks (bersitasi, Python murni).

Jalankan run_pipeline.py --stage index sekali, lalu:
    python ask.py "Metode retrieval apa yang paling sering dipakai untuk tiket troubleshooting?"
    python ask.py                # mode interaktif
    python ask.py --paper P11 "Berapa persen chunk benar masuk top-4?"
"""
from __future__ import annotations

import argparse

from src import ingest
from src.config import DOMAIN, PROMPT_DIR, Settings
from src.llm import build_embedder, build_llm
from src.retriever import HybridIndex


def answer(question: str, index: HybridIndex, llm, embedder, k: int, paper: str | None = None) -> dict:
    hits = index.search(question, embedder, paper_id=paper, k=k)
    context = "\n\n".join(f"[{c.paper_id}, hlm. {c.page}] ({c.chunk_id})\n{c.text}" for c in hits)
    prompt = (PROMPT_DIR / "answer.txt").read_text().format(domain=DOMAIN, question=question, context=context)
    out = llm.generate_json(prompt)
    out["retrieved"] = [f"{c.chunk_id} (hlm. {c.page})" for c in hits]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--paper", help="Batasi pencarian pada satu paper, mis. P05")
    args = ap.parse_args()

    s = Settings()
    index = HybridIndex.load(ingest.load_chunks())
    llm, embedder = build_llm(s), build_embedder(s)

    def show(q: str) -> None:
        res = answer(q, index, llm, embedder, k=s.top_k, paper=args.paper)
        print(f"\n{res.get('answer', '')}\n")
        print("Sitasi :", ", ".join(res.get("citations", [])) or "-")
        print("Chunk  :", "; ".join(res["retrieved"]))

    if args.question:
        show(" ".join(args.question))
        return
    print("Ketik pertanyaan (kosongkan untuk keluar).")
    while q := input("\n> ").strip():
        show(q)


if __name__ == "__main__":
    main()
