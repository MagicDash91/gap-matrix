"""Tahap 3 — Ekstraksi: untuk tiap paper × kolom, retrieve chunk paper itu lalu minta LLM mengisi sel (dengan bukti)."""
from __future__ import annotations

import json

import pandas as pd

from .config import ASPECTS, DOMAIN, FIELDS, OUTPUT_DIR, PROMPT_DIR, Settings
from .retriever import HybridIndex
from .ingest import Chunk, load_papers

CACHE = OUTPUT_DIR / "extraction_cache.jsonl"


def _context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id} | hlm. {c.page}]\n{c.text}" for c in chunks)


def _load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    rows = [json.loads(l) for l in CACHE.read_text().splitlines() if l.strip()]
    return {r["key"]: r for r in rows}


def _save(rec: dict) -> None:
    with CACHE.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(s: Settings, index: HybridIndex, llm, embedder, only: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tpl_field = (PROMPT_DIR / "extract_field.txt").read_text()
    tpl_aspect = (PROMPT_DIR / "extract_aspects.txt").read_text()
    cache = _load_cache()           # bisa dilanjutkan bila terputus (rate limit / kuota)
    papers = load_papers()
    if only:
        papers = papers[papers["id"].isin(only)]

    matrix_rows, aspect_rows = [], []
    for _, p in papers.iterrows():
        print(f"→ {p['id']}  {p['title'][:70]}")
        meta = {"title": p["title"], "authors": p["authors"], "year": p["year"], "domain": DOMAIN}
        row = {"ID": p["id"], "Paper": f"{p['authors']} ({p['year']}). {p['title']}. {p['venue']}",
               "DOI": p["doi"] or p["url"]}

        # ---- kolom deskriptif --------------------------------------------
        for f in FIELDS:
            key = f"{p['id']}::{f.key}"
            if key not in cache:
                hits = index.search(f.query, embedder, paper_id=p["id"], k=s.top_k)
                out = llm.generate_json(tpl_field.format(label=f.label, instruction=f.instruction,
                                                         context=_context(hits), **meta))
                cache[key] = {"key": key, "value": out.get("value", ""), "evidence": out.get("evidence", ""),
                              "chunk_ids": out.get("chunk_ids", [])}
                _save(cache[key])
            row[f.label] = cache[key]["value"]
            row[f"_bukti_{f.key}"] = cache[key]["evidence"]
        matrix_rows.append(row)

        # ---- aspek biner (satu panggilan per paper) ------------------------
        key = f"{p['id']}::aspects"
        if key not in cache:
            seen, hits = set(), []
            for a in ASPECTS:   # gabungan chunk teratas dari tiap pertanyaan aspek
                for c in index.search(a.question + " " + a.label, embedder, paper_id=p["id"], k=2):
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        hits.append(c)
            aspects_txt = "\n".join(f"- {a.key}: {a.question}" for a in ASPECTS)
            out = llm.generate_json(tpl_aspect.format(aspects=aspects_txt, context=_context(hits[:14]), **meta))
            cache[key] = {"key": key, "aspects": out}
            _save(cache[key])
        arow = {"ID": p["id"]}
        for a in ASPECTS:
            v = cache[key]["aspects"].get(a.key, {})
            arow[a.label] = bool(v.get("present")) if isinstance(v, dict) else bool(v)
        aspect_rows.append(arow)

    return pd.DataFrame(matrix_rows), pd.DataFrame(aspect_rows)
