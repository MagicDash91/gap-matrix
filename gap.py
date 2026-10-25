"""Tahap 4 — Analisis gap: hitung cakupan aspek lalu minta LLM menulis draf analisis paragraf.

Catatan: keluaran LLM adalah DRAF. Wajib dibaca, diverifikasi terhadap matriks, dan disunting oleh kelompok.
"""
from __future__ import annotations

import pandas as pd

from .config import DOMAIN, PROMPT_DIR


def coverage(aspects: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in aspects.columns if c != "ID"]
    cov = pd.DataFrame({
        "Aspek": cols,
        "Jumlah paper": [int(aspects[c].sum()) for c in cols],
        "Persentase": [round(100 * aspects[c].mean(), 1) for c in cols],
        "Paper": [", ".join(aspects.loc[aspects[c], "ID"]) for c in cols],
    })
    return cov.sort_values("Jumlah paper").reset_index(drop=True)


def analyse(matrix: pd.DataFrame, cov: pd.DataFrame, llm, focus: str = "") -> str:
    keep = ["ID", "Paper", "Problem", "Metode", "Dataset", "Temuan", "Keterbatasan", "Future work"]
    compact = matrix[[c for c in keep if c in matrix.columns]].to_csv(index=False)
    prompt = (PROMPT_DIR / "gap_analysis.txt").read_text().format(
        focus=focus or "-", domain=DOMAIN, n=len(matrix), matrix=compact, coverage=cov.to_csv(index=False))
    return llm.generate_json(prompt).get("analysis", "")
