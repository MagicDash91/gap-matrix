"""Tahap 5 — Ekspor: Excel (3 sheet) + Markdown."""
from __future__ import annotations

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import OUTPUT_DIR

HEAD = PatternFill("solid", fgColor="1F3A5F")
YES = PatternFill("solid", fgColor="D9EAD3")
NO = PatternFill("solid", fgColor="F4CCCC")


def _style(ws, widths: dict[int, int] | None = None, default: int = 28) -> None:
    for cell in ws[1]:
        cell.fill, cell.font = HEAD, Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = (widths or {}).get(col, default)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"


def to_excel(matrix: pd.DataFrame, aspects: pd.DataFrame, cov: pd.DataFrame, analysis: str, name: str = "gap_matrix") -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.xlsx"
    visible = matrix[[c for c in matrix.columns if not c.startswith("_bukti_")]]
    evidence = matrix[["ID"] + [c for c in matrix.columns if c.startswith("_bukti_")]]
    marks = aspects.copy()
    for c in marks.columns[1:]:
        marks[c] = marks[c].map({True: "✓", False: "✗"})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        visible.to_excel(xw, sheet_name="Gap Matrix", index=False)
        marks.to_excel(xw, sheet_name="Cakupan Aspek", index=False)
        cov.to_excel(xw, sheet_name="Ringkasan Gap", index=False)
        evidence.to_excel(xw, sheet_name="Bukti (evidence)", index=False)
        pd.DataFrame({"Analisis gap (draf LLM — wajib disunting)": analysis.split("\n\n")}).to_excel(
            xw, sheet_name="Analisis", index=False)
        _style(xw.sheets["Gap Matrix"], {1: 6, 2: 44, 3: 26}, default=34)
        _style(xw.sheets["Cakupan Aspek"], {1: 6}, default=14)
        for row in xw.sheets["Cakupan Aspek"].iter_rows(min_row=2, min_col=2):
            for cell in row:
                cell.fill = YES if cell.value == "✓" else NO
                cell.alignment = Alignment(horizontal="center")
        _style(xw.sheets["Ringkasan Gap"], {1: 32, 2: 14, 3: 12, 4: 60})
        _style(xw.sheets["Bukti (evidence)"], {1: 6}, default=40)
        _style(xw.sheets["Analisis"], {1: 120})
    return str(path)


def to_markdown(matrix: pd.DataFrame, cov: pd.DataFrame, analysis: str, name: str = "gap_matrix") -> str:
    visible = matrix[[c for c in matrix.columns if not c.startswith("_bukti_")]]
    md = ["# Literature Gap Matrix — RAG Domain Industri\n", visible.to_markdown(index=False),
          "\n\n## Ringkasan Cakupan Aspek\n", cov.to_markdown(index=False),
          "\n\n## Analisis Gap (draf)\n", analysis]
    path = OUTPUT_DIR / f"{name}.md"
    path.write_text("\n".join(md))
    return str(path)
